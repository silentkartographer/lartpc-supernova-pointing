# Authored by Hilary Utaegbulam

"""Train the two-point DSNT head-tail model (UNet++ / DenseNet169, one network per view).

Each view predicts Point A (the muon-Michel vertex, supervised) and Point B (unsupervised; it only
has to make A->B point along the electron's initial direction). Training couples the three views
through cross-view tick constraints; see docs/losses.md.
"""
from __future__ import annotations
import os
import random
import argparse
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime

from .constants import Config, PARTITION_SEED
from .training_utils import (
    replace_cfg, save_checkpoint, load_checkpoint, build_models,
)
from .models import DenseModel, DenseModelSMP, DenseModelConvNeXt
from .dataset import make_dense_loader
from .training import run_epoch_dense, run_epoch_multi
from .metrics import _fmt, _fmt_multi, _fmt_perview, _csv_log_epoch
from .evaluation import (
    run_testing, test_worker_single_view, run_head_tail_analysis,
    plot_multiview_error_correlation, plot_multiview_consistency,
)
from .visualization import save_consistency_analysis_plots
from multiprocessing import Manager
import traceback
import pickle

def main_worker_single_view(gpu, cfg, view):
    """
    DDP worker for single-view training.
    Each GPU processes a portion of the batch for one view.
    """
    rank = gpu
    world_size = torch.cuda.device_count()
    
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(minutes=30)
    )
    
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    
    if rank == 0:
        print(f"\n==== View {view} (DDP with {world_size} GPUs) ====")
    
    # Data loaders with DDP
    train_ds, train_ld = make_dense_loader(
        'train', view, cfg,
        is_distributed=True, rank=rank, world_size=world_size
    )
    val_ds, val_ld = make_dense_loader(
        'val', view, cfg,
        is_distributed=False, rank=rank, world_size=world_size
    )
    
    # Build model 
    models = build_models(cfg, device, use_sync_bn=True)
    model = models["dense"]
    model = DDP(model, device_ids=[gpu])
    
    view_dir = os.path.join(cfg.out_dir, f"runs_{view}")
    if rank == 0:
        os.makedirs(view_dir, exist_ok=True)
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5, verbose=(rank==0))
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if cfg.resume:
        if cfg.resume_path and os.path.exists(cfg.resume_path):
            ckpt_path = cfg.resume_path
        else:
            ckpt_path = os.path.join(view_dir, 'last_checkpoint.pt')
        
        if os.path.exists(ckpt_path):
            start_epoch, best_val_loss = load_checkpoint(
                ckpt_path, {'dense': model.module}, {'dense': opt}, {'dense': scheduler}, device
            )
            start_epoch += 1 
    
    warmup_epochs = 20
    early_stopping_patience = 15 #10
    
    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        train_ld.sampler.set_epoch(epoch)
        
        # Train
        tr_logs = run_epoch_dense(
            model, train_ld, opt, device, cfg, epoch, cfg.epochs, train=True, is_distributed=True
        )
        
        # Validation
        va_logs = run_epoch_dense(
            model, val_ld, None, device, cfg, epoch, cfg.epochs, train=False, is_distributed=False
        )
        val_loss = va_logs["loss"]

        dist.barrier() # Sync before saving/logging

        # Generate validation plots periodically (this breaks ddp b/c no gather)
        # if cfg.viz_val and (epoch % 10 == 0):
        #     val_plot_dir = os.path.join(view_dir, "val_plots")
        #     save_validation_plots(model.module, val_ds, device, cfg, epoch, val_plot_dir, k=cfg.viz_k)

        
        # LR scheduling
        scheduler.step(val_loss)
        
        # Print (rank 0 only)
        if rank == 0:
            msg = f"[{view}] Epoch {epoch:03d} | TR({_fmt(tr_logs)}) | VA({_fmt(va_logs)})"
            print(msg)
        
        if rank == 0:
            # Save last
            save_checkpoint({
                'dense': {
                    'model': model.module.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                },
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'rng_states': {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all(),
                }
            }, os.path.join(view_dir, 'last_checkpoint.pt'))
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                save_checkpoint({
                    'dense': {
                        'model': model.module.state_dict(),
                        'optimizer': opt.state_dict(),
                        'scheduler': scheduler.state_dict(),
                    },
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'rng_states': {
                        'python': random.getstate(),
                        'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all(),
                    }
                }, os.path.join(view_dir, 'best_checkpoint.pt'))
                print(f"New best validation loss: {best_val_loss:.4f}. Checkpoint saved.")
            else:
                if epoch > warmup_epochs:
                    epochs_no_improve += 1
                    print(f"Val loss did not improve for {epochs_no_improve} epoch(s). Best is {best_val_loss:.4f}.")
            
            # Early stopping
            if epoch > warmup_epochs and epochs_no_improve >= early_stopping_patience:
                print(f"\n    !!! Early stopping triggered after {early_stopping_patience} epochs with no improvement. !!!")
                break
        
        # Sync all processes
        dist.barrier()
        
        # Broadcast early stopping decision
        if epoch > warmup_epochs:
            stop_flag = torch.tensor(epochs_no_improve >= early_stopping_patience, dtype=torch.int, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                break
    
    if rank == 0:
        print(f"[{view}] Training complete. Best val loss={best_val_loss:.4f}")
    
    dist.destroy_process_group()


def main_worker_multi_view(gpu, cfg):
    """
    DDP worker for multi-view training.
    All GPUs process batches for all three views together.
    """
    rank = gpu
    world_size = torch.cuda.device_count()
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:29500',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(gpu)
    device = torch.device(f'cuda:{gpu}')
    
    if rank == 0:
        print(f"Multi-View Training (DDP with {world_size} GPUs)")
    
    if rank == 0:
        print("\nLoading data...")
    train_ds, train_ld = make_dense_loader(
        'train', view=None, cfg=cfg,
        is_distributed=True, rank=rank, world_size=world_size
    )
    val_ds, val_ld = make_dense_loader(
        'val', view=None, cfg=cfg,
        is_distributed=False, rank=rank, world_size=world_size
    )
    
    if rank == 0:
        print(f"Train samples: {len(train_ds)}")
        print(f"Val samples: {len(val_ds)}")
        print("\nBuilding models...")
    
    # Build models (one per view) 
    models = {}
    for v in ['U', 'V', 'Z']:
        arch = cfg.dense_arch.lower()
        
        if arch == "convnextv2_unet":
            model = DenseModelConvNeXt(
                in_ch=1, feat_ch=cfg.dense_feat_ch, num_classes=cfg.num_classes,
                enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
                variant=cfg.cnv2_variant, decoder_scale=cfg.cnv2_decoder_scale,
                use_transpose=cfg.cnv2_use_transpose, skip_proj=cfg.cnv2_skip_proj,
            )
        elif arch == "smp_unet":
            model = DenseModelSMP(
                in_ch=cfg.smp_in_channels, feat_ch=cfg.dense_feat_ch, num_classes=cfg.num_classes,
                enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg,
                encoder_name=cfg.smp_encoder, encoder_weights=cfg.smp_encoder_weights,
                decoder_channels=cfg.smp_decoder_channels,
            )
        else:
            model = DenseModel(
                in_ch=1, base=cfg.dense_base, feat_ch=cfg.dense_feat_ch,
                num_classes=cfg.num_classes, enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg
            )
        
        # Convert to SyncBatchNorm and wrap with DDP
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)
        models[v] = model
        
        if rank == 0:
            print(f"  View {v}: {arch}")
    
    # Optimizers & Schedulers
    if rank == 0:
        print("\nSetting up optimizers...")
    opts = {v: torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=1e-4) 
            for v, m in models.items()}
    schedulers = {v: ReduceLROnPlateau(o, mode='min', factor=0.1, patience=5, verbose=(rank==0)) 
                  for v, o in opts.items()}
    
    # Output directory
    view_dir = os.path.join(cfg.out_dir, "runs_multi")
    if rank == 0:
        os.makedirs(view_dir, exist_ok=True)
        print(f"\nOutput directory: {view_dir}")
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    if cfg.resume:
        if cfg.resume_path and os.path.exists(cfg.resume_path):
            ckpt_path = cfg.resume_path
        else:
            ckpt_path = os.path.join(view_dir, 'last_checkpoint.pt')
        
        if os.path.exists(ckpt_path):
            # Extract module from DDP wrapper for loading
            models_unwrapped = {v: m.module for v, m in models.items()}
            start_epoch, best_val_loss = load_checkpoint(
                ckpt_path, models_unwrapped, opts, schedulers, device
            )
            start_epoch += 1
    
    warmup_epochs = 20
    early_stopping_patience = 15
    
    if rank == 0:
        print(f"\nTraining configuration:")
        print(f"  Epochs: {cfg.epochs}")
        print(f"  Batch size: {cfg.batch_size}")
        print(f"  Learning rate: {cfg.lr}")
        print(f"  X-constraint weight: {cfg.lambda_x_constraint}")
        print(f"  X-GT weight: {cfg.lambda_x_gt}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print("\n" + "="*80)
    
    # Training loop
    val_consistency_metrics_over_time = []
    for epoch in range(start_epoch, cfg.epochs):
        # Set epoch for distributed sampler
        train_ld.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\nEpoch {epoch}/{cfg.epochs}")
            print("-" * 80)
        
        # Train
        tr_logs = run_epoch_multi(
            models, train_ld, opts, device, cfg, epoch, cfg.epochs, train=True, is_distributed=True
        )
        
        # Validation
        va_logs = run_epoch_multi(
            models, val_ld, None, device, cfg, epoch, cfg.epochs, train=False, is_distributed=False
        )
        val_loss = va_logs["loss"]
        # Accumulate consistency metrics
        if getattr(cfg, 'use_consistency_checks', False):
            if 'consistency_metrics' in va_logs:
                val_consistency_metrics_over_time.append(va_logs['consistency_metrics'])

        dist.barrier()

        # Generate validation plots periodically; breaks ddp because no gather
        # if cfg.viz_val and (epoch % 10 == 0):
        #     for v in ['U', 'V', 'Z']:
        #         val_plot_dir = os.path.join(view_dir, f"val_plots_{v}")
        #         save_validation_plots(models[v].module, val_ds.ds[v], device, cfg, epoch, val_plot_dir, k=cfg.viz_k)
        
        
        
        # LR scheduling
        for sch in schedulers.values():
            sch.step(val_loss)
        
        if rank == 0:
            print(f"[Multi] TR: {_fmt_multi(tr_logs)}")
            if isinstance(tr_logs.get("per_view", None), dict):
                for v in ("U","V","Z"):
                    if v in tr_logs["per_view"]:
                        print("    " + _fmt_perview(f"TR-{v}", tr_logs["per_view"][v]))

            print(f"[Multi] VA: {_fmt_multi(va_logs)}")
            if isinstance(va_logs.get("per_view", None), dict):
                for v in ("U","V","Z"):
                    if v in va_logs["per_view"]:
                        print("    " + _fmt_perview(f"VA-{v}", va_logs["per_view"][v]))
        
        # Save checkpoints (rank 0 only)
        if rank == 0:
            # Save last
            save_checkpoint({
                'U': {
                    'model': models['U'].module.state_dict(),
                    'optimizer': opts['U'].state_dict(),
                    'scheduler': schedulers['U'].state_dict(),
                },
                'V': {
                    'model': models['V'].module.state_dict(),
                    'optimizer': opts['V'].state_dict(),
                    'scheduler': schedulers['V'].state_dict(),
                },
                'Z': {
                    'model': models['Z'].module.state_dict(),
                    'optimizer': opts['Z'].state_dict(),
                    'scheduler': schedulers['Z'].state_dict(),
                },
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'rng_states': {
                    'python': random.getstate(),
                    'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all(),
                }
            }, os.path.join(view_dir, 'last_checkpoint.pt'))
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                
                save_checkpoint({
                    'U': {
                        'model': models['U'].module.state_dict(),
                        'optimizer': opts['U'].state_dict(),
                        'scheduler': schedulers['U'].state_dict(),
                    },
                    'V': {
                        'model': models['V'].module.state_dict(),
                        'optimizer': opts['V'].state_dict(),
                        'scheduler': schedulers['V'].state_dict(),
                    },
                    'Z': {
                        'model': models['Z'].module.state_dict(),
                        'optimizer': opts['Z'].state_dict(),
                        'scheduler': schedulers['Z'].state_dict(),
                    },
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'rng_states': {
                        'python': random.getstate(),
                        'numpy': np.random.get_state(),
                        'torch': torch.get_rng_state(),
                        'cuda': torch.cuda.get_rng_state_all(),
                    }
                }, os.path.join(view_dir, 'best_checkpoint.pt'))
                
                print(f"  -> ✓ New best validation loss: {best_val_loss:.4f}. Checkpoints saved.")
            else:
                if epoch > warmup_epochs:
                    epochs_no_improve += 1
                    print(f"  -> Val loss did not improve for {epochs_no_improve} epoch(s). Best: {best_val_loss:.4f}")
            
            # Early stopping
            if epoch > warmup_epochs and epochs_no_improve >= early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"!!! Early stopping triggered after {early_stopping_patience} epochs with no improvement !!!")
                print(f"{'='*80}")
        
        # Sync all processes
        dist.barrier()
        
        # Broadcast early stopping decision
        if epoch > warmup_epochs:
            stop_flag = torch.tensor(epochs_no_improve >= early_stopping_patience, dtype=torch.int, device=device)
            dist.broadcast(stop_flag, src=0)
            if stop_flag.item():
                break
    
    if rank == 0:
        print(f"[Multi] Training complete. Best val loss: {best_val_loss:.4f}")


    if rank == 0 and getattr(cfg, 'use_consistency_checks', False):
        if len(val_consistency_metrics_over_time) > 0:
            metrics_file = os.path.join(cfg.out_dir, 'runs_multi', 'consistency_metrics.pkl')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            
            metrics_to_save = []
            for m in val_consistency_metrics_over_time:
                m_copy = {}
                for k, v in m.items():
                    if k == 'per_view':
                        m_copy[k] = {}
                        for view, view_data in v.items():
                            m_copy[k][view] = {}
                            for metric_name, metric_val in view_data.items():
                                if isinstance(metric_val, np.ndarray):
                                    m_copy[k][view][metric_name] = metric_val
                                elif torch.is_tensor(metric_val):
                                    m_copy[k][view][metric_name] = metric_val.cpu().numpy()
                                else:
                                    m_copy[k][view][metric_name] = metric_val
                    else:
                        m_copy[k] = v
                metrics_to_save.append(m_copy)
            
            with open(metrics_file, 'wb') as f:
                pickle.dump(metrics_to_save, f)
            
            print(f"\n Saved consistency metrics to: {metrics_file}")
    
    dist.destroy_process_group()

def main(cfg: Config):
    # Set random seeds
    random.seed(PARTITION_SEED)
    np.random.seed(PARTITION_SEED)
    torch.manual_seed(PARTITION_SEED)
    torch.cuda.manual_seed_all(PARTITION_SEED)
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This DDP script requires GPUs.")
        return
    
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    
    os.makedirs(cfg.out_dir, exist_ok=True)
    
    # TRAINING PHASE 
    if cfg.multi_view:
        print("\n>>> MULTI-VIEW MODE <<<\n")
        
        mp.spawn(
            main_worker_multi_view,
            args=(cfg,),
            nprocs=n_gpus,
            join=True
        )

        # CREATE CONSISTENCY PLOTS AFTER TRAINING 
        if getattr(cfg, 'use_consistency_checks', False):
            metrics_file = os.path.join(cfg.out_dir, 'runs_multi', 'consistency_metrics.pkl')
            if os.path.exists(metrics_file):
                print("Creating A→B consistency analysis plots...")
                with open(metrics_file, 'rb') as f:
                    val_consistency_metrics_over_time = pickle.load(f)
                
                plot_dir = os.path.join(cfg.out_dir, 'runs_multi', 'consistency_analysis')
                save_consistency_analysis_plots(
                    metrics_over_time=val_consistency_metrics_over_time,
                    save_dir=plot_dir,
                    prefix="val_"
                )
                
                print(f"Consistency plots saved to: {plot_dir}")
        
        # TESTING PHASE (Parallel: one GPU per view)
        print("Starting parallel testing (U→GPU0, V→GPU1, Z→GPU2)...")
        
        if n_gpus >= 3:
            # Parallel testing with shared results
            manager = Manager()
            results_dict = manager.dict()
            
            ctx = mp.get_context('spawn')
            
            processes = []
            for i, view in enumerate(['U', 'V', 'Z']):
                p = ctx.Process(target=test_worker_single_view, args=(i, view, cfg, results_dict))
                p.start()
                processes.append(p)
            
            # Wait for all to finish
            for p in processes:
                p.join()
            
            # MULTI-VIEW ANALYSIS (NO CSV LOADING)
            print("MULTI-VIEW CORRELATION ANALYSIS")
            
            if len(results_dict) == 3:
                view_errors = {v: results_dict[v]['angular_errors'] for v in ['U', 'V', 'Z']}
                view_predictions = {v: results_dict[v]['u_pred'] for v in ['U', 'V', 'Z']}
                
                n_U = len(view_errors['U'])
                n_V = len(view_errors['V'])
                n_Z = len(view_errors['Z'])
                
                if n_U == n_V == n_Z:
                    print(f"  Found {n_U} common events across U/V/Z")
                    
                    save_dir = os.path.join(cfg.out_dir, "runs_multi", "test_plots")
                    
                    # Plot error correlation
                    plot_multiview_error_correlation(
                        view_errors,
                        os.path.join(save_dir, "multiview_error_correlation.png")
                    )
                    
                    # Plot 3-view consistency
                    plot_multiview_consistency(
                        view_predictions,
                        os.path.join(save_dir, "multiview_consistency.png")
                    )
                    
                    print("Multi-view analysis complete!")
                else:
                    print(f"WARNING: View counts don't match: U={n_U}, V={n_V}, Z={n_Z}")
            else:
                print("WARNING: Not all views completed successfully.")
            
        else:
            print(f"WARNING: Only {n_gpus} GPU(s) available. Running tests sequentially on GPU 0.")
            device = torch.device('cuda:0')
            results_dict = {}
            
            for view in ['U', 'V', 'Z']:
                try:
                    result = run_testing(view, device, cfg)
                    results_dict[view] = result
                except Exception as e:
                    print(f"[{view}] ERROR during testing: {e}")
                    import traceback
                    traceback.print_exc()
            
            # MULTI-VIEW ANALYSIS (NO CSV LOADING)
            if len(results_dict) == 3:
                print("MULTI-VIEW CORRELATION ANALYSIS")
                
                view_errors = {v: results_dict[v]['angular_errors'] for v in ['U', 'V', 'Z']}
                view_predictions = {v: results_dict[v]['u_pred'] for v in ['U', 'V', 'Z']}
                
                n_U = len(view_errors['U'])
                n_V = len(view_errors['V'])
                n_Z = len(view_errors['Z'])
                
                if n_U == n_V == n_Z:
                    print(f"  Found {n_U} common events across U/V/Z")
                    
                    save_dir = os.path.join(cfg.out_dir, "runs_multi", "test_plots")
                    
                    plot_multiview_error_correlation(
                        view_errors,
                        os.path.join(save_dir, "multiview_error_correlation.png")
                    )
                    
                    plot_multiview_consistency(
                        view_predictions,
                        os.path.join(save_dir, "multiview_consistency.png")
                    )
                    
                    print("  Multi-view analysis complete!")
                else:
                    print(f"  WARNING: View counts don't match: U={n_U}, V={n_V}, Z={n_Z}")
    
    else:
        print("\n>>> SINGLE-VIEW MODE <<<\n")
        
        # Train each view sequentially using all GPUs 
        for view in cfg.views:
            try:
                print(f"Training view: {view}")
                
                mp.spawn(
                    main_worker_single_view,
                    args=(cfg, view),
                    nprocs=n_gpus,
                    join=True
                )
                
                # Test on GPU 0 after training
                print(f"\nTesting view: {view}")
                device = torch.device('cuda:0')
                run_testing(view, device, cfg)
                
            except Exception as e:
                print(f"[{view}] ERROR: {e}")
                traceback.print_exc()

    # HEAD-TAIL DISAMBIGUATION ANALYSIS
    if cfg.run_head_tail_analysis:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        run_head_tail_analysis(cfg, device)


def str2bool(v):
    """Parse boolean values from command line."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected, got {v}')

def parse_tuple_float(s):
    return tuple(float(x.strip()) for x in s.split(','))

def parse_tuple_int(s):
    return tuple(int(x.strip()) for x in s.split(','))

def parse_tuple_str(s):
    return tuple(x.strip() for x in s.split(','))

def parse_optional_float(s):
    if s is None or s.lower() in ('none', 'null'):
        return None
    return float(s)

def parse_optional_str(s):
    if s is None or s.lower() in ('none', 'null'):
        return None
    return s

def parse_optional_tuple_int(s):
    if s is None or s.lower() in ('none', 'null'):
        return None
    return tuple(int(x.strip()) for x in s.split(','))

def build_parser():
    p = argparse.ArgumentParser(
        description="Direction-Only Training with Optional Segmentation (Dense, DSNT Vector Head)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data paths
    p.add_argument('--preprocessed_root', type=str, required=True,
                    help='Path to preprocessed NPZ directory')
    p.add_argument('--out_dir', type=str, required=True,
                    help='Output directory for runs, checkpoints, plots')

    # Views
    p.add_argument('--views', type=parse_tuple_str, default='U,V,Z',
                    help='Comma-separated list of views, e.g. U,V,Z')
    p.add_argument('--use_raw_bg', type=str2bool, default=True,
                    help='Load raw background data for visualization')

    # Resume
    p.add_argument('--resume', type=str2bool, default=False,
                    help='Resume training from checkpoint')
    p.add_argument('--resume_path', type=parse_optional_str, default=None,
                    help='Path to checkpoint file (or "none")')

    # Partition
    p.add_argument('--partition_frac', type=float, default=1.0)
    p.add_argument('--partition_seed', type=int, default=12345)
    p.add_argument('--train_fraction', type=float, default=1.0)

    # Training
    p.add_argument('--epochs', type=int, default=1000000000000,
                    help='Max epochs (relies on early stopping)')
    p.add_argument('--batch_size', type=int, default=6)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', type=str2bool, default=True,
                    help='Automatic mixed precision')

    # Augmentation & normalization
    p.add_argument('--augment', type=str2bool, default=False)
    p.add_argument('--zscore_norm', type=str2bool, default=False)

    # Vector heads
    p.add_argument('--enable_dirreg', type=str2bool, default=False)
    p.add_argument('--lambda_ang', type=float, default=4.2)
    p.add_argument('--dirreg_weight', type=float, default=0.0)

    # DSNT stabilizers
    p.add_argument('--dsnt_temp', type=float, default=1.0)
    p.add_argument('--sep_margin_px', type=float, default=2.0)
    p.add_argument('--overlap_weight', type=float, default=0.01)
    p.add_argument('--entropy_weight', type=float, default=0.001)

    # Segmentation
    p.add_argument('--enable_seg', type=str2bool, default=True)
    p.add_argument('--enable_brems', type=str2bool, default=True)
    p.add_argument('--num_classes', type=int, default=5)
    p.add_argument('--lambda_seg', type=float, default=0.5)
    p.add_argument('--seg_dice_weight', type=float, default=1.0)
    p.add_argument('--seg_loss_type', type=str, default='focal+dice')
    p.add_argument('--seg_focal_gamma', type=float, default=2.0)
    p.add_argument('--seg_focal_alpha', type=parse_optional_float, default=0.25,
                    help='Focal alpha (float or "none")')

    # Priors
    p.add_argument('--prior_mode', type=str, default='off')
    p.add_argument('--prior_alpha', type=float, default=0.3)
    p.add_argument('--prior_topk_percent', type=float, default=20)

    # Dense model
    p.add_argument('--dense_base', type=int, default=32)
    p.add_argument('--dense_arch', type=str, default='smp_unet')
    p.add_argument('--dense_feat_ch', type=int, default=256,
                    help='Dense feature channels (64*4=256)')

    # SMP options
    p.add_argument('--smp_encoder', type=str, default='densenet169')
    p.add_argument('--smp_encoder_weights', type=parse_optional_str, default=None,
                    help='SMP encoder pretrained weights (or "none")')
    p.add_argument('--smp_in_channels', type=int, default=1)
    p.add_argument('--smp_decoder_channels', type=parse_tuple_int, default='256,128,64,32,16')
    p.add_argument('--smp_use_deeplabv3', type=str2bool, default=False)

    # # ConvNeXtV2-UNet knobs
    # p.add_argument('--cnv2_variant', type=str, default='femto')
    # p.add_argument('--cnv2_decoder_scale', type=float, default=1)
    # p.add_argument('--cnv2_use_transpose', type=str2bool, default=False)
    # p.add_argument('--cnv2_skip_proj', type=str2bool, default=True)

    # Ground truth
    p.add_argument('--gt_vector_mode', type=str, default='truth_momentum_pdf')
    p.add_argument('--use_truth_vertex', type=str2bool, default=True)
    p.add_argument('--vertex_tick_mode', type=str, default='relative')
    p.add_argument('--t0_offset_ticks', type=int, default=0)
    p.add_argument('--wire_index_base', type=int, default=0)

    # Visualization
    p.add_argument('--viz_val', type=str2bool, default=True)
    p.add_argument('--viz_k', type=int, default=10)

    # Bragg-aware figure settings
    p.add_argument('--bragg_make', type=str2bool, default=True)
    p.add_argument('--bragg_dpi', type=int, default=600)
    p.add_argument('--bragg_contour_levels', type=int, default=8)
    p.add_argument('--bragg_contour_alpha', type=float, default=0.28)
    p.add_argument('--bragg_topk_percent', type=float, default=10.0)
    p.add_argument('--bragg_zscore_patch', type=int, default=7)
    p.add_argument('--bragg_roi', type=parse_optional_tuple_int, default=None,
                    help='ROI as y0,x0,y1,x1 or "none"')
    p.add_argument('--plot_show_error_text', type=str2bool, default=True)
    p.add_argument('--resize_to', type=parse_optional_tuple_int, default=None,
                    help='Resize images to H,W or "none"')

    # Physics-informed loss
    p.add_argument('--use_brems_A', type=str2bool, default=True)
    p.add_argument('--use_brems_B', type=str2bool, default=False)
    p.add_argument('--use_brems_C', type=str2bool, default=True)
    p.add_argument('--lambda_brems', type=float, default=10.0)
    p.add_argument('--brems_margin', type=float, default=0.10)
    p.add_argument('--brems_lambda_min', type=float, default=0.25)
    p.add_argument('--brems_C_mode', type=str, default='twoheaded')
    p.add_argument('--brems_C_T', type=float, default=0.5)
    p.add_argument('--brems_C_gamma', type=float, default=1.0)
    p.add_argument('--brems_C_K', type=int, default=64)
    p.add_argument('--brems_inner_weights', type=parse_tuple_float, default='1.0,1.0,1.0')
    p.add_argument('--brems_warmup_epochs', type=int, default=10)

    # Channel ordering
    p.add_argument('--dsnt_order_mode', type=str, default='fixed')
    p.add_argument('--perm_inv_warmup_epochs', type=int, default=5)

    # Auxiliary heads/losses
    p.add_argument('--lambda_coord', type=float, default=0.10)
    p.add_argument('--lambda_kl', type=float, default=0.05)
    p.add_argument('--hm_sigma', type=float, default=1.5)

    # DSNT temperature + loss schedules
    p.add_argument('--dsnt_temp_hi', type=float, default=2.0)
    p.add_argument('--dsnt_temp_lo', type=float, default=1.0)
    p.add_argument('--overlap_weight0', type=parse_optional_float, default=None)
    p.add_argument('--entropy_weight0', type=parse_optional_float, default=None)
    p.add_argument('--sep_margin_px0', type=parse_optional_float, default=None)

    # Multi-view training
    p.add_argument('--multi_view', type=str2bool, default=True)
    p.add_argument('--lambda_x_constraint', type=float, default=200)
    p.add_argument('--lambda_x_gt', type=float, default=1.0)

    # Ray-based losses
    p.add_argument('--lambda_ray_perp', type=float, default=1.0)
    p.add_argument('--lambda_ray_fwd', type=float, default=1.0)
    p.add_argument('--lambda_ray_len', type=float, default=0.1)
    p.add_argument('--lambda_ray_cos', type=float, default=0.5)
    p.add_argument('--ray_t_min', type=float, default=0.0)
    p.add_argument('--ray_warmup_epochs', type=float, default=0.0)

    # Analysis
    p.add_argument('--run_head_tail_analysis', type=str2bool, default=True)
    p.add_argument('--skip_per_event_plots', type=str2bool, default=False)
    p.add_argument('--test_plot_fraction', type=float, default=0.10)

    # Consistency pipeline
    p.add_argument('--use_consistency_checks', type=str2bool, default=True)
    p.add_argument('--consistency_flip_threshold', type=float, default=0.90)
    p.add_argument('--consistency_drop_threshold', type=float, default=0.25)
    p.add_argument('--lambda_consistency', type=float, default=0.5)

    # # Label-invariant augmentations
    # p.add_argument('--aug_dead_stripes', type=str2bool, default=True)
    # p.add_argument('--aug_dead_p_col', type=float, default=0.2)
    # p.add_argument('--aug_dead_p_row', type=float, default=0.2)
    # p.add_argument('--aug_dead_max_cols', type=int, default=6)
    # p.add_argument('--aug_dead_max_rows', type=int, default=6)
    # p.add_argument('--aug_random_erasing', type=str2bool, default=True)
    # p.add_argument('--aug_erasing_p', type=float, default=0.2)
    # p.add_argument('--aug_erasing_area', type=parse_tuple_float, default='0.005,0.03')
    # p.add_argument('--aug_bragg_jitter', type=str2bool, default=True)
    # p.add_argument('--aug_bragg_p', type=float, default=0.3)
    # p.add_argument('--aug_bragg_percentile', type=float, default=95.0)
    # p.add_argument('--aug_bragg_scale', type=parse_tuple_float, default='0.8,1.2')
    # p.add_argument('--aug_gain_offset', type=str2bool, default=True)
    # p.add_argument('--aug_gain_range', type=parse_tuple_float, default='0.9,1.1')
    # p.add_argument('--aug_offset_range', type=parse_tuple_float, default='-0.02,0.02')
    # p.add_argument('--aug_gaussian_noise', type=str2bool, default=False)
    # p.add_argument('--aug_noise_sigma', type=float, default=0.02)

    # Multi-view tick consistency
    p.add_argument('--lambda_tick_sign', type=float, default=10.0)
    p.add_argument('--lambda_tick_val', type=float, default=5.0)
    p.add_argument('--tick_tau', type=float, default=2.0)
    p.add_argument('--tick_scale', type=float, default=250.0)

    return p

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    parser = build_parser()
    cfg = parser.parse_args()

    # Initialize schedule baselines
    if cfg.overlap_weight0 is None: cfg.overlap_weight0 = cfg.overlap_weight
    if cfg.entropy_weight0 is None: cfg.entropy_weight0 = cfg.entropy_weight
    if cfg.sep_margin_px0 is None:  cfg.sep_margin_px0  = cfg.sep_margin_px

    main(cfg)

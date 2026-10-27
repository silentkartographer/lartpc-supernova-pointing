# Authored by Hilary Utaegbulam

from __future__ import annotations
import os
import copy
import random
import numpy as np
import torch
from typing import Optional

from .constants import Config
from .models import (
    DenseModel, DenseModelSMP, DenseModelConvNeXt,
)

def replace_cfg(cfg, **kwargs):
    new = copy.copy(cfg)
    for k, v in kwargs.items():
        setattr(new, k, v)
    return new

def save_checkpoint(state, filename):
    torch.save(state, filename)
    
def load_checkpoint(checkpoint_path, models, opts, schedulers, device):
    if not os.path.exists(checkpoint_path):
        print(f"    -> No checkpoint found at {checkpoint_path}")
        return 0, float('inf')
    
    print(f"    -> Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    for view in models.keys():
        if view in checkpoint:
            model_state = checkpoint[view]['model']
            if list(model_state.keys())[0].startswith('module.'):
                model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
            models[view].load_state_dict(model_state)
            
            if view in opts and 'optimizer' in checkpoint[view]:
                opts[view].load_state_dict(checkpoint[view]['optimizer'])
            
            if view in schedulers and 'scheduler' in checkpoint[view]:
                schedulers[view].load_state_dict(checkpoint[view]['scheduler'])
    
    if 'rng_states' in checkpoint:
        rng = checkpoint['rng_states']
        try:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            
            torch_state = rng['torch']
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch_state.byte() if hasattr(torch_state, 'byte') else torch.ByteTensor(torch_state)
            torch.set_rng_state(torch_state)
            
            if torch.cuda.is_available() and 'cuda' in rng:
                cuda_states = rng['cuda']
                if isinstance(cuda_states, list):
                    cuda_states = [s.byte() if (hasattr(s, 'byte') and not isinstance(s, torch.ByteTensor)) else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
            
            print(f"    -> RNG states restored")
        except Exception as e:
            print(f"    -> WARNING: Could not restore RNG states: {e}")
            print(f"    -> Continuing without RNG state restoration")
    
    print(f"    -> Resumed from epoch {epoch}, best_val_loss={best_val_loss:.4f}")
    return epoch, best_val_loss

def build_models(cfg: Config, device, use_sync_bn=False):
    models = {}
    arch = cfg.dense_arch.lower()
    model = None
    if arch == "convnextv2_unet":
        model = DenseModelConvNeXt(
            in_ch=1,
            feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg,
            enable_dirreg=cfg.enable_dirreg,
            variant=cfg.cnv2_variant,
            decoder_scale=cfg.cnv2_decoder_scale,
            use_transpose=cfg.cnv2_use_transpose,
            skip_proj=cfg.cnv2_skip_proj,
        )
    elif arch == "smp_unet":
        model = DenseModelSMP(
            in_ch=cfg.smp_in_channels,
            feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes,
            enable_seg=cfg.enable_seg,
            enable_dirreg=cfg.enable_dirreg,
            encoder_name=cfg.smp_encoder,
            encoder_weights=cfg.smp_encoder_weights,
            decoder_channels=cfg.smp_decoder_channels,
        )
    else:
        model = DenseModel(
            in_ch=1, base=cfg.dense_base, feat_ch=cfg.dense_feat_ch,
            num_classes=cfg.num_classes, enable_seg=cfg.enable_seg, enable_dirreg=cfg.enable_dirreg
        )

    if model is not None:
        if use_sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        models["dense"] = model

    return models

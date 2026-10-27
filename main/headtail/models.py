# Authored by Hilary Utaegbulam

"""Neural network architectures: Dense U-Net, SMP variants, ConvNeXtV2."""
from __future__ import annotations
import torch
import torch.nn as nn

try:
    from convnextv2Unet import ConvNeXtV2_UNet
    import convnextv2_for_import as cnv2_factory
    HAVE_CNV2_DENSE = True
except Exception:
    HAVE_CNV2_DENSE = False

try:
    import segmentation_models_pytorch as smp
    HAVE_SMP = True
except Exception:
    HAVE_SMP = False

class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class DenseUNetFeatures(nn.Module):
    def __init__(self, in_ch=1, base=32, feat_ch=64):
        super().__init__()
        self.down1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base, base*2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(base*2, base*4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, feat_ch)  # final features @ full res
    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b  = self.bottleneck(self.pool3(d3))
        u3 = self.up3(b); c3 = torch.cat([u3, d3], 1)
        u2 = self.up2(self.dec3(c3)); c2 = torch.cat([u2, d2], 1)
        u1 = self.up1(self.dec2(c2)); c1 = torch.cat([u1, d1], 1)
        feats = self.dec1(c1)  # [B,feat_ch,H,W]
        return feats

class DenseHeads(nn.Module):
    def __init__(self, feat_ch: int, num_classes: int, enable_seg: bool, enable_dirreg: bool):
        super().__init__()
        self.dsnt_head = nn.Conv2d(feat_ch, 2, kernel_size=1)  # A/B logits
        self.enable_seg = enable_seg
        self.enable_dirreg = enable_dirreg
        if enable_seg:
            self.seg_head = nn.Conv2d(feat_ch, num_classes, kernel_size=1)
        if enable_dirreg:
            self.dir_head = nn.Sequential(
                nn.Conv2d(feat_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(32, 2)
            )
    def forward(self, feats):
        out = {"logits_ab": self.dsnt_head(feats)}
        if self.enable_seg:
            out["seg_logits"] = self.seg_head(feats)
        if self.enable_dirreg:
            out["dir_vec"] = self.dir_head(feats)  # [B,2]
        return out

class DenseModel(nn.Module):
    def __init__(self, in_ch, base, feat_ch, num_classes, enable_seg, enable_dirreg):
        super().__init__()
        self.backbone = DenseUNetFeatures(in_ch=in_ch, base=base, feat_ch=feat_ch)
        self.heads = DenseHeads(feat_ch, num_classes, enable_seg, enable_dirreg)
    def forward(self, x):
        feats = self.backbone(x)
        return self.heads(feats)

class DenseModelSMP(nn.Module):
    """
    SMP Unet backbone that outputs a full-res feature map with 'feat_ch' channels.
    I set classes=feat_ch and I use the result as features.
    """
    def __init__(self, in_ch, feat_ch, num_classes, enable_seg, enable_dirreg,
                 encoder_name="resnet152",
                 encoder_weights=None,
                 decoder_channels=(256,128,64,32,16)):
        super().__init__()
        if not HAVE_SMP:
            raise RuntimeError("segmentation_models_pytorch not available. `pip install segmentation-models-pytorch timm`")

        # UNet that returns (B, feat_ch, H, W)
        # self.backbone = smp.Unet(
        #     encoder_name=encoder_name,
        #     encoder_weights=encoder_weights,     # 'imagenet' or None
        #     in_channels=in_ch,                   # 1 because 1 channel input
        #     classes=feat_ch,                     # these are features
        #     decoder_channels=decoder_channels,
        #     activation=None,
        # )
        # self.heads = DenseHeads(feat_ch, num_classes, enable_seg, enable_dirreg)

        # UNet++ that returns (B, feat_ch, H, W). (The "smp_unet" arch flag selects this model.)
        self.backbone = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,     # 'imagenet' or None
            in_channels=in_ch,                   # 1 because 1 channel input
            classes=feat_ch,                     # these are features
            decoder_channels=decoder_channels,
            activation=None,
        )
        self.heads = DenseHeads(feat_ch, num_classes, enable_seg, enable_dirreg)


    def forward(self, x):
        feats = self.backbone(x)   # (B, feat_ch, H, W)
        return self.heads(feats)

        
class DenseModelConvNeXt(nn.Module):
    """
    Dense model that uses ConvNeXtV2_UNet as the backbone to produce a full-res
    feature map with 'feat_ch' channels. Like above, I set UNet's num_classes=feat_ch and
    treat its output as features (no softmax).
    """
    def __init__(self, in_ch, feat_ch, num_classes, enable_seg, enable_dirreg,
                 variant="tiny", decoder_scale=0.5, use_transpose=False, skip_proj=True):
        super().__init__()
        if not HAVE_CNV2_DENSE:
            raise RuntimeError("ConvNeXtV2_UNet not importable — check convnextv2Unet.py and convnextv2_for_import.")
        self.backbone = ConvNeXtV2_UNet(
            variant=variant,
            in_chans=in_ch,
            num_classes=feat_ch,            
            decoder_scale=decoder_scale,
            use_transpose=use_transpose,
            skip_proj=skip_proj,
            factory_module=cnv2_factory,    
        )
        self.heads = DenseHeads(feat_ch, num_classes, enable_seg, enable_dirreg)

    def forward(self, x):
        feats = self.backbone(x)         # (B, feat_ch, H, W)
        return self.heads(feats)


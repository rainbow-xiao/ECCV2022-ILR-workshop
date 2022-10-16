import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.IMG_SIZE = 224
_C.SEED = 999
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'SWIN'
_C.MODEL.hidden_dim = 3072
_C.MODEL.Embedding_dim = 64
_C.MODEL.num_classes = 274330
_C.MODEL.finetune = None
_C.MODEL.output_dir = '/root/swin_out'

#head
_C.MODEL.head = CN()
_C.MODEL.head.name = 'Arc_face'
_C.MODEL.head.K = 3

#neck
_C.MODEL.neck = CN()
_C.MODEL.neck.style = 'simple'

#pool
_C.MODEL.pool = CN()
_C.MODEL.pool.GeM_p_trainable = False

# backbone
_C.MODEL.backbone = CN()
_C.MODEL.backbone.name = 'swinv1'
_C.MODEL.backbone.from_timm = False
_C.MODEL.backbone.pretrained = '/root/pretrained_models/swin_large_patch4_window7_224_22k.pth'

# Swin Transformer V1
_C.MODEL.backbone.SWINV1 = CN()
_C.MODEL.backbone.SWINV1.PATCH_SIZE = 4
_C.MODEL.backbone.SWINV1.IN_CHANS = 3
_C.MODEL.backbone.SWINV1.EMBED_DIM = 192
_C.MODEL.backbone.SWINV1.DEPTHS = [2, 2, 18, 2]
_C.MODEL.backbone.SWINV1.NUM_HEADS = [6, 12, 24, 48]
_C.MODEL.backbone.SWINV1.WINDOW_SIZE = 7
_C.MODEL.backbone.SWINV1.DROP_RATE = 0.0
_C.MODEL.backbone.SWINV1.DROP_PATH_RATE = 0.2
_C.MODEL.backbone.SWINV1.MLP_RATIO = 4.
_C.MODEL.backbone.SWINV1.QKV_BIAS = True
_C.MODEL.backbone.SWINV1.QK_SCALE = None
_C.MODEL.backbone.SWINV1.APE = False
_C.MODEL.backbone.SWINV1.PATCH_NORM = True
_C.MODEL.backbone.SWINV1.USE_CHECKPOINT = False
_C.MODEL.backbone.SWINV1.FUSED_WINDOW_PROCESS = False

# Swin Transformer V2
_C.MODEL.backbone.SWINV2 = CN()
_C.MODEL.backbone.SWINV2.PATCH_SIZE = 4
_C.MODEL.backbone.SWINV2.IN_CHANS = 3
_C.MODEL.backbone.SWINV2.EMBED_DIM = 192
_C.MODEL.backbone.SWINV2.DEPTHS = [2, 2, 18, 2]
_C.MODEL.backbone.SWINV2.NUM_HEADS = [6, 12, 24, 48]
_C.MODEL.backbone.SWINV2.WINDOW_SIZE = 24
_C.MODEL.backbone.SWINV1.DROP_RATE = 0.0
_C.MODEL.backbone.SWINV1.DROP_PATH_RATE = 0.2
_C.MODEL.backbone.SWINV2.MLP_RATIO = 4.
_C.MODEL.backbone.SWINV2.QKV_BIAS = True
_C.MODEL.backbone.SWINV2.APE = False
_C.MODEL.backbone.SWINV2.PATCH_NORM = True
_C.MODEL.backbone.SWINV2.USE_CHECKPOINT = False
_C.MODEL.backbone.SWINV2.PRETRAINED_WINDOW_SIZES = [12, 12, 12, 6]

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'AdamW'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-5
_C.Optimizer.backbone_lr_scale_factor = 1e-3

def get_config():
    config = _C.clone()
    return config
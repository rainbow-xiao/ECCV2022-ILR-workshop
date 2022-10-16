import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']
_C.IMG_SIZE = 224
_C.SEED = 999
# -----------------------------------------------------------------------------
# Model settings 126827
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = 'CLIP_ViT'
_C.MODEL.hidden_dim = 3072
_C.MODEL.Embedding_dim = 64
_C.MODEL.num_classes = 274330
_C.MODEL.finetune = None
_C.MODEL.output_dir = '/root/autodl-tmp/clip_vit224_final_0.001'

#head
_C.MODEL.head = CN()
_C.MODEL.head.name = 'Arc_face'
_C.MODEL.head.K = 2

#neck
_C.MODEL.neck = CN()
_C.MODEL.neck.style = 'simple'

#pool
_C.MODEL.pool = CN()
_C.MODEL.pool.GeM_p_trainable = False

# backbone
_C.MODEL.backbone = CN()
_C.MODEL.backbone.name = 'clip_vit'
_C.MODEL.backbone.frozen = False
_C.MODEL.backbone.from_timm = False
_C.MODEL.backbone.output_dim = 1280
_C.MODEL.backbone.pretrained = 'pretrained_models/ViT_H_14_2B_vision_model.pt'

#CLIP-H-16
_C.MODEL.backbone.VIT = CN()
_C.MODEL.backbone.VIT.image_size = 224
_C.MODEL.backbone.VIT.patch_size = 14
_C.MODEL.backbone.VIT.width = 1280
_C.MODEL.backbone.VIT.layers = 32
_C.MODEL.backbone.VIT.heads = 16
_C.MODEL.backbone.VIT.mlp_ratio = 4.0
_C.MODEL.backbone.VIT.output_dim = 1024

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'AdamW'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-5
_C.Optimizer.backbone_lr_scale_factor = 1e-3

#Loss
_C.Loss = CN()
_C.Loss.name = 'ArcFaceLossAdaptiveMargin'
_C.Loss.s = 30.0
_C.Loss.m = 0.3
_C.Loss.stride_m = 0.1
_C.Loss.max_m = 0.8

def get_config():
    config = _C.clone()
    return config
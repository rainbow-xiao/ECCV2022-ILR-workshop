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
_C.MODEL.NAME = 'DOLG'
_C.MODEL.hidden_dim = 3072
_C.MODEL.Embedding_dim = 64
_C.MODEL.num_classes = 68863
_C.MODEL.finetune = None
_C.MODEL.output_dir = '/root/autodl-tmp/DOLG'

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
_C.MODEL.backbone.name = 'tv_resnet101'
_C.MODEL.backbone.frozen = False
_C.MODEL.backbone.from_timm = True
_C.MODEL.backbone.output_dim = 1024
_C.MODEL.backbone.pretrained = True

#Optimizer
_C.Optimizer = CN()
_C.Optimizer.name = 'SGD'
_C.Optimizer.momentum = 0.9
_C.Optimizer.weight_decay = 1e-5

def get_config():
    config = _C.clone()
    return config
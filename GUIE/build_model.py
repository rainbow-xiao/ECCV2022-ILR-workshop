import torch
import torch.nn as nn
import torch.nn.functional as F
from models import *
import apex
from utils import *
import timm
def build_swin(config, logger):
    model_type = config.MODEL.backbone.name
    layernorm = apex.normalization.FusedLayerNorm
    if model_type == 'swinv1':
        model = SwinTransformerV1(img_size=config.IMG_SIZE,
                                  patch_size=config.MODEL.backbone.SWINV1.PATCH_SIZE,
                                  in_chans=config.MODEL.backbone.SWINV1.IN_CHANS,
                                  embed_dim=config.MODEL.backbone.SWINV1.EMBED_DIM,
                                  depths=config.MODEL.backbone.SWINV1.DEPTHS,
                                  num_heads=config.MODEL.backbone.SWINV1.NUM_HEADS,
                                  window_size=config.MODEL.backbone.SWINV1.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.backbone.SWINV1.MLP_RATIO,
                                  qkv_bias=config.MODEL.backbone.SWINV1.QKV_BIAS,
                                  qk_scale=config.MODEL.backbone.SWINV1.QK_SCALE,
                                  drop_rate=config.MODEL.backbone.SWINV1.DROP_RATE,
                                  drop_path_rate=config.MODEL.backbone.SWINV1.DROP_PATH_RATE,
                                  ape=config.MODEL.backbone.SWINV1.APE,
                                  norm_layer=layernorm,
                                  patch_norm=config.MODEL.backbone.SWINV1.PATCH_NORM,
                                  use_checkpoint=config.MODEL.backbone.SWINV1.USE_CHECKPOINT,
                                  fused_window_process=config.MODEL.backbone.SWINV1.FUSED_WINDOW_PROCESS)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.IMG_SIZE,
                                  patch_size=config.MODEL.backbone.SWINV2.PATCH_SIZE,
                                  in_chans=config.MODEL.backbone.SWINV2.IN_CHANS,
                                  embed_dim=config.MODEL.backbone.SWINV2.EMBED_DIM,
                                  depths=config.MODEL.backbone.SWINV2.DEPTHS,
                                  num_heads=config.MODEL.backbone.SWINV2.NUM_HEADS,
                                  window_size=config.MODEL.backbone.SWINV2.WINDOW_SIZE,
                                  mlp_ratio=config.MODEL.backbone.SWINV2.MLP_RATIO,
                                  qkv_bias=config.MODEL.backbone.SWINV2.QKV_BIAS,
                                  drop_rate=config.MODEL.backbone.SWINV2.DROP_RATE,
                                  drop_path_rate=config.MODEL.backbone.SWINV2.DROP_PATH_RATE,
                                  ape=config.MODEL.backbone.SWINV2.APE,
                                  patch_norm=config.MODEL.backbone.SWINV2.PATCH_NORM,
                                  use_checkpoint=config.MODEL.backbone.SWINV2.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.MODEL.backbone.SWINV2.PRETRAINED_WINDOW_SIZES)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    if config.MODEL.backbone.pretrained != None:
        model = load_pretrained_backbone(config, model, logger)
    return model

def build_vit(config, logger):
    model_type = config.MODEL.backbone.name
    if model_type=='clip_vit':
        model = VisionTransformer(
                  image_size=config.MODEL.backbone.VIT.image_size,
                  patch_size=config.MODEL.backbone.VIT.patch_size,
                  width=config.MODEL.backbone.VIT.width,
                  layers=config.MODEL.backbone.VIT.layers,
                  heads=config.MODEL.backbone.VIT.heads,
                  mlp_ratio=config.MODEL.backbone.VIT.mlp_ratio,
                  output_dim=config.MODEL.backbone.VIT.output_dim
              )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")
    if config.MODEL.backbone.pretrained != None:
        model.load_state_dict(torch.load(config.MODEL.backbone.pretrained), strict=True)
        logger.info(f"=> Load pretrained vit_backbone '{config.MODEL.backbone.pretrained}' successfully")
    return model
    
def build_head(config):
    name = config.MODEL.head.name
    if name=='Arc_face':
        head = ArcMarginProduct(config.MODEL.Embedding_dim, config.MODEL.num_classes)
    elif name=='Sub_Arc_face':
        head = ArcMarginProduct_subcenter(config.MODEL.Embedding_dim, config.MODEL.num_classes, k=config.MODEL.head.K)
    else:
        raise NotImplementedError(f"Unkown head: {name}")
    return head

class Backbone(nn.Module):
    def __init__(self, config=None, logger=None):
        super(Backbone, self).__init__()
        name = config.MODEL.backbone.name
        self.from_timm = config.MODEL.backbone.from_timm
        if self.from_timm==True:
            self.net = timm.create_model(name,
                                         pretrained=config.MODEL.backbone.pretrained,
                                         features_only=True,
                                         in_chans=3,
                                         out_indices=(2, 3))
        else:
            if 'swin' in name:
                self.net = build_swin(config, logger)
            elif 'vit' in name:
                self.net = build_vit(config, logger)
            else:
                raise NotImplementedError(f"Unkown model_name: {name}")
    def forward(self, x):
        if self.from_timm:
            x = self.net(x)
        else:
            x = self.net.forward_features(x)
#             x = self.net(x)
        return x

class XL_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
        self.global_pool = GeM_Pooling(p_trainable=config.MODEL.pool.GeM_p_trainable)
        self.Neck = Neck(config.MODEL.backbone.output_dim, config.MODEL.Embedding_dim, style=config.MODEL.neck.style)
        self.head = build_head(config)
    
    def forward_embedding(self, input_dict):
        x = self.backbone(x)
        x = self.global_pool(x[-1])
        embedding = self.Neck(x.squeeze())
        return embedding
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.global_pool(x[-1])
        embedding = self.Neck(x.squeeze())
        logits = self.head(embedding)
        return logits, F.normalize(embedding)
    
class XL_CLIP_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_CLIP_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
        self.Neck = Neck(config.MODEL.backbone.output_dim, config.MODEL.Embedding_dim, style=config.MODEL.neck.style)
        self.head = build_head(config)
        if config.MODEL.backbone.frozen:
            for param in self.backbone.parameters():
                param.requires_grad=False
        
    def forward_embedding(self, x):
        x = self.backbone(x)
        embedding = self.Neck(x)
        return embedding
    
    def forward(self, x):
        x = self.backbone(x)
        embedding = self.Neck(x)
        logits = self.head(embedding)
        return logits


class XL_DOLG_Net(nn.Module):
    def __init__(self, config, logger):
        super(XL_DOLG_Net, self).__init__()        
        self.backbone = Backbone(config, logger)
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, config.MODEL.hidden_dim, size=int(config.IMG_SIZE/8))
        self.global_pool = GeM_Pooling(p_trainable=config.MODEL.pool.GeM_p_trainable)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.neck_glob = Neck(1024, config.MODEL.hidden_dim, style=config.MODEL.neck.style)
        self.fc_cat = nn.Linear(int(2*config.MODEL.hidden_dim), config.MODEL.Embedding_dim)
        self.head = build_head(config)
        if config.MODEL.backbone.frozen:
            for param in self.backbone.parameters():
                param.requires_grad=False
    
    def forward_embedding(self, x):
        features = self.backbone(x)
        global_feat = self.neck_glob(self.global_pool(features[1]).squeeze())
        local_feat = self.local_branch(features[0])
        local_feat = self.orthogonal_fusion(local_feat, global_feat)
        local_feat = self.gap(local_feat).squeeze()
        feats = torch.cat([global_feat, local_feat], dim=1)
        feats = self.fc_cat(feats)
        return feats
    
    def forward(self, x):
        features = self.backbone(x)
        global_feat = self.neck_glob(self.global_pool(features[1]).squeeze())
        local_feat = self.local_branch(features[0])
        local_feat = self.orthogonal_fusion(local_feat, global_feat)
        local_feat = self.gap(local_feat).squeeze()
        feats = torch.cat([global_feat, local_feat], dim=1)
        feats = self.fc_cat(feats)
        logits = self.head(feats)
        return logits
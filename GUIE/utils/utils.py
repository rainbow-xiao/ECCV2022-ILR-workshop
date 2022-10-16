import torch
import numpy as np

def load_checkpoint(config, model, logger):
    logger.info(f"==============> Load model from {config.MODEL.finetune}")
    checkpoint = torch.load(config.MODEL.finetune, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def load_pretrained_backbone(config, model, logger):
    state_dict = torch.load(config.MODEL.backbone.pretrained, map_location='cpu')
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"=> Load pretrained backbone '{config.MODEL.backbone.pretrained}' successfully")
    del state_dict
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    return model
    
def save_checkpoint(model, save_path, save_optim=False, optimizer=None, epoch=None, config=None):
    if save_optim:
        save_state = {'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'epoch': epoch,
                      'config': config}
    else:
        save_state = {'state_dict': model.state_dict(),
                      'optimizer': {},
                      'epoch': {},
                      'config': {}}
    torch.save(save_state, save_path)

def get_train_epoch_lr(c_epoch, max_epoch, init_lr):
    return 0.5 * init_lr * (1.0 + np.cos(np.pi * c_epoch / max_epoch))

def get_warm_up_lr(warm_up_epochs, c_epoch, warm_up_step, init_lr, iters_per_epoch=None):
    c_step = (c_epoch-1)*iters_per_epoch
    total_step = warm_up_epochs*iters_per_epoch
    alpha = (c_step+warm_up_step)/total_step
    factor = 0.01 * (1.0 - alpha) + alpha
    lr = init_lr*factor
    return lr

def set_lr(optimizer, lr, s):
    for g_idx, pg in enumerate(optimizer.param_groups):
        if g_idx==0:
            pg["lr"] = s*lr
        else:
            pg["lr"] = lr
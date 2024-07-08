# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin import SwinTransformer

from .resnet import ResNet50
from .vit import ViT
# from .CSwin import CSWinTransformer
from .cswin_boat import CSWinTransformer
def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'swin':
        model = SwinTransformer(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            rpe=config.MODEL.SWIN.RPE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            use_multiscale=config.TRAIN.USE_MULTISCALE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=config.TRAIN.USE_ABS)

    elif model_type == 'resnet50':
        model = ResNet50(
            num_classes=config.MODEL.NUM_CLASSES,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=config.TRAIN.USE_ABS
        )
    elif model_type == "vit":
        model = ViT(
            image_size=224,
            patch_size=16,
            num_classes=config.MODEL.NUM_CLASSES,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072,
            dropout=0.1,
            emb_dropout=0,
            use_drloc=config.TRAIN.USE_DRLOC,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=config.TRAIN.USE_ABS
        )
        # cswintransformer without Boat
    elif model_type == "cswin":
        model = CSWinTransformer(
            patch_size=4 ,
            embed_dim=64,
            depth=[2,4,32,2],
            drop_path_rate=0.4,
            split_size=[1,2,7,7],
            num_heads=[2,4,8,16],
            mlp_ratio=4.,
            use_drloc=True,
            use_multiscale=False,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=config.TRAIN.USE_ABS

        )
    # Cswintransformer with Boat
    elif model_type == "cswin_boat":
        model = CSWinTransformer(
            patch_size=4,
            embed_dim=64,
            depth=[2,4,32,2],
            drop_path_rate=0.4,
            split_size=[1,2,7,7],
            num_heads=[2,4,8,16],
            mlp_ratio=4.,
            use_drloc=True,
            use_multiscale=False,
            sample_size=config.TRAIN.SAMPLE_SIZE,
            drloc_mode=config.TRAIN.DRLOC_MODE,
            use_abs=config.TRAIN.USE_ABS,
            model_class=config.MODEL.NUM_CLASSES

        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

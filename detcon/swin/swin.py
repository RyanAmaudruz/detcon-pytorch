



teacher =  SwinTransformer(
    img_size=224,
    patch_size=16,
    in_chans=13,
    embed_dim=384,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=7,
    mlp_ratio=4.,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0,
    ape=False,
    patch_norm=True,
    use_checkpoint=False,
    norm_befor_mlp='ln',
)



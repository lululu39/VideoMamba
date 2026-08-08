import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights


class PatchEmbed(nn.Module):
    """2D image to patch embedding, matching the VideoMamba stem."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        self.grid_size = tuple(
            (image - patch) // step + 1
            for image, patch, step in zip(self.img_size, self.patch_size, stride)
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=self.patch_size,
            stride=stride,
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input image size ({height}*{width}) does not match model "
                f"({self.img_size[0]}*{self.img_size[1]})."
            )
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return self.norm(x)


class SoftmaxAttention(nn.Module):
    """Multi-head scaled dot-product softmax attention."""

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **factory_kwargs)
        self.out_proj = nn.Linear(dim, dim, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_drop(self.out_proj(x))


class SwiGLUMLP(nn.Module):
    """Bias-free slow MLP matching the LACT reference implementation."""

    def __init__(self, dim, mlp_ratio=3.0, device=None, dtype=None):
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError(f"mlp_ratio must be positive, got {mlp_ratio}")
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_dim = int(dim * mlp_ratio)
        self.gate = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.up = nn.Linear(dim, hidden_dim, bias=False, **factory_kwargs)
        self.down = nn.Linear(hidden_dim, dim, bias=False, **factory_kwargs)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm softmax attention followed by a slow SwiGLU MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        norm_cls=nn.RMSNorm,
        residual_in_fp32=True,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        mlp_ratio=3.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.mixer = SoftmaxAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **factory_kwargs,
        )
        self.norm = norm_cls(dim, **factory_kwargs)
        self.norm_mlp = norm_cls(dim, **factory_kwargs)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, **factory_kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, hidden_states, inference_params=None):
        del inference_params
        hidden_states = hidden_states + self.drop_path(
            self.mixer(
                self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
            )
        )
        hidden_states = hidden_states + self.drop_path(
            self.mlp(
                self.norm_mlp(
                    hidden_states.to(dtype=self.norm_mlp.weight.dtype)
                )
            )
        )
        if self.residual_in_fp32:
            hidden_states = hidden_states.float()
        return hidden_states


def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear) and module.bias is not None:
        if not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, parameter in module.named_parameters():
            if name == "out_proj.weight":
                nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
                with torch.no_grad():
                    parameter /= math.sqrt(n_residuals_per_layer * n_layer)


def _base_init(module):
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class VisionTransformer(nn.Module):
    """VideoMamba architecture with bidirectional Mamba replaced by attention."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=16,
        depth=24,
        embed_dim=192,
        num_heads=3,
        mlp_ratio=3.0,
        channels=3,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        qkv_bias=True,
        norm_epsilon=1e-5,
        initializer_cfg=None,
        rms_norm=True,
        residual_in_fp32=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            stride=stride,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim, **factory_kwargs)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim, **factory_kwargs)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes, **factory_kwargs)
            if num_classes > 0
            else nn.Identity()
        )

        norm_type = nn.RMSNorm if rms_norm else nn.LayerNorm
        norm_cls = partial(norm_type, eps=norm_epsilon)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        inter_dpr = [0.0] + dpr
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.layers = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    norm_cls=norm_cls,
                    residual_in_fp32=residual_in_fp32,
                    drop_path=inter_dpr[index],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    **factory_kwargs,
                )
                for index in range(depth)
            ]
        )
        self.norm_f = norm_cls(embed_dim, **factory_kwargs)

        self.apply(_base_init)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def get_num_layers(self):
        return len(self.layers)

    def forward_features(self, x, inference_params=None):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        hidden_states = torch.cat((cls_token, x), dim=1)
        hidden_states = self.pos_drop(hidden_states + self.pos_embed)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                inference_params=inference_params,
            )

        hidden_states = self.norm_f(
            hidden_states.to(dtype=self.norm_f.weight.dtype)
        )
        return hidden_states[:, 0]

    def forward(self, x, inference_params=None):
        return self.head(self.forward_features(x, inference_params))


def _create_videovit(pretrained=False, **kwargs):
    for metadata_key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(metadata_key, None)
    model = VisionTransformer(**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise ValueError("No pretrained VideoViT checkpoint is available")
    return model


@register_model
def videovit_tiny(pretrained=False, **kwargs):
    return _create_videovit(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=3,
        **kwargs,
    )


@register_model
def videovit_small(pretrained=False, **kwargs):
    return _create_videovit(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=6,
        **kwargs,
    )


@register_model
def videovit_middle(pretrained=False, **kwargs):
    return _create_videovit(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=432,
        depth=32,
        num_heads=9,
        **kwargs,
    )


@register_model
def videovit_base(pretrained=False, **kwargs):
    return _create_videovit(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=768,
        depth=24,
        num_heads=12,
        **kwargs,
    )

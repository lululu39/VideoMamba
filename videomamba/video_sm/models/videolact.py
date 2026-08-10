import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import to_2tuple, trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights


def inverse_softplus(value):
    return value + math.log(-math.expm1(-value))


def zeropower_via_newtonschulz5(matrix, steps=5):
    """Muon's batched quintic Newton-Schulz zeroth-power iteration."""
    if steps == 0:
        return matrix
    if matrix.ndim != 3:
        raise ValueError(f"Expected a 3D matrix batch, got shape {matrix.shape}")

    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.bfloat16()
    transpose = matrix.shape[-2] > matrix.shape[-1]
    if transpose:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        polynomial = b * gram + (c * gram) @ gram
        x = a * x + polynomial @ x
    if transpose:
        x = x.transpose(-2, -1)
    return x


class SoftmaxAttention(nn.Module):
    """Per-tubelet multi-head scaled dot-product softmax attention."""

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
    """Bias-free slow MLP matching image and video VideoViT blocks."""

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


class TensorDropPath(nn.Module):
    """Stochastic depth with a tensor probability shared by compiled graphs."""

    _PROBABILITY_SCALE = 1 << 24

    def __init__(self, drop_prob):
        super().__init__()
        self.register_buffer(
            "drop_prob_scaled",
            torch.tensor(
                round(float(drop_prob) * self._PROBABILITY_SCALE),
                dtype=torch.int32,
            ),
            persistent=False,
        )

    def forward(self, x):
        if not self.training:
            return x
        drop_prob = self.drop_prob_scaled.float() / self._PROBABILITY_SCALE
        keep_prob = 1.0 - drop_prob
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(mask_shape).bernoulli_(keep_prob)
        return x * random_tensor.div_(keep_prob)


class FastWeightSwiGLU(nn.Module):
    """Multi-head SwiGLU fast weights with optional private projections."""

    def __init__(
        self,
        dim,
        inter_multi=2,
        num_heads=1,
        share_proj=False,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if inter_multi <= 0:
            raise ValueError(f"inter_multi must be positive, got {inter_multi}")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.hidden_dim = int(dim * inter_multi)
        self.share_proj = share_proj

        if share_proj:
            self.apply_proj = None
            self.update_proj = None
            self.output_proj = None
        else:
            self.apply_proj = nn.Sequential(
                nn.Linear(dim, dim, bias=False, **factory_kwargs),
                nn.SiLU(),
            )
            self.update_proj = nn.Sequential(
                nn.Linear(dim, dim, bias=False, **factory_kwargs),
                nn.SiLU(),
            )
            self.output_proj = nn.Linear(
                dim,
                dim,
                bias=False,
                **factory_kwargs,
            )
        self.apply_norm = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            elementwise_affine=False,
            **factory_kwargs,
        )
        self.output_norm = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            elementwise_affine=False,
            **factory_kwargs,
        )

        parameter_kwargs = {
            key: value for key, value in factory_kwargs.items() if value is not None
        }
        self.w0 = nn.Parameter(
            torch.randn(
                num_heads,
                self.head_dim,
                self.hidden_dim,
                **parameter_kwargs,
            )
            / math.sqrt(self.head_dim)
        )
        self.w1 = nn.Parameter(
            torch.randn(
                num_heads,
                self.hidden_dim,
                self.head_dim,
                **parameter_kwargs,
            )
            / math.sqrt(self.hidden_dim)
        )
        self.w2 = nn.Parameter(
            torch.randn(
                num_heads,
                self.head_dim,
                self.hidden_dim,
                **parameter_kwargs,
            )
            / math.sqrt(self.head_dim)
        )

    def init_fast_weights(self, batch_size):
        master_weights = tuple(
            weight.float().unsqueeze(0).repeat(batch_size, 1, 1, 1)
            for weight in (self.w0, self.w1, self.w2)
        )
        fast_dtype = (
            torch.bfloat16 if master_weights[0].is_cuda else master_weights[0].dtype
        )
        fast_weights = tuple(
            F.normalize(weight, dim=2, eps=1e-5).to(fast_dtype)
            for weight in master_weights
        )
        return fast_weights, master_weights

    def _apply_fast_weights(self, x, fast_weights):
        w0, w1, w2 = fast_weights
        output_dtype = x.dtype
        x = x.to(w0.dtype)
        batch_size, seq_len, _ = x.shape
        x_heads = x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        gate = torch.einsum("blhd,bhdk->blhk", x_heads, w0)
        up = torch.einsum("blhd,bhdk->blhk", x_heads, w2)
        hidden = F.silu(gate) * up
        output = torch.einsum("blhk,bhkd->blhd", hidden, w1)
        output = output.reshape(batch_size, seq_len, self.dim).to(output_dtype)
        return output, x_heads, gate, up, hidden

    def forward(self, x, fast_weights):
        if self.apply_proj is not None:
            x = self.apply_norm(self.apply_proj(x))
        output, _, _, _, _ = self._apply_fast_weights(x, fast_weights)
        output = self.output_norm(output)
        if self.output_proj is not None:
            output = self.output_proj(output)
        return output

    def update(
        self,
        memory_input,
        target,
        learning_rates,
        fast_weights,
        master_weights,
        muon_update_steps,
    ):
        w0, w1, w2 = fast_weights
        master_w0, master_w1, master_w2 = master_weights
        batch_size, seq_len, _ = memory_input.shape

        key = memory_input
        if self.update_proj is not None:
            key = self.apply_norm(self.update_proj(key))
        output, key_heads, gate, up, hidden = self._apply_fast_weights(
            key,
            fast_weights,
        )

        error = (target.float() - output.float()) / seq_len
        error_heads = error.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        d_hidden = torch.einsum(
            "blhd,bhdk->blhk",
            error_heads,
            w1.float().transpose(-1, -2),
        )
        d_up = d_hidden * F.silu(gate)
        d_gate = d_hidden * up
        sigmoid = torch.sigmoid(gate)
        d_gate_pre = d_gate * sigmoid * (1 + gate * (1 - sigmoid))

        lr0, lr1, lr2 = learning_rates.float().split(1, dim=-1)
        key_heads = key_heads.float()
        hidden = hidden.float()
        error_heads = error_heads.float()
        d_gate_pre = d_gate_pre.float()
        d_up = d_up.float()
        w1_grad = torch.einsum(
            "blhk,blhd->bhkd",
            hidden * lr1.unsqueeze(2),
            error_heads,
        )
        w0_grad = torch.einsum(
            "blhd,blhk->bhdk",
            key_heads * lr0.unsqueeze(2),
            d_gate_pre,
        )
        w2_grad = torch.einsum(
            "blhd,blhk->bhdk",
            key_heads * lr2.unsqueeze(2),
            d_up,
        )

        def muon_update(gradient):
            flat = gradient.flatten(0, 1)
            flat = zeropower_via_newtonschulz5(flat, muon_update_steps)
            return flat.reshape_as(gradient)

        w0_update = muon_update(w0_grad)
        w1_update = muon_update(w1_grad)
        w2_update = muon_update(w2_grad)
        master_weights = (
            master_w0 + w0_update.to(master_w0.dtype),
            master_w1 + w1_update.to(master_w1.dtype),
            master_w2 + w2_update.to(master_w2.dtype),
        )
        fast_weights = tuple(
            F.normalize(weight, dim=2, eps=1e-5).to(w0.dtype)
            for weight in master_weights
        )
        return fast_weights, master_weights


class LACTBlock(nn.Module):
    """One image-initializable window-attention plus fast-weight LACT block."""

    def __init__(
        self,
        dim,
        num_heads,
        norm_cls,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        mlp_ratio=3.0,
        residual_in_fp32=True,
        fw_inter_multi=2,
        fw_num_heads=1,
        fw_base_lr=0.01,
        muon_update_steps=5,
        share_proj=False,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if fw_base_lr <= 0:
            raise ValueError(f"fw_base_lr must be positive, got {fw_base_lr}")
        if muon_update_steps < 0:
            raise ValueError(
                f"muon_update_steps must be non-negative, got {muon_update_steps}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32
        self.share_proj = share_proj

        # Keep these names aligned with image VideoViT checkpoints.
        self.norm = norm_cls(dim, **factory_kwargs)
        self.mixer = SoftmaxAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **factory_kwargs,
        )
        self.drop_path = (
            TensorDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.memory_norm = norm_cls(dim, **factory_kwargs)
        self.memory = FastWeightSwiGLU(
            dim,
            inter_multi=fw_inter_multi,
            num_heads=fw_num_heads,
            share_proj=share_proj,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
        )
        self.value_proj = (
            None
            if share_proj
            else nn.Linear(dim, dim, bias=False, **factory_kwargs)
        )
        self.lr_proj = nn.Linear(dim, 3, bias=False, **factory_kwargs)
        self.base_lr_inverse = inverse_softplus(fw_base_lr)
        self.muon_update_steps = muon_update_steps

        self.norm_mlp = norm_cls(dim, **factory_kwargs)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, **factory_kwargs)

    def init_fast_weights(self, batch_size):
        return self.memory.init_fast_weights(batch_size)

    def _shared_qkv(self, memory_input):
        query, key, value = self.mixer.qkv(memory_input).chunk(3, dim=-1)
        query = self.memory.apply_norm(F.silu(query))
        key = self.memory.apply_norm(F.silu(key))
        return query, key, value.contiguous()

    def _apply_window_attention(self, x):
        batch_size, tubelets, seq_len, dim = x.shape
        x = x.reshape(batch_size * tubelets, seq_len, dim)
        x = x + self.drop_path(
            self.mixer(self.norm(x.to(dtype=self.norm.weight.dtype)))
        )
        return x.reshape(batch_size, tubelets, seq_len, dim)

    @torch.compile
    def _compiled_window_attention(self, x):
        return self._apply_window_attention(x)

    @torch.compile
    def _compiled_checkpoint_window_attention(self, x):
        return checkpoint.checkpoint(
            self._apply_window_attention,
            x,
            preserve_rng_state=True,
            use_reentrant=False,
        )

    def apply_window_attention(self, x, use_checkpoint=False):
        if use_checkpoint:
            if x.is_cuda:
                return self._compiled_checkpoint_window_attention(x)
            return checkpoint.checkpoint(
                self._apply_window_attention,
                x,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        if x.is_cuda:
            return self._compiled_window_attention(x)
        return self._apply_window_attention(x)

    def _apply_memory_mlp_chunk(self, x, w0, w1, w2):
        if x.ndim not in (3, 4):
            raise ValueError(
                "Expected memory input shaped [B, L, D] or [B, G, L, D], "
                f"got {tuple(x.shape)}"
            )
        grouped = x.ndim == 4
        if not grouped:
            x = x.unsqueeze(1)
        batch_size, group_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size, group_size * seq_len, dim)
        memory_input = self.memory_norm(
            flat_x.to(dtype=self.memory_norm.weight.dtype)
        )
        if self.share_proj:
            query, key, target = self._shared_qkv(memory_input)
            memory_output = self.memory(query, (w0, w1, w2))
            memory_output = self.mixer.proj_drop(
                self.mixer.out_proj(memory_output)
            )
        else:
            key = target = memory_input
            memory_output = self.memory(memory_input, (w0, w1, w2))
        memory_output = memory_output.reshape(
            batch_size * group_size,
            seq_len,
            dim,
        )
        x = x + self.drop_path(memory_output).reshape_as(x)
        flat_group_x = x.reshape(batch_size * group_size, seq_len, dim)
        slow_output = self.mlp(
            self.norm_mlp(
                flat_group_x.to(dtype=self.norm_mlp.weight.dtype)
            )
        )
        x = x + self.drop_path(slow_output).reshape_as(x)
        if self.residual_in_fp32:
            x = x.float()
        if not grouped:
            x = x.squeeze(1)
        return x, memory_input, key, target

    @torch.compile
    def _compiled_memory_mlp_chunk(self, x, *fast_weights):
        return self._apply_memory_mlp_chunk(x, *fast_weights)

    @torch.compile
    def _compiled_checkpoint_memory_mlp_chunk(self, x, *fast_weights):
        return checkpoint.checkpoint(
            self._apply_memory_mlp_chunk,
            x,
            *fast_weights,
            preserve_rng_state=True,
            use_reentrant=False,
        )

    def apply_memory_mlp_chunk(self, x, fast_weights, use_checkpoint=False):
        if use_checkpoint:
            if x.is_cuda:
                return self._compiled_checkpoint_memory_mlp_chunk(
                    x,
                    *fast_weights,
                )
            return checkpoint.checkpoint(
                self._apply_memory_mlp_chunk,
                x,
                *fast_weights,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        if x.is_cuda:
            return self._compiled_memory_mlp_chunk(x, *fast_weights)
        return self._apply_memory_mlp_chunk(x, *fast_weights)

    def _apply_chunk(self, x, w0, w1, w2):
        x = self._apply_window_attention(x.unsqueeze(1)).squeeze(1)
        return self._apply_memory_mlp_chunk(x, w0, w1, w2)

    def apply_chunk(self, x, fast_weights, use_checkpoint=False):
        if use_checkpoint:
            return checkpoint.checkpoint(
                self._apply_chunk,
                x,
                *fast_weights,
                use_reentrant=False,
            )
        return self._apply_chunk(x, *fast_weights)

    def _update_fast_weights(
        self,
        memory_input,
        key,
        target,
        w0,
        w1,
        w2,
        master_w0,
        master_w1,
        master_w2,
    ):
        prediction_input = F.rms_norm(
            memory_input,
            normalized_shape=(memory_input.shape[-1],),
            eps=1e-5,
        )
        update_input = key
        if not self.share_proj:
            update_input = memory_input
            target = self.value_proj(prediction_input)
        with torch.autocast(device_type=memory_input.device.type, enabled=False):
            learning_rates = F.softplus(
                F.linear(
                    prediction_input.float(),
                    self.lr_proj.weight.float(),
                )
                + self.base_lr_inverse
            )
        fast_weights, master_weights = self.memory.update(
            update_input,
            target,
            learning_rates,
            (w0, w1, w2),
            (master_w0, master_w1, master_w2),
            self.muon_update_steps,
        )
        return (*fast_weights, *master_weights)

    @torch.compile
    def _compiled_update_fast_weights(self, *args):
        return self._update_fast_weights(*args)

    @torch.compile
    def _compiled_checkpoint_update_fast_weights(self, *args):
        return checkpoint.checkpoint(
            self._update_fast_weights,
            *args,
            preserve_rng_state=False,
            use_reentrant=False,
        )

    def update_fast_weights(
        self,
        memory_input,
        key,
        target,
        fast_weights,
        master_weights,
        use_checkpoint=False,
    ):
        args = (memory_input, key, target, *fast_weights, *master_weights)
        if use_checkpoint:
            if memory_input.is_cuda:
                outputs = self._compiled_checkpoint_update_fast_weights(*args)
            else:
                outputs = checkpoint.checkpoint(
                    self._update_fast_weights,
                    *args,
                    preserve_rng_state=False,
                    use_reentrant=False,
                )
        elif memory_input.is_cuda:
            outputs = self._compiled_update_fast_weights(*args)
        else:
            outputs = self._update_fast_weights(*args)
        return outputs[:3], outputs[3:]

    def _forward_scan(self, x, fw_update_group_size):
        """Run one complete layer, including its recurrent fast-weight scan."""
        batch_size, tubelets, _, _ = x.shape
        x = self.apply_window_attention(x, use_checkpoint=False)

        fast_weights, master_weights = self.init_fast_weights(batch_size)
        group_outputs = []
        for group_start in range(0, tubelets, fw_update_group_size):
            group_end = min(group_start + fw_update_group_size, tubelets)
            chunk_output, memory_input, key, target = (
                self.apply_memory_mlp_chunk(
                    x[:, group_start:group_end],
                    fast_weights,
                    use_checkpoint=False,
                )
            )
            # The final update has no later token to consume it, so skipping
            # it preserves the output while avoiding dead computation.
            if group_end < tubelets:
                fast_weights, master_weights = self.update_fast_weights(
                    memory_input,
                    key,
                    target,
                    fast_weights,
                    master_weights,
                    use_checkpoint=False,
                )
            group_outputs.append(chunk_output)
        return torch.cat(group_outputs, dim=1)

    def forward_scan(self, x, fw_update_group_size, use_checkpoint=False):
        if not use_checkpoint:
            return self._forward_scan(x, fw_update_group_size)

        # Checkpoint the complete recurrent scan, rather than each individual
        # apply/update. The latter still has to retain every updated state for
        # the next group and therefore does not reduce recurrent-state memory.
        def checkpointed_scan(layer_input):
            return self._forward_scan(layer_input, fw_update_group_size)

        return checkpoint.checkpoint(
            checkpointed_scan,
            x,
            preserve_rng_state=True,
            use_reentrant=False,
        )


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
        if getattr(module, "weight", None) is not None:
            nn.init.ones_(module.weight)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        kernel_size=1,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.num_patches = (
            self.img_size[0]
            // self.patch_size[0]
            * (self.img_size[1] // self.patch_size[1])
        )
        self.tubelet_size = kernel_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(kernel_size, *self.patch_size),
            stride=(kernel_size, *self.patch_size),
        )

    def forward(self, x):
        _, _, _, height, width = x.shape
        if (height, width) != self.img_size:
            raise ValueError(
                f"Input frame size ({height}*{width}) does not match model "
                f"({self.img_size[0]}*{self.img_size[1]})."
            )
        return self.proj(x)


class VisionLACT(nn.Module):
    """Tubelet-chunked LACT for supervised video classification/regression."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
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
        kernel_size=1,
        num_frames=8,
        fc_drop_rate=0.0,
        fw_inter_multi=2,
        fw_num_heads=1,
        fw_base_lr=0.01,
        muon_update_steps=5,
        share_proj=False,
        fw_update_group_size=1,
        device=None,
        dtype=None,
        use_checkpoint=False,
        checkpoint_num=0,
    ):
        super().__init__()
        if num_frames % kernel_size != 0:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by kernel_size={kernel_size}"
            )
        if not 0 <= checkpoint_num <= depth:
            raise ValueError(
                f"checkpoint_num={checkpoint_num} must be between 0 and depth={depth}"
            )
        if fw_update_group_size <= 0:
            raise ValueError(
                "fw_update_group_size must be positive, got "
                f"{fw_update_group_size}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            in_chans=channels,
            embed_dim=embed_dim,
        )
        temporal_tokens = num_frames // kernel_size
        self.tokens_per_tubelet = self.patch_embed.num_patches + 1
        self.window_size = self.tokens_per_tubelet
        self.fw_update_group_size = int(fw_update_group_size)
        self.chunk_size = self.tokens_per_tubelet * self.fw_update_group_size
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim, **factory_kwargs)
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.tokens_per_tubelet, embed_dim, **factory_kwargs)
        )
        self.temporal_pos_embedding = nn.Parameter(
            torch.zeros(1, temporal_tokens, embed_dim, **factory_kwargs)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.head_drop = (
            nn.Dropout(fc_drop_rate) if fc_drop_rate > 0.0 else nn.Identity()
        )
        self.head = (
            nn.Linear(embed_dim, num_classes, **factory_kwargs)
            if num_classes > 0
            else nn.Identity()
        )

        norm_type = nn.RMSNorm if rms_norm else nn.LayerNorm
        norm_cls = partial(norm_type, eps=norm_epsilon)
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        inter_dpr = [0.0] + dpr
        self.layers = nn.ModuleList(
            [
                LACTBlock(
                    embed_dim,
                    num_heads=num_heads,
                    norm_cls=norm_cls,
                    drop_path=inter_dpr[index],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    residual_in_fp32=residual_in_fp32,
                    fw_inter_multi=fw_inter_multi,
                    fw_num_heads=fw_num_heads,
                    fw_base_lr=fw_base_lr,
                    muon_update_steps=muon_update_steps,
                    share_proj=share_proj,
                    norm_epsilon=norm_epsilon,
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
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def get_num_layers(self):
        return len(self.layers)

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, channels, tubelets, height, width = x.shape
        if tubelets != self.temporal_pos_embedding.shape[1]:
            raise ValueError(
                f"Embedded tubelet count {tubelets} does not match model "
                f"{self.temporal_pos_embedding.shape[1]}"
            )
        x = x.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            tubelets,
            height * width,
            channels,
        )
        cls_tokens = self.cls_token.expand(batch_size, tubelets, -1, -1)
        x = torch.cat((cls_tokens, x), dim=2)
        x = x + self.pos_embed.unsqueeze(1)
        x = x + self.temporal_pos_embedding.unsqueeze(2)
        x = self.pos_drop(x)

        # Process layers outermost so every layer can evaluate all independent
        # tubelet attention windows in one batched SDPA call. Fast weights are
        # still applied and updated recurrently, in temporal group order, within
        # each layer. Since fast-weight state is private to a layer, this is the
        # same dependency graph as the corresponding group-major schedule.
        for layer_index, layer in enumerate(self.layers):
            use_checkpoint = (
                self.use_checkpoint and layer_index < self.checkpoint_num
            )
            x = layer.forward_scan(
                x,
                self.fw_update_group_size,
                use_checkpoint=use_checkpoint,
            )

        x = self.norm_f(
            x.to(dtype=self.norm_f.weight.dtype)
        )
        return x[:, -1, 0]

    def forward(self, x):
        return self.head(self.head_drop(self.forward_features(x)))


def _create_videolact(pretrained=False, **kwargs):
    for metadata_key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(metadata_key, None)
    model = VisionLACT(**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise ValueError("No pretrained VideoLACT checkpoint is available")
    return model


@register_model
def videolact_tiny(pretrained=False, **kwargs):
    return _create_videolact(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=3,
        **kwargs,
    )


@register_model
def videolact_small(pretrained=False, **kwargs):
    return _create_videolact(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=6,
        **kwargs,
    )


@register_model
def videolact_middle(pretrained=False, **kwargs):
    return _create_videolact(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=432,
        depth=32,
        num_heads=9,
        **kwargs,
    )

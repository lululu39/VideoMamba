import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights

from .videolact import (
    PatchEmbed,
    SoftmaxAttention,
    SwiGLUMLP,
    TensorDropPath,
    _base_init,
    _init_weights,
    zeropower_via_newtonschulz5,
)


def _sincos_position_embedding(length, dim, device=None):
    """Return a non-learned 1D position embedding for the tiny decoder."""
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    half_dim = dim // 2
    if half_dim == 0:
        return torch.zeros(length, dim, device=device)
    positions = torch.arange(length, dtype=torch.float32, device=device)
    frequencies = torch.arange(half_dim, dtype=torch.float32, device=device)
    frequencies = torch.exp(
        -math.log(10000.0) * frequencies / max(half_dim - 1, 1)
    )
    angles = positions[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding


class FastWeightEncoder(nn.Module):
    """Per-sample multi-head SwiGLU state, matching LACT state size."""

    def __init__(
        self,
        dim,
        inter_multi=2,
        num_heads=1,
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
        parameter_kwargs = {
            key: value for key, value in factory_kwargs.items() if value is not None
        }
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.hidden_dim = int(dim * inter_multi)
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
        self.output_norm = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            elementwise_affine=False,
            **factory_kwargs,
        )

    def init_state(self, batch_size):
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

    def forward(self, x, fast_weights):
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
        return self.output_norm(output)

    def muon_descent(self, gradients, steps):
        """Apply the same batched Muon/NS orientation used by VideoLACT."""
        grad_w0, grad_w1, grad_w2 = gradients
        transpose_w02 = self.head_dim > self.hidden_dim
        if transpose_w02:
            oriented = torch.stack(
                (
                    grad_w0.transpose(-1, -2),
                    grad_w1,
                    grad_w2.transpose(-1, -2),
                )
            )
        else:
            oriented = torch.stack(
                (
                    grad_w0,
                    grad_w1.transpose(-1, -2),
                    grad_w2,
                )
            )
        updates = zeropower_via_newtonschulz5(
            oriented.flatten(0, 2),
            steps,
        ).reshape_as(oriented)
        update_w0, update_w1_oriented, update_w2 = updates.unbind(0)
        if transpose_w02:
            update_w0 = update_w0.transpose(-1, -2)
            update_w1 = update_w1_oriented
            update_w2 = update_w2.transpose(-1, -2)
        else:
            update_w1 = update_w1_oriented.transpose(-1, -2)
        return update_w0, update_w1, update_w2


class TinyDecoderAttention(nn.Module):
    """Explicit attention with double backward for the inner-loss gradient."""

    def __init__(
        self,
        dim,
        num_heads,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, 3 * dim, **factory_kwargs)
        self.out_proj = nn.Linear(dim, dim, **factory_kwargs)

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = torch.softmax(
            (query @ key.transpose(-1, -2)) * self.scale,
            dim=-1,
        )
        output = attention @ value
        output = output.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.out_proj(output)


class TinyMaskedDecoder(nn.Module):
    """One small attention block used only to form the state-update loss."""

    def __init__(
        self,
        input_dim,
        decoder_dim,
        num_heads,
        max_chunk_size,
        mlp_ratio=2.0,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if decoder_dim % num_heads != 0:
            raise ValueError(
                f"decoder_dim={decoder_dim} must be divisible by num_heads={num_heads}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_dim = int(decoder_dim * mlp_ratio)
        self.decoder_dim = decoder_dim
        self.encoder_to_decoder = nn.Linear(
            input_dim,
            decoder_dim,
            bias=False,
            **factory_kwargs,
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_dim, **factory_kwargs)
        )
        self.norm_attn = nn.RMSNorm(
            decoder_dim,
            eps=norm_epsilon,
            **factory_kwargs,
        )
        self.attn = TinyDecoderAttention(
            decoder_dim,
            num_heads=num_heads,
            **factory_kwargs,
        )
        self.norm_mlp = nn.RMSNorm(
            decoder_dim,
            eps=norm_epsilon,
            **factory_kwargs,
        )
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim, hidden_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(hidden_dim, decoder_dim, **factory_kwargs),
        )
        self.norm_out = nn.RMSNorm(
            decoder_dim,
            eps=norm_epsilon,
            **factory_kwargs,
        )
        self.predict = nn.Linear(
            decoder_dim,
            input_dim,
            **factory_kwargs,
        )
        self.register_buffer(
            "position_embedding",
            _sincos_position_embedding(
                max_chunk_size,
                decoder_dim,
                device=device,
            ),
            persistent=False,
        )
        trunc_normal_(self.mask_token, std=0.02)

    def forward(self, visible_latents, visible_indices, seq_len):
        batch_size = visible_latents.shape[0]
        visible_tokens = self.encoder_to_decoder(visible_latents)
        tokens = self.mask_token.to(dtype=visible_tokens.dtype).expand(
            batch_size,
            seq_len,
            -1,
        )
        scatter_indices = visible_indices.unsqueeze(-1).expand(
            -1,
            -1,
            self.decoder_dim,
        )
        tokens = tokens.scatter(1, scatter_indices, visible_tokens)
        positions = self.position_embedding[:seq_len].to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        tokens = tokens + positions.unsqueeze(0)
        tokens = tokens + self.attn(self.norm_attn(tokens))
        tokens = tokens + self.mlp(self.norm_mlp(tokens))
        return self.predict(self.norm_out(tokens))


class MARSBlock(nn.Module):
    """Window attention plus masked-autoencoding recurrent fast state."""

    def __init__(
        self,
        dim,
        num_heads,
        norm_cls,
        tokens_per_tubelet,
        max_chunk_size,
        layer_index,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        mlp_ratio=3.0,
        residual_in_fp32=True,
        fw_inter_multi=2,
        fw_num_heads=1,
        muon_update_steps=5,
        mask_ratio=0.5,
        decoder_dim=64,
        decoder_num_heads=1,
        decoder_mlp_ratio=2.0,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if muon_update_steps < 0:
            raise ValueError(
                f"muon_update_steps must be non-negative, got {muon_update_steps}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.tokens_per_tubelet = int(tokens_per_tubelet)
        self.layer_index = int(layer_index)
        self.mask_ratio = float(mask_ratio)
        self.muon_update_steps = int(muon_update_steps)
        self.residual_in_fp32 = residual_in_fp32

        # These names intentionally match image/video VideoViT checkpoints.
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
        self.state_encoder = FastWeightEncoder(
            dim,
            inter_multi=fw_inter_multi,
            num_heads=fw_num_heads,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
        )
        self.decoder = TinyMaskedDecoder(
            dim,
            decoder_dim=decoder_dim,
            num_heads=decoder_num_heads,
            max_chunk_size=max_chunk_size,
            mlp_ratio=decoder_mlp_ratio,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
        )
        self.memory_gate = nn.Parameter(torch.zeros(dim, **factory_kwargs))
        self.norm_mlp = norm_cls(dim, **factory_kwargs)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, **factory_kwargs)

    def _apply_window_attention(self, x, window_group_size):
        batch_size, tubelets, seq_len, dim = x.shape
        full_window_count = tubelets // window_group_size
        full_tubelet_count = full_window_count * window_group_size
        outputs = []
        if full_window_count:
            windows = x[:, :full_tubelet_count].reshape(
                batch_size * full_window_count,
                window_group_size * seq_len,
                dim,
            )
            windows = windows + self.drop_path(
                self.mixer(self.norm(windows.to(dtype=self.norm.weight.dtype)))
            )
            outputs.append(
                windows.reshape(batch_size, full_tubelet_count, seq_len, dim)
            )
        if full_tubelet_count < tubelets:
            tail_count = tubelets - full_tubelet_count
            tail = x[:, full_tubelet_count:].reshape(
                batch_size,
                tail_count * seq_len,
                dim,
            )
            tail = tail + self.drop_path(
                self.mixer(self.norm(tail.to(dtype=self.norm.weight.dtype)))
            )
            outputs.append(tail.reshape(batch_size, tail_count, seq_len, dim))
        return torch.cat(outputs, dim=1)

    def _apply_memory_mlp_chunk(self, x, fast_weights):
        batch_size, group_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size, group_size * seq_len, dim)
        memory_input = self.memory_norm(
            flat_x.to(dtype=self.memory_norm.weight.dtype)
        )
        memory_output = self.state_encoder(memory_input, fast_weights)
        memory_output = memory_output * self.memory_gate
        x = x + self.drop_path(memory_output).reshape_as(x)
        flat_group_x = x.reshape(batch_size * group_size, seq_len, dim)
        slow_output = self.mlp(
            self.norm_mlp(flat_group_x.to(dtype=self.norm_mlp.weight.dtype))
        )
        x = x + self.drop_path(slow_output).reshape_as(x)
        if self.residual_in_fp32:
            x = x.float()
        return x, memory_input

    @staticmethod
    def _gather_tokens(x, indices):
        return torch.gather(
            x,
            1,
            indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]),
        )

    def _masked_indices(self, batch_size, seq_len, update_index, device):
        token_indices = torch.arange(seq_len, device=device)
        patch_indices = token_indices[
            token_indices.remainder(self.tokens_per_tubelet) != 0
        ]
        patch_count = patch_indices.numel()
        if patch_count < 2:
            raise ValueError(
                "MARS requires at least two non-CLS patch tokens per chunk"
            )
        mask_count = int(round(patch_count * self.mask_ratio))
        mask_count = min(max(mask_count, 1), patch_count - 1)
        if self.training:
            scores = torch.rand(batch_size, patch_count, device=device)
            masked_local = scores.argsort(dim=1)[:, :mask_count]
        else:
            local_indices = torch.arange(patch_count, device=device)
            # A deterministic modular hash gives every layer/chunk a stable
            # pseudo-random mask without consuming evaluation RNG state.
            scores = (
                local_indices * 1103515245
                + (update_index + 1) * 12345
                + (self.layer_index + 1) * 2654435761
            ).remainder(2147483647)
            masked_local = scores.argsort()[:mask_count]
            masked_local = masked_local.unsqueeze(0).expand(batch_size, -1)
        masked_indices = patch_indices[masked_local]
        mask = torch.zeros(
            batch_size,
            seq_len,
            dtype=torch.bool,
            device=device,
        )
        mask.scatter_(1, masked_indices, True)
        all_indices = token_indices.unsqueeze(0).expand(batch_size, -1)
        visible_indices = all_indices.masked_select(~mask).reshape(batch_size, -1)
        masked_indices = all_indices.masked_select(mask).reshape(batch_size, -1)
        return visible_indices, masked_indices

    def _reconstruction_gradients(
        self,
        memory_input,
        fast_weights,
        update_index,
        create_graph,
    ):
        batch_size, seq_len, _ = memory_input.shape
        visible_indices, masked_indices = self._masked_indices(
            batch_size,
            seq_len,
            update_index,
            memory_input.device,
        )
        visible_input = self._gather_tokens(memory_input, visible_indices)
        visible_latents = self.state_encoder(visible_input, fast_weights)
        prediction = self.decoder(visible_latents, visible_indices, seq_len)
        prediction = self._gather_tokens(prediction, masked_indices).float()
        # The target is the current normalized hidden state itself. It is a
        # stop-gradient reconstruction target, not an EMA/teacher feature.
        target = memory_input.detach().float()
        target = self._gather_tokens(target, masked_indices)
        reconstruction_loss = (prediction - target).square().mean(dim=(1, 2))
        return torch.autograd.grad(
            reconstruction_loss.sum(),
            fast_weights,
            create_graph=create_graph,
            retain_graph=create_graph,
        )

    def _update_state(
        self,
        memory_input,
        fast_weights,
        master_weights,
        update_index,
    ):
        outer_grad_enabled = torch.is_grad_enabled()
        create_graph = self.training and outer_grad_enabled
        with torch.enable_grad():
            differentiable_fast_weights = tuple(
                weight
                if weight.requires_grad
                else weight.detach().requires_grad_(True)
                for weight in fast_weights
            )
            gradients = self._reconstruction_gradients(
                memory_input,
                differentiable_fast_weights,
                update_index,
                create_graph=create_graph,
            )
            updates = self.state_encoder.muon_descent(
                gradients,
                self.muon_update_steps,
            )
            master_weights = tuple(
                master - update.to(master.dtype)
                for master, update in zip(master_weights, updates)
            )
            fast_dtype = differentiable_fast_weights[0].dtype
            fast_weights = tuple(
                F.normalize(master, dim=2, eps=1e-5).to(fast_dtype)
                for master in master_weights
            )
        return fast_weights, master_weights

    def _forward_scan(self, x, fw_update_group_size):
        """Strict apply-then-update scan over one complete MARS layer."""
        batch_size, tubelets, _, _ = x.shape
        x = self._apply_window_attention(x, fw_update_group_size)
        fast_weights, master_weights = self.state_encoder.init_state(batch_size)
        outputs = []
        update_index = 0
        for group_start in range(0, tubelets, fw_update_group_size):
            group_end = min(group_start + fw_update_group_size, tubelets)
            chunk_output, memory_input = self._apply_memory_mlp_chunk(
                x[:, group_start:group_end],
                fast_weights,
            )
            # The current group is always encoded with the old state. The
            # reconstruction update becomes visible only to a later group.
            if group_end < tubelets:
                fast_weights, master_weights = self._update_state(
                    memory_input,
                    fast_weights,
                    master_weights,
                    update_index,
                )
                update_index += 1
            outputs.append(chunk_output)
        return torch.cat(outputs, dim=1)

    def forward_scan(self, x, fw_update_group_size, use_checkpoint=False):
        if not use_checkpoint:
            return self._forward_scan(x, fw_update_group_size)

        def checkpointed_scan(layer_input):
            return self._forward_scan(layer_input, fw_update_group_size)

        return checkpoint.checkpoint(
            checkpointed_scan,
            x,
            preserve_rng_state=True,
            use_reentrant=False,
        )


class VisionMARS(nn.Module):
    """Masked Autoencoding Recurrent State model for supervised video tasks."""

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
        muon_update_steps=5,
        fw_update_group_size=1,
        mars_mask_ratio=0.5,
        mars_decoder_dim=64,
        mars_decoder_num_heads=1,
        mars_decoder_mlp_ratio=2.0,
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
        self.fw_update_group_size = int(fw_update_group_size)
        self.chunk_size = self.tokens_per_tubelet * self.fw_update_group_size
        self.window_size = self.chunk_size
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
                MARSBlock(
                    embed_dim,
                    num_heads=num_heads,
                    norm_cls=norm_cls,
                    tokens_per_tubelet=self.tokens_per_tubelet,
                    max_chunk_size=self.chunk_size,
                    layer_index=index,
                    drop_path=inter_dpr[index],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    residual_in_fp32=residual_in_fp32,
                    fw_inter_multi=fw_inter_multi,
                    fw_num_heads=fw_num_heads,
                    muon_update_steps=muon_update_steps,
                    mask_ratio=mars_mask_ratio,
                    decoder_dim=mars_decoder_dim,
                    decoder_num_heads=mars_decoder_num_heads,
                    decoder_mlp_ratio=mars_decoder_mlp_ratio,
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
        # Preserve the image VideoViT function while the MARS branch learns
        # to open. Decoder parameters remain trainable through meta-gradients
        # once later chunks consume an updated state.
        for layer in self.layers:
            nn.init.zeros_(layer.memory_gate)

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

        for index, layer in enumerate(self.layers):
            x = layer.forward_scan(
                x,
                self.fw_update_group_size,
                use_checkpoint=(
                    self.use_checkpoint and index < self.checkpoint_num
                ),
            )
        x = self.norm_f(x.to(dtype=self.norm_f.weight.dtype))
        return x[:, -1, 0]

    def forward(self, x):
        return self.head(self.head_drop(self.forward_features(x)))


def _create_videomars(pretrained=False, **kwargs):
    for metadata_key in ("pretrained_cfg", "pretrained_cfg_overlay", "cache_dir"):
        kwargs.pop(metadata_key, None)
    model = VisionMARS(**kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        raise ValueError("No pretrained VideoMARS checkpoint is available")
    return model


@register_model
def videomars_tiny(pretrained=False, **kwargs):
    return _create_videomars(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=192,
        depth=24,
        num_heads=3,
        **kwargs,
    )


@register_model
def videomars_small(pretrained=False, **kwargs):
    return _create_videomars(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=384,
        depth=24,
        num_heads=6,
        **kwargs,
    )


@register_model
def videomars_middle(pretrained=False, **kwargs):
    return _create_videomars(
        pretrained=pretrained,
        patch_size=16,
        embed_dim=432,
        depth=32,
        num_heads=9,
        **kwargs,
    )

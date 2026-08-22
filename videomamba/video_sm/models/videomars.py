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


class _MuonStraightThrough(torch.autograd.Function):
    """Use the exact Muon value with an identity Jacobian to its input."""

    @staticmethod
    def forward(ctx, exact_update, raw_gradient, backward_scale):
        ctx.save_for_backward(backward_scale)
        ctx.raw_dtype = raw_gradient.dtype
        return exact_update

    @staticmethod
    def backward(ctx, output_gradient):
        (backward_scale,) = ctx.saved_tensors
        raw_gradient = (output_gradient.float() * backward_scale).to(
            ctx.raw_dtype
        )
        return None, raw_gradient, None


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


class FastTransformerStateLayer(nn.Module):
    """Base parameters for one per-sample fast Transformer encoder layer."""

    def __init__(self, dim, inter_multi=2, device=None, dtype=None):
        super().__init__()
        if inter_multi <= 0:
            raise ValueError(f"inter_multi must be positive, got {inter_multi}")
        parameter_kwargs = {
            key: value
            for key, value in {"device": device, "dtype": dtype}.items()
            if value is not None
        }
        hidden_dim = int(dim * inter_multi)
        if hidden_dim <= 0:
            raise ValueError(
                f"inter_multi={inter_multi} gives invalid hidden dim {hidden_dim}"
            )
        self.wq = nn.Parameter(
            torch.randn(dim, dim, **parameter_kwargs) / math.sqrt(dim)
        )
        self.wk = nn.Parameter(
            torch.randn(dim, dim, **parameter_kwargs) / math.sqrt(dim)
        )
        self.wv = nn.Parameter(
            torch.randn(dim, dim, **parameter_kwargs) / math.sqrt(dim)
        )
        self.wo = nn.Parameter(
            torch.randn(dim, dim, **parameter_kwargs) / math.sqrt(dim)
        )
        self.w0 = nn.Parameter(
            torch.randn(dim, hidden_dim, **parameter_kwargs) / math.sqrt(dim)
        )
        self.w1 = nn.Parameter(
            torch.randn(hidden_dim, dim, **parameter_kwargs)
            / math.sqrt(hidden_dim)
        )
        self.w2 = nn.Parameter(
            torch.randn(dim, hidden_dim, **parameter_kwargs) / math.sqrt(dim)
        )

    def state_parameters(self):
        return (self.wq, self.wk, self.wv, self.wo, self.w0, self.w1, self.w2)


class FastWeightTransformerEncoder(nn.Module):
    """Per-sample explicit-attention Transformer used as recurrent state."""

    weights_per_layer = 7

    def __init__(
        self,
        input_dim,
        encoder_dim,
        depth=2,
        num_heads=1,
        inter_multi=2,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if encoder_dim <= 0:
            raise ValueError(f"encoder_dim must be positive, got {encoder_dim}")
        if encoder_dim % num_heads != 0:
            raise ValueError(
                f"encoder_dim={encoder_dim} must be divisible by num_heads={num_heads}"
            )
        parameter_kwargs = {
            key: value
            for key, value in {"device": device, "dtype": dtype}.items()
            if value is not None
        }
        self.input_dim = input_dim
        self.encoder_dim = encoder_dim
        self.depth = depth
        self.num_heads = num_heads
        self.head_dim = encoder_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.norm_epsilon = norm_epsilon
        self.input_proj = nn.Parameter(
            torch.randn(input_dim, encoder_dim, **parameter_kwargs)
            / math.sqrt(input_dim)
        )
        self.transformer_layers = nn.ModuleList(
            [
                FastTransformerStateLayer(
                    encoder_dim,
                    inter_multi=inter_multi,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.output_proj = nn.Parameter(
            torch.randn(encoder_dim, input_dim, **parameter_kwargs)
            / math.sqrt(encoder_dim)
        )

    def state_parameters(self):
        weights = [self.input_proj]
        for layer in self.transformer_layers:
            weights.extend(layer.state_parameters())
        weights.append(self.output_proj)
        return tuple(weights)

    def init_state(self, batch_size):
        master_weights = tuple(
            weight.float().unsqueeze(0).repeat(batch_size, 1, 1)
            for weight in self.state_parameters()
        )
        fast_dtype = (
            torch.bfloat16 if master_weights[0].is_cuda else master_weights[0].dtype
        )
        fast_weights = tuple(
            F.normalize(weight, dim=1, eps=1e-5).to(fast_dtype)
            for weight in master_weights
        )
        return fast_weights, master_weights

    @staticmethod
    def _linear(x, weight):
        return torch.einsum("bld,bdk->blk", x.to(weight.dtype), weight)

    def _attention(self, x, wq, wk, wv, wo):
        batch_size, seq_len, _ = x.shape
        query = self._linear(x, wq).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        key = self._linear(x, wk).reshape_as(query)
        value = self._linear(x, wv).reshape_as(query)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        attention = torch.softmax(
            (query @ key.transpose(-1, -2)) * self.scale,
            dim=-1,
        )
        output = (attention @ value).transpose(1, 2).reshape(
            batch_size,
            seq_len,
            self.encoder_dim,
        )
        return self._linear(output, wo)

    def forward(self, x, fast_weights):
        expected_weights = 2 + self.depth * self.weights_per_layer
        if len(fast_weights) != expected_weights:
            raise ValueError(
                f"Expected {expected_weights} fast weights, got {len(fast_weights)}"
            )
        output_dtype = x.dtype
        hidden = self._linear(x, fast_weights[0])
        offset = 1
        for _ in range(self.depth):
            wq, wk, wv, wo, w0, w1, w2 = fast_weights[
                offset : offset + self.weights_per_layer
            ]
            offset += self.weights_per_layer
            attention_input = F.rms_norm(
                hidden,
                normalized_shape=(self.encoder_dim,),
                eps=self.norm_epsilon,
            )
            hidden = hidden + self._attention(
                attention_input,
                wq,
                wk,
                wv,
                wo,
            )
            mlp_input = F.rms_norm(
                hidden,
                normalized_shape=(self.encoder_dim,),
                eps=self.norm_epsilon,
            )
            gate = self._linear(mlp_input, w0)
            up = self._linear(mlp_input, w2)
            hidden = hidden + self._linear(F.silu(gate) * up, w1)
        hidden = F.rms_norm(
            hidden,
            normalized_shape=(self.encoder_dim,),
            eps=self.norm_epsilon,
        )
        output = self._linear(hidden, fast_weights[-1]).to(output_dtype)
        return F.rms_norm(
            output,
            normalized_shape=(self.input_dim,),
            eps=self.norm_epsilon,
        )

    def muon_descent(
        self,
        gradients,
        steps,
        backward_mode="exact",
        backward_gain=1.0,
    ):
        """Batch same-shaped matrices for LACT-style quintic NS updates."""
        if backward_mode not in (
            "exact",
            "straight_through",
            "normalized_straight_through",
        ):
            raise ValueError(
                "Muon backward mode must be 'exact', 'straight_through', or "
                "'normalized_straight_through', "
                f"got {backward_mode!r}"
            )
        if backward_gain <= 0:
            raise ValueError(
                f"Muon backward gain must be positive, got {backward_gain}"
            )
        grouped = {}
        for index, gradient in enumerate(gradients):
            transpose = gradient.shape[-2] > gradient.shape[-1]
            oriented = gradient.transpose(-1, -2) if transpose else gradient
            key = tuple(oriented.shape[-2:])
            grouped.setdefault(key, []).append((index, transpose, oriented))

        updates = [None] * len(gradients)
        for entries in grouped.values():
            oriented = torch.stack([entry[2] for entry in entries])
            oriented_flat = oriented.flatten(0, 1)
            transformed_flat = zeropower_via_newtonschulz5(
                oriented_flat,
                steps,
            )
            if backward_mode != "exact":
                # Preserve the exact Muon/NS forward update while replacing
                # its ill-conditioned quintic Jacobian with a diagonal
                # surrogate. The normalized variant retains the detached
                # scale of Muon's initial matrix normalization.
                if backward_mode == "normalized_straight_through":
                    backward_scale = (
                        oriented_flat.float()
                        .norm(dim=(-2, -1), keepdim=True)
                        .add(1e-7)
                        .reciprocal()
                        .detach()
                    )
                else:
                    backward_scale = torch.ones_like(
                        oriented_flat[:, :1, :1],
                        dtype=torch.float32,
                    )
                backward_scale = backward_scale * backward_gain
                transformed_flat = _MuonStraightThrough.apply(
                    transformed_flat.detach(),
                    oriented_flat,
                    backward_scale,
                )
            transformed = transformed_flat.reshape_as(oriented)
            for transformed_weight, (index, transpose, _) in zip(
                transformed,
                entries,
            ):
                updates[index] = (
                    transformed_weight.transpose(-1, -2)
                    if transpose
                    else transformed_weight
                )
        return tuple(updates)


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


class TinyDecoderBlock(nn.Module):
    """One persistent explicit-attention Transformer decoder block."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=2.0,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        hidden_dim = int(dim * mlp_ratio)
        if hidden_dim <= 0:
            raise ValueError(
                f"mlp_ratio={mlp_ratio} gives invalid hidden dim {hidden_dim}"
            )
        self.norm_attn = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            **factory_kwargs,
        )
        self.attn = TinyDecoderAttention(
            dim,
            num_heads=num_heads,
            **factory_kwargs,
        )
        self.norm_mlp = nn.RMSNorm(
            dim,
            eps=norm_epsilon,
            **factory_kwargs,
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim, **factory_kwargs),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, **factory_kwargs),
        )

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        return x + self.mlp(self.norm_mlp(x))


class TinyMaskedDecoder(nn.Module):
    """Small configurable Transformer used only to form the inner loss."""

    def __init__(
        self,
        encoder_output_dim,
        prediction_dim,
        decoder_dim,
        num_heads,
        max_chunk_size,
        depth=1,
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
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        factory_kwargs = {"device": device, "dtype": dtype}
        self.decoder_dim = decoder_dim
        self.encoder_to_decoder = nn.Linear(
            encoder_output_dim,
            decoder_dim,
            bias=False,
            **factory_kwargs,
        )
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, decoder_dim, **factory_kwargs)
        )
        self.blocks = nn.ModuleList(
            [
                TinyDecoderBlock(
                    decoder_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    norm_epsilon=norm_epsilon,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )
        self.norm_out = nn.RMSNorm(
            decoder_dim,
            eps=norm_epsilon,
            **factory_kwargs,
        )
        self.predict = nn.Linear(
            decoder_dim,
            prediction_dim,
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
        for block in self.blocks:
            tokens = block(tokens)
        return self.predict(self.norm_out(tokens))


class MARSWindowBlock(nn.Module):
    """VideoViT block whose attention is restricted to temporal windows."""

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
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
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

    def _forward_block(self, x, window_group_size):
        x = self._apply_window_attention(x, window_group_size)
        batch_size, tubelets, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * tubelets, seq_len, dim)
        flat_x = flat_x + self.drop_path(
            self.mlp(
                self.norm_mlp(flat_x.to(dtype=self.norm_mlp.weight.dtype))
            )
        )
        x = flat_x.reshape(batch_size, tubelets, seq_len, dim)
        return x.float() if self.residual_in_fp32 else x

    def forward(self, x, window_group_size, use_checkpoint=False):
        if not use_checkpoint:
            return self._forward_block(x, window_group_size)

        def checkpointed_block(layer_input):
            return self._forward_block(layer_input, window_group_size)

        return checkpoint.checkpoint(
            checkpointed_block,
            x,
            preserve_rng_state=True,
            use_reentrant=False,
        )


class SharedMARSState(nn.Module):
    """One model-wide recurrent state trained by masked pixel reconstruction."""

    def __init__(
        self,
        dim,
        pixel_dim,
        norm_cls,
        tokens_per_tubelet,
        max_chunk_size,
        fw_inter_multi=2,
        encoder_dim=288,
        encoder_depth=2,
        encoder_num_heads=1,
        muon_update_steps=5,
        mask_ratio=0.5,
        decoder_dim=64,
        decoder_depth=1,
        decoder_num_heads=1,
        decoder_mlp_ratio=2.0,
        muon_backward_mode="normalized_straight_through",
        muon_backward_gain=2.0,
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
        self.mask_ratio = float(mask_ratio)
        self.muon_update_steps = int(muon_update_steps)
        if muon_backward_mode not in (
            "exact",
            "straight_through",
            "normalized_straight_through",
        ):
            raise ValueError(
                "Muon backward mode must be 'exact', 'straight_through', or "
                "'normalized_straight_through', "
                f"got {muon_backward_mode!r}"
            )
        self.muon_backward_mode = muon_backward_mode
        if muon_backward_gain <= 0:
            raise ValueError(
                f"Muon backward gain must be positive, got {muon_backward_gain}"
            )
        self.muon_backward_gain = float(muon_backward_gain)
        self.memory_norm = norm_cls(dim, **factory_kwargs)
        self.state_encoder = FastWeightTransformerEncoder(
            input_dim=dim,
            encoder_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            inter_multi=fw_inter_multi,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
        )
        self.decoder = TinyMaskedDecoder(
            encoder_output_dim=dim,
            prediction_dim=pixel_dim,
            decoder_dim=decoder_dim,
            num_heads=decoder_num_heads,
            max_chunk_size=max_chunk_size,
            depth=decoder_depth,
            mlp_ratio=decoder_mlp_ratio,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
        )
        self.memory_gate = nn.Parameter(torch.zeros(dim, **factory_kwargs))

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
            scores = (
                local_indices * 1103515245 + (update_index + 1) * 12345
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
        pixel_targets,
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
        target = self._gather_tokens(
            pixel_targets.detach(),
            masked_indices,
        ).float()
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
        pixel_targets,
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
                pixel_targets,
                differentiable_fast_weights,
                update_index,
                create_graph=create_graph,
            )
            updates = self.state_encoder.muon_descent(
                gradients,
                self.muon_update_steps,
                backward_mode=self.muon_backward_mode,
                backward_gain=self.muon_backward_gain,
            )
            master_weights = tuple(
                master - update.to(master.dtype)
                for master, update in zip(master_weights, updates)
            )
            fast_dtype = differentiable_fast_weights[0].dtype
            fast_weights = tuple(
                # MARS stores every matrix as [batch, input, output], unlike
                # LACT's [batch, head, input, output] layout.  Keep the input
                # dimension normalized before and after every state update.
                F.normalize(master, dim=1, eps=1e-5).to(fast_dtype)
                for master in master_weights
            )
        return fast_weights, master_weights

    def forward(self, x, pixel_targets, fw_update_group_size):
        """For every group, update from masked pixels and then apply state."""
        batch_size, tubelets, seq_len, dim = x.shape
        patch_count = seq_len - 1
        if pixel_targets.shape[:3] != (batch_size, tubelets, patch_count):
            raise ValueError(
                "Pixel target shape does not match embedded video: "
                f"{tuple(pixel_targets.shape)} versus "
                f"({batch_size}, {tubelets}, {patch_count}, pixel_dim)"
            )
        fast_weights, master_weights = self.state_encoder.init_state(batch_size)
        outputs = []
        update_index = 0
        for group_start in range(0, tubelets, fw_update_group_size):
            group_end = min(group_start + fw_update_group_size, tubelets)
            group_size = group_end - group_start
            memory_input = self.memory_norm(
                x[:, group_start:group_end]
                .reshape(batch_size, group_size * seq_len, dim)
                .to(dtype=self.memory_norm.weight.dtype)
            )
            group_pixels = pixel_targets[:, group_start:group_end]
            target_tokens = group_pixels.new_zeros(
                batch_size,
                group_size,
                seq_len,
                group_pixels.shape[-1],
            )
            target_tokens[:, :, 1:] = group_pixels
            target_tokens = target_tokens.reshape(
                batch_size,
                group_size * seq_len,
                group_pixels.shape[-1],
            )
            fast_weights, master_weights = self._update_state(
                memory_input,
                target_tokens,
                fast_weights,
                master_weights,
                update_index,
            )
            outputs.append(
                self.state_encoder(memory_input, fast_weights).reshape(
                    batch_size,
                    group_size,
                    seq_len,
                    dim,
                )
            )
            update_index += 1
        memory_output = torch.cat(outputs, dim=1)
        return memory_output * self.memory_gate


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
        muon_update_steps=5,
        fw_update_group_size=1,
        mars_mask_ratio=0.5,
        mars_encoder_dim=None,
        mars_encoder_depth=2,
        mars_encoder_num_heads=None,
        mars_decoder_dim=64,
        mars_decoder_depth=1,
        mars_decoder_num_heads=1,
        mars_decoder_mlp_ratio=2.0,
        mars_muon_backward="normalized_straight_through",
        mars_muon_backward_gain=2.0,
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
        if mars_encoder_dim is None:
            mars_encoder_dim = 2 * embed_dim // 3
        if mars_encoder_num_heads is None:
            mars_encoder_num_heads = 1
        factory_kwargs = {"device": device, "dtype": dtype}
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim
        self.mars_encoder_dim = mars_encoder_dim
        self.mars_encoder_depth = mars_encoder_depth
        self.mars_encoder_num_heads = mars_encoder_num_heads
        self.mars_decoder_depth = mars_decoder_depth
        self.channels = int(channels)
        self.tubelet_size = int(kernel_size)
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
        patch_height, patch_width = self.patch_embed.patch_size
        self.pixel_dim = (
            self.channels * self.tubelet_size * patch_height * patch_width
        )
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
                MARSWindowBlock(
                    embed_dim,
                    num_heads=num_heads,
                    norm_cls=norm_cls,
                    drop_path=inter_dpr[index],
                    attn_drop=attn_drop_rate,
                    proj_drop=drop_rate,
                    qkv_bias=qkv_bias,
                    mlp_ratio=mlp_ratio,
                    residual_in_fp32=residual_in_fp32,
                    **factory_kwargs,
                )
                for index in range(depth)
            ]
        )
        self.shared_state = SharedMARSState(
            dim=embed_dim,
            pixel_dim=self.pixel_dim,
            norm_cls=norm_cls,
            tokens_per_tubelet=self.tokens_per_tubelet,
            max_chunk_size=self.chunk_size,
            fw_inter_multi=fw_inter_multi,
            encoder_dim=mars_encoder_dim,
            encoder_depth=mars_encoder_depth,
            encoder_num_heads=mars_encoder_num_heads,
            muon_update_steps=muon_update_steps,
            mask_ratio=mars_mask_ratio,
            decoder_dim=mars_decoder_dim,
            decoder_depth=mars_decoder_depth,
            decoder_num_heads=mars_decoder_num_heads,
            decoder_mlp_ratio=mars_decoder_mlp_ratio,
            muon_backward_mode=mars_muon_backward,
            muon_backward_gain=mars_muon_backward_gain,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
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
        # Preserve the image VideoViT function while the shared MARS branch
        # learns to open through the supervised outer objective.
        nn.init.zeros_(self.shared_state.memory_gate)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}

    @torch.jit.ignore
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def get_num_layers(self):
        return len(self.layers)

    def patchify_pixels(self, video):
        """Convert video into targets aligned with PatchEmbed's tubelets."""
        batch_size, channels, frames, height, width = video.shape
        patch_height, patch_width = self.patch_embed.patch_size
        if channels != self.channels:
            raise ValueError(
                f"Input channels {channels} do not match model {self.channels}"
            )
        if frames % self.tubelet_size != 0:
            raise ValueError(
                f"Input frames {frames} are not divisible by tubelet size "
                f"{self.tubelet_size}"
            )
        if height % patch_height != 0 or width % patch_width != 0:
            raise ValueError(
                f"Input size ({height}, {width}) is not divisible by patch "
                f"size ({patch_height}, {patch_width})"
            )
        tubelets = frames // self.tubelet_size
        grid_height = height // patch_height
        grid_width = width // patch_width
        return (
            video.reshape(
                batch_size,
                channels,
                tubelets,
                self.tubelet_size,
                grid_height,
                patch_height,
                grid_width,
                patch_width,
            )
            .permute(0, 2, 4, 6, 3, 5, 7, 1)
            .reshape(
                batch_size,
                tubelets,
                grid_height * grid_width,
                self.pixel_dim,
            )
        )

    def forward_features(self, x):
        pixel_targets = self.patchify_pixels(x)
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
        state_output = self.shared_state(
            x,
            pixel_targets,
            self.fw_update_group_size,
        )

        for index, layer in enumerate(self.layers):
            x = layer(
                x,
                self.fw_update_group_size,
                use_checkpoint=(
                    self.use_checkpoint and index < self.checkpoint_num
                ),
            )
        x = x + state_output
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

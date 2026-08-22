from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import trunc_normal_
from timm.models import register_model
from timm.models.vision_transformer import _cfg, _load_weights

from .videolact import (
    FastWeightSwiGLU,
    PatchEmbed,
    SoftmaxAttention,
    SwiGLUMLP,
    TensorDropPath,
    _base_init,
    _init_weights,
    inverse_softplus,
    zeropower_via_newtonschulz5,
)


class MaskedFastWeightAutoencoder(FastWeightSwiGLU):
    """One LACT SwiGLU viewed as an encoder followed by a decoder.

    ``w0`` and ``w2`` encode a token into the gated hidden representation;
    ``w1`` decodes that representation back to model width. The only change
    from LACT's update objective is that input features are masked and the
    reconstruction error is evaluated on those masked coordinates.
    """

    num_weights = 3

    def __init__(
        self,
        dim,
        inter_multi=2,
        num_heads=1,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__(
            dim,
            inter_multi=inter_multi,
            num_heads=num_heads,
            share_proj=True,
            norm_epsilon=norm_epsilon,
            device=device,
            dtype=dtype,
        )

    def state_parameters(self):
        return (self.w0, self.w1, self.w2)

    def encode(self, x, fast_weights):
        """Apply the trained encoder/decoder state as the memory mapping."""
        if len(fast_weights) != self.num_weights:
            raise ValueError(
                f"Expected {self.num_weights} fast weights, got "
                f"{len(fast_weights)}"
            )
        return self.forward(x, fast_weights)

    def reconstruct(self, masked_input, fast_weights):
        """Run the raw LACT encoder/decoder used by the inner objective."""
        reconstruction, input_heads, gate, up, hidden = self._apply_fast_weights(
            masked_input,
            fast_weights,
        )
        return reconstruction, (input_heads, gate, up, hidden)

    @staticmethod
    def _silu_backward(output_direction, pre_activation):
        sigmoid = torch.sigmoid(pre_activation)
        return output_direction * sigmoid * (
            1 + pre_activation * (1 - sigmoid)
        )

    def reconstruction_directions(
        self,
        masked_input,
        target,
        reconstruction_mask,
        learning_rates,
        fast_weights,
    ):
        """LACT analytic update direction for masked reconstruction."""
        if learning_rates.shape[-1] != self.num_weights:
            raise ValueError(
                f"Expected {self.num_weights} learning rates, got "
                f"{learning_rates.shape[-1]}"
            )
        reconstruction, cache = self.reconstruct(masked_input, fast_weights)
        input_heads, gate, up, hidden = cache
        batch_size, seq_len, num_heads, head_dim = input_heads.shape
        # LACT uses (target - output) / L. MARS changes only the objective by
        # retaining that error on masked hidden coordinates.
        error = (
            (target.detach().float() - reconstruction.float())
            * reconstruction_mask.float()
            / seq_len
        )
        error_heads = error.reshape(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        ).float()
        w0, w1, w2 = fast_weights
        w0 = w0.float()
        w1 = w1.float()
        w2 = w2.float()
        input_heads = input_heads.float()
        gate = gate.float()
        up = up.float()
        hidden = hidden.float()

        d_hidden = torch.einsum("blhd,bhkd->blhk", error_heads, w1)
        d_up = d_hidden * F.silu(gate)
        d_gate = self._silu_backward(d_hidden * up, gate)
        lr0, lr1, lr2 = learning_rates.float().split(1, dim=-1)
        # Fold the three identically shaped update GEMMs into one batch, as
        # in LACT. This changes only launch geometry, not the update math.
        batch_heads = batch_size * num_heads
        gradient_left = torch.stack(
            (
                input_heads * lr0.unsqueeze(2),
                error_heads,
                input_heads * lr2.unsqueeze(2),
            )
        ).permute(0, 1, 3, 4, 2).reshape(
            3 * batch_heads,
            head_dim,
            seq_len,
        )
        gradient_right = torch.stack(
            (
                d_gate,
                hidden * lr1.unsqueeze(2),
                d_up,
            )
        ).permute(0, 1, 3, 2, 4).reshape(
            3 * batch_heads,
            seq_len,
            self.hidden_dim,
        )
        grad_w0, grad_w1_transposed, grad_w2 = torch.bmm(
            gradient_left,
            gradient_right,
        ).reshape(
            3,
            batch_size,
            num_heads,
            head_dim,
            self.hidden_dim,
        ).unbind(0)
        grad_w1 = grad_w1_transposed.transpose(-1, -2)
        return grad_w0, grad_w1, grad_w2

    def muon_updates(self, gradients, steps):
        """Apply exact LACT quintic Newton-Schulz to all three matrices."""
        if len(gradients) != self.num_weights:
            raise ValueError(
                f"Expected {self.num_weights} gradients, got {len(gradients)}"
            )
        grad_w0, grad_w1, grad_w2 = gradients
        oriented = torch.stack((grad_w0, grad_w1.transpose(-1, -2), grad_w2))
        transformed = zeropower_via_newtonschulz5(
            oriented.flatten(0, 2),
            steps,
        ).reshape_as(oriented)
        update_w0, update_w1, update_w2 = transformed.unbind(0)
        return update_w0, update_w1.transpose(-1, -2), update_w2

    def update(
        self,
        masked_input,
        target,
        reconstruction_mask,
        learning_rates,
        fast_weights,
        master_weights,
        muon_update_steps,
    ):
        gradients = self.reconstruction_directions(
            masked_input,
            target,
            reconstruction_mask,
            learning_rates,
            fast_weights,
        )
        updates = self.muon_updates(gradients, muon_update_steps)
        master_weights = tuple(
            master + update.to(master.dtype)
            for master, update in zip(master_weights, updates)
        )
        fast_weights = tuple(
            F.normalize(weight, dim=2, eps=1e-5).to(fast_weights[0].dtype)
            for weight in master_weights
        )
        return fast_weights, master_weights


class MARSBlock(nn.Module):
    """Window attention, recurrent masked-reconstruction state, slow MLP."""

    def __init__(
        self,
        dim,
        num_heads,
        norm_cls,
        layer_index,
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
        mask_ratio=0.5,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if fw_base_lr <= 0:
            raise ValueError(f"fw_base_lr must be positive, got {fw_base_lr}")
        if muon_update_steps < 0:
            raise ValueError(
                f"muon_update_steps must be non-negative, got {muon_update_steps}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim = dim
        self.layer_index = int(layer_index)
        self.mask_ratio = float(mask_ratio)
        self.muon_update_steps = int(muon_update_steps)
        self.residual_in_fp32 = residual_in_fp32

        # Image/video VideoViT-compatible window-attention names.
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
        self.state = MaskedFastWeightAutoencoder(
            dim,
            inter_multi=fw_inter_multi,
            num_heads=fw_num_heads,
            norm_epsilon=norm_epsilon,
            **factory_kwargs,
        )
        self.memory_gate = nn.Parameter(torch.zeros(dim, **factory_kwargs))
        self.lr_proj = nn.Linear(
            dim,
            MaskedFastWeightAutoencoder.num_weights,
            bias=False,
            **factory_kwargs,
        )
        self.base_lr_inverse = inverse_softplus(fw_base_lr)
        self.norm_mlp = norm_cls(dim, **factory_kwargs)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, **factory_kwargs)

    def init_fast_weights(self, batch_size):
        return self.state.init_fast_weights(batch_size)

    def _attend_flat_windows(self, x):
        return x + self.drop_path(
            self.mixer(self.norm(x.to(dtype=self.norm.weight.dtype)))
        )

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
            windows = self._attend_flat_windows(windows)
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
            tail = self._attend_flat_windows(tail)
            outputs.append(tail.reshape(batch_size, tail_count, seq_len, dim))
        return torch.cat(outputs, dim=1)

    @torch.compile
    def _compiled_window_attention(self, x, window_group_size):
        return self._apply_window_attention(x, window_group_size)

    def apply_window_attention(self, x, window_group_size):
        if x.is_cuda:
            return self._compiled_window_attention(x, window_group_size)
        return self._apply_window_attention(x, window_group_size)

    def _apply_memory_mlp_chunk(self, x, *fast_weights):
        batch_size, group_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size, group_size * seq_len, dim)
        memory_input = self.memory_norm(
            flat_x.to(dtype=self.memory_norm.weight.dtype)
        )
        memory_output = self.state.encode(memory_input, fast_weights)
        memory_output = memory_output * self.memory_gate
        x = x + self.drop_path(memory_output.reshape_as(x))
        flat_x = x.reshape(batch_size * group_size, seq_len, dim)
        slow_output = self.mlp(
            self.norm_mlp(flat_x.to(dtype=self.norm_mlp.weight.dtype))
        )
        x = x + self.drop_path(slow_output).reshape_as(x)
        if self.residual_in_fp32:
            x = x.float()
        return x, memory_input

    @torch.compile
    def _compiled_memory_mlp_chunk(self, x, *fast_weights):
        return self._apply_memory_mlp_chunk(x, *fast_weights)

    def apply_memory_mlp_chunk(self, x, fast_weights):
        if x.is_cuda:
            return self._compiled_memory_mlp_chunk(x, *fast_weights)
        return self._apply_memory_mlp_chunk(x, *fast_weights)

    def _apply_window_memory_mlp_chunk(
        self,
        x,
        window_group_size,
        *fast_weights,
    ):
        x = self._apply_window_attention(x, window_group_size)
        return self._apply_memory_mlp_chunk(x, *fast_weights)

    @torch.compile
    def _compiled_window_memory_mlp_chunk(
        self,
        x,
        window_group_size,
        *fast_weights,
    ):
        return self._apply_window_memory_mlp_chunk(
            x,
            window_group_size,
            *fast_weights,
        )

    @torch.compile
    def _compiled_checkpoint_window_memory_mlp_chunk(
        self,
        x,
        window_group_size,
        *fast_weights,
    ):
        return checkpoint.checkpoint(
            self._apply_window_memory_mlp_chunk,
            x,
            window_group_size,
            *fast_weights,
            preserve_rng_state=True,
            use_reentrant=False,
        )

    def apply_window_memory_mlp_chunk(
        self,
        x,
        window_group_size,
        fast_weights,
        use_checkpoint=False,
    ):
        args = (x, window_group_size, *fast_weights)
        if use_checkpoint:
            if x.is_cuda:
                return self._compiled_checkpoint_window_memory_mlp_chunk(*args)
            return checkpoint.checkpoint(
                self._apply_window_memory_mlp_chunk,
                *args,
                preserve_rng_state=True,
                use_reentrant=False,
            )
        if x.is_cuda:
            return self._compiled_window_memory_mlp_chunk(*args)
        return self._apply_window_memory_mlp_chunk(*args)

    def _feature_mask(self, batch_size, seq_len, update_index, device):
        """Mask an exact fraction of hidden channels for the whole chunk."""
        mask_count = int(round(self.dim * self.mask_ratio))
        mask_count = min(max(mask_count, 1), self.dim - 1)
        if self.training:
            scores = torch.rand(batch_size, self.dim, device=device)
            masked_channels = scores.argsort(dim=-1)[:, :mask_count]
        else:
            channels = torch.arange(self.dim, device=device, dtype=torch.int64)
            scores = (
                channels * 1103515245
                + (update_index + 1) * 12345
                + (self.layer_index + 1) * 2654435761
            ).remainder(2147483647)
            masked_channels = scores.argsort()[:mask_count]
            masked_channels = masked_channels.unsqueeze(0).expand(
                batch_size,
                -1,
            )
        channel_mask = torch.zeros(
            batch_size,
            self.dim,
            dtype=torch.bool,
            device=device,
        )
        channel_mask.scatter_(1, masked_channels, True)
        return channel_mask.unsqueeze(1).expand(-1, seq_len, -1)

    def _update_fast_weights(
        self,
        memory_input,
        update_index,
        *weights,
    ):
        fast_weights = weights[: MaskedFastWeightAutoencoder.num_weights]
        master_weights = weights[MaskedFastWeightAutoencoder.num_weights :]
        prediction_input = F.rms_norm(
            memory_input,
            normalized_shape=(self.dim,),
            eps=1e-5,
        )
        with torch.autocast(
            device_type=memory_input.device.type,
            enabled=False,
        ):
            learning_rates = F.softplus(
                F.linear(
                    prediction_input.float(),
                    self.lr_proj.weight.float(),
                )
                + self.base_lr_inverse
            )
        reconstruction_mask = self._feature_mask(
            memory_input.shape[0],
            memory_input.shape[1],
            update_index,
            memory_input.device,
        )
        masked_input = memory_input.masked_fill(reconstruction_mask, 0)
        fast_weights, master_weights = self.state.update(
            masked_input,
            memory_input,
            reconstruction_mask,
            learning_rates,
            fast_weights,
            master_weights,
            self.muon_update_steps,
        )
        return (*fast_weights, *master_weights)

    @torch.compile
    def _compiled_update_fast_weights(self, *args):
        return self._update_fast_weights(*args)

    def update_fast_weights(
        self,
        memory_input,
        update_index,
        fast_weights,
        master_weights,
    ):
        args = (memory_input, update_index, *fast_weights, *master_weights)
        if memory_input.is_cuda:
            outputs = self._compiled_update_fast_weights(*args)
        else:
            outputs = self._update_fast_weights(*args)
        split = MaskedFastWeightAutoencoder.num_weights
        return outputs[:split], outputs[split:]

    def _forward_scan(self, x, fw_update_group_size):
        """Apply old state to each group, then update it for later groups."""
        batch_size, tubelets, _, _ = x.shape
        x = self.apply_window_attention(x, fw_update_group_size)
        fast_weights, master_weights = self.init_fast_weights(batch_size)
        outputs = []
        update_index = 0
        for group_start in range(0, tubelets, fw_update_group_size):
            group_end = min(group_start + fw_update_group_size, tubelets)
            chunk_output, memory_input = self.apply_memory_mlp_chunk(
                x[:, group_start:group_end],
                fast_weights,
            )
            if group_end < tubelets:
                fast_weights, master_weights = self.update_fast_weights(
                    memory_input,
                    update_index,
                    fast_weights,
                    master_weights,
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
    """Per-layer masked-autoencoding recurrent state for supervised video."""

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
        fw_update_group_size=1,
        fw_update_layer_group_size=1,
        mars_mask_ratio=0.5,
        device=None,
        dtype=None,
        use_checkpoint=False,
        checkpoint_num=0,
    ):
        super().__init__()
        if num_frames % kernel_size != 0:
            raise ValueError(
                f"num_frames={num_frames} must be divisible by "
                f"kernel_size={kernel_size}"
            )
        if not 0 <= checkpoint_num <= depth:
            raise ValueError(
                f"checkpoint_num={checkpoint_num} must be between 0 and "
                f"depth={depth}"
            )
        if fw_update_group_size <= 0:
            raise ValueError(
                "fw_update_group_size must be positive, got "
                f"{fw_update_group_size}"
            )
        if fw_update_layer_group_size <= 0:
            raise ValueError(
                "fw_update_layer_group_size must be positive, got "
                f"{fw_update_layer_group_size}"
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
        self.fw_update_layer_group_size = int(fw_update_layer_group_size)
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
                    layer_index=index,
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
                    mask_ratio=mars_mask_ratio,
                    norm_epsilon=norm_epsilon,
                    **factory_kwargs,
                )
                for index in range(depth)
            ]
        )
        self.norm_f = norm_cls(embed_dim, **factory_kwargs)
        self.register_buffer(
            "_layer_indices",
            torch.arange(depth, dtype=torch.int64),
            persistent=False,
        )

        self.apply(_base_init)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        # Preserve the pretrained VideoViT function at initialization. The
        # gate learns first; state meta-gradients appear as it opens.
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

    def _batched_feature_mask(
        self,
        num_layers,
        batch_size,
        seq_len,
        layer_indices,
        update_index,
        device,
    ):
        """Construct one exact channel mask per layer and sample."""
        dim = self.embed_dim
        mask_count = int(round(dim * self.layers[0].mask_ratio))
        mask_count = min(max(mask_count, 1), dim - 1)
        if self.training:
            scores = torch.rand(num_layers, batch_size, dim, device=device)
            masked_channels = scores.argsort(dim=-1)[..., :mask_count]
        else:
            channels = torch.arange(dim, device=device, dtype=torch.int64)
            scores = (
                channels.view(1, -1) * 1103515245
                + (update_index + 1) * 12345
                + (layer_indices.view(-1, 1) + 1) * 2654435761
            ).remainder(2147483647)
            masked_channels = scores.argsort(dim=-1)[:, :mask_count]
            masked_channels = masked_channels.unsqueeze(1).expand(
                -1,
                batch_size,
                -1,
            )
        channel_mask = torch.zeros(
            num_layers,
            batch_size,
            dim,
            dtype=torch.bool,
            device=device,
        )
        channel_mask.scatter_(2, masked_channels, True)
        return channel_mask.unsqueeze(2).expand(-1, -1, seq_len, -1)

    def _batched_update_fast_weights(
        self,
        memory_inputs,
        w0,
        w1,
        w2,
        master_w0,
        master_w1,
        master_w2,
        lr_weights,
        layer_indices,
        update_index,
    ):
        """Update several independent MARS layers in one GEMM batch."""
        num_layers, batch_size, seq_len, dim = memory_inputs.shape
        prediction_inputs = F.rms_norm(
            memory_inputs,
            normalized_shape=(dim,),
            eps=1e-5,
        )
        with torch.autocast(
            device_type=memory_inputs.device.type,
            enabled=False,
        ):
            learning_rates = torch.bmm(
                prediction_inputs.float().reshape(
                    num_layers,
                    batch_size * seq_len,
                    dim,
                ),
                lr_weights.float().transpose(1, 2),
            ).reshape(num_layers, batch_size, seq_len, 3)
            learning_rates = F.softplus(
                learning_rates + self.layers[0].base_lr_inverse
            )
        reconstruction_mask = self._batched_feature_mask(
            num_layers,
            batch_size,
            seq_len,
            layer_indices,
            update_index,
            memory_inputs.device,
        )
        masked_inputs = memory_inputs.masked_fill(reconstruction_mask, 0)
        fast_weights, master_weights = self.layers[0].state.update(
            masked_inputs.flatten(0, 1),
            memory_inputs.detach().flatten(0, 1),
            reconstruction_mask.flatten(0, 1),
            learning_rates.flatten(0, 1),
            tuple(weight.flatten(0, 1) for weight in (w0, w1, w2)),
            tuple(
                weight.flatten(0, 1)
                for weight in (master_w0, master_w1, master_w2)
            ),
            self.layers[0].muon_update_steps,
        )
        fast_weights = tuple(
            weight.reshape(num_layers, batch_size, *weight.shape[1:])
            for weight in fast_weights
        )
        master_weights = tuple(
            weight.reshape(num_layers, batch_size, *weight.shape[1:])
            for weight in master_weights
        )
        return (*fast_weights, *master_weights)

    @torch.compile
    def _compiled_batched_update_fast_weights(self, *args):
        return self._batched_update_fast_weights(*args)

    @torch.compile
    def _compiled_checkpoint_batched_update_fast_weights(self, *args):
        return checkpoint.checkpoint(
            self._batched_update_fast_weights,
            *args,
            preserve_rng_state=True,
            use_reentrant=False,
        )

    def _update_layer_group(
        self,
        layers,
        update_records,
        fast_weights,
        master_weights,
        lr_weights,
        layer_indices,
        update_index,
        use_checkpoint=False,
    ):
        memory_inputs = torch.stack(update_records)
        args = (
            memory_inputs,
            *fast_weights,
            *master_weights,
            lr_weights,
            layer_indices,
            update_index,
        )
        full_group = len(layers) == self.fw_update_layer_group_size
        if use_checkpoint:
            if memory_inputs.is_cuda and full_group:
                outputs = self._compiled_checkpoint_batched_update_fast_weights(
                    *args
                )
            else:
                outputs = checkpoint.checkpoint(
                    self._batched_update_fast_weights,
                    *args,
                    preserve_rng_state=True,
                    use_reentrant=False,
                )
        elif memory_inputs.is_cuda and full_group:
            outputs = self._compiled_batched_update_fast_weights(*args)
        else:
            outputs = self._batched_update_fast_weights(*args)
        return outputs[:3], outputs[3:]

    @staticmethod
    def _init_layer_group_fast_weights(layers, batch_size):
        layer_states = [layer.init_fast_weights(batch_size) for layer in layers]
        fast_weights = tuple(
            torch.stack([state[0][index] for state in layer_states])
            for index in range(3)
        )
        master_weights = tuple(
            torch.stack([state[1][index] for state in layer_states])
            for index in range(3)
        )
        return fast_weights, master_weights

    def _iter_cross_layer_groups(self):
        for layer_start in range(
            0,
            len(self.layers),
            self.fw_update_layer_group_size,
        ):
            yield (
                layer_start,
                min(
                    layer_start + self.fw_update_layer_group_size,
                    len(self.layers),
                ),
            )

    def _forward_cross_layer_scan(self, x):
        """Run strict chunk-major apply with cross-layer batched updates."""
        batch_size, tubelets, _, _ = x.shape
        layer_groups = []
        layer_states = []
        for layer_start, layer_end in self._iter_cross_layer_groups():
            layers = self.layers[layer_start:layer_end]
            fast_weights, master_weights = (
                self._init_layer_group_fast_weights(layers, batch_size)
            )
            layer_groups.append(
                {
                    "start": layer_start,
                    "layers": layers,
                    "fast_weights": fast_weights,
                    "master_weights": master_weights,
                    "lr_weights": torch.stack(
                        [layer.lr_proj.weight for layer in layers]
                    ),
                    "layer_indices": self._layer_indices[layer_start:layer_end],
                }
            )
            layer_states.extend(
                (len(layer_groups) - 1, layer_index)
                for layer_index in range(len(layers))
            )

        group_outputs = []
        update_index = 0
        for group_start in range(0, tubelets, self.fw_update_group_size):
            group_end = min(
                group_start + self.fw_update_group_size,
                tubelets,
            )
            hidden_states = x[:, group_start:group_end]
            update_records = []
            for layer_index, layer in enumerate(self.layers):
                state_group_index, state_layer_index = layer_states[layer_index]
                state_group = layer_groups[state_group_index]
                hidden_states, memory_input = (
                    layer.apply_window_memory_mlp_chunk(
                        hidden_states,
                        window_group_size=self.fw_update_group_size,
                        fast_weights=tuple(
                            weight[state_layer_index]
                            for weight in state_group["fast_weights"]
                        ),
                        use_checkpoint=(
                            self.use_checkpoint
                            and layer_index < self.checkpoint_num
                        ),
                    )
                )
                update_records.append(memory_input)

            if group_end < tubelets:
                for state_group in layer_groups:
                    layer_start = state_group["start"]
                    layer_end = layer_start + len(state_group["layers"])
                    fast_weights, master_weights = self._update_layer_group(
                        state_group["layers"],
                        update_records[layer_start:layer_end],
                        state_group["fast_weights"],
                        state_group["master_weights"],
                        state_group["lr_weights"],
                        state_group["layer_indices"],
                        update_index,
                        use_checkpoint=(
                            self.use_checkpoint
                            and layer_start < self.checkpoint_num
                        ),
                    )
                    state_group["fast_weights"] = fast_weights
                    state_group["master_weights"] = master_weights
                update_index += 1
            group_outputs.append(hidden_states)
        return torch.cat(group_outputs, dim=1)

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

        if self.fw_update_layer_group_size == 1:
            for index, layer in enumerate(self.layers):
                x = layer.forward_scan(
                    x,
                    self.fw_update_group_size,
                    use_checkpoint=(
                        self.use_checkpoint and index < self.checkpoint_num
                    ),
                )
        else:
            x = self._forward_cross_layer_scan(x)
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

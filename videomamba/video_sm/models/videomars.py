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
    inverse_softplus,
    zeropower_via_newtonschulz5,
)


class MaskedResidualDenoiserConv3d(nn.Module):
    """Per-sample residual CNN state trained by masked reconstruction.

    One ``down -> depthwise 3D convolution -> up`` state is used for both the
    recurrent apply and the reconstruction update.  Masked patch tokens are
    replaced before the same denoiser is evaluated, so the full-dimensional
    identity skip cannot leak their targets.
    """

    num_weights = 3

    def __init__(
        self,
        dim,
        hidden_dim=128,
        kernel_size=(3, 3, 3),
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if any(size <= 0 or size % 2 == 0 for size in kernel_size):
            raise ValueError(
                "Conv3d kernel sizes must be positive odd integers, got "
                f"{kernel_size}"
            )
        parameter_kwargs = {
            key: value
            for key, value in {"device": device, "dtype": dtype}.items()
            if value is not None
        }
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.kernel_size = tuple(int(size) for size in kernel_size)
        self.norm_epsilon = float(norm_epsilon)

        def matrix(input_dim, output_dim):
            return nn.Parameter(
                torch.randn(input_dim, output_dim, **parameter_kwargs)
                / math.sqrt(input_dim)
            )

        def depthwise_kernel():
            volume = math.prod(self.kernel_size)
            return nn.Parameter(
                torch.randn(
                    self.hidden_dim,
                    *self.kernel_size,
                    **parameter_kwargs,
                )
                / math.sqrt(volume)
            )

        self.down = matrix(self.dim, self.hidden_dim)
        self.kernel = depthwise_kernel()
        self.up = matrix(self.hidden_dim, self.dim)
        self.mask_token = nn.Parameter(torch.zeros(self.dim, **parameter_kwargs))
        nn.init.normal_(self.mask_token, std=0.02)

    def state_parameters(self):
        return self.down, self.kernel, self.up

    @staticmethod
    def _normalize_weight(weight, index):
        if index == 1:
            flat = weight.flatten(2)
            return F.normalize(flat, dim=2, eps=1e-5).reshape_as(weight)
        return F.normalize(weight, dim=1, eps=1e-5)

    def init_fast_weights(self, batch_size):
        master_weights = tuple(
            weight.float().unsqueeze(0).repeat(batch_size, *([1] * weight.ndim))
            for weight in self.state_parameters()
        )
        fast_dtype = (
            torch.bfloat16 if master_weights[0].is_cuda else master_weights[0].dtype
        )
        fast_weights = tuple(
            self._normalize_weight(weight, index).to(fast_dtype)
            for index, weight in enumerate(master_weights)
        )
        return fast_weights, master_weights

    @staticmethod
    def _pointwise(x, weight):
        return torch.einsum("b...d,bdk->b...k", x, weight)

    def _depthwise_conv3d(self, x, kernel):
        batch_size, tubelets, height, width, channels = x.shape
        kernel_t, kernel_h, kernel_w = self.kernel_size
        convolution_input = F.pad(
            x.permute(0, 4, 1, 2, 3),
            (
                kernel_w // 2,
                kernel_w // 2,
                kernel_h // 2,
                kernel_h // 2,
                kernel_t // 2,
                kernel_t // 2,
            ),
        )
        patches = (
            convolution_input.unfold(2, kernel_t, 1)
            .unfold(3, kernel_h, 1)
            .unfold(4, kernel_w, 1)
            .reshape(
                batch_size * channels,
                tubelets * height * width,
                kernel_t * kernel_h * kernel_w,
            )
        )
        output = torch.bmm(
            patches,
            kernel.reshape(
                batch_size * channels,
                kernel_t * kernel_h * kernel_w,
                1,
            ),
        ).squeeze(-1)
        return output.reshape(
            batch_size,
            channels,
            tubelets,
            height,
            width,
        ).permute(0, 2, 3, 4, 1)

    def _denoise_grid(self, x, down, kernel, up):
        normalized_input = F.rms_norm(
            x,
            normalized_shape=(self.dim,),
            eps=self.norm_epsilon,
        )
        hidden = F.silu(self._pointwise(normalized_input, down))
        normalized = F.rms_norm(
            hidden,
            normalized_shape=(self.hidden_dim,),
            eps=self.norm_epsilon,
        )
        hidden = hidden + F.silu(self._depthwise_conv3d(normalized, kernel))
        return F.rms_norm(
            x + self._pointwise(hidden, up),
            normalized_shape=(self.dim,),
            eps=self.norm_epsilon,
        )

    def denoise_grid(self, x, fast_weights):
        return self._denoise_grid(x, *fast_weights)

    @staticmethod
    def _split_tokens(x, group_size, height, width):
        batch_size, seq_len, dim = x.shape
        tokens_per_tubelet = height * width + 1
        if seq_len != group_size * tokens_per_tubelet:
            raise ValueError(
                f"Expected {group_size * tokens_per_tubelet} tokens, got "
                f"{seq_len}"
            )
        x = x.reshape(batch_size, group_size, tokens_per_tubelet, dim)
        cls_tokens = x[:, :, 0]
        patch_tokens = x[:, :, 1:].reshape(
            batch_size,
            group_size,
            height,
            width,
            dim,
        )
        return cls_tokens, patch_tokens

    @staticmethod
    def _merge_tokens(cls_tokens, patch_tokens):
        batch_size, group_size, height, width, dim = patch_tokens.shape
        return torch.cat(
            (
                cls_tokens.unsqueeze(2),
                patch_tokens.reshape(batch_size, group_size, height * width, dim),
            ),
            dim=2,
        ).flatten(1, 2)

    @staticmethod
    def _expand_mask_token(token, batch_size, ndim):
        if token.ndim == 1:
            token = token.unsqueeze(0).expand(batch_size, -1)
        return token.view(batch_size, *([1] * (ndim - 2)), token.shape[-1])

    def apply_denoiser(self, x, fast_weights, group_size, height, width):
        cls_tokens, patch_tokens = self._split_tokens(
            x,
            group_size,
            height,
            width,
        )
        denoised_patches = self.denoise_grid(patch_tokens, fast_weights)
        # CLS has no natural 3D-grid location.  It shares both pointwise
        # weights and receives the current tubelet's pooled patch latent.
        normalized_cls = F.rms_norm(
            cls_tokens,
            normalized_shape=(self.dim,),
            eps=self.norm_epsilon,
        )
        normalized_patches = F.rms_norm(
            patch_tokens,
            normalized_shape=(self.dim,),
            eps=self.norm_epsilon,
        )
        cls_hidden = F.silu(self._pointwise(normalized_cls, fast_weights[0]))
        patch_hidden = F.silu(
            self._pointwise(normalized_patches, fast_weights[0])
        )
        cls_hidden = cls_hidden + patch_hidden.mean(dim=(2, 3))
        denoised_cls = F.rms_norm(
            cls_tokens + self._pointwise(cls_hidden, fast_weights[2]),
            normalized_shape=(self.dim,),
            eps=self.norm_epsilon,
        )
        return self._merge_tokens(denoised_cls, denoised_patches).to(x.dtype)

    def reconstruct(
        self,
        x,
        token_mask,
        fast_weights,
        group_size,
        height,
        width,
        mask_token=None,
    ):
        _, patch_tokens = self._split_tokens(x, group_size, height, width)
        batch_size = patch_tokens.shape[0]
        if token_mask.shape != patch_tokens.shape[:-1]:
            raise ValueError(
                f"Token mask shape {tuple(token_mask.shape)} does not match "
                f"patch grid {tuple(patch_tokens.shape[:-1])}"
            )
        if mask_token is None:
            mask_token = self.mask_token
        mask_token = self._expand_mask_token(
            mask_token,
            batch_size,
            patch_tokens.ndim,
        )
        masked_input = torch.where(
            token_mask.unsqueeze(-1),
            mask_token,
            patch_tokens,
        )
        return self.denoise_grid(masked_input, fast_weights), patch_tokens

    def reconstruction_directions(
        self,
        x,
        token_mask,
        learning_rates,
        fast_weights,
        group_size,
        height,
        width,
        mask_token=None,
        create_graph=True,
    ):
        """Return the exact negative gradients of masked-token MSE."""
        with torch.enable_grad():
            differentiable_weights = tuple(
                weight
                if weight.requires_grad
                else weight.detach().requires_grad_(True)
                for weight in fast_weights
            )
            reconstruction, target = self.reconstruct(
                x,
                token_mask,
                differentiable_weights,
                group_size,
                height,
                width,
                mask_token,
            )
            if learning_rates.shape != (x.shape[0], x.shape[1], 1):
                raise ValueError(
                    "Expected per-token learning rates "
                    f"{(x.shape[0], x.shape[1], 1)}, got "
                    f"{tuple(learning_rates.shape)}"
                )
            _, patch_learning_rates = self._split_tokens(
                learning_rates,
                group_size,
                height,
                width,
            )
            sequence_length = group_size * height * width
            loss = (
                0.5
                * (
                    (reconstruction.float() - target.detach().float()).square()
                    * token_mask.unsqueeze(-1)
                    * patch_learning_rates.float()
                ).sum()
                / sequence_length
            )
            gradients = torch.autograd.grad(
                loss,
                differentiable_weights,
                create_graph=create_graph,
                retain_graph=create_graph,
            )
        return tuple(-gradient for gradient in gradients)

    def muon_updates(self, directions, steps):
        if len(directions) != self.num_weights:
            raise ValueError(
                f"Expected {self.num_weights} directions, got {len(directions)}"
            )
        down, kernel, up = directions
        matrices = torch.stack(
            (
                down,
                up.transpose(-1, -2),
            )
        )
        matrix_updates = zeropower_via_newtonschulz5(
            matrices.flatten(0, 1).float(),
            steps,
        ).reshape_as(matrices)
        convolution_updates = zeropower_via_newtonschulz5(
            kernel.flatten(2).float(),
            steps,
        ).reshape_as(kernel)
        return (
            matrix_updates[0],
            convolution_updates,
            matrix_updates[1].transpose(-1, -2),
        )

    def update(
        self,
        x,
        token_mask,
        learning_rates,
        fast_weights,
        master_weights,
        muon_update_steps,
        group_size,
        height,
        width,
        update_scale=0.03,
        mask_token=None,
    ):
        if learning_rates.shape != (x.shape[0], x.shape[1], 1):
            raise ValueError(
                "Expected per-token learning rates "
                f"{(x.shape[0], x.shape[1], 1)}, "
                f"got {tuple(learning_rates.shape)}"
            )
        if update_scale <= 0:
            raise ValueError(
                f"update_scale must be positive, got {update_scale}"
            )
        directions = self.reconstruction_directions(
            x,
            token_mask,
            learning_rates,
            fast_weights,
            group_size,
            height,
            width,
            mask_token,
            create_graph=self.training and torch.is_grad_enabled(),
        )
        updates = self.muon_updates(directions, muon_update_steps)
        master_weights = tuple(
            master + float(update_scale) * update.to(master.dtype)
            for master, update in zip(master_weights, updates)
        )
        fast_weights = tuple(
            self._normalize_weight(weight, index).to(fast_weights[0].dtype)
            for index, weight in enumerate(master_weights)
        )
        return fast_weights, master_weights


class MARSBlock(nn.Module):
    """Window attention, recurrent convolutional MAE state, slow MLP."""

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
        mars_cnn_dim=64,
        spatial_size=(14, 14),
        fw_base_lr=0.01,
        muon_update_steps=5,
        mask_ratio=0.5,
        tube_mask_fraction=0.5,
        update_scale=0.03,
        no_fw=False,
        norm_epsilon=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if not 0.0 <= tube_mask_fraction <= 1.0:
            raise ValueError(
                "tube_mask_fraction must be in [0, 1], got "
                f"{tube_mask_fraction}"
            )
        if update_scale <= 0:
            raise ValueError(
                f"update_scale must be positive, got {update_scale}"
            )
        if fw_base_lr <= 0:
            raise ValueError(f"fw_base_lr must be positive, got {fw_base_lr}")
        if muon_update_steps < 0:
            raise ValueError(
                f"muon_update_steps must be non-negative, got {muon_update_steps}"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.dim = dim
        self.layer_index = int(layer_index)
        self.no_fw = bool(no_fw)
        self.mask_ratio = float(mask_ratio)
        self.tube_mask_fraction = float(tube_mask_fraction)
        self.update_scale = float(update_scale)
        self.spatial_size = tuple(int(size) for size in spatial_size)
        if len(self.spatial_size) != 2 or any(
            size <= 0 for size in self.spatial_size
        ):
            raise ValueError(
                f"spatial_size must contain two positive values, got "
                f"{spatial_size}"
            )
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
        if not self.no_fw:
            self.memory_norm = norm_cls(dim, **factory_kwargs)
            self.state = MaskedResidualDenoiserConv3d(
                dim,
                hidden_dim=mars_cnn_dim,
                norm_epsilon=norm_epsilon,
                **factory_kwargs,
            )
            self.memory_gate = nn.Parameter(torch.zeros(dim, **factory_kwargs))
            self.lr_proj = nn.Linear(
                dim,
                1,
                bias=False,
                **factory_kwargs,
            )
            self.base_lr_inverse = inverse_softplus(fw_base_lr)
        self.norm_mlp = norm_cls(dim, **factory_kwargs)
        self.mlp = SwiGLUMLP(dim, mlp_ratio=mlp_ratio, **factory_kwargs)

    def init_fast_weights(self, batch_size):
        if self.no_fw:
            raise RuntimeError("Fast weights are disabled for this MARS block")
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

    def _forward_no_fw(self, x, window_group_size):
        """Apply grouped window attention and the slow MLP without memory."""
        x = self.apply_window_attention(x, window_group_size)
        batch_size, tubelets, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * tubelets, seq_len, dim)
        slow_output = self.mlp(
            self.norm_mlp(flat_x.to(dtype=self.norm_mlp.weight.dtype))
        )
        x = x + self.drop_path(slow_output).reshape_as(x)
        return x.float() if self.residual_in_fp32 else x

    def forward_no_fw(self, x, window_group_size, use_checkpoint=False):
        if not self.no_fw:
            raise RuntimeError("forward_no_fw requires a no-FW MARS block")
        if not use_checkpoint:
            return self._forward_no_fw(x, window_group_size)

        def checkpointed_block(layer_input):
            return self._forward_no_fw(layer_input, window_group_size)

        return checkpoint.checkpoint(
            checkpointed_block,
            x,
            preserve_rng_state=True,
            use_reentrant=False,
        )

    def _apply_memory_mlp_chunk(self, x, *fast_weights):
        batch_size, group_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size, group_size * seq_len, dim)
        memory_input = self.memory_norm(
            flat_x.to(dtype=self.memory_norm.weight.dtype)
        )
        memory_output = self.state.apply_denoiser(
            memory_input,
            fast_weights,
            group_size,
            *self.spatial_size,
        )
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

    def _token_mask(self, batch_size, group_size, update_index, device):
        """Mask an exact mix of full tubes and independent patch tokens."""
        height, width = self.spatial_size
        patch_count = height * width
        total_count = group_size * patch_count
        mask_count = int(round(total_count * self.mask_ratio))
        mask_count = min(max(mask_count, 1), total_count - 1)
        tube_count = int(mask_count * self.tube_mask_fraction) // group_size
        tube_count = min(tube_count, patch_count - 1)
        if self.training:
            scores = torch.rand(batch_size, patch_count, device=device)
            tube_tokens = scores.argsort(dim=-1)[:, :tube_count]
        else:
            tokens = torch.arange(patch_count, device=device, dtype=torch.int64)
            scores = (
                tokens * 1103515245
                + (update_index + 1) * 12345
                + (self.layer_index + 1) * 2654435761
            ).remainder(2147483647)
            tube_tokens = scores.argsort()[:tube_count]
            tube_tokens = tube_tokens.unsqueeze(0).expand(
                batch_size,
                -1,
            )
        token_mask = torch.zeros(
            batch_size,
            group_size,
            patch_count,
            dtype=torch.bool,
            device=device,
        )
        tube_mask = torch.zeros(
            batch_size,
            patch_count,
            dtype=torch.bool,
            device=device,
        )
        tube_mask.scatter_(1, tube_tokens, True)
        token_mask |= tube_mask.unsqueeze(1)
        remaining_count = mask_count - tube_count * group_size
        if self.training:
            remaining_scores = torch.rand(
                batch_size,
                group_size,
                patch_count,
                device=device,
            )
        else:
            positions = torch.arange(
                total_count,
                device=device,
                dtype=torch.int64,
            ).reshape(1, group_size, patch_count)
            remaining_scores = (
                positions * 1664525
                + (update_index + 1) * 1013904223
                + (self.layer_index + 1) * 2246822519
            ).remainder(2147483647).float()
            remaining_scores = remaining_scores.expand(batch_size, -1, -1)
        remaining_scores = remaining_scores.masked_fill(token_mask, float("inf"))
        remaining_tokens = remaining_scores.flatten(1).argsort(dim=-1)[
            :, :remaining_count
        ]
        token_mask.flatten(1).scatter_(1, remaining_tokens, True)
        return token_mask.reshape(batch_size, group_size, height, width)

    def _update_fast_weights(
        self,
        memory_input,
        update_index,
        *weights,
    ):
        fast_weights = weights[: MaskedResidualDenoiserConv3d.num_weights]
        master_weights = weights[MaskedResidualDenoiserConv3d.num_weights :]
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
        height, width = self.spatial_size
        tokens_per_tubelet = height * width + 1
        if memory_input.shape[1] % tokens_per_tubelet:
            raise ValueError(
                f"Chunk length {memory_input.shape[1]} is not divisible by "
                f"tokens_per_tubelet={tokens_per_tubelet}"
            )
        group_size = memory_input.shape[1] // tokens_per_tubelet
        token_mask = self._token_mask(
            memory_input.shape[0],
            group_size,
            update_index,
            memory_input.device,
        )
        fast_weights, master_weights = self.state.update(
            memory_input,
            token_mask,
            learning_rates,
            fast_weights,
            master_weights,
            self.muon_update_steps,
            group_size,
            height,
            width,
            self.update_scale,
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
        split = MaskedResidualDenoiserConv3d.num_weights
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
        mars_cnn_dim=64,
        fw_base_lr=0.01,
        muon_update_steps=5,
        fw_update_group_size=1,
        fw_update_layer_group_size=1,
        mars_mask_ratio=0.5,
        mars_tube_mask_fraction=0.5,
        mars_update_scale=0.03,
        mars_no_fw=False,
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
        self.mars_no_fw = bool(mars_no_fw)
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
        spatial_side = int(math.isqrt(self.patch_embed.num_patches))
        if spatial_side * spatial_side != self.patch_embed.num_patches:
            raise ValueError(
                "VideoMARS convolutional state currently requires a square "
                f"patch grid, got {self.patch_embed.num_patches} patches"
            )
        self.spatial_size = (spatial_side, spatial_side)
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
                    mars_cnn_dim=mars_cnn_dim,
                    spatial_size=self.spatial_size,
                    fw_base_lr=fw_base_lr,
                    muon_update_steps=muon_update_steps,
                    mask_ratio=mars_mask_ratio,
                    tube_mask_fraction=mars_tube_mask_fraction,
                    update_scale=mars_update_scale,
                    no_fw=self.mars_no_fw,
                    norm_epsilon=norm_epsilon,
                    **factory_kwargs,
                )
                for index in range(depth)
            ]
        )
        self.norm_f = norm_cls(embed_dim, **factory_kwargs)
        if not self.mars_no_fw:
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
        if not self.mars_no_fw:
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

    def _batched_token_mask(
        self,
        num_layers,
        batch_size,
        group_size,
        layer_indices,
        update_index,
        device,
    ):
        """Construct one exact mixed tube/random mask per layer and sample."""
        height, width = self.spatial_size
        patch_count = height * width
        total_count = group_size * patch_count
        mask_count = int(round(total_count * self.layers[0].mask_ratio))
        mask_count = min(max(mask_count, 1), total_count - 1)
        tube_count = int(
            mask_count * self.layers[0].tube_mask_fraction
        ) // group_size
        tube_count = min(tube_count, patch_count - 1)
        if self.training:
            scores = torch.rand(
                num_layers,
                batch_size,
                patch_count,
                device=device,
            )
            tube_tokens = scores.argsort(dim=-1)[..., :tube_count]
        else:
            tokens = torch.arange(
                patch_count,
                device=device,
                dtype=torch.int64,
            )
            scores = (
                tokens.view(1, -1) * 1103515245
                + (update_index + 1) * 12345
                + (layer_indices.view(-1, 1) + 1) * 2654435761
            ).remainder(2147483647)
            tube_tokens = scores.argsort(dim=-1)[:, :tube_count]
            tube_tokens = tube_tokens.unsqueeze(1).expand(
                -1,
                batch_size,
                -1,
            )
        token_mask = torch.zeros(
            num_layers,
            batch_size,
            group_size,
            patch_count,
            dtype=torch.bool,
            device=device,
        )
        tube_mask = torch.zeros(
            num_layers,
            batch_size,
            patch_count,
            dtype=torch.bool,
            device=device,
        )
        tube_mask.scatter_(2, tube_tokens, True)
        token_mask |= tube_mask.unsqueeze(2)
        remaining_count = mask_count - tube_count * group_size
        if self.training:
            remaining_scores = torch.rand(
                num_layers,
                batch_size,
                group_size,
                patch_count,
                device=device,
            )
        else:
            positions = torch.arange(
                total_count,
                device=device,
                dtype=torch.int64,
            ).reshape(1, 1, group_size, patch_count)
            remaining_scores = (
                positions * 1664525
                + (update_index + 1) * 1013904223
                + (layer_indices.view(-1, 1, 1, 1) + 1) * 2246822519
            ).remainder(2147483647).float()
            remaining_scores = remaining_scores.expand(
                -1,
                batch_size,
                -1,
                -1,
            )
        remaining_scores = remaining_scores.masked_fill(token_mask, float("inf"))
        remaining_tokens = remaining_scores.flatten(2).argsort(dim=-1)[
            ..., :remaining_count
        ]
        token_mask.flatten(2).scatter_(2, remaining_tokens, True)
        return token_mask.reshape(
            num_layers,
            batch_size,
            group_size,
            height,
            width,
        )

    def _batched_update_fast_weights(
        self,
        memory_inputs,
        *args,
    ):
        """Update several independent convolutional MARS layers together."""
        num_weights = MaskedResidualDenoiserConv3d.num_weights
        fast_weights = args[:num_weights]
        master_weights = args[num_weights : 2 * num_weights]
        (
            lr_weights,
            mask_tokens,
            layer_indices,
            update_index,
        ) = args[2 * num_weights :]
        num_layers, batch_size, seq_len, dim = memory_inputs.shape
        height, width = self.spatial_size
        tokens_per_tubelet = height * width + 1
        if seq_len % tokens_per_tubelet:
            raise ValueError(
                f"Chunk length {seq_len} is not divisible by "
                f"tokens_per_tubelet={tokens_per_tubelet}"
            )
        group_size = seq_len // tokens_per_tubelet
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
            ).reshape(
                num_layers,
                batch_size,
                seq_len,
                1,
            )
            learning_rates = F.softplus(
                learning_rates + self.layers[0].base_lr_inverse
            )
        token_mask = self._batched_token_mask(
            num_layers,
            batch_size,
            group_size,
            layer_indices,
            update_index,
            memory_inputs.device,
        )
        flat_batch_size = num_layers * batch_size
        expanded_mask_tokens = mask_tokens.unsqueeze(1).expand(
            -1,
            batch_size,
            -1,
        ).reshape(flat_batch_size, dim)
        updated_fast, updated_master = self.layers[0].state.update(
            memory_inputs.flatten(0, 1),
            token_mask.flatten(0, 1),
            learning_rates.flatten(0, 1),
            tuple(weight.flatten(0, 1) for weight in fast_weights),
            tuple(weight.flatten(0, 1) for weight in master_weights),
            self.layers[0].muon_update_steps,
            group_size,
            height,
            width,
            self.layers[0].update_scale,
            expanded_mask_tokens,
        )
        updated_fast = tuple(
            weight.reshape(num_layers, batch_size, *weight.shape[1:])
            for weight in updated_fast
        )
        updated_master = tuple(
            weight.reshape(num_layers, batch_size, *weight.shape[1:])
            for weight in updated_master
        )
        return (*updated_fast, *updated_master)

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
            torch.stack([layer.state.mask_token for layer in layers]),
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
        split = MaskedResidualDenoiserConv3d.num_weights
        return outputs[:split], outputs[split:]

    @staticmethod
    def _init_layer_group_fast_weights(layers, batch_size):
        layer_states = [layer.init_fast_weights(batch_size) for layer in layers]
        fast_weights = tuple(
            torch.stack([state[0][index] for state in layer_states])
            for index in range(MaskedResidualDenoiserConv3d.num_weights)
        )
        master_weights = tuple(
            torch.stack([state[1][index] for state in layer_states])
            for index in range(MaskedResidualDenoiserConv3d.num_weights)
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

        if self.mars_no_fw:
            for index, layer in enumerate(self.layers):
                x = layer.forward_no_fw(
                    x,
                    self.fw_update_group_size,
                    use_checkpoint=(
                        self.use_checkpoint and index < self.checkpoint_num
                    ),
                )
        elif self.fw_update_layer_group_size == 1:
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

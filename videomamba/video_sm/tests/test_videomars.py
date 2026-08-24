import copy
import unittest
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from videomamba.video_sm.models.videomars import (
    MARSBlock,
    MaskedResidualDenoiserConv3d,
    VisionMARS,
)
from videomamba.video_sm.models.videovit import VisionTransformer


class MaskedResidualDenoiserConv3dTest(unittest.TestCase):
    @staticmethod
    def _make_state():
        return MaskedResidualDenoiserConv3d(dim=8, hidden_dim=4)

    @staticmethod
    def _input():
        # Two tubelets, each with one CLS and a 2x2 patch grid.
        return torch.randn(2, 10, 8)

    @staticmethod
    def _mask():
        mask = torch.zeros(2, 2, 2, 2, dtype=torch.bool)
        mask[..., 0, 0] = True
        mask[..., 1, 1] = True
        return mask

    @staticmethod
    def _learning_rates():
        return torch.linspace(0.005, 0.02, 20).reshape(2, 10, 1)

    def test_weighted_reconstruction_directions_match_autograd(self):
        torch.manual_seed(0)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=2)
        fast_weights = tuple(
            weight.detach().requires_grad_(True) for weight in fast_weights
        )
        update_input = self._input()
        mask = self._mask()
        learning_rates = self._learning_rates()
        reconstruction, target = state.reconstruct(
            update_input,
            mask,
            fast_weights,
            group_size=2,
            height=2,
            width=2,
        )
        _, patch_learning_rates = state._split_tokens(
            learning_rates,
            group_size=2,
            height=2,
            width=2,
        )
        loss = (
            0.5
            * (
                (reconstruction.float() - target.detach().float()).square()
                * mask.unsqueeze(-1)
                * patch_learning_rates
            ).sum()
            / 8
        )
        expected = torch.autograd.grad(loss, fast_weights)
        directions = state.reconstruction_directions(
            update_input,
            mask,
            learning_rates,
            fast_weights,
            group_size=2,
            height=2,
            width=2,
            create_graph=False,
        )
        for direction, gradient in zip(directions, expected):
            torch.testing.assert_close(direction, -gradient)

    def test_apply_and_reconstruction_use_same_denoiser(self):
        torch.manual_seed(1)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=1)
        update_input = torch.randn(1, 10, 8)
        no_mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
        reconstruction, _ = state.reconstruct(
            update_input,
            no_mask,
            fast_weights,
            2,
            2,
            2,
        )
        _, patches = state._split_tokens(update_input, 2, 2, 2)
        torch.testing.assert_close(
            reconstruction,
            state.denoise_grid(patches, fast_weights),
        )
        applied = state.apply_denoiser(update_input, fast_weights, 2, 2, 2)
        _, applied_patches = state._split_tokens(applied, 2, 2, 2)
        torch.testing.assert_close(applied_patches, reconstruction)

    def test_denoiser_identity_skip_survives_zero_adapter(self):
        torch.manual_seed(2)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=1)
        x = torch.randn(1, 2, 2, 2, 8)
        zero_up = torch.zeros_like(fast_weights[2])
        actual = state.denoise_grid(
            x,
            (fast_weights[0], fast_weights[1], zero_up),
        )
        expected = F.rms_norm(x, normalized_shape=(8,), eps=1e-5)
        torch.testing.assert_close(actual, expected)

    def test_denoiser_mixes_neighbor_tokens(self):
        torch.manual_seed(3)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=1)
        update_input = torch.randn(1, 10, 8)
        mask = self._mask()[:1]
        first, _ = state.reconstruct(update_input, mask, fast_weights, 2, 2, 2)
        changed_input = update_input.clone()
        changed_input[:, 2] += 1.0
        second, _ = state.reconstruct(
            changed_input,
            mask,
            fast_weights,
            2,
            2,
            2,
        )
        self.assertGreater((second - first)[mask].abs().sum().item(), 0.0)

    def test_unfold_bmm_depthwise_conv_matches_conv3d(self):
        torch.manual_seed(4)
        state = self._make_state()
        x = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
        kernel = torch.randn(2, 4, 3, 3, 3, requires_grad=True)
        actual = state._depthwise_conv3d(x, kernel)
        expected = torch.stack(
            [
                F.conv3d(
                    x[index].permute(3, 0, 1, 2).unsqueeze(0),
                    kernel[index].unsqueeze(1),
                    padding=1,
                    groups=4,
                )
                .squeeze(0)
                .permute(1, 2, 3, 0)
                for index in range(2)
            ]
        )
        torch.testing.assert_close(actual, expected)
        actual_gradients = torch.autograd.grad(
            actual.square().sum(),
            (x, kernel),
            retain_graph=True,
        )
        expected_gradients = torch.autograd.grad(
            expected.square().sum(),
            (x, kernel),
        )
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients
        ):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_update_scale_controls_master_step_and_normalizes_fast_weights(self):
        torch.manual_seed(5)
        state = self._make_state().eval()
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        update_input = self._input()
        token_mask = self._mask()
        learning_rates = self._learning_rates()

        def update(scale):
            return state.update(
                update_input,
                token_mask,
                learning_rates,
                fast_weights,
                master_weights,
                muon_update_steps=1,
                group_size=2,
                height=2,
                width=2,
                update_scale=scale,
            )

        small_fast, small_master = update(0.01)
        large_fast, large_master = update(0.03)
        for index, (initial, small, large) in enumerate(
            zip(master_weights, small_master, large_master)
        ):
            torch.testing.assert_close(large - initial, 3 * (small - initial))
            expected = state._normalize_weight(large, index).to(
                large_fast[index].dtype
            )
            torch.testing.assert_close(large_fast[index], expected)

    def test_one_update_reduces_masked_reconstruction_loss(self):
        torch.manual_seed(2)
        state = MaskedResidualDenoiserConv3d(dim=48, hidden_dim=16).eval()
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        update_input = torch.randn(2, 10, 48)
        token_mask = self._mask()
        learning_rates = torch.full((2, 10, 1), 0.01)

        def loss(weights):
            prediction, target = state.reconstruct(
                update_input,
                token_mask,
                weights,
                2,
                2,
                2,
            )
            return (
                0.5
                * (
                    (prediction.float() - target.float()).square()
                    * token_mask.unsqueeze(-1)
                ).sum()
                / 8
            )

        before = loss(fast_weights)
        updated_fast, _ = state.update(
            update_input,
            token_mask,
            learning_rates,
            fast_weights,
            master_weights,
            muon_update_steps=5,
            group_size=2,
            height=2,
            width=2,
            update_scale=0.03,
        )
        self.assertLess(loss(updated_fast).item(), before.item())


class PerLayerMARSTest(unittest.TestCase):
    @staticmethod
    def _make_block():
        return MARSBlock(
            dim=12,
            num_heads=3,
            norm_cls=partial(nn.RMSNorm, eps=1e-5),
            layer_index=0,
            mars_cnn_dim=6,
            spatial_size=(2, 2),
            fw_base_lr=0.01,
            muon_update_steps=1,
            mask_ratio=0.5,
            tube_mask_fraction=0.5,
            update_scale=0.03,
        )

    def test_mixed_mask_is_exact_and_eval_deterministic(self):
        block = self._make_block().eval()
        first = block._token_mask(2, 3, update_index=3, device="cpu")
        second = block._token_mask(2, 3, update_index=3, device="cpu")
        torch.testing.assert_close(first, second)
        self.assertEqual(first.flatten(1).sum(dim=-1).tolist(), [6, 6])
        # One spatial location is a full tube, while the random half differs
        # over time.
        full_tubes = first.all(dim=1).flatten(1).sum(dim=-1)
        self.assertTrue(torch.all(full_tubes >= 1))
        self.assertTrue(torch.any(first[:, 1:] != first[:, :1]))

    def test_two_groups_give_all_state_meta_gradients(self):
        torch.manual_seed(6)
        block = self._make_block().train()
        with torch.no_grad():
            block.memory_gate.fill_(0.1)
        x = torch.randn(2, 2, 5, 12, requires_grad=True)
        output = block.forward_scan(x, fw_update_group_size=1)
        output.square().mean().backward()

        gradients = [parameter.grad for parameter in block.state.state_parameters()]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        self.assertTrue(all(gradient.abs().sum() > 0 for gradient in gradients))
        self.assertIsNotNone(block.state.mask_token.grad)
        self.assertGreater(block.state.mask_token.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(block.lr_proj.weight.grad)
        self.assertGreater(block.lr_proj.weight.grad.abs().sum().item(), 0.0)

    def test_model_has_independent_state_per_layer(self):
        model = VisionMARS(
            img_size=32,
            patch_size=16,
            depth=2,
            embed_dim=12,
            num_heads=3,
            num_classes=4,
            num_frames=2,
            fw_update_group_size=1,
            muon_update_steps=1,
            mars_cnn_dim=6,
        )
        self.assertFalse(hasattr(model, "shared_state"))
        self.assertIsNot(model.layers[0].state, model.layers[1].state)
        self.assertEqual(model.layers[0].state.hidden_dim, 6)
        output = model(torch.randn(1, 3, 2, 32, 32))
        self.assertEqual(output.shape, (1, 4))

    def test_no_fw_model_is_parameter_identical_to_videovit(self):
        common = dict(
            img_size=32,
            patch_size=16,
            depth=2,
            embed_dim=12,
            num_heads=3,
            num_classes=4,
            num_frames=2,
            kernel_size=1,
        )
        model = VisionMARS(
            **common,
            fw_update_group_size=1,
            mars_no_fw=True,
        )
        vit = VisionTransformer(**common)
        model_state = model.state_dict()
        vit_state = vit.state_dict()
        self.assertEqual(model_state.keys(), vit_state.keys())
        for name in model_state:
            self.assertEqual(model_state[name].shape, vit_state[name].shape)
        parameter_names = dict(model.named_parameters())
        self.assertFalse(any(".state." in name for name in parameter_names))
        self.assertFalse(any("memory" in name for name in parameter_names))
        self.assertFalse(any("lr_proj" in name for name in parameter_names))
        output = model(torch.randn(1, 3, 2, 32, 32))
        self.assertEqual(output.shape, (1, 4))

    def test_no_fw_checkpoint_matches_eager_with_stochastic_depth(self):
        torch.manual_seed(12)
        eager = MARSBlock(
            dim=12,
            num_heads=3,
            norm_cls=partial(nn.RMSNorm, eps=1e-5),
            layer_index=0,
            drop_path=0.2,
            spatial_size=(2, 2),
            no_fw=True,
        ).train()
        checked = copy.deepcopy(eager)
        eager_input = torch.randn(2, 3, 5, 12, requires_grad=True)
        checked_input = eager_input.detach().clone().requires_grad_(True)

        torch.manual_seed(13)
        eager_output = eager.forward_no_fw(eager_input, 2, use_checkpoint=False)
        torch.manual_seed(13)
        checked_output = checked.forward_no_fw(
            checked_input,
            2,
            use_checkpoint=True,
        )
        torch.testing.assert_close(checked_output, eager_output)
        eager_output.square().mean().backward()
        checked_output.square().mean().backward()
        torch.testing.assert_close(checked_input.grad, eager_input.grad)
        for eager_parameter, checked_parameter in zip(
            eager.parameters(), checked.parameters()
        ):
            torch.testing.assert_close(
                checked_parameter.grad,
                eager_parameter.grad,
            )

    def test_full_scan_checkpoint_preserves_mask_rng_and_gradients(self):
        torch.manual_seed(7)
        eager = self._make_block().train()
        checked = copy.deepcopy(eager)
        with torch.no_grad():
            eager.memory_gate.fill_(0.1)
            checked.memory_gate.fill_(0.1)
        eager_input = torch.randn(1, 2, 5, 12, requires_grad=True)
        checked_input = eager_input.detach().clone().requires_grad_(True)

        torch.manual_seed(8)
        eager_output = eager.forward_scan(eager_input, 1, use_checkpoint=False)
        torch.manual_seed(8)
        checked_output = checked.forward_scan(checked_input, 1, use_checkpoint=True)
        torch.testing.assert_close(checked_output, eager_output)
        eager_output.square().mean().backward()
        checked_output.square().mean().backward()
        torch.testing.assert_close(checked_input.grad, eager_input.grad)
        for eager_parameter, checked_parameter in zip(
            eager.parameters(), checked.parameters()
        ):
            if eager_parameter.grad is None:
                self.assertIsNone(checked_parameter.grad)
            else:
                torch.testing.assert_close(
                    checked_parameter.grad,
                    eager_parameter.grad,
                    atol=2e-5,
                    rtol=2e-4,
                )

    @staticmethod
    def _make_cross_layer_model(use_checkpoint=False):
        model = VisionMARS(
            img_size=32,
            patch_size=16,
            depth=2,
            embed_dim=12,
            num_heads=3,
            num_classes=4,
            num_frames=2,
            fw_update_group_size=1,
            fw_update_layer_group_size=2,
            muon_update_steps=1,
            mars_cnn_dim=6,
            use_checkpoint=use_checkpoint,
            checkpoint_num=2 if use_checkpoint else 0,
        )
        with torch.no_grad():
            for layer in model.layers:
                layer.memory_gate.fill_(0.1)
        return model

    def test_cross_layer_g2_matches_layer_major_output_and_gradients(self):
        torch.manual_seed(9)
        layer_major = self._make_cross_layer_model().eval()
        layer_major.fw_update_layer_group_size = 1
        cross_layer = copy.deepcopy(layer_major)
        cross_layer.fw_update_layer_group_size = 2
        layer_input = torch.randn(1, 3, 2, 32, 32)

        layer_output = layer_major(layer_input)
        cross_output = cross_layer(layer_input.detach().clone())
        torch.testing.assert_close(cross_output, layer_output)
        layer_output.square().mean().backward()
        cross_output.square().mean().backward()
        for layer_parameter, cross_parameter in zip(
            layer_major.parameters(), cross_layer.parameters()
        ):
            if layer_parameter.grad is None:
                self.assertIsNone(cross_parameter.grad)
            else:
                torch.testing.assert_close(
                    cross_parameter.grad,
                    layer_parameter.grad,
                    atol=3e-5,
                    rtol=3e-4,
                )

    def test_cross_layer_checkpoint_preserves_mask_rng_and_gradients(self):
        torch.manual_seed(10)
        eager = self._make_cross_layer_model().train()
        checked = copy.deepcopy(eager)
        checked.use_checkpoint = True
        checked.checkpoint_num = 2
        eager_input = torch.randn(1, 3, 2, 32, 32, requires_grad=True)
        checked_input = eager_input.detach().clone().requires_grad_(True)

        torch.manual_seed(11)
        eager_output = eager(eager_input)
        torch.manual_seed(11)
        checked_output = checked(checked_input)
        torch.testing.assert_close(checked_output, eager_output)
        eager_output.square().mean().backward()
        checked_output.square().mean().backward()
        torch.testing.assert_close(checked_input.grad, eager_input.grad)
        for eager_parameter, checked_parameter in zip(
            eager.parameters(), checked.parameters()
        ):
            if eager_parameter.grad is None:
                self.assertIsNone(checked_parameter.grad)
            else:
                torch.testing.assert_close(
                    checked_parameter.grad,
                    eager_parameter.grad,
                    atol=3e-5,
                    rtol=3e-4,
                )


if __name__ == "__main__":
    unittest.main()

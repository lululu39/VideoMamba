import copy
import unittest
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from videomamba.video_sm.models.videomars import (
    MARSBlock,
    MaskedAutoencoderConv3d,
    VisionMARS,
)


class MaskedAutoencoderConv3dTest(unittest.TestCase):
    @staticmethod
    def _make_state():
        return MaskedAutoencoderConv3d(dim=8, hidden_dim=4)

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

    def test_exact_reconstruction_directions_match_autograd(self):
        torch.manual_seed(0)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=2)
        fast_weights = tuple(
            weight.detach().requires_grad_(True) for weight in fast_weights
        )
        update_input = self._input()
        mask = self._mask()
        reconstruction, target = state.reconstruct(
            update_input,
            mask,
            fast_weights,
            group_size=2,
            height=2,
            width=2,
        )
        loss = (
            0.5
            * (
                (reconstruction.float() - target.detach().float()).square()
                * mask.unsqueeze(-1)
            ).sum()
            / 8
        )
        expected = torch.autograd.grad(loss, fast_weights)
        directions = state.reconstruction_directions(
            update_input,
            mask,
            fast_weights,
            group_size=2,
            height=2,
            width=2,
            create_graph=False,
        )
        for direction, gradient in zip(directions, expected):
            torch.testing.assert_close(direction, -gradient)

    def test_encoder_and_decoder_both_mix_neighbor_tokens(self):
        torch.manual_seed(1)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=1)
        update_input = torch.randn(1, 10, 8)
        mask = self._mask()[:1]
        first, _ = state.reconstruct(
            update_input,
            mask,
            fast_weights,
            2,
            2,
            2,
        )
        changed_input = update_input.clone()
        # Change a visible patch neighboring a masked patch.
        changed_input[:, 2] += 1.0
        second, _ = state.reconstruct(
            changed_input,
            mask,
            fast_weights,
            2,
            2,
            2,
        )
        masked_difference = (second - first)[mask].abs().sum()
        self.assertGreater(masked_difference.item(), 0.0)

    def test_unfold_bmm_depthwise_conv_matches_conv3d(self):
        torch.manual_seed(10)
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

    def test_update_normalizes_every_fast_weight(self):
        torch.manual_seed(2)
        state = self._make_state()
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        updated_fast, updated_master = state.update(
            self._input(),
            self._mask(),
            torch.full((2, state.num_weights), 0.01),
            fast_weights,
            master_weights,
            muon_update_steps=1,
            group_size=2,
            height=2,
            width=2,
        )
        for index, (fast_weight, master_weight) in enumerate(
            zip(updated_fast, updated_master)
        ):
            expected = state._normalize_weight(master_weight, index).to(
                fast_weight.dtype
            )
            torch.testing.assert_close(fast_weight, expected)

    def test_one_muon_update_reduces_reconstruction_loss(self):
        torch.manual_seed(2)
        state = MaskedAutoencoderConv3d(dim=48, hidden_dim=16).train()
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        update_input = torch.randn(2, 10, 48)
        mask = self._mask()

        def reconstruction_loss(weights):
            reconstruction, target = state.reconstruct(
                update_input,
                mask,
                weights,
                group_size=2,
                height=2,
                width=2,
            )
            return (
                0.5
                * (
                    (reconstruction.float() - target.float()).square()
                    * mask.unsqueeze(-1)
                ).sum()
                / 8
            )

        before = reconstruction_loss(fast_weights)
        updated_fast, _ = state.update(
            update_input,
            mask,
            torch.full((2, state.num_weights), 0.01),
            fast_weights,
            master_weights,
            muon_update_steps=5,
            group_size=2,
            height=2,
            width=2,
        )
        after = reconstruction_loss(updated_fast)
        self.assertLess(after.item(), before.item())


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
        )

    def test_token_mask_is_tube_shaped_exact_and_eval_deterministic(self):
        block = self._make_block().eval()
        first = block._token_mask(2, 3, update_index=3, device="cpu")
        second = block._token_mask(2, 3, update_index=3, device="cpu")
        torch.testing.assert_close(first, second)
        self.assertEqual(first[:, 0].flatten(1).sum(dim=-1).tolist(), [2, 2])
        torch.testing.assert_close(first[:, :1].expand_as(first), first)

    def test_two_groups_give_all_fast_cnn_weights_exact_meta_gradients(self):
        torch.manual_seed(3)
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
        for token in (
            block.state.encoder_mask_token,
            block.state.decoder_mask_token,
        ):
            self.assertIsNotNone(token.grad)
            self.assertGreater(token.grad.abs().sum().item(), 0.0)

    def test_model_has_independent_state_per_layer_and_no_shared_state(self):
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

    def test_full_scan_checkpoint_preserves_mask_rng_and_gradients(self):
        torch.manual_seed(4)
        eager = self._make_block().train()
        checked = copy.deepcopy(eager)
        with torch.no_grad():
            eager.memory_gate.fill_(0.1)
            checked.memory_gate.fill_(0.1)
        eager_input = torch.randn(1, 2, 5, 12, requires_grad=True)
        checked_input = eager_input.detach().clone().requires_grad_(True)

        torch.manual_seed(5)
        eager_output = eager.forward_scan(
            eager_input,
            fw_update_group_size=1,
            use_checkpoint=False,
        )
        torch.manual_seed(5)
        checked_output = checked.forward_scan(
            checked_input,
            fw_update_group_size=1,
            use_checkpoint=True,
        )
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
        torch.manual_seed(7)
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
        torch.manual_seed(8)
        eager = self._make_cross_layer_model().train()
        checked = copy.deepcopy(eager)
        checked.use_checkpoint = True
        checked.checkpoint_num = 2
        eager_input = torch.randn(1, 3, 2, 32, 32, requires_grad=True)
        checked_input = eager_input.detach().clone().requires_grad_(True)

        torch.manual_seed(9)
        eager_output = eager(eager_input)
        torch.manual_seed(9)
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

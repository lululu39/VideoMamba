import copy
import unittest
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from videomamba.video_sm.models.videomars import (
    MARSBlock,
    MaskedReconstructionSwiGLU,
    VisionMARS,
)
from videomamba.video_sm.models.videolact import (
    FastWeightSwiGLU,
    zeropower_via_newtonschulz5,
)


class MaskedReconstructionSwiGLUTest(unittest.TestCase):
    @staticmethod
    def _make_state():
        return MaskedReconstructionSwiGLU(
            dim=8,
            inter_multi=2,
            num_heads=1,
        )

    def test_analytic_reconstruction_directions_match_autograd(self):
        torch.manual_seed(0)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=2)
        fast_weights = tuple(
            weight.detach().requires_grad_(True) for weight in fast_weights
        )
        masked_input = torch.randn(2, 5, 8)
        target = torch.randn_like(masked_input)
        mask = torch.rand_like(masked_input) > 0.5
        learning_rates = torch.ones(2, 5, 3)

        reconstruction, _ = state.reconstruct(masked_input, fast_weights)
        loss = (
            0.5
            * ((reconstruction - target).square() * mask).sum()
            / masked_input.shape[1]
        )
        autograd_gradients = torch.autograd.grad(loss, fast_weights)
        directions = state.reconstruction_directions(
            masked_input,
            target,
            mask,
            learning_rates,
            fast_weights,
        )
        for direction, gradient in zip(directions, autograd_gradients):
            torch.testing.assert_close(
                direction,
                -gradient,
                atol=2e-5,
                rtol=2e-4,
            )

    def test_muon_batches_all_three_matrices_without_changing_math(self):
        torch.manual_seed(1)
        state = self._make_state()
        fast_weights, _ = state.init_fast_weights(batch_size=2)
        gradients = tuple(torch.randn_like(weight) for weight in fast_weights)
        updates = state.muon_updates(gradients, steps=3)
        for index, (gradient, update) in enumerate(zip(gradients, updates)):
            is_w1 = index == 1
            oriented = gradient.transpose(-1, -2) if is_w1 else gradient
            expected = zeropower_via_newtonschulz5(oriented.flatten(0, 1), 3)
            expected = expected.reshape_as(oriented)
            if is_w1:
                expected = expected.transpose(-1, -2)
            torch.testing.assert_close(update, expected)

    def test_unmasked_error_path_is_exactly_the_lact_update(self):
        torch.manual_seed(6)
        state = self._make_state()
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        update_input = torch.randn(2, 5, 8)
        target = torch.randn_like(update_input)
        learning_rates = torch.rand(2, 5, 3).add(0.01)
        reconstruction_mask = torch.ones_like(update_input, dtype=torch.bool)
        mars_fast, mars_master = state.update(
            update_input,
            target,
            reconstruction_mask,
            learning_rates,
            fast_weights,
            master_weights,
            muon_update_steps=5,
        )
        lact_fast, lact_master = FastWeightSwiGLU.update_preprojected(
            state,
            update_input,
            target,
            learning_rates,
            fast_weights,
            master_weights,
            5,
        )
        for mars, lact in zip(
            (*mars_fast, *mars_master),
            (*lact_fast, *lact_master),
        ):
            torch.testing.assert_close(mars, lact)

    def test_updated_fast_weights_use_lact_input_dimension_normalization(self):
        torch.manual_seed(2)
        state = self._make_state()
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        masked_input = torch.randn(2, 5, 8)
        target = torch.randn_like(masked_input)
        mask = torch.rand_like(masked_input) > 0.5
        learning_rates = torch.full((2, 5, 3), 0.01)
        updated_fast, updated_master = state.update(
            masked_input,
            target,
            mask,
            learning_rates,
            fast_weights,
            master_weights,
            muon_update_steps=1,
        )
        for fast_weight, master_weight in zip(updated_fast, updated_master):
            expected = F.normalize(master_weight, dim=2, eps=1e-5).to(
                fast_weight.dtype
            )
            torch.testing.assert_close(fast_weight, expected)

    def test_one_muon_update_reduces_masked_reconstruction_loss(self):
        torch.manual_seed(0)
        state = MaskedReconstructionSwiGLU(48, inter_multi=2, num_heads=1)
        fast_weights, master_weights = state.init_fast_weights(batch_size=2)
        target = torch.randn(2, 64, 48)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[..., ::2] = True
        masked_input = target.masked_fill(mask, 0)
        learning_rates = torch.full((2, 64, 3), 0.01)

        def reconstruction_loss(weights):
            prediction, _ = state.reconstruct(masked_input, weights)
            return 0.5 * (
                (prediction.float() - target).square() * mask
            ).sum() / masked_input.shape[1]

        before = reconstruction_loss(fast_weights)
        updated_fast, _ = state.update(
            masked_input,
            target,
            mask,
            learning_rates,
            fast_weights,
            master_weights,
            muon_update_steps=5,
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
            fw_inter_multi=2,
            fw_num_heads=1,
            fw_base_lr=0.01,
            muon_update_steps=1,
            mask_ratio=0.5,
        )

    def test_feature_mask_is_exact_and_eval_deterministic(self):
        block = self._make_block().eval()
        first = block._feature_mask(2, 5, update_index=3, device="cpu")
        second = block._feature_mask(2, 5, update_index=3, device="cpu")
        torch.testing.assert_close(first, second)
        self.assertEqual(first[:, 0].sum(dim=-1).tolist(), [6, 6])
        torch.testing.assert_close(first[:, :1].expand_as(first), first)

    def test_two_groups_give_all_state_weights_exact_meta_gradients(self):
        torch.manual_seed(3)
        block = self._make_block().train()
        with torch.no_grad():
            block.memory_gate.fill_(0.1)
        x = torch.randn(2, 2, 5, 12, requires_grad=True)
        output = block.forward_scan(x, fw_update_group_size=1)
        output.square().mean().backward()

        # Gate/up/down are all part of one state mapping.
        for parameters in (
            (block.state.w0, block.state.w2),
            (block.state.w1,),
        ):
            gradients = [parameter.grad for parameter in parameters]
            self.assertTrue(gradients)
            self.assertTrue(all(grad is not None for grad in gradients))
            self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
            self.assertGreater(
                sum(grad.float().square().sum().item() for grad in gradients),
                0.0,
            )

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
        )
        self.assertFalse(hasattr(model, "shared_state"))
        self.assertIsNot(model.layers[0].state, model.layers[1].state)
        self.assertEqual(model.layers[0].state.hidden_dim, 24)
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
            eager.parameters(),
            checked.parameters(),
        ):
            if eager_parameter.grad is None:
                self.assertIsNone(checked_parameter.grad)
            else:
                torch.testing.assert_close(
                    checked_parameter.grad,
                    eager_parameter.grad,
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
        cross_input = layer_input.detach().clone()

        layer_output = layer_major(layer_input)
        cross_output = cross_layer(cross_input)
        torch.testing.assert_close(cross_output, layer_output)
        layer_output.square().mean().backward()
        cross_output.square().mean().backward()
        for layer_parameter, cross_parameter in zip(
            layer_major.parameters(),
            cross_layer.parameters(),
        ):
            if layer_parameter.grad is None:
                self.assertIsNone(cross_parameter.grad)
            else:
                torch.testing.assert_close(
                    cross_parameter.grad,
                    layer_parameter.grad,
                    atol=2e-5,
                    rtol=2e-4,
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
            eager.parameters(),
            checked.parameters(),
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


if __name__ == "__main__":
    unittest.main()

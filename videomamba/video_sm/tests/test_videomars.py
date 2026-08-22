import unittest
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from videomamba.video_sm.models.videomars import (
    FastWeightTransformerEncoder,
    SharedMARSState,
)


class SharedMARSStateTest(unittest.TestCase):
    @staticmethod
    def _make_state():
        return SharedMARSState(
            dim=12,
            pixel_dim=6,
            norm_cls=partial(nn.RMSNorm, eps=1e-5),
            tokens_per_tubelet=5,
            max_chunk_size=5,
            encoder_dim=8,
            encoder_depth=1,
            encoder_num_heads=1,
            fw_inter_multi=2,
            muon_update_steps=1,
            mask_ratio=0.5,
            decoder_dim=4,
            decoder_depth=1,
            decoder_num_heads=1,
        )

    def test_muon_straight_through_preserves_forward_and_uses_identity_backward(self):
        encoder = FastWeightTransformerEncoder(
            input_dim=6,
            encoder_dim=4,
            depth=1,
            num_heads=1,
        )
        gradient = torch.randn(2, 6, 4, requires_grad=True)
        exact = encoder.muon_descent(
            (gradient,),
            steps=3,
            backward_mode="exact",
        )[0]
        straight_through = encoder.muon_descent(
            (gradient,),
            steps=3,
            backward_mode="straight_through",
        )[0]

        torch.testing.assert_close(straight_through, exact)
        straight_through.sum().backward()
        torch.testing.assert_close(gradient.grad, torch.ones_like(gradient))

        gradient.grad = None
        normalized_straight_through = encoder.muon_descent(
            (gradient,),
            steps=3,
            backward_mode="normalized_straight_through",
            backward_gain=7.0,
        )[0]
        torch.testing.assert_close(normalized_straight_through, exact)
        normalized_straight_through.sum().backward()
        expected_scale = (
            gradient.detach()
            .float()
            .norm(dim=(-2, -1), keepdim=True)
            .add(1e-7)
            .reciprocal()
        )
        torch.testing.assert_close(
            gradient.grad,
            (7.0 * expected_scale).expand_as(gradient),
        )

    def test_updated_fast_weights_normalize_the_input_dimension(self):
        torch.manual_seed(0)
        state = self._make_state().eval()
        memory_input = torch.randn(2, 5, 12)
        pixel_targets = torch.randn(2, 5, 6)
        fast_weights, master_weights = state.state_encoder.init_state(2)

        updated_fast_weights, updated_master_weights = state._update_state(
            memory_input,
            pixel_targets,
            fast_weights,
            master_weights,
            update_index=0,
        )

        for fast_weight, master_weight in zip(
            updated_fast_weights,
            updated_master_weights,
        ):
            expected = F.normalize(master_weight, dim=1, eps=1e-5).to(
                fast_weight.dtype
            )
            torch.testing.assert_close(fast_weight, expected)
            torch.testing.assert_close(
                torch.linalg.vector_norm(fast_weight.float(), dim=1),
                torch.ones_like(fast_weight[:, 0, :], dtype=torch.float32),
                atol=1e-5,
                rtol=1e-5,
            )

    def test_default_surrogate_keeps_encoder_and_decoder_meta_gradients(self):
        torch.manual_seed(1)
        state = self._make_state().train()
        with torch.no_grad():
            state.memory_gate.fill_(0.1)
        memory_input = torch.randn(2, 2, 5, 12, requires_grad=True)
        pixel_targets = torch.randn(2, 2, 4, 6)

        output = state(memory_input, pixel_targets, fw_update_group_size=1)
        output.square().mean().backward()

        for module in (state.state_encoder, state.decoder):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.grad is not None
            ]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
            self.assertGreater(
                sum(grad.float().square().sum().item() for grad in gradients),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()

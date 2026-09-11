"""CPU tests for the optional candidate-C scale projection."""
import unittest
from types import SimpleNamespace

import torch

from runtime_v3 import fresh_topology, prune
from scale_bounds_v3 import apply_scale_bounds


def config(enabled=True):
    return SimpleNamespace(
        scale_bounds=enabled,
        size_ratio=2.0,
        thickness_ratio=0.1,
        pruning=True,
        prune_start=0,
        prune_interval=1,
        prune_opacity=0.005,
        prune_min_views=1,
        prune_patience=1,
        prune_max_fraction=0.5,
    )


class FakeGaussian:
    def __init__(self, scales):
        self._scaling = torch.nn.Parameter(torch.as_tensor(scales, dtype=torch.float32).log())
        self.optimizer = None

    @property
    def get_scaling(self):
        return self._scaling.exp()


class ScaleBoundsTests(unittest.TestCase):
    def test_off_is_exact_noop(self):
        g = FakeGaussian([[9.0, 8.0, 7.0]])
        identity = g._scaling
        before = g._scaling.detach().clone()
        stats = apply_scale_bounds(g, {"spacing": torch.tensor([1.0])}, config(False))
        self.assertIs(g._scaling, identity)
        torch.testing.assert_close(g._scaling, before, rtol=0, atol=0)
        self.assertFalse(bool(stats["enabled"]))
        self.assertEqual(int(stats["clipped_coordinates"]), 0)

    def test_zero_confidence_is_still_bounded(self):
        g = FakeGaussian([[3.0, 1.0, 0.2]])
        reference = {"spacing": torch.tensor([1.0]), "confidence": torch.tensor([0.0])}
        stats = apply_scale_bounds(g, reference, config())
        torch.testing.assert_close(g.get_scaling, torch.tensor([[2.0, 1.0, 0.1]]))
        self.assertEqual(int(stats["clipped_coordinates"]), 2)

    def test_single_extreme_is_clipped_and_small_values_stay(self):
        g = FakeGaussian([[1.0, 1.5, 0.05], [200.0, 1.0, 4.0]])
        before = g.get_scaling.detach().clone()
        stats = apply_scale_bounds(g, {"spacing": torch.ones(2)}, config())
        torch.testing.assert_close(g.get_scaling[0], before[0], rtol=0, atol=0)
        torch.testing.assert_close(g.get_scaling[1], torch.tensor([2.0, 1.0, 0.1]))
        self.assertEqual(int(stats["clipped_points"]), 1)
        self.assertEqual(int(stats["clipped_coordinates"]), 2)
        self.assertGreater(float(stats["max_log_excess_before"]), 0)
        self.assertEqual(float(stats["max_log_excess_after"]), 0)

    def test_adam_state_is_cleared_only_for_clipped_coordinates(self):
        g = FakeGaussian([[1.0, 3.0, 0.2], [1.0, 1.0, 0.05]])
        original_parameter = g._scaling
        g.optimizer = torch.optim.Adam([g._scaling], lr=0.01, amsgrad=True)
        g._scaling.grad = torch.ones_like(g._scaling)
        g.optimizer.step()
        state = g.optimizer.state[g._scaling]
        for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            state[name].copy_(torch.arange(1, 7, dtype=torch.float32).reshape(2, 3))
        before = {name: state[name].clone() for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")}

        apply_scale_bounds(g, {"spacing": torch.ones(2)}, config())

        self.assertIs(g._scaling, original_parameter)
        clipped = torch.tensor([[False, True, True], [False, False, False]])
        for name in before:
            self.assertTrue(torch.equal(state[name][clipped], torch.zeros(2)))
            torch.testing.assert_close(state[name][~clipped], before[name][~clipped])

    def test_repeated_outward_updates_remain_finite_and_legal(self):
        g = FakeGaussian([[1.0, 1.0, 0.05], [0.5, 0.5, 0.02]])
        g.optimizer = torch.optim.Adam([g._scaling], lr=0.5)
        reference = {"spacing": torch.tensor([1.0, 0.25])}
        a = config()
        for _ in range(20):
            g.optimizer.zero_grad(set_to_none=True)
            g._scaling.grad = -torch.ones_like(g._scaling)
            g.optimizer.step()
            apply_scale_bounds(g, reference, a)
        limits = torch.tensor([[2.0, 2.0, 0.1], [0.5, 0.5, 0.025]])
        self.assertTrue(torch.isfinite(g.get_scaling).all())
        self.assertTrue((g.get_scaling <= limits).all())

    def test_pruning_keeps_reference_alignment_for_projection(self):
        class Prunable(FakeGaussian):
            def __init__(self):
                super().__init__([[1.0, 1.0, 0.05], [9.0, 9.0, 9.0], [5.0, 5.0, 5.0]])
                self.get_xyz = torch.zeros(3, 3)
                self.get_opacity = torch.tensor([[0.1], [0.001], [0.1]])

            def prune_points(self, mask):
                keep = ~mask
                self._scaling = torch.nn.Parameter(self._scaling.detach()[keep].clone())
                self.get_xyz = self.get_xyz[keep]
                self.get_opacity = self.get_opacity[keep]

        g = Prunable()
        reference = {
            "spacing": torch.tensor([1.0, 4.0, 0.5]),
            "confidence": torch.tensor([1.0, 0.0, 1.0]),
            "source_id": torch.arange(3),
        }
        state = fresh_topology(g)
        state["epoch_views"].fill_(2)
        reference, state, event = prune(g, reference, state, config(), 1)
        self.assertEqual(event["removed"], 1)
        self.assertEqual(reference["source_id"].tolist(), [0, 2])

        apply_scale_bounds(g, reference, config())

        expected_limits = torch.tensor([[2.0, 2.0, 0.1], [1.0, 1.0, 0.05]])
        self.assertEqual(len(g.get_xyz), len(reference["spacing"]))
        self.assertTrue((g.get_scaling <= expected_limits * (1 + 1e-6)).all())


if __name__ == "__main__":
    unittest.main()

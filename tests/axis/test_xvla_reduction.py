"""X-VLA's per-sample loss (reduction="none") averages to its scalar loss: every sample has the same
number of elements (chunk x real action dims), so the mean of the per-sample means is the mean."""

import types

import torch

from lerobot.policies.xvla.action_hub import AutoActionSpace
from lerobot.policies.xvla.modeling_xvla import XVLAModel


def test_the_per_sample_mean_is_the_scalar_loss():
    torch.manual_seed(0)
    space = AutoActionSpace(real_dim=8, max_dim=20)
    pred, target = torch.randn(4, 30, 20), torch.randn(4, 30, 8)
    per_sample = XVLAModel._per_sample_loss(types.SimpleNamespace(action_space=space), pred, target)
    assert per_sample.shape == (4,)
    torch.testing.assert_close(
        per_sample.mean(), space.compute_loss(pred, target)["joints_loss"], rtol=1e-6, atol=0
    )

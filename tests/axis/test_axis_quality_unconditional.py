"""Pure conditioning: an artifact with no NO_TAG row loads only when the run declares it."""

import pytest
import torch

from lerobot.configs.train import TrainPipelineConfig
from tests.axis.test_axis_quality import _quality_npz


def test_an_all_tagged_artifact_needs_the_declaration(tmp_path):
    from lerobot.utils.axis_quality import AxisQualityTags, apply_quality_tags

    path = str(_quality_npz(tmp_path, [3, 3, 5, 1]))
    with pytest.raises(ValueError, match="no NO_TAG(.|\n)*axis_quality_unconditional = false"):
        AxisQualityTags(path, expected_frames=4)
    q = AxisQualityTags(path, expected_frames=4, require_unconditional=False)
    batch = {"task": ["pick"], "index": torch.tensor([2])}
    apply_quality_tags(batch, q)
    assert batch["task"] == ["pick\nQuality: 5"]


def test_the_default_keeps_the_paper_behaviour():
    field = TrainPipelineConfig.__dataclass_fields__["axis_quality_unconditional"]
    assert field.default is True

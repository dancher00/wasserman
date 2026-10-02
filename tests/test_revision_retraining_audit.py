import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from audit_revision_retraining import tensor_differences  # noqa: E402

pytestmark = pytest.mark.unit


def test_retraining_tensor_comparison_keeps_real_differences_and_equal_integer_counters():
    left = {"weight": torch.tensor([1.0, 2.0]), "counter": torch.tensor(10000)}
    right = {"weight": torch.tensor([1.0, 2.25]), "counter": torch.tensor(10000)}
    assert tensor_differences(left, left) == {}
    assert tensor_differences(left, right) == {"weight": 0.25}


@pytest.mark.parametrize("right", [torch.tensor([float("nan")]), torch.tensor([1.0, 2.0]), torch.tensor([1])])
def test_retraining_tensor_comparison_rejects_nonfinite_shape_or_dtype_changes(right):
    with pytest.raises(ValueError, match="Invalid tensor"):
        tensor_differences({"weight": torch.tensor([1.0])}, {"weight": right})

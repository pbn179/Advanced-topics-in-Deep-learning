"""Checks for architecture accounting and the iterative pruning invariants."""

import math

import pytest
import torch

from models import ARCHITECTURES, MLP, parameter_count
from pruning import apply_masks, make_masks, prune_masks, rewind_weights, weight_statistics


@pytest.mark.parametrize("name,count", [("A", 200013), ("B", 200054), ("C", 199979),
                                        ("LeNet300100", 266610), ("legacy_A", 266730),
                                        ("legacy_B", 266730), ("D", 266436)])
def test_parameter_counts_and_output_shape(name, count):
    model = MLP(*ARCHITECTURES[name])
    assert parameter_count(model) == count
    assert model(torch.zeros(2, 1, 28, 28)).shape == (2, 10)
    assert model(torch.zeros(2, 784)).shape == (2, 10)
    for layer in (model.fc1, model.fc2, model.fc3):
        assert torch.count_nonzero(layer.bias) == 0


def test_layerwise_rates_and_monotone_masks():
    model = MLP()
    original_masks = make_masks(model)
    masks = original_masks
    for _ in range(13):
        next_masks = prune_masks(model, masks)
        for name, previous in masks.items():
            rate = 0.1 if name == "fc3" else 0.2
            active = int(previous.sum())
            assert int(next_masks[name].sum()) == active - math.floor(rate * active)
            assert not torch.any(next_masks[name] & ~previous)
            assert next_masks[name].data_ptr() != previous.data_ptr()
        masks = next_masks
    assert all(torch.all(mask) for mask in original_masks.values())
    assert weight_statistics(model, original_masks)["remaining_weights"] == 266200


def test_smallest_surviving_weights_and_stable_ties():
    model = MLP(2, 2)
    with torch.no_grad():
        for layer in (model.fc1, model.fc2, model.fc3):
            layer.weight.fill_(1.0)
        model.fc2.weight.copy_(torch.tensor([[0.0, -2.0], [1.0, 4.0]]))
    masks = make_masks(model)
    masks["fc2"][0, 0] = False
    new_masks = prune_masks(model, masks, hidden_rate=1 / 3, output_rate=0)
    assert torch.equal(new_masks["fc2"], torch.tensor([[False, True], [False, True]]))
    # Tied magnitudes in fc1 are removed in their flattened index order.
    count = math.floor(model.fc1.weight.numel() / 3)
    assert not new_masks["fc1"].flatten()[:count].any()
    assert new_masks["fc1"].flatten()[count:].all()


def test_rewind_preserves_initial_survivors_and_restores_biases():
    model = MLP(12, 4)
    initial = {key: value.detach().clone() for key, value in model.state_dict().items()}
    masks = prune_masks(model, make_masks(model))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(5.0)
    rewind_weights(model, initial, masks)
    for name in ("fc1", "fc2", "fc3"):
        layer = getattr(model, name)
        assert torch.equal(layer.weight[masks[name]], initial[f"{name}.weight"][masks[name]])
        assert not layer.weight[~masks[name]].any()
        assert torch.equal(layer.bias, initial[f"{name}.bias"])
    # Stored initialization is not aliased with the restored model state.
    with torch.no_grad():
        model.fc1.weight.add_(1.0)
    assert not torch.equal(model.fc1.weight, initial["fc1.weight"])


def test_mask_application_prevents_regrowth_over_optimizer_updates():
    torch.manual_seed(7)
    model = MLP(12, 4)
    masks = prune_masks(model, make_masks(model))
    apply_masks(model, masks)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)
    inputs = torch.randn(8, 1, 28, 28)
    labels = torch.arange(8) % 10
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(inputs), labels).backward()
        optimizer.step()
        apply_masks(model, masks)
        for name in masks:
            assert torch.count_nonzero(getattr(model, name).weight[~masks[name]]) == 0


def test_extreme_rates_and_statistics():
    model = MLP(5, 3)
    masks = make_masks(model)
    no_pruning = prune_masks(model, masks, 0.0, 0.0)
    assert all(torch.equal(mask, no_pruning[name]) for name, mask in masks.items())
    all_pruned = prune_masks(model, masks, 1.0, 1.0)
    stats = weight_statistics(model, all_pruned)
    assert stats["remaining_weights"] == 0
    assert stats["sparsity"] == 1.0
    assert stats["remaining_percent"] == 0.0
    again = prune_masks(model, all_pruned)
    assert all(torch.equal(mask, again[name]) for name, mask in all_pruned.items())


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_rates_rejected(rate):
    model = MLP(5, 3)
    with pytest.raises(ValueError, match="Pruning rates"):
        prune_masks(model, make_masks(model), rate, 0.1)


def test_invalid_mask_rejected():
    model = MLP(5, 3)
    masks = make_masks(model)
    masks["fc1"] = masks["fc1"].float()
    with pytest.raises(ValueError, match="bool tensor"):
        apply_masks(model, masks)


def test_c_exact_one_round_counts_and_dense_biases():
    model = MLP()
    masks = prune_masks(model, make_masks(model))
    assert {name: int(mask.sum()) for name, mask in masks.items()} == {
        "fc1": 188160, "fc2": 24000, "fc3": 900,
    }
    assert weight_statistics(model, masks)["remaining_weights"] == 213060
    assert sum(layer.bias.numel() for layer in (model.fc1, model.fc2, model.fc3)) == 410

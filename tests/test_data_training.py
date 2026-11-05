"""Fast, offline checks; synthetic examples here are never experiment results."""

import random

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

import data
import train
from models import MLP
from pruning import make_masks, prune_masks, rewind_weights


class FakeMNIST(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.full((1, 28, 28), index / self.length), index % 10


def toy_loaders(seed=7):
    generator = torch.Generator().manual_seed(10)
    labels = torch.arange(64) % 2
    images = torch.randn(64, 1, 28, 28, generator=generator) * 0.01
    images += labels[:, None, None, None] * 0.8
    dataset = TensorDataset(images, labels)
    return (
        DataLoader(dataset, batch_size=16, shuffle=True, generator=torch.Generator().manual_seed(seed)),
        DataLoader(dataset, batch_size=23),
    )


def test_fixed_partition_reproducible_disjoint_and_complete():
    first = data.split_indices()
    data.set_seed(999)
    second = data.split_indices()
    assert first == second
    training, validation = first
    assert len(training) == 55_000
    assert len(validation) == 5_000
    assert not set(training) & set(validation)
    assert set(training) | set(validation) == set(range(60_000))
    assert first != data.split_indices(split_seed=43)


def test_all_rng_seeds_repeat():
    data.set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(4))
    data.set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(4))
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_loaders_full_sizes_and_fixed_split_across_model_seeds(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "_mnist_dataset", lambda root, train, **kwargs: FakeMNIST(60_000 if train else 10_000))
    first = data.make_loaders(tmp_path, seed=1, download=False)
    second = data.make_loaders(tmp_path, seed=3, download=False)
    assert {name: len(loader.dataset) for name, loader in first.items()} == {
        "train": 55_000, "validation": 5_000, "test": 10_000,
    }
    assert first["train"].dataset.indices == second["train"].dataset.indices
    assert first["validation"].dataset.indices == second["validation"].dataset.indices
    first_batch = next(iter(first["train"]))[0]
    assert not torch.equal(first_batch, next(iter(second["train"]))[0])
    replay = data.make_loaders(tmp_path, seed=1, download=False)
    assert torch.equal(first_batch, next(iter(replay["train"]))[0])
    limited = data.make_loaders(tmp_path, train_limit=120, val_limit=40, test_limit=30, download=False)
    assert [len(limited[name].dataset) for name in ("train", "validation", "test")] == [120, 40, 30]


@pytest.mark.parametrize("validation_size", [0, 60_000, 60_001])
def test_invalid_partition_rejected(validation_size):
    with pytest.raises(ValueError):
        data.split_indices(validation_size=validation_size)


def test_evaluate_is_example_weighted_and_restores_mode():
    model = MLP(8, 4)
    _, loader = toy_loaders()
    model.train()
    metrics = train.evaluate(model, loader)
    images, labels = loader.dataset.tensors
    with torch.no_grad():
        logits = model(images)
        expected_loss = torch.nn.functional.cross_entropy(logits, labels).item()
        expected_accuracy = (logits.argmax(1) == labels).float().mean().item()
    assert metrics["loss"] == pytest.approx(expected_loss, abs=1e-6)
    assert metrics["accuracy"] == pytest.approx(expected_accuracy)
    assert model.training


def test_stage_keeps_full_budget_and_selects_accuracy_at_minimum_loss(monkeypatch):
    data.set_seed(2)
    model = MLP(8, 4)
    loaders = toy_loaders()
    sequence = iter([{"loss": 1.0, "accuracy": 0.3}, {"loss": 0.5, "accuracy": 0.8}, {"loss": 0.7, "accuracy": 0.9}])
    monkeypatch.setattr(train, "evaluate", lambda *args, **kwargs: next(sequence))
    seen = []
    handle = model.register_forward_hook(lambda *args: seen.append(1))
    try:
        result = train.train_stage(model, *loaders, make_masks(model), iterations=7, eval_interval=3, progress=False)
    finally:
        handle.remove()
    assert len(seen) == 7
    assert [row["iteration"] for row in result["history"]] == [3, 6, 7]
    assert result["best_iteration"] == 6
    assert result["best_validation_accuracy"] == 0.8
    assert result["best_validation_loss"] == 0.5
    assert result["runtime_seconds"] > 0


def test_masked_adam_training_never_regrows_pruned_weights():
    data.set_seed(5)
    model = MLP(8, 4)
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    masks = prune_masks(model, make_masks(model))
    rewind_weights(model, initial, masks)
    train.train_stage(model, *toy_loaders(), masks, iterations=8, eval_interval=4, progress=False)
    for name, mask in masks.items():
        layer = getattr(model, name)
        assert torch.count_nonzero(layer.weight[~mask]).item() == 0
        assert torch.count_nonzero(layer.weight.grad[~mask]).item() == 0


def test_short_training_learns_and_repeats_exactly_on_cpu():
    results = []
    states = []
    for _ in range(2):
        data.set_seed(123)
        model = MLP(16, 8)
        loaders = toy_loaders()
        before = train.evaluate(model, loaders[1])["loss"]
        result = train.train_stage(model, *loaders, make_masks(model), iterations=40,
                                   learning_rate=0.01, eval_interval=10, progress=False)
        assert result["best_validation_loss"] < before
        assert result["best_validation_accuracy"] >= 0.95
        assert np.isfinite(result["training_loss"])
        results.append(result)
        states.append({name: value.clone() for name, value in model.state_dict().items()})
    assert results[0]["history"] == results[1]["history"]
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])


def test_cpu_device_selection():
    assert train.choose_device("cpu") == torch.device("cpu")


def test_each_training_stage_creates_fresh_adam(monkeypatch):
    original = torch.optim.Adam
    optimizers = []
    def observed(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        assert not optimizer.state
        optimizers.append(optimizer)
        return optimizer
    monkeypatch.setattr(torch.optim, "Adam", observed)
    model = MLP(8, 4)
    for _ in range(2):
        train.train_stage(model, *toy_loaders(), make_masks(model), iterations=2,
                          eval_interval=1, progress=False)
    assert len(optimizers) == 2
    assert optimizers[0] is not optimizers[1]
    for optimizer in optimizers:
        assert all(int(state["step"]) == 2 for state in optimizer.state.values())


def test_reported_best_iteration_does_not_restore_earlier_weights(monkeypatch):
    model = MLP(8, 4)
    snapshots = []
    def validation(*args, **kwargs):
        snapshots.append({name: value.clone() for name, value in model.state_dict().items()})
        return {"loss": float(len(snapshots)), "accuracy": .5}
    monkeypatch.setattr(train, "evaluate", validation)
    result = train.train_stage(model, *toy_loaders(), make_masks(model), iterations=4,
                               eval_interval=2, progress=False)
    assert result["best_iteration"] == 2
    assert all(torch.equal(value, snapshots[-1][name]) for name, value in model.state_dict().items())
    assert any(not torch.equal(value, snapshots[0][name]) for name, value in model.state_dict().items())


def test_preloaded_pixels_exactly_match_totorchvision_totensor():
    from types import SimpleNamespace
    from torchvision.transforms import ToTensor
    from PIL import Image
    images = torch.randint(0, 256, (5, 28, 28), dtype=torch.uint8)
    labels = torch.arange(5)
    materialized = data.tensorize_mnist(SimpleNamespace(data=images, targets=labels))
    for index in range(5):
        image, label = materialized[index]
        assert torch.equal(image, ToTensor()(Image.fromarray(images[index].numpy())))
        assert label == labels[index]


def test_diagnostic_validation_does_not_select_off_grid_minimum(monkeypatch):
    model = MLP(8, 4)
    observations = iter([.1, .5, .2, .4])
    monkeypatch.setattr(train, 'evaluate', lambda *args, **kwargs: {'loss': next(observations), 'accuracy': .5})
    result = train.train_stage(model, *toy_loaders(), make_masks(model), iterations=4,
                               eval_interval=2, trace_interval=1, progress=False)
    assert result['best_iteration'] == 4
    assert result['best_validation_loss'] == .4
    assert [row['selection_eligible'] for row in result['history']] == [False, True, False, True]


def test_diagnostic_sampling_does_not_change_training_trajectory():
    states = []
    results = []
    for trace in (None, 1):
        data.set_seed(7)
        model = MLP(8, 4)
        result = train.train_stage(model, *toy_loaders(), make_masks(model), iterations=6,
                                   eval_interval=3, trace_interval=trace, progress=False)
        states.append({name: value.clone() for name, value in model.state_dict().items()})
        results.append(result)
    assert all(torch.equal(states[0][name], states[1][name]) for name in states[0])
    assert results[0]['best_iteration'] == results[1]['best_iteration']
    assert results[0]['best_validation_accuracy'] == results[1]['best_validation_accuracy']

"""Layer-wise iterative magnitude pruning and rewinding to initialization.

Masks contain only Linear weight tensors. Biases are never pruned. Pruning
uses the *final* trained weights of a stage; validation checkpoint selection
must not affect which weights survive into the next stage.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn


def _linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    layers = {name: layer for name, layer in model.named_modules()
              if isinstance(layer, nn.Linear)}
    if not layers:
        raise ValueError("The model must contain at least one Linear layer.")
    return layers


def _validate_masks(model: nn.Module, masks: Mapping[str, Tensor]) -> dict[str, nn.Linear]:
    layers = _linear_layers(model)
    if set(masks) != set(layers):
        raise ValueError("Masks must contain exactly one entry per Linear layer.")
    for name, layer in layers.items():
        if masks[name].shape != layer.weight.shape or masks[name].dtype != torch.bool:
            raise ValueError(f"Mask {name!r} must be a bool tensor shaped like its weights.")
    return layers


def make_masks(model: nn.Module) -> dict[str, Tensor]:
    """Create all-active masks on the same device as each model weight."""
    return {name: torch.ones_like(layer.weight, dtype=torch.bool)
            for name, layer in _linear_layers(model).items()}


@torch.no_grad()
def apply_masks(model: nn.Module, masks: Mapping[str, Tensor]) -> None:
    """Force pruned weights to zero; call before training and after each update."""
    for name, layer in _validate_masks(model, masks).items():
        layer.weight.masked_fill_(~masks[name].to(layer.weight.device), 0.0)


@torch.no_grad()
def prune_masks(
    model: nn.Module,
    masks: Mapping[str, Tensor],
    hidden_rate: float = 0.2,
    output_rate: float = 0.1,
) -> dict[str, Tensor]:
    """Return new masks, dropping floor(rate * active_count) weights per layer.

    Previously removed entries remain removed. Stable sorting resolves equal
    magnitudes by flattened weight index, ensuring deterministic tie-breaking.
    The last Linear layer is the output layer. Neither input masks nor trained
    parameters are modified, so callers may save them before rewinding.
    """
    for rate in (hidden_rate, output_rate):
        if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
            raise ValueError("Pruning rates must be finite fractions between 0 and 1.")
    layers = _validate_masks(model, masks)
    output_name = next(reversed(layers))
    result: dict[str, Tensor] = {}
    for name, layer in layers.items():
        mask = masks[name].to(layer.weight.device).clone()
        flat_mask = mask.flatten()
        active_indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        rate = output_rate if name == output_name else hidden_rate
        count = math.floor(rate * active_indices.numel())
        magnitudes = layer.weight.detach().flatten()[active_indices].abs()
        if not torch.isfinite(magnitudes).all():
            raise ValueError(f"Layer {name!r} has non-finite surviving weights.")
        if count:
            order = torch.argsort(magnitudes, stable=True)
            flat_mask[active_indices[order[:count]]] = False
        result[name] = mask
    return result


@torch.no_grad()
def rewind_weights(
    model: nn.Module,
    initial_state: Mapping[str, Tensor],
    masks: Mapping[str, Tensor],
) -> None:
    """Restore every parameter (including biases), then reapply the mask.

    Create a new optimizer after calling this function; Adam's accumulated
    moments must not carry over between pruning stages.
    """
    _validate_masks(model, masks)
    model.load_state_dict(initial_state, strict=True)
    apply_masks(model, masks)


def weight_statistics(model: nn.Module, masks: Mapping[str, Tensor]) -> dict[str, int | float]:
    """Count mask-active weights, excluding biases from the sparsity denominator.

    An active position is counted even when its numerical weight happens to be
    zero. ``sparsity`` is a fraction; ``remaining_percent`` is a percentage.
    """
    _validate_masks(model, masks)
    total = sum(mask.numel() for mask in masks.values())
    remaining = sum(int(mask.count_nonzero().item()) for mask in masks.values())
    return {
        "total_weights": total,
        "remaining_weights": remaining,
        "remaining_percent": 100.0 * remaining / total,
        "sparsity": 1.0 - remaining / total,
    }

"""Fixed-budget training and validation for an individual pruning stage."""

from __future__ import annotations

import time

import torch
from torch import nn
from tqdm.auto import tqdm

from pruning import apply_masks


def choose_device(requested: str = "cpu") -> torch.device:
    """CPU is the reproducible default; CUDA and Apple MPS are opt-in or auto."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, mps, or auto")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no compatible CUDA device is available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but this machine has no supported Apple MPS device")
    return device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str | torch.device = "cpu") -> dict[str, float]:
    """Return example-weighted cross entropy and accuracy in [0, 1]."""
    device = torch.device(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    try:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += targets.numel()
    finally:
        model.train(was_training)
    if not total_examples:
        raise ValueError("Cannot evaluate an empty dataset")
    return {"loss": total_loss / total_examples, "accuracy": total_correct / total_examples}


def train_stage(
    model: nn.Module,
    train_loader,
    val_loader,
    masks: dict[str, torch.Tensor],
    *,
    iterations: int = 10_000,
    learning_rate: float = 0.0012,
    eval_interval: int = 500,
    device: str | torch.device = "cpu",
    progress: bool = True,
    trace_interval: int | None = None,
    history_callback=None,
) -> dict:
    """Train every requested update, selecting validation metrics retrospectively.

    There is no actual early stopping and no best-checkpoint restore: subsequent
    magnitude pruning must inspect the final trained weights, as in Appendix G.1.
    Every invocation creates a fresh Adam optimizer (including fresh moments).
    """
    if iterations <= 0 or eval_interval <= 0 or learning_rate <= 0:
        raise ValueError("iterations, eval_interval, and learning_rate must be positive")
    if trace_interval is not None and trace_interval <= 0:
        raise ValueError("trace_interval must be positive")
    if not len(train_loader):
        raise ValueError("Training loader is empty")
    device = choose_device(str(device))
    model.to(device)
    # The masks may have been built before the model was moved to an accelerator.
    active_masks = {name: mask.to(device=device, dtype=torch.bool) for name, mask in masks.items()}
    apply_masks(model, active_masks)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    parameters = dict(model.named_parameters())
    masked_parameters = []
    for name, mask in active_masks.items():
        parameter_name = name if name.endswith(".weight") else f"{name}.weight"
        masked_parameters.append((parameters[parameter_name], mask))

    history = []
    evaluation_seconds = 0.0
    diagnostic_evaluation_seconds = 0.0
    best = None
    weighted_training_loss = 0.0
    examples_seen = 0
    final_training_loss = float("nan")
    iterator = iter(train_loader)
    _synchronize(device)
    started = time.perf_counter()
    with tqdm(total=iterations, desc="Training", unit="step", disable=not progress, leave=False) as bar:
        for iteration in range(1, iterations + 1):
            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at iteration {iteration}")
            loss.backward()
            for parameter, mask in masked_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(mask)
            optimizer.step()
            apply_masks(model, active_masks)
            final_training_loss = loss.item()
            weighted_training_loss += final_training_loss * targets.numel()
            examples_seen += targets.numel()

            eligible = iteration % eval_interval == 0 or iteration == iterations
            diagnostic = trace_interval is not None and iteration % trace_interval == 0
            if eligible or diagnostic:
                _synchronize(device)
                evaluation_started = time.perf_counter()
                validation = evaluate(model, val_loader, device)
                _synchronize(device)
                evaluation_elapsed = time.perf_counter() - evaluation_started
                evaluation_seconds += evaluation_elapsed
                if not eligible:
                    diagnostic_evaluation_seconds += evaluation_elapsed
                if not all(torch.isfinite(torch.tensor(value)).item() for value in validation.values()):
                    raise RuntimeError(f"Non-finite validation metrics at iteration {iteration}")
                row = {
                    "iteration": iteration,
                    "selection_eligible": eligible,
                    "training_loss": final_training_loss,
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                }
                history.append(row)
                if eligible and (best is None or row["validation_loss"] < best["validation_loss"]):
                    best = row.copy()
                if history_callback is not None:
                    history_callback(history)
                bar.set_postfix(loss=f"{loss.item():.3f}", val_acc=f"{validation['accuracy']:.3f}")
            bar.update(1)
    _synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "best_iteration": best["iteration"],
        "best_validation_loss": best["validation_loss"],
        "best_validation_accuracy": best["validation_accuracy"],
        "final_training_loss": final_training_loss,
        "training_loss": weighted_training_loss / examples_seen,
        "runtime_seconds": elapsed,
        "evaluation_seconds": evaluation_seconds,
        "diagnostic_evaluation_seconds": diagnostic_evaluation_seconds,
        "history": history,
    }

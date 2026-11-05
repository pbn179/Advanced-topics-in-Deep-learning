# Stage 2 verification — full experiment not started

The implementation passes the G.1 review and 38 unit/integration tests. The executed `stage2.ipynb` contains the longer pilot, validation plots, runtime probes and projections. The reusable traces and figures are in `reports/stage2/`; start/final checkpoints and masks are retained locally under `results/C-seed0-zugxz1by/` (ignored by Git).

## Method review against Appendix G.1 (PDF page 24)

| Requirement | Implementation / evidence |
|---|---|
| Fully connected 784–300–100–10, ReLU | `models.MLP`; Gaussian Glorot/Xavier normal, gain 1, zero biases |
| 55,000 training / 5,000 validation, batches 60, reshuffle per epoch | `data.make_loaders`; fixed split seed 42 independent of model seed; last batch contains 40 examples |
| Adam 0.0012 and cross entropy | `train.train_stage`; optimizer freshly created on each call |
| Train fixed budget despite best validation iteration | Entire loop always executes; no early stopping or best-weight restoration |
| Validation every 100 updates, select accuracy at minimum validation loss | Example-weighted full-validation metrics; retrospective minimum selection; final update also evaluated if not a multiple of 100 |
| Per-layer 20% hidden / 10% output survivor pruning | `pruning.prune_masks`; smallest absolute surviving weights, stable ties, floor rounding, old mask zeros preserved |
| Reset surviving weights and optimizer | Clone original state before training; rewind survivors and biases; new Adam each stage |
| Keep pruned weights zero throughout training | Apply masks before training, to gradients, and after each optimizer update |
| Prune final trained weights | No best-checkpoint restore; later masks use fixed-budget final ticket weights |
| Three seeds, mean with min/max bars | Implemented in `run_suite` and plots; **not executed** for this single-seed pilot |

For C, one pruning round retains 188,160 first-layer weights, 24,000 second-layer weights and 900 output weights: **213,060 / 266,200 = 80.0376%**. All 410 biases remain dense. Total parameters including biases are 266,610.

## Unit and integration evidence

Final command: `.venv/bin/python -m pytest -q` — **38 passed**.

Tests demonstrate exact architecture counts; smallest-magnitude selection and per-layer survivor counts; monotone masks across **13 rounds**; exact original survivor/bias restoration; no mask regrowth under Adam; masked gradients; fresh Adam moment state per stage; fixed iteration budget without best-weight restoration; deterministic seeds and split; same-mask control initialization and paired batch order; controls do not alter later ticket masks or weights; unique run directories; incremental persistence and failed-run status; aggregation without mixing conditions; correct 13-stage cost arithmetic. Synthetic data are used only for fast offline tests, never for the recorded pilot.

The saved real-MNIST checkpoints were also checked directly: the sparse mask exactly matches magnitude pruning of the final dense checkpoint, all surviving start weights exactly equal the original initialization, biases reset exactly, and every pruned final weight is zero.

The complete notebook executed successfully on Linux CPU with no cell errors. macOS/Windows/CUDA/MPS have not been validated natively.

## Longer real-MNIST pilot

C (300–100), model seed 0, split seed 42, one pruning round, **10,000 updates per stage**. Batch size 60, learning rate 0.0012, validation every 100: **100 full-validation observations per stage**. Train/validation/test sizes: 55,000/5,000/10,000. Test data were not evaluated.

| Stage | Loss at 200 | Accuracy at 200 | Minimum validation loss | Accuracy at that minimum | Minimum iteration | Final loss | Training + validation runtime |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 0.25241 | 92.52% | 0.07736 | 97.72% | 2,600 | 0.10181 | 120.60 s |
| One-round ticket | 0.21712 | 93.52% | 0.07381 | 97.66% | 2,500 | 0.10714 | 150.37 s |

The pilot's stage training/evaluation total was **270.97 s (4.52 minutes)**, excluding data loading, checkpointing and figure generation. The complete pilot runner took 274.27 s. Both stages improved substantially after update 200. Their final losses exceeded the earlier minima, consistent with overfitting/validation variability rather than continuing monotonic generalization improvement. This demonstrates that 200 was insufficient; it does **not** establish convergence at other widths, seeds or higher sparsity. A single noisy validation curve cannot prove convergence.

The one-round mask here comes from the final 10,000-update dense model; it differs from Stage 1's mask derived after 200 dense updates. The ticket's comparison at 200 versus later uses the SAME longer-pilot mask, avoiding that confound. Accuracy is reported at minimum loss, not maximum accuracy. Dense-versus-ticket accuracy difference at each stage's minimum is only -0.06 percentage points, which is not meaningful evidence about winning tickets from one seed.

## Runtime and full-study projections

Measurements on this four-vCPU Linux CPU: A dense, 500 updates, 5.570 s; B dense, 500 updates, 5.513 s; C dense/pruned, 10,000 each, 120.598/150.367 s. Estimates include evaluation every 100 updates, assume serial execution, and give no speedup credit for sparsity: masks still use dense matrix operations. CPU load and short A/B probes make these approximate. Some verification tests overlapped with the C pilot, so timings include that contention. The added 25% is a planning allowance, not a confidence interval. Loading/plot/checkpoint overhead is small in the pilot but not explicitly extrapolated. GPU or native desktop times require separate measurements.

| Interpretation, A/B/C × 3 seeds | Budget per stage | Stage trainings | Estimated CPU hours | With 25% allowance |
|---|---:|---:|---:|---:|
| 13 total stages = dense + 12 rounds | 10,000 | 117 | 3.87 | 4.84 |
| Same, plus one control per sparse stage | 10,000 | 225 | 7.44 | 9.30 |
| 13 total stages = dense + 12 rounds | 50,000 | 117 | 19.35 | 24.18 |
| Same, plus one control per sparse stage | 50,000 | 225 | 37.20 | 46.51 |
| Dense + 13 pruning rounds | 50,000 | 126 | 20.83 | 26.04 |
| Same, plus one control per sparse stage | 50,000 | 243 | 40.18 | 50.23 |

At 50,000 updates, 13 total stages is 5,850,000 optimizer updates without controls, or 11,250,000 with controls. Use `rounds=12` for 13 total stages; `rounds=13` means 14 total stages. The full study has **not** been launched.

## Bugs, deviations and recommendation

No core training/pruning/rewinding defect was found. The old plotter would mix ticket and reinitialization conditions if given new control rows; it now separates them while retaining the shared dense baseline on the ticket curve. The runner now records actual device, initialization seeds, split hash and sizes, package versions, condition, and complete/failed status, and preserves start/final checkpoints on CPU. CSV writes are atomic after completed stages. Automatic checkpoint resumption is not implemented; long full runs should retain their result directories and checkpoints.

Explicit departures/choices relative to G.1: one seed and a shorter pilot budget; a new fixed split because the original split indices are unavailable; test monitoring omitted to preserve the held-out test set; [0,1] pixels, zero biases, unpruned biases, and floor rounding documented where G.1 does not fully specify details. A/B are the intended width-ratio extension, with approximately equal total parameters (266,730 for A/B versus 266,610 for C). Optional D is not run. The same-mask control is implemented and verified with offline tests but has **not** yet been run as a real-MNIST comparison.

**Recommended full-study budget: 50,000 fixed updates per stage**, with evaluation every 100, for comparability to Appendix G.1. Do not replace that with validation-triggered stopping, because it would change which weights get pruned. **10,000 is suitable for further development pilots** and covered both observed C loss minima, but reducing the scientific study to 10,000 would require labeling the deviation and checking A/B and high-sparsity stages first. A shorter budget cannot be justified for all 13 stages from this single dense/lightly-pruned seed.

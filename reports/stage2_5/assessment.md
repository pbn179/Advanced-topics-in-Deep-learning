# Stage 2.5 assessment

**C — run additional targeted A/B pilots before deciding.** 10,000 updates looks promising for this C seed, but is not established for A/B. Propose A and B, seed 0 only, 12 pruning rounds at 10,000 updates and validation every 100, downsampling the same traces to 500 for comparison. The 500-grid change is not yet scientifically neutral. These are two single-seed pilots, not the three-seed study. Extend flagged high-sparsity stages to 15k/20k in separate fixed-mask diagnostic probes if minima remain late.

No Stage 3/full three-seed experiment has been started. The completed scientific pilot is C (231–77), seed 0, 12 pruning rounds / 13 training stages, 10,000 fixed updates per stage. All 55,000/5,000/10,000 MNIST examples are retained; the test set was not evaluated.

## Parameter counts and method

PyTorch verifies A=200,013, B=200,054, C=199,979 total parameters (including biases). The old LeNet-300-100 remains available as `LeNet300100` (266,610). The default main architecture names A/B/C now refer to the requested 200k models; legacy notebooks use explicit legacy names. Historical result files are unchanged.

The Glorot-normal/Adam 0.0012/batch-60 protocol and validated layer-wise 20%/20%/10% pruning, mask enforcement, initialization rewind and optimizer resets are preserved. No early stopping is used; pruning always uses the final fixed-budget ticket weights. New deviations from G.1 are 10,000 rather than 50,000 updates and primary evaluation every 500 rather than 100. Bias, preprocessing, rounding and split choices remain as documented in Stage 2.

## Convergence and validation frequency

- Primary (500-grid) minima in updates 8,000–10,000: **0/13**.
- Fine (100-grid) minima in updates 8,000–10,000: **0/13**.
- Primary minima exactly at update 10,000: **0/13**.
- Final-stage retained weights: **13,891** (6.957%); sparsity is about 93.04%.
- Largest extra selected loss from using the coarser grid: **0.002093**.
- Largest absolute change in accuracy at the selected minimum: **0.560 percentage points**.

**Sampling-frequency caution:** The dense 100-grid selection is 98.02% at update 4,100; the 500-grid selection is 97.46% at update 2,500. That is a **0.56 percentage-point** difference and **1,600 updates** in the reported trainability statistic. At stage 12, the ticket-minus-dense accuracy is +0.52 pp on the 500 grid but -0.12 pp on the 100 grid. The coarser grid therefore changes even the sign of this descriptive single-seed comparison. This does not establish a population-level advantage, but it shows that 500-step evaluation cannot be assumed harmless for this research question. Keep the requested code default of 500; recommend retaining every-100 evaluation for scientific comparisons unless the follow-up establishes grid robustness.

These grid comparisons use the SAME trajectory and masks, not independently trained runs. The additional 100-grid evaluations never select the primary result, alter training duration, or feed into pruning. Low coarse-grid loss cannot be smaller than the fine-grid minimum. A coarser grid can shift the reported best iteration and accuracy without changing the trained network. These are descriptive single-seed differences, not significance tests. Earlier minima alone do not prove convergence; late validation fluctuations and transient minima must be interpreted alongside the full loss/accuracy traces. All per-stage observations are saved in `validation_history.csv` and `convergence_sampling.csv`.

## Measured CPU cost

C's full pilot training/evaluation runtime was **13.67 minutes**, including fine diagnostic evaluations. Measured diagnostic-only evaluation time was **68.53 seconds**. Removing that cost gives a primary-500-grid estimate of **12.53 minutes** for C. This subtraction is approximate and retains CSV/logging overhead.

For A/B/C × three seeds × 13 stages × 10,000 updates (**117 stage trainings; 1,170,000 updates**, no optional controls):

| Measurement basis | Estimated serial CPU hours | With 25% planning allowance |
|---|---:|---:|
| New 200k setup, eval500 | 1.54 | 1.92 |
| Legacy 266k models, matched optimized pipeline / eval500 | 1.51 | 1.89 |
| Previous Stage 2 266k pipeline / eval100 | 3.87 | 4.84 |

The primary comparison uses measured new A/B 1,000-update dense/pruned timings and C's complete sparsity trajectory. The matched legacy probes use the same CPU thread count, pixel cache and 500 evaluation grid; they cover dense plus one pruning round and are less representative of mask-related overhead at higher sparsity. A directly matched new-200k short-probe estimate is also saved in the cost table to isolate this workload difference. Therefore these are planning estimates, not a controlled attribution of savings to model size alone. The historical comparison combines model-size, validation-frequency, tensor-caching and CPU-thread changes. Smaller parameter count alone is not used to predict speed. A/B behavior at high sparsity remains unmeasured. No sparse-kernel speedup is assumed. Initialization/loading/checkpoint/plot overhead and hardware load add uncertainty; the 25% allowance is not a statistical confidence interval.

The directly matched short-probe new-200k estimate is **1.47 h**, versus **1.51 h** for 266k: only about **2.8%** faster in these measured CPU workloads. Parameter reduction alone did not deliver a 25% runtime improvement. Most of the historical speedup comes from the combined pipeline/thread/evaluation changes. Keeping every-100 validation instead is estimated at **1.69 h**, or **2.12 h** with allowance. This sensitivity uses measured per-evaluation time for A/B probes and the C trace, adding 80 evaluations per stage; it is an extrapolation, not a completed full run.

GPU timing: **not measured**; GPU hardware is unavailable in this runtime. No GPU speed estimate is claimed.

## Validation, resumption and reproducibility

See `tests.txt` for the final passing test suite, including exact model counts, pruning/mask invariants through 13 rounds, original survivor restoration, optimizer reset, identical interrupted/resumed trajectories with and without controls, no retraining of committed stages, corruption/configuration rejection, and exact tensor-cache pixel equivalence. `stage2_5.ipynb` executed the benchmarks and the full authorized C pilot.

Completed stages commit checksummed original weights, split, masks, start/final states, validation histories and cumulative result rows atomically. Per-stage RNG/loader seeds reset explicitly and Adam is intentionally fresh, so those saved states suffice for exact same-environment stage-boundary resumption. A partial stage is rerun; completed stages are skipped. Automatic mid-stage Adam-state resume is not implemented. Code, versions, widths, split, device and CPU thread count must match. Do not use simultaneous writers to a run directory. Legacy Stage 1/2 results cannot be resumed/migrated and are left unchanged.

Scientific data sizes and pixel values are unchanged. The optional tensor cache materializes exactly equivalent float32 pixels once (~220 MB) and one CPU thread avoids overhead for these small dense matrices. Timing and parameter tables, configuration and device availability are saved alongside all figures and results in this directory. Native Windows/macOS and CUDA/MPS have not been tested.

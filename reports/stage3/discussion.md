# Stage 3 — hidden-layer ratio experiment

The main experiment is complete: **9 architecture–seed combinations, 117 training stages, 11,700 validation observations**. Exactly **78 new stages** (seeds 1 and 2) were executed; all **39 seed-0 stages were reused**, including their original masks/checkpoints. The held-out test set and random-reinitialization controls were not evaluated/run. Final tests: **77 passed in 7.72s**.

## Protocol

MNIST 55,000 train / 5,000 validation / 10,000 held-out test; split seed 42; A 162–421, B 200–204, C 231–77; seeds 0/1/2; Adam 0.0012; batch 60; Gaussian Glorot; 10,000 fixed updates; validation every 100; 12 layer-wise pruning rounds (20% hidden, 10% output survivors), original-initialization rewind and fresh Adam. No protocol was changed in response to accuracy or warnings. **10,000 replaces the original paper's 50,000 updates**, so this is a reduced-budget extension, not an exact replication.

## Observed accuracy and trainability

| Model | Stage | Active parameters | Accuracy (%), mean ± SD | Change (pp), mean ± SD | Best iteration, mean ± SD |
|---|---:|---:|---:|---:|---:|
| A | 0 | 200,013 | 97.93 ± 0.08 | 0.00 ± 0.00 | 4300.00 ± 1637.07 |
| A | 6 | 54,008 | 98.00 ± 0.11 | 0.07 ± 0.19 | 2266.67 ± 351.19 |
| A | 12 | 15,203 | 97.93 ± 0.01 | 0.00 ± 0.09 | 2833.33 ± 416.33 |
| B | 0 | 200,054 | 97.79 ± 0.11 | 0.00 ± 0.00 | 5033.33 ± 3106.98 |
| B | 6 | 53,301 | 98.10 ± 0.11 | 0.31 ± 0.20 | 2400.00 ± 458.26 |
| B | 12 | 14,575 | 98.09 ± 0.05 | 0.30 ± 0.06 | 2966.67 ± 305.51 |
| C | 0 | 199,979 | 97.95 ± 0.07 | 0.00 ± 0.00 | 4600.00 ± 458.26 |
| C | 6 | 52,870 | 98.02 ± 0.12 | 0.07 ± 0.16 | 2466.67 ± 152.75 |
| C | 12 | 14,209 | 97.89 ± 0.13 | -0.06 ± 0.10 | 3966.67 ± 378.59 |

These are observed points, with **mean ± sample SD across three seeds**, not confidence intervals. Changes are first computed relative to each model/seed's own dense accuracy. Equal rounds leave different active parameter counts, so this table does not establish a parameter-matched winner. Full per-seed values and all rounds are in `individual_results.csv`; all 39 mean/SD rows are in `summary_mean_std.csv`. Faint figure lines show individual seed trajectories.

## Accuracy at common high-sparsity parameter budgets

| Budget (interpolated) | Model | Accuracy (%), mean ± SD | Change (pp), mean ± SD | Seeds retaining dense accuracy | Interpolated best iteration, mean ± SD |
|---:|---|---:|---:|---:|---:|
| 16,000 | A | 97.96 ± 0.05 | 0.03 ± 0.11 | 2/3 | 2734.33 ± 351.76 |
| 16,000 | B | 98.07 ± 0.05 | 0.28 ± 0.08 | 3/3 | 2820.65 ± 187.92 |
| 16,000 | C | 97.89 ± 0.17 | -0.06 ± 0.16 | 1/3 | 3546.98 ± 171.34 |
| 20,000 | A | 98.06 ± 0.12 | 0.13 ± 0.17 | 2/3 | 2433.33 ± 305.51 |
| 20,000 | B | 98.07 ± 0.08 | 0.28 ± 0.12 | 3/3 | 2633.33 ± 115.47 |
| 20,000 | C | 97.99 ± 0.14 | 0.04 ± 0.15 | 2/3 | 3161.76 ± 66.23 |

Each seed is interpolated linearly in log(active parameters), within its observed support only; these are descriptive estimates, not additional trained networks or exact-mask matches. Saved interpolation brackets document the approximation. Actual-count curves and individual observations remain the primary evidence.

- At **16,000 active parameters** (interpolated), the descriptive mean accuracy leader is **B**; the mean dense-relative retention leader is **B**. The largest mean accuracy separation is **0.177 pp**, approximately 8.8 of 5,000 validation classifications. Nonnegative dense-relative changes: A: 2/3, B: 3/3, C: 1/3. The smallest interpolated mean minimum-loss iteration is for **A**. These are descriptive comparisons, not new trained networks or held-out measurements.
- At **20,000 active parameters** (interpolated), the descriptive mean accuracy leader is **B**; the mean dense-relative retention leader is **B**. The largest mean accuracy separation is **0.079 pp**, approximately 4.0 of 5,000 validation classifications. Nonnegative dense-relative changes: A: 2/3, B: 3/3, C: 2/3. The smallest interpolated mean minimum-loss iteration is for **A**. These are descriptive comparisons, not new trained networks or held-out measurements.

Consistency of those descriptive leaders across individual seed indices:

- 16,000, validation_accuracy: individual-seed leaders/ties: A 0/3, B 2/3, C 1/3.
- 16,000, accuracy_change_pp: individual-seed leaders/ties: A 0/3, B 3/3, C 0/3.
- 16,000, best_iteration: individual-seed leaders/ties: A 2/3, B 2/3, C 0/3.
- 20,000, validation_accuracy: individual-seed leaders/ties: A 2/3, B 0/3, C 1/3.
- 20,000, accuracy_change_pp: individual-seed leaders/ties: A 1/3, B 2/3, C 0/3.
- 20,000, best_iteration: individual-seed leaders/ties: A 3/3, B 2/3, C 0/3.

## Answer to the research question

**Dense-accuracy retention:** B was the most consistent descriptive retention candidate at the two shared high-sparsity budgets. Its mean changes were +0.277 ± 0.080 pp at 16k and +0.279 ± 0.125 pp at 20k; it retained/exceeded its own dense accuracy in 3/3 and 3/3 seeds, respectively. This is a descriptive result for these seeds, not statistically established superiority.

**Absolute sparse accuracy:** at 20k active parameters the means were A 98.060 ± 0.119%, B 98.072 ± 0.083%, C 97.993 ± 0.138%. The largest separation was 0.079 pp (about 4.0 validation classifications), smaller than each architecture's measured seed SD (0.083–0.138 pp). Individual-seed accuracy leaders/ties at 20k were A 2/3, B 0/3, C 1/3. At 16k, the A-minus-B accuracy contrast was -0.109 ± 0.020 pp; it was positive in 0/3 and negative in 3/3 seed indices. Thus the descriptive ordering depends on parameter budget and seed. Dense means were A 97.93%, B 97.79%, C 97.95%; B's weaker dense baseline is important when interpreting its larger pruning gains.

**Trainability:** at the same 20k budget, interpolated minimum-loss iterations were A 2433 ± 306, B 2633 ± 115, C 3162 ± 66. C reached its minimum later than A in 3/3 seeds and later than B in 3/3 seeds. A and B tied in 2/3 seeds and A was earlier in 1/3 at this budget. This is a more consistent descriptive learning-pattern difference than the absolute-accuracy ordering. It suggests the 3:1 reference reached its best sampled loss later in these pruned runs; it is not a population-level or wall-clock speed claim. All models still trained for 10,000 updates. Common-count numbers are interpolated approximations; observed actual-count curves remain primary.

**Implication:** the data favor examining B for robust dense-relative accuracy preservation and A/B for earlier minimum-loss iteration under this protocol, but do not justify declaring one hidden-layer ratio universally superior. The small seed count, shared validation set, reduced training budget, one late dense-B warning, interpolation and missing reinitialization controls limit stronger conclusions. In particular, the experiment has not established an advantage over random reinitialization or held-out test performance.

Observed accuracy and minimum-loss iteration vary among the hidden-layer ratios in this experiment, but the scope of that statement is descriptive: the tables/curves quantify the differences under this fixed split, optimizer and pruning schedule. Absolute accuracy leadership and preservation relative to the dense parent are different questions; a model with a weaker dense baseline can show a larger improvement after pruning without having the highest final accuracy. Use the retention means, worst individual changes and 0–3 seed-retention counts together, rather than selecting a winner from a single curve.

The consistency counts above show where ordering repeats and where it changes between seeds. There is **no statistically established superior ratio** from this study: n=3 provides limited precision, repeated rounds are correlated observations, all seeds share the same validation examples, and many comparisons are examined. No hypothesis tests or population-level significance claims are made. A descriptive leader is not proof of superiority. Practical relevance should be judged against the reported pp gaps, seed SDs and individual trajectories; the small-example-count translation above is a scale reference, not an uncertainty estimate. Remaining parameters are genuinely approximately matched initially (A 200,013; B 200,054; C 199,979), but output/bias allocations and slower output pruning make later counts differ.

Minimum-validation-loss iteration is a retrospective trainability proxy measured on a 100-step grid, not a wall-clock speed measurement. All executed stages still cost the full 10,000 updates, regardless of when the minimum was observed. Its mean, SD and individual variation show the ratio-dependent learning patterns. Dense masked matrix kernels do not become faster simply because fewer weights are active, and parallel host contention prevents treating timing differences here as controlled architecture-speed effects. Controls are still needed before attributing IMP-ticket behavior to the original initialization rather than the mask alone.

## Late-training warnings

- Minima in updates 8,000–10,000: **1/117**; new seeds: **1/78**.
- Minima exactly at update 10,000: **0/117**.
- Budget-concern screen: **1/117**. The screen flags a minimum in the last 20%, or a jointly decreasing late slope/mean whose final loss is within 2% of the global minimum. These are screening heuristics, not formal convergence tests.

Late-budget warnings: **B/seed 1/round 0**. All runs still used the locked 10,000 updates. Do not reinterpret these stages as fully converged; warnings can affect trainability comparisons. Consider separately approved fixed-mask 15k/20k diagnostics, not architecture-specific changes to this completed study.

Detailed flags and late slopes are in `late_training_warnings.csv` and `individual_results.csv`. Earlier minima never terminated training: final fixed-budget weights determined every next mask.

## Runtime and verification

The six missing runs used three isolated spawned processes, each with one PyTorch CPU thread. Actual orchestration wall time: **32.57 minutes**. Summed new-run training/evaluation wall times: **92.81 minutes**; these overlap under parallel execution and are not CPU-seconds. Reused seed-0 timing is historical. Only CPU timing was measured; no GPU estimate is claimed.

All 117 real checkpoints were checked for final-parent magnitude pruning, correct active counts and initial seeds, exact survivor/bias reset, old-mask persistence and zero masked entries. Complete new-run resumption was tested with training forbidden. Configurations, code/software hashes and split identity agree; C/seed0 uses its approved every-100 fine observations without editing its original 500-primary/100-diagnostic metadata. **610 prior report/result/notebook files** passed preservation checks. See `verification.json`, `run_provenance.csv`, `reproducibility.md/json`, `protocol.lock.json` and the checksum snapshot.

## Proposed small same-mask control study — not executed

Recommend starting with **A/B/C, seed 0, rounds 6 and 12**: **6 new reinitialization trainings / 60,000 updates**. Reuse the existing ticket measurements and saved masks; do not retrain tickets or run an entire IMP chain again. Each control should use fresh Gaussian Glorot weights, zero biases, the exact same fixed mask and batch order, fresh Adam, 10,000 updates and every-100 validation. Use the recorded derived seed convention `(seed + 1000003 + stage*1009) % 2**32`; controls never feed into pruning.

Estimated serial CPU cost for those six controls: **6.44 minutes**, or **8.05 minutes** with a 25% allowance. Extending the same two rounds to all three seeds would be **18 trainings / 180,000 updates**, approximately **20.52 minutes** (25.65 with allowance). These estimates use matched ticket wall times; contention, new initialization and overhead add uncertainty. They are estimates, not executions or confidence intervals. No random-reinitialization controls have been launched.

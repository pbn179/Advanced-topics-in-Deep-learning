# Four-architecture exploratory findings

This new study freshly trained **156 stages**: four fixed architectures × three model seeds × 13 observed stages. Every stage received 10,000 Adam updates with 100-update validation. All 156 actual best-validation states passed reload validation before all-stage test evaluation. No historical A/B/C checkpoint or random-control measurement was substituted, and no new random-reinitialization controls were run.

The main figure shows complete observed trajectories of minimum-validation-loss iteration and held-out test accuracy against each architecture's actual remaining-weight percentage. Lines join observed means; bars are **sample SD across three seeds**, not confidence intervals. Identical pruning rounds do not have exactly equal remaining counts.

| Architecture | Dense mean test (%) | Stage 12 test (%) | Stage 12 change from own dense (pp) | Stage 12 selected iteration | Stage 12 active parameters |
|---|---:|---:|---:|---:|---:|
| A | 97.923 | 98.143 ± 0.078 | +0.220 ± 0.182 | 2833 ± 416 | 15,203 |
| B | 97.850 | 98.293 ± 0.140 | +0.443 ± 0.202 | 2967 ± 306 | 14,575 |
| C | 97.947 | 98.040 ± 0.026 | +0.093 ± 0.067 | 3967 ± 379 | 14,209 |
| D | 97.877 | 97.940 ± 0.098 | +0.063 ± 0.172 | 4533 ± 971 | 14,115 |

Across the full trajectories, the observed between-architecture mean test-accuracy spread ranged from **0.040 to 0.353 percentage points**, depending on stage. This describes magnitude, not statistical significance or equal-count comparison.

| Architecture | Range of observed stage-mean test accuracy (%) | Mean across 12 pruned stage means (%) | Pruned stages with nonnegative mean dense-relative change |
|---|---:|---:|---:|
| A | 97.893–98.183 | 98.099 | 11/12 |
| B | 97.850–98.293 | 98.175 | 12/12 |
| C | 97.893–98.187 | 98.085 | 11/12 |
| D | 97.850–98.187 | 98.033 | 11/12 |

Across the **13 observed stage means**, counts of highest mean test accuracy were A: 0/13, B: 10/13, C: 3/13, D: 0/13 (ties count for each tied architecture). This is a descriptive trajectory summary, not 13 independent statistical replications. `trajectory_summary.csv` also reports the range of stage means and the arithmetic mean across the 12 pruned-stage means, which weights pruning rounds equally rather than matching parameter counts. Interpret these alongside the full curves and individual-seed CSV, not as proof of architectural superiority from a selected point.

Width allocation can show different observed accuracy and minimum-loss-iteration trajectories. Their practical size is expressed directly in percentage-point changes and the three-seed variability; no p-values or statistical-significance claims are made. A later selected minimum does not prove failure to converge, and an earlier minimum is not actual early stopping. All models continued to 10,000 updates. 1/156 stages selected minima in the last 20% of the budget; diagnostic slopes are recorded without changing the protocol.

**This is an exploratory extension using the established, previously examined MNIST test set.** It is not an independent unseen confirmation of architecture D. Architectures and protocol were fixed before training and were not adjusted after test inspection. Only three model seeds, one data split, one dataset and approximately matched dense sizes were studied. A/B/C reuse the prior predefined seeds and split, so old and new runs must not be pooled as additional independent replications. Surviving parameter counts and mask geometry differ. The budget is **10,000 rather than the original paper's 50,000 updates**. No universally superior ratio is statistically established by this design.

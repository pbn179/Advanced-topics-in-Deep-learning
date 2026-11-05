# Final held-out research findings

All 27 preselected ticket states were reconstructed only to their recorded minimum-validation-loss iteration. All 27 tickets and 18 saved control states passed validation before any held-out evaluation. The 45 selected states were evaluated on all 10,000 MNIST test images; no test-based selection or hyperparameter changes were made.

## Test accuracy and preservation of each seed's dense baseline

Mean ± sample standard deviation across three model seeds. Accuracy is percent; dense-relative changes are percentage points. Random controls use the corresponding original architecture/seed dense baseline, not a separately trained random dense model.

| Architecture | Observed stage | Initialization | Test accuracy (%) | Change from dense (pp) |
|---|---:|---|---:|---:|
| A | 0 | Original initialization | 97.923 ± 0.112 | +0.000 ± 0.000 |
| A | 6 | Original initialization | 98.170 ± 0.036 | +0.247 ± 0.135 |
| A | 6 | Random reinitialization | 97.980 ± 0.113 | +0.057 ± 0.178 |
| A | 12 | Original initialization | 98.143 ± 0.078 | +0.220 ± 0.182 |
| A | 12 | Random reinitialization | 97.570 ± 0.069 | -0.353 ± 0.051 |
| B | 0 | Original initialization | 97.850 ± 0.118 | +0.000 ± 0.000 |
| B | 6 | Original initialization | 98.227 ± 0.045 | +0.377 ± 0.140 |
| B | 6 | Random reinitialization | 97.967 ± 0.168 | +0.117 ± 0.076 |
| B | 12 | Original initialization | 98.293 ± 0.140 | +0.443 ± 0.202 |
| B | 12 | Random reinitialization | 97.420 ± 0.130 | -0.430 ± 0.100 |
| C | 0 | Original initialization | 97.947 ± 0.055 | +0.000 ± 0.000 |
| C | 6 | Original initialization | 98.127 ± 0.098 | +0.180 ± 0.044 |
| C | 6 | Random reinitialization | 97.733 ± 0.107 | -0.213 ± 0.085 |
| C | 12 | Original initialization | 98.040 ± 0.026 | +0.093 ± 0.067 |
| C | 12 | Random reinitialization | 97.197 ± 0.310 | -0.750 ± 0.361 |

## Same-mask paired initialization comparison

Positive ticket-minus-control differences favor original initialization. Differences are paired before computing sample SD.

| Architecture | Observed stage | Ticket − control test accuracy (pp) | Ticket wins |
|---|---:|---:|---:|
| A | 6 | +0.190 ± 0.140 | 3/3 |
| A | 12 | +0.573 ± 0.146 | 3/3 |
| B | 6 | +0.260 ± 0.202 | 3/3 |
| B | 12 | +0.873 ± 0.126 | 3/3 |
| C | 6 | +0.393 ± 0.099 | 3/3 |
| C | 12 | +0.843 ± 0.294 | 3/3 |

Original tickets had higher test accuracy in **18/18** pairs, lower accuracy in **0/18**, with **0/18** ties. At stage 12 they won **9/9** pairs. The greatest mean stage-12 initialization advantage was **B: +0.873 ± 0.126 pp**. Largest per-seed high-sparsity gaps: seed 0: B, seed 1: B, seed 2: C.

## Relation to Stage 3 and Stage 4A

At stage 6, B had the highest mean dense-relative test accuracy change (+0.377 pp). At stage 12, B had the highest mean dense-relative test accuracy change (+0.443 pp). B's stage-12 individual dense-relative changes were seed 0: +0.560 pp; seed 1: +0.210 pp; seed 2: +0.560 pp. Assessing preservation relative to dense and assessing absolute accuracy are different comparisons; the per-seed and aggregate tables report both. B led dense-relative preservation in 2/3 seeds at stage 6 and 3/3 at stage 12. All nine stage-12 original tickets exceeded their own dense test accuracy, so these are improvements rather than degradation on this held-out set. B's larger relative gain also reflects its somewhat lower dense baseline; both absolute accuracy and change should be considered.

C's claim to the largest initialization sensitivity **does not hold uniformly on test**. At stage 6 C has the largest mean gap (+0.393 pp). At stage 12, B's mean gap (+0.873 ± 0.126 pp) is slightly larger than C's (+0.843 ± 0.294 pp), reversing the Stage 4A validation mean ordering. This small B–C difference is not evidence of statistical superiority. A larger ticket–control advantage measures initialization sensitivity, not architectural superiority: B has the highest original-ticket mean accuracy at stage 12, while A has the highest random-control mean accuracy there. The ranking therefore depends on initialization condition as well as sparsity and metric.

`validation_test_comparison.csv` and the paired CSV preserve both validation and held-out measurements, allowing direct comparison without treating agreement as a statistical test. Conclusions depend on sparsity level, initialization condition, and whether the metric is absolute accuracy, dense-relative change, or paired initialization benefit. They are descriptive: three model seeds do not establish statistical significance, and all seeds share one dataset split and one test set.

## Active parameters and sparsity

Counts are identical between each ticket and its paired control. These are mask-active positions, not counts of numerically nonzero weights.

| Architecture | Observed stage | Active parameters (including biases) | Weight sparsity | Parameter sparsity |
|---|---:|---:|---:|---:|
| A | 0 | 200,013 | 0.000% | 0.000% |
| A | 6 | 54,008 | 73.215% | 72.998% |
| A | 12 | 15,203 | 92.674% | 92.399% |
| B | 0 | 200,054 | 0.000% | 0.000% |
| B | 6 | 53,301 | 73.509% | 73.357% |
| B | 12 | 14,575 | 92.907% | 92.714% |
| C | 0 | 199,979 | 0.000% | 0.000% |
| C | 6 | 52,870 | 73.679% | 73.562% |
| C | 12 | 14,209 | 93.043% | 92.895% |

## Scope and limitations

Only **observed pruning stages 0, 6 and 12** were evaluated. The prior Stage 3 **16k/20k validation estimates were interpolations** between observed points; they are neither measured models nor test-set results. Actual active parameter counts differ between architectures at the same pruning stage. Counts include unpruned biases; weight sparsity excludes biases, while parameter sparsity includes them. These denominators are separately recorded in CSVs.

All original experiments used **10,000 fixed updates per stage rather than the original paper's 50,000**. Replays stopped only at preselected iterations to recover missing states; they did not change the original training budget, prune models again, or select new checkpoints. Controls were also trained for 10,000 updates. In Stage 4A, 16/18 control minima occurred within the last 20% of that budget, so random reinitialization may have been training-budget limited. These results do not establish asymptotic inferiority of random initialization or a universally superior hidden-layer ratio.

Experimental execution is complete. No further training experiments are automatically launched.

# Stage 2.6 assessment

**Recommend 10,000 fixed updates per stage for all A/B/C in Stage 3, with primary validation every 100 updates, subject to your review.** All three seed-0 pilots covered their observed validation-loss minima before the final 20% of the budget, and no stage met the predefined late-training concern screen. There is no pilot evidence requiring a 15k/20k increase. This is supporting evidence from one seed per architecture, not a guarantee of convergence for every future seed.

**Stage 3 has not been started.** A and B were executed here with seed 0, 13 stages each and 10,000 fixed updates per stage. C was reused from all 1,300 recorded Stage 2.5 fine-grid observations and not retrained. All 39 stages use primary selection every **100 updates**. No pruning, optimizer, initialization or architecture-specific budget changes were made.

## Comparable results

| Model | Dense accuracy | Stage 12 accuracy | Degradation (pp) | Best-iteration range | Late minima | Runtime (min) |
|---|---:|---:|---:|---:|---:|---:|
| A | 97.86% | 97.94% | -0.08 | 1,900–3,300 | 0/13 | 13.77 |
| B | 97.80% | 98.10% | -0.30 | 2,500–4,300 | 0/13 | 14.42 |
| C | 98.02% | 97.90% | +0.12 | 2,500–4,100 | 0/13 | 13.67 |

Accuracy is selected at the iteration of **minimum validation loss**, not maximum accuracy. Positive degradation means worse than that architecture's own fine-grid dense baseline. The final column is measured training+validation time, excluding loading/initialization/checkpoint/plot overhead; C's time is the original historical measurement including every-100 evaluations. Seed-0 curves have no between-seed error bars and do not establish statistical significance or a superior width ratio.

C's corrected dense accuracy is **98.02% at update 4,100**; stage 12 is **97.90%**, giving **+0.12 pp** degradation. The original 500-grid selections remain in source provenance columns and all Stage 2.5 files are unchanged. No C training time was newly incurred here.

## Budget and late-training evidence

- Loss minima in updates 8,000–10,000: **0/39**.
- Minima exactly at update 10,000: **0/39**.
- Late decreasing trends (both negative slope and negative mean change): **5/39**.
- Budget-concern flags: **0/39**, including **0** in highly pruned stages 9–12.

The screen is deliberately descriptive: flag a minimum in the last 20%, or a decreasing trend over updates 8,000–10,000 whose final loss is within 2% of the best observed loss. The slope is an ordinary linear fit (loss units per 1,000 updates). Mean change compares updates 9,100–10,000 with 8,000–8,900. These thresholds are heuristics, not formal convergence tests. A late decline well above an earlier global minimum can reflect fluctuation/overfitting recovery and is not by itself evidence that the budget is insufficient. Full per-stage losses and accuracies are in `validation_history.csv`; all derived measures and remaining parameter counts are in `results.csv` and `stage_diagnostics.csv`.

No targeted longer-training diagnostic is currently indicated by the screen. If later seeds show late minima, investigate saved masks before changing the common budget.

## Runtime

- Newly executed A+B training/evaluation: **28.19 minutes**.
- Total recorded seed-0 A+B+C training/evaluation (including reused C): **41.86 minutes**.
- Three architectures × three seeds × 13 stages × 10,000 updates: **117 stage trainings / 1,170,000 updates**. Projected serial CPU time: **2.09 hours**, or **2.62 hours** with a 25% planning allowance. Optional random-reinitialization controls are not included and would add work.

If the reviewed Stage 3 protocol remains exactly the same, all three validated seed-0 pilots can be reused. Only seeds 1 and 2 would then remain: **78 stage trainings / 780,000 new updates**, estimated **1.40 CPU hours**, or **1.74 hours** with allowance. This is conditional planning, not a launch or a claim that a new budget could reuse these masks.

The complete-study projection is three times the sum of the three measured seed-0 runtimes, rather than an assumption that parameter count predicts speed. Host load, seed-dependent variation and overhead add uncertainty; C was measured in the earlier stage. The allowance is not a statistical confidence interval. Runs use one CPU thread, exactly equivalent preloaded MNIST tensors, and dense matrix operations despite masking. GPU timings are unavailable and are not inferred. Native macOS/Windows/CUDA/MPS were not tested in this Linux CPU session.

## Verification and preservation

Final tests: **62 passed in 7.06s**. Every real stage checkpoint was checked against pruning its parent's final weights, original surviving initialization/biases and persistent zero masks. Completed A/B pilots were resumed with training forbidden to confirm no stages rerun. All **319 Stage 2.5 files** passed the before/after checksum comparison. The six core training/pruning/data/checkpoint source hashes, software versions and fixed split hash agree between A/B and reused C. The held-out test set was not evaluated.

All results, live histories, masks/checkpoints, verification evidence and PDF/PNG figures are saved under `reports/stage2_6/`. `stage2_6.ipynb` executes the A/B pilots (or resumes their commits) and performs this same comparison. The original source C files are referenced through new provenance metadata, not modified.

## Methodological limits

The pilot budgets are uniformly 10,000 rather than G.1's 50,000 updates. Retrospective evaluation frequency is restored to G.1's 100. The new widths are the intended extension at approximately equal initial total parameter counts; biases stay unpruned. Because output weights are pruned more slowly and output-layer widths differ, the same pruning round leaves slightly different total remaining counts across architectures. Compare curves on actual remaining parameter counts; the stage-12 summary is not an exactly parameter-matched comparison. A fixed new split, zero biases, [0,1] pixels and floor rounding remain the documented choices. Scientific budget adequacy is judged from validation behavior, not a training-loss stopping rule. Earlier minima do not justify ending a stage early: final fixed-budget weights still determine the next mask. One model seed cannot rule out later minima in other seeds or guarantee the paper's lottery-ticket conclusions.

## Per-stage observations

| Model | Stage | Weights remaining (%) | Minimum iteration | Loss at minimum | Accuracy at minimum | Late loss slope / 1k | Late mean change | Budget flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 0 | 100.00 | 2,500 | 0.07586 | 97.86% | +0.00800 | +0.00861 | False |
| A | 1 | 80.21 | 2,300 | 0.07153 | 97.62% | +0.00691 | +0.00782 | False |
| A | 2 | 64.36 | 2,600 | 0.06854 | 98.20% | +0.00164 | -0.00060 | False |
| A | 3 | 51.66 | 2,600 | 0.06697 | 98.12% | -0.00339 | -0.00663 | False |
| A | 4 | 41.48 | 2,500 | 0.06677 | 97.96% | +0.00241 | -0.00165 | False |
| A | 5 | 33.32 | 2,300 | 0.06797 | 98.16% | +0.00085 | -0.00428 | False |
| A | 6 | 26.79 | 2,600 | 0.06554 | 98.10% | +0.01404 | +0.01476 | False |
| A | 7 | 21.54 | 1,900 | 0.06609 | 98.14% | +0.00419 | +0.00557 | False |
| A | 8 | 17.33 | 2,500 | 0.06320 | 98.20% | -0.00001 | -0.00156 | False |
| A | 9 | 13.96 | 2,500 | 0.06583 | 98.14% | +0.00897 | +0.01283 | False |
| A | 10 | 11.25 | 2,500 | 0.06732 | 98.16% | +0.00349 | +0.00276 | False |
| A | 11 | 9.07 | 2,500 | 0.07034 | 97.98% | +0.00882 | +0.01140 | False |
| A | 12 | 7.33 | 3,300 | 0.06922 | 97.94% | +0.00752 | +0.00863 | False |
| B | 0 | 100.00 | 4,100 | 0.07528 | 97.80% | +0.00949 | +0.00799 | False |
| B | 1 | 80.10 | 4,300 | 0.06836 | 97.82% | +0.00582 | +0.00247 | False |
| B | 2 | 64.17 | 2,500 | 0.06563 | 97.96% | +0.00775 | +0.00292 | False |
| B | 3 | 51.42 | 2,500 | 0.06377 | 98.14% | +0.00415 | -0.00156 | False |
| B | 4 | 41.21 | 2,500 | 0.06001 | 98.18% | +0.00249 | -0.00127 | False |
| B | 5 | 33.04 | 2,500 | 0.06217 | 98.18% | -0.00079 | +0.00029 | False |
| B | 6 | 26.49 | 2,500 | 0.06092 | 98.20% | -0.00042 | -0.00366 | False |
| B | 7 | 21.25 | 2,500 | 0.06404 | 98.08% | -0.00051 | -0.00049 | False |
| B | 8 | 17.05 | 2,500 | 0.06281 | 98.14% | +0.00605 | +0.00688 | False |
| B | 9 | 13.68 | 2,500 | 0.06454 | 98.14% | +0.00267 | +0.00323 | False |
| B | 10 | 10.99 | 2,500 | 0.06473 | 98.00% | +0.00365 | +0.00364 | False |
| B | 11 | 8.83 | 2,500 | 0.06646 | 97.96% | +0.00733 | +0.00763 | False |
| B | 12 | 7.09 | 2,900 | 0.06791 | 98.10% | +0.00733 | +0.00915 | False |
| C | 0 | 100.00 | 4,100 | 0.07508 | 98.02% | +0.00947 | +0.01012 | False |
| C | 1 | 80.04 | 2,600 | 0.07451 | 97.82% | +0.00734 | +0.00404 | False |
| C | 2 | 64.07 | 2,600 | 0.07051 | 97.92% | +0.00576 | +0.00487 | False |
| C | 3 | 51.29 | 2,500 | 0.06503 | 98.14% | -0.00219 | -0.00483 | False |
| C | 4 | 41.06 | 2,500 | 0.06351 | 98.12% | +0.00894 | +0.01081 | False |
| C | 5 | 32.87 | 2,500 | 0.06351 | 98.20% | +0.00045 | +0.00124 | False |
| C | 6 | 26.32 | 2,500 | 0.06238 | 97.90% | +0.00283 | +0.00407 | False |
| C | 7 | 21.08 | 2,500 | 0.06264 | 97.96% | +0.01376 | +0.01201 | False |
| C | 8 | 16.88 | 2,500 | 0.06393 | 97.96% | +0.00237 | +0.00509 | False |
| C | 9 | 13.52 | 2,500 | 0.06597 | 97.82% | +0.00365 | +0.00125 | False |
| C | 10 | 10.83 | 3,000 | 0.06563 | 98.02% | +0.00092 | -0.00012 | False |
| C | 11 | 8.68 | 3,200 | 0.06970 | 97.78% | +0.00516 | +0.00501 | False |
| C | 12 | 6.96 | 3,800 | 0.07107 | 97.90% | +0.00655 | +0.00647 | False |

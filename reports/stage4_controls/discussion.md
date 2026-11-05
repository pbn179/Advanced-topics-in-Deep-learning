# Stage 4A — same-mask random-reinitialization controls

**Exactly 18 approved controls completed**, across A/B/C × seeds 0/1/2 × pruning rounds 6/12. No original tickets were retrained, no masks regenerated, and the held-out test set was not evaluated. Final tests: **90 passed in 7.22s**.

## Does original initialization help?

Original-initialization tickets had higher selected validation accuracy in **16/18** pairs, lower accuracy in **2/18**, and tied in **0/18**. Original initialization was beneficial in many observed cases, but it did not uniformly outperform random reinitialization. Individual outcomes and each architecture/round's three-seed results matter.

Accuracy exceptions where random reinitialization was higher: **A/seed 2/round 6 (-0.060 pp), B/seed 1/round 6 (-0.120 pp)**.

| Model | Round | Active parameters | Ticket accuracy (%), mean ± SD | Control accuracy (%), mean ± SD | Ticket − control (pp), mean ± SD | Ticket wins/ties/losses |
|---|---:|---:|---:|---:|---:|---|
| A | 6 | 54,008 | 98.00 ± 0.11 | 97.91 ± 0.08 | +0.093 ± 0.136 | 2/0/1 |
| B | 6 | 53,301 | 98.10 ± 0.11 | 97.87 ± 0.22 | +0.233 ± 0.332 | 2/0/1 |
| C | 6 | 52,870 | 98.02 ± 0.12 | 97.74 ± 0.02 | +0.280 ± 0.131 | 3/0/0 |
| A | 12 | 15,203 | 97.93 ± 0.01 | 97.33 ± 0.23 | +0.600 ± 0.227 | 3/0/0 |
| B | 12 | 14,575 | 98.09 ± 0.05 | 97.33 ± 0.23 | +0.767 ± 0.223 | 3/0/0 |
| C | 12 | 14,209 | 97.89 ± 0.13 | 96.79 ± 0.08 | +1.107 ± 0.133 | 3/0/0 |

Differences are **ticket minus its corresponding control**, calculated within architecture, model seed and exact mask before aggregation. SD is the **sample SD of the three paired differences**, not a combination of independent SDs or a confidence interval. Accuracy is at each run's own minimum validation loss, sampled every 100 updates. The original metrics are reused from Stage 3's approved fine grid.

- Round 6: paired accuracy differences A +0.093 ± 0.136 pp, B +0.233 ± 0.332 pp, C +0.280 ± 0.131 pp; ticket wins 7/9, losses 2/9, ties 0/9.
- Round 12: paired accuracy differences A +0.600 ± 0.227 pp, B +0.767 ± 0.223 pp, C +1.107 ± 0.133 pp; ticket wins 9/9, losses 0/9, ties 0/9.

## Loss and trainability

| Model | Round | Ticket minimum loss, mean ± SD | Control minimum loss, mean ± SD | Ticket minimum iteration, mean ± SD | Control minimum iteration, mean ± SD |
|---|---:|---:|---:|---:|---:|
| A | 6 | 0.0658 ± 0.0004 | 0.0761 ± 0.0049 | 2267 ± 351 | 7067 ± 1966 |
| B | 6 | 0.0620 ± 0.0020 | 0.0796 ± 0.0029 | 2400 ± 458 | 8100 ± 529 |
| C | 6 | 0.0647 ± 0.0024 | 0.0805 ± 0.0040 | 2467 ± 153 | 8233 ± 115 |
| A | 12 | 0.0693 ± 0.0019 | 0.0879 ± 0.0004 | 2833 ± 416 | 9400 ± 436 |
| B | 12 | 0.0664 ± 0.0020 | 0.0949 ± 0.0044 | 2967 ± 306 | 9300 ± 866 |
| C | 12 | 0.0724 ± 0.0023 | 0.1067 ± 0.0044 | 3967 ± 379 | 9267 ± 681 |

Tickets reached their selected minimum earlier in **18/18** pairs and had lower minimum validation loss in **18/18**. These counts are descriptive, not 18 independent hypothesis-test replicates. The two pruning rounds share parent experiments, seeds and a fixed validation dataset.

Control minima in updates 8,000–10,000: **16/18**; exactly at update 10,000: **0/18**. Late control minima can indicate budget-limited learning. Consequently, accuracy gaps establish behavior at the specified **10,000-update budget**, not an asymptotic inability of randomly initialized networks to catch up. The budget was not changed for any architecture or outcome. The original paper's 50,000-update setting remains a documented deviation.

## What this adds to the width-ratio research question

The initialization benefit increased from round 6 to round 12 in **9/9** architecture–seed trajectories. At round 12, **C** showed the largest observed mean paired accuracy advantage, **+1.107 ± 0.133 pp**. This identifies stronger initialization sensitivity for those saved masks, not the best absolute ticket architecture. In Stage 3, B had the strongest dense-relative preservation; here the size of the initialization advantage is a separate quantity. In particular, C can depend more on its original initialization while still having lower absolute ticket accuracy than B. The ratios therefore appear to differ in initialization sensitivity under the specified sparsity/budget, but active counts/mask geometry differ and only three seeds were tested. No universally optimal ratio or statistically established cross-architecture superiority follows.

The control isolates initialization **within each saved mask**: architecture, active positions, data split, exact batch order, optimizer settings and budget are matched. It separates the behavior of the original-initialization IMP ticket from what a fresh initialization can do with the same selected connectivity. The paired tables show how initialization sensitivity differs among the tested width allocations; they must not be read as evidence that one ratio is universally best.

Stage 3's dense-relative preservation and Stage 4's initialization advantage answer different questions. A ratio may preserve its dense accuracy well yet have a small initialization advantage, or show a large ticket/control gap because its random control learns slowly. Compare absolute ticket accuracy, paired gaps, minimum-loss iteration and individual seeds together. At round 12, active counts are A 15,203, B 14,575, C 14,209; identical rounds are **not exactly parameter-matched across architectures**. Thus cross-architecture differences in control benefit combine allocation and the resulting mask/sparsity geometry. Within each pair the positions and count are exactly identical.

Only three model seeds and one predetermined fresh initialization per mask were used. Mean ± SD and individual paired observations describe these runs; there are no population-level significance claims or post-hoc seed searches. The fixed validation split, many correlated comparisons, finite budget and unevaluated test set limit stronger generalization. This is evidence about initialization under this protocol, not a claim that every ticket beats every random initialization or that any ratio is statistically superior.

## Verification and preservation

- Exact source masks were copied, not regenerated. Start, selected-best and final control states match all saved active positions and keep masked weights zero; masks were also checked at all 100 training validation events.
- Full fresh Gaussian Glorot states were reproduced bit for bit from the predetermined seed rule `(model_seed + 1000003 + round*1009) % 2**32`, with zero biases, and differ from the corresponding original initialization on surviving weights.
- Each control's **observed** training fetch stream and final sampler RNG state match replay of the source ticket's exact split/loader protocol for all 10,000 updates, including partial batches and epoch reshuffles. Original Stage 3 fetch hashes were not recorded, so this is verified reconstruction from the frozen code, versions, split and seeds; no ticket training was replayed.
- A fresh Adam is created for each run. All original source checkpoints/masks and **1072 prior files** remained unchanged by checksum comparison.
- Actual saved control `best_validation.pt` states reproduce the selected validation loss/accuracy when reloaded. This verification used validation only, never the held-out test loader.

See `verification.json`, `protocol.lock.json`, `batch_order_reference.json` and each run's committed artifacts. All tables/figures and full histories are under `reports/stage4_controls/`.

## Can final test evaluation proceed without retraining?

**Not at Stage 3's reported minimum-loss ticket states.** The audit inspected all 117 source stages: only original/start/final 10,000-update tensors were saved, and no selected minimum occurred at update 10,000. Thus **0/117 selected ticket states are currently available** without replay. Histories and metrics cannot substitute for weights.

Saved starts/masks/seeds allow a separately authorized deterministic training replay to each selected iteration, followed by validation-metric reproduction. That would be ticket retraining/replay and is **not performed here**. Do not regenerate masks or silently replace the reported states.

The saved final 10,000-update ticket checkpoints can be evaluated without retraining **if a distinct final-weight test estimand is explicitly chosen**. It would not be the validation-selected evaluation currently reported. Controls now have actual minimum-loss snapshots, so their selected test evaluation would not require control retraining; paired selected-state ticket testing still needs the missing ticket states. See `stage3_checkpoint_audit.csv/md` for the complete audit.

## Runtime and scope

Actual orchestration wall time: **6.44 minutes**, with three isolated one-thread CPU worker processes. Summed control training/validation wall times: **17.85 minutes**, overlapping under parallel execution and including audit/checkpoint callback overhead. No GPU timings are claimed. Only these 18 control trainings were executed; unit/integration tests use synthetic data and are not extra MNIST experiments. No additional diagnostics, ticket replays or test-set evaluations were launched.

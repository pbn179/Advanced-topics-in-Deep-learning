# Short interpretation

Original-initialization tickets had higher selected validation accuracy in **16/18** pairs, lower accuracy in **2/18**, and tied in **0/18**. Original initialization was beneficial in many observed cases, but it did not uniformly outperform random reinitialization. Individual outcomes and each architecture/round's three-seed results matter.

- Round 6: paired accuracy differences A +0.093 ± 0.136 pp, B +0.233 ± 0.332 pp, C +0.280 ± 0.131 pp; ticket wins 7/9, losses 2/9, ties 0/9.
- Round 12: paired accuracy differences A +0.600 ± 0.227 pp, B +0.767 ± 0.223 pp, C +1.107 ± 0.133 pp; ticket wins 9/9, losses 0/9, ties 0/9.

The initialization benefit increased from round 6 to round 12 in **9/9** architecture–seed trajectories. At round 12, **C** showed the largest observed mean paired accuracy advantage, **+1.107 ± 0.133 pp**. This identifies stronger initialization sensitivity for those saved masks, not the best absolute ticket architecture. In Stage 3, B had the strongest dense-relative preservation; here the size of the initialization advantage is a separate quantity. In particular, C can depend more on its original initialization while still having lower absolute ticket accuracy than B. The ratios therefore appear to differ in initialization sensitivity under the specified sparsity/budget, but active counts/mask geometry differ and only three seeds were tested. No universally optimal ratio or statistically established cross-architecture superiority follows.

Original tickets selected their minimum loss earlier in 18/18 pairs. 16/18 controls selected minima in the last 20% of their budget, so the comparison concerns fixed-budget accuracy/trainability, not guaranteed asymptotic superiority.

The evidence quantifies initialization sensitivity for each ratio, but does not by itself establish a universally superior hidden-layer allocation. Active counts differ across architectures, and n=3 supports descriptive paired results rather than strong significance claims.

Stage 3 did not save minimum-loss weight snapshots (0/117 available); validation-selected ticket test evaluation needs separately authorized deterministic replay. Final 10k weights are available for a different, explicitly labeled final-weight evaluation. Control best snapshots were saved. No test evaluation or ticket replay was performed.

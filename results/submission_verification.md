# Final submission verification

The intended submission is [final-four-architecture-review](https://github.com/pbn179/Advanced-topics-in-Deep-learning/tree/final-four-architecture-review). Complete model artifacts are on [final-four-architecture-checkpoints](https://github.com/pbn179/Advanced-topics-in-Deep-learning/tree/final-four-architecture-checkpoints).

## Verification results

| Check | Result |
|---|---:|
| Unit and integration tests | **48 passed**, no failures or skips |
| Completed experimental training stages | **156** |
| Existing recorded test evaluations | **156**, 10,000 examples each |
| Architectures × seeds × observed stages | **4 × 3 × 13** |
| Aggregate rows / measured figure points per panel | **52** |
| Selected checkpoint hashes matched | **156/156** |
| Remote checkpoint artifact hashes matched | **1,134/1,134** |
| Required `.pt` files | **492/492** |
| Default notebook | **7/7 code cells executed** |
| Main PNG | Regenerated **bit-for-bit identically** |
| Main PDF | Regenerated successfully; content matches except creation date |
| Scientific CSVs | All **5 unchanged**, including both final result tables |
| Required reproducibility files missing | **None** |

## Remote branch and checkpoint audit

Both branches were freshly cloned through the configured authenticated GitHub connection, rather than inferred from local branch names. The inspected remote snapshots were `638e06f553bca19e9d41e2eb05f282e38d33237e` (submission) and `42733f5f1e32e34936c9e5ad6269a16e3b1aeb0a` (checkpoints). This submission-cleanup revision only updates documentation, tests/verification presentation and redundant operational records; it does not change the frozen measurements or model artifacts.

Every entry in `archive/new_checkpoint_artifacts.json` exists in the fresh checkpoint clone and matches SHA256. Inventory: 156 masked starts, 156 actual best-validation states, 156 final 10k states, 12 original initializations and 12 saved splits. All 156 stage commit records and their referenced files were checked. Complete histories have updates 100, 200, ..., 10,000. The recorded minimum-loss iteration matches the history; initializer and split hashes and previous-final-state chains match. All existing test records match the corresponding selected-state hash and the submission CSV. Frozen training source/config hashes also match the checkpoint protocol.

These checks read saved files and existing metrics only. **No model forward pass, new validation/test evaluation, scientific training, pruning, architecture change or tuning was performed.** The experimental validation-reload results remain those in `results/verification.json`.

## Default notebook isolation

The notebook was executed from the fresh submission clone with **neither MNIST nor an artifacts/checkpoint directory present**. Project modules were confirmed to come from that clone. Default settings were `MODE="saved"` and `ENABLE_TRAINING=False`.

Runtime profiling guards were installed before executing the actual cells. Attempts to call training, Adam optimizer steps, full-study execution, dataset loading, checkpoint loading, or model evaluation would fail verification. **Every guarded call counter stayed zero.** All four architectures, all three seeds and all 13 observed stages were loaded. Aggregate tables were recomputed and checked against saved results. All scientific CSV hashes remained unchanged. Only derived figure files and the executed notebook outputs were regenerated in the temporary clone. The PDF differed only in creation-date metadata; the original frozen PDF bytes were retained afterward.

Execution used IPython in-process with real rich-output capture because the environment restricts ordinary Jupyter-kernel TCP sockets. The installed dependency environment was reused. An independent dependency installation and native Windows/macOS execution were **not tested**; no cross-platform bitwise training claim is made. GitHub access was verified with authenticated Git clones, not anonymous/public browser access.

## Submission cleanup and recovery

Redundant preflight/console/old cosmetic-update records were removed from the clean view. Two old console transcripts contained machine-specific paths; none were runtime dependencies. The negative-path test now uses pytest's platform-specific temporary path rather than a literal Unix path. `.gitattributes` pins text checkouts to LF and marks checkpoint/image/PDF files as binary, protecting frozen SHA256 manifests from automatic line-ending conversion. All removals and exact recovery references are listed in `archive/README.md`; their prior contents remain in Git history at `638e06f553bca19e9d41e2eb05f282e38d33237e`.

No scientific CSV, selected model, original initialization, mask, protocol or original research branch was deleted. No history rewrite, force-push or main merge is part of this revision. The final runtime code/notebook has no hardcoded Codex workspace path or dependency on historical development files. Historical references are documentation/recovery metadata only.

The README gives exact saved-figure, full-reproduction and checkpoint retrieval commands. The dataset is intentionally excluded and automatically downloadable for an explicitly enabled new reproduction; default saved-results mode needs no dataset or checkpoints. Large artifact files are intentionally absent from the clean branch and complete on the separate checkpoint branch. No required artifact was silently omitted.

The result remains an exploratory extension using the previously examined MNIST test set, with only three seeds and a 10k-update budget. These submission checks do not create additional independent experiments or test measurements.

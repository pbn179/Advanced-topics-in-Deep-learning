# Preserved research history and large checkpoints

The clean review branch removes development-stage material from its ordinary working tree, not from history or the preserved original workspace.

## Previous study

- Historical branch: `research-package-download`.
- Frozen historical reference before this redesign: `ab3b53c830dbeba1174507b142919ab09eadbe5b`.
- Earlier complete package snapshot: `cf25bc3d8e9e1606a6858cfd543e010822e2e725` (all 469 historical `.pt` files).
- The final four-architecture cleanup commit descends from the preserved historical branch, so earlier history remains recoverable even without relying solely on a branch name.

Historical material moved out of the new main workflow includes Stage 1–4B notebooks, old report folders, previous CSVs/figures, A/B/C checkpoints and reinitialization-control outputs, packaging helpers, old configurations and duplicate live outputs. These have **not** been irreversibly deleted. The original cloud project remains unchanged. `previous_artifact_checksums.json` records its preservation snapshot; `implementation_provenance.json` records validated modules copied unchanged. Historical D aliases refer to a different optional architecture; new D is fixed to 242–40.

Recover a complete historical snapshot without overwriting this clean working tree:

```sh
git archive --format=zip --output=../historical-research.zip ab3b53c830dbeba1174507b142919ab09eadbe5b
```

Or inspect individual historical files:

```sh
git show ab3b53c830dbeba1174507b142919ab09eadbe5b:reports/stage4_test/findings.md
```

In a shallow clone, fetch the historical branch first. Avoid restoring old `results/` over the new four-architecture results. No previous random-control measurements are mixed into the new study.

## New run tensors

The freshly trained tensors are retained under `artifacts/main_study/` in the execution workspace. Each of 12 runs has original initialization and split indices, and all 13 stages have exact masks, start state, actual best-validation state, final 10k state, full history, metrics and SHA256 commit marker. All 156 selected state hashes are also directly recorded in `results/verification.json`.

The complete new checkpoint archive is on branch `final-four-architecture-checkpoints` of the same repository. All 492 tensor files (156 starts, 156 best states, 156 finals, 12 initializations and 12 splits) are included, with stage history and SHA256 commit records. See `new_checkpoint_artifacts.json` for every artifact hash. The lightweight review branch deliberately omits these large tensor files; this does not impede the default CSV-based notebook. Use GitHub Code → Download ZIP on `final-four-architecture-checkpoints` for a separate full artifact copy. Alternatively, fetch that branch and restore `artifacts/main_study/` into a separate working directory. Preserve any local unrelated run files first. GitHub Releases/LFS remain alternatives, but release API access was unavailable in the prior environment. No unverified external artifact URL is asserted.

## Submission-verification cleanup

The following redundant operational records were removed from the latest clean working tree, with their original contents preserved at commit `638e06f553bca19e9d41e2eb05f282e38d33237e` and its ancestors:

- `results/preflight_tests.txt`
- `results/notebook_execution.txt`
- `results/clean_checkout_notebook_execution.txt`
- `results/clean_checkout_tests.txt`
- `results/axis_label_revision.json`

They are superseded by the current `results/tests.txt`, `results/submission_verification.json/.md`, and the executed notebook. Two console transcripts contained machine-specific execution paths; none were runtime dependencies. Recover any one with `git show 638e06f553bca19e9d41e2eb05f282e38d33237e:PATH`. No scientific CSV, checkpoint, initialization, pruning mask, protocol record or original research branch was removed.

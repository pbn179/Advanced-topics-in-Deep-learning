# Final packaging verification

This release freezes the completed Stage 1–4B study. **No new MNIST training, ticket replay, control run or held-out evaluation was performed during packaging.**

## Verified

- All **112 existing and added unit/integration tests passed**; see `tests.txt`. Small synthetic training tests validate checkpoint behavior and are not additional scientific experiments.
- The consolidated notebook was executed from beginning to end in default **Mode A**. Actual IPython cell execution and rich-output capture were used in-process because the sandbox cannot open the TCP sockets required for an ordinary Jupyter kernel. The delivered notebook records its execution engine and counts; no outputs were fabricated.
- The saved-data workflow independently recomputed Stage 3 means/sample SDs, Stage 4A paired means/sample SDs and Stage 4B condition/paired aggregates. They match approved CSV tables to absolute tolerance **1e-10**, allowing CSV float rounding only.
- All **117 main-stage observations**, **18 control pairs** and **45 previously selected test measurements** are present. All **45 selected checkpoint SHA256 values** match the historical validation gate. No test-set evaluation is repeated.
- All **11 figure pairs (PDF and 300-dpi PNG)** and **14 CSV/LaTeX table pairs** were generated from real approved measurements; no simulated scientific data were used. CSVs preserve machine-readable numbers; LaTeX tables use 12 significant digits.
- All **1,382 original report/result/notebook/core-source files** match the SHA256 preservation snapshot taken before packaging. Checkpoints and masks were not changed. The archive contains **all 469 original `.pt` checkpoints**, including initializations, starts/finals, masks, control best states and reconstructed selected tickets.
- Original notebook copies are consolidated under `archive/notebooks/`. Byte-identical notebook/live-history duplicates are recorded as aliases; logs, locks, caches, environments and downloaded MNIST are excluded. Canonical experimental data remain included.
- The final notebook, analysis module, full reproduction entry point and final configuration use project-relative paths. Historical immutable CSV/JSON/notebook metadata may contain absolute provenance paths; these are retained to preserve original bytes and are not resolved by the portable workflow. Historical runners/notebooks are archival entry points, not the final portable workflow. Copied paper-table provenance paths are normalized to relative paths.
- Mode B requires explicit `enabled=True`/`--enable-full-reproduction` and a separate relative `reproductions/NAME` output. The notebook additionally requires `MODE="B"` and `ENABLE_FULL_REPRODUCTION=True`. Guards, output protection, completed-stage checksum resumption, best-state capture, and the new-results analysis adapter are covered by tests.
- ZIP CRCs and all packaged data-manifest checksums are checked. The release is extracted into a fresh directory, and both the full test suite and the actual default notebook workflow are exercised there using the installed release dependencies. Regenerated approved numerical tables match. See `verification.json` for the extraction result. All old result/checkpoint bytes remain unchanged after analysis. The archive SHA256 is provided in the external `.zip.sha256` sidecar to avoid a self-hash cycle.

## Exact limits of clean-environment verification

A **new isolated virtual environment** was created and an independent CPU dependency installation was attempted with:

```sh
python -m venv /tmp/lottery-final-clean-env
/tmp/lottery-final-clean-env/bin/python -m pip install --disable-pip-version-check --retries 0 --timeout 20 --index-url https://download.pytorch.org/whl/cpu torch==2.6.0+cpu torchvision==0.21.0+cpu
```

The index returned **“No matching distribution found for torch==2.6.0+cpu”**; see `clean_install_attempt.txt`. The dependencies were therefore **not successfully installed into that isolated environment**, and a completely independent clean dependency installation is **not verified**. No explanation beyond this observed failure is inferred. The extracted-project checks instead use the already-installed project Python dependency environment, from the fresh extraction's working directory with its own modules/artifacts. They do not require the original repository's files, datasets or virtual-environment path in the delivered notebook/code.

The tested interpreter was Linux Python 3.12 with the versions in `requirements.txt` and `requirements-tested-linux.txt`. Native macOS/Windows installation, cross-platform numerical replay, ordinary socket-based Jupyter kernel startup, and the **full 135-stage Mode B scientific sweep** were not executed. Mode B is supplied as tested orchestration around the frozen training implementation; testing it on synthetic data does not substitute for a complete scientific rerun. No new experiment was started to resolve these limits.

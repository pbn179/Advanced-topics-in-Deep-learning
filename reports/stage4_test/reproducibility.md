# Verification and reproducibility

- Exactly 27 ticket replays: A/B/C × seeds 0/1/2 × observed stages 0/6/12.
- Exactly 18 saved Stage 4A controls: A/B/C × seeds 0/1/2 × stages 6/12.
- All **45** states passed validation before test evaluation; no failed checkpoint was tested.
- Predeclared absolute tolerance: validation loss **1e-08**, accuracy **1e-12**; no relative tolerance. With 5,000 validation examples this accuracy tolerance permits no changed total classification count.
- Maximum observed loss discrepancy **9.02e-17**; maximum accuracy discrepancy **0**.
- Every replay's entire 100-step validation-history prefix matched historical losses/accuracies within these tolerances, including C seed 0's original fine diagnostic trace. Its old 500-step primary eligibility flags do not alter the recovered 100-step selected iteration.
- Starts equal the saved original initialization with the exact saved mask applied, including restored biases. Source start/final masks match; original commit checksums and frozen source/software versions were checked.
- All reconstructed checkpoint tensors were checked for finite values, masked zeros, and lossless save/reload. Mask persistence was checked at every replay validation point and again after loading selected states.
- Actual replay sample-fetch hashes, sample counts and final DataLoader-generator states matched index-only reconstruction of the original seed/split/loader protocol. Historical Stage 3 fetch hashes were not recorded; the claim is protocol reconstruction, not comparison to an unavailable historical trace.
- Fresh Adam, learning rate .0012, batch size 60, split seed 42, CPU/one Torch thread, exact saved split, unchanged Gaussian-Glorot original starts; original core source SHA256 and versions are in `protocol.lock.json`.
- Test data uses the original [0,1] pixel transform, no augmentation, all 10,000 examples, and the existing example-weighted evaluation code. Test loss is reported as an additional measurement; it did not select weights.
- **1268** previous results/report/notebook files retain their SHA256 hashes. `previous_results_snapshot.json` is the preservation manifest. New recovered states live only under `reports/stage4_test/replays/`.
- All new selected-state checksums are in `validation_gate.json` and `verification.json`. Complete replay markers are written after verification; cached complete replays are checksum-checked and skipped. Cached test results are reused for the same checkpoint set, without reselection.
- Wall-clock replay/verification/test execution: **4.21 minutes**, up to three replay worker processes. Package versions match the historical run; cross-platform bitwise replay is not promised. The notebook and spawn-safe runner use pathlib and workers=0 for data loading for Windows/macOS portability.

Reproduce or validate artifacts with `.venv/bin/python stage4_test_runner.py --workers 3`, then `.venv/bin/python stage4_test_analysis.py`, or execute `stage4_test.ipynb`. Completed checkpoints/results are reused. Run `.venv/bin/python -m pytest -q` for unit and synthetic integration tests; test results are saved in `tests.txt`. No synthetic test is an additional MNIST training experiment.

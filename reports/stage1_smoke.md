# Stage 1 smoke validation

Linux CPU, Python 3.12, PyTorch 2.6.0+cpu; seed 0, split seed 42.
Full MNIST split; C architecture; 200 updates per stage, validation every 100.
25 unit tests passed. Dense accuracy 92.52%, one-round ticket 93.24%; both minimum validation losses occurred at update 200. Training runtimes: 1.856s dense, 2.109s pruned (excluding download/plotting). Remaining weights: 213,060/266,200 (80.0376%).

This is a short implementation check, not a convergence or winning-ticket claim. Native Windows/macOS and accelerators have not been validated. Next: longer baseline pilots, then three seeds and the same-mask random-reinitialization control before the full architecture sweep. requirements-tested-linux.txt records this test environment; use requirements.txt for platform-specific installation.

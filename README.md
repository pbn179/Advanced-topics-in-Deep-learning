# Lottery Ticket Hypothesis: four hidden-layer width ratios

**Research question:** How does the ratio between hidden-layer widths affect the accuracy and trainability of winning tickets when the total number of parameters is held approximately constant?

This is the clean, fresh four-architecture experimental extension. All four architectures were fixed before training. Earlier A/B/C experiments and random-reinitialization controls are historical work, preserved in Git history and the identified archive branch; their checkpoints and measurements are not substituted into this run. No new random controls are run.

| Architecture | Hidden widths | Width ratio | Initial parameters including biases |
|---|---|---|---:|
| A | 162–421 | 1:2.60 | 200,013 |
| B | 200–204 | 1:1.02 | 200,054 |
| C | 231–77 | 3:1 | 199,979 |
| D | 242–40 | 6.05:1 | 200,100 |

`study.config()` and the tests verify counts programmatically. The maximum dense-count spread is 121 parameters, 0.0605% of 200k. The MLP remains 784 → hidden1 → hidden2 → 10 with ReLU hidden activations and unnormalized output logits.

## Main deliverables

- `final_experiment.ipynb`: consolidated executed notebook; default is saved-results analysis without training or test reevaluation.
- `paper_assets/main_results.pdf` and `.png`: one two-panel figure, all 13 observed stages for each architecture, at approximately two-column paper width. Panel (a), labelled `Early-Stop Iteration (Val.)`, shows iteration of minimum validation loss; panel (b), labelled `Accuracy at Early-Stop (Test)%`, shows held-out accuracy at that checkpoint. These labels follow the paper; selection is retrospective and training never stopped early. The shared x-axis label is `Percent of Weights Remaining`. Bars are sample SD across three seeds. X positions are actual percentages of original **weights**, excluding biases, remaining; they are not rounded common parameter budgets.
- `results/individual_results.csv`: all 156 new architecture/seed/stage measurements, selected-state identifiers and integrity metadata.
- `results/aggregated_results.csv`: mean, sample SD, min/max by architecture and stage (52 rows).
- `paper_assets/architectures.tex`: concise architecture table; `main_results_caption.txt/.tex`: caption.
- `results/findings.md`, `verification.md/.json`, `protocol.json`, `tests.txt`: findings, checks, configuration/source versions and test evidence.
- `results/validation_history.csv`: all 15,600 validation observations (100 per stage). `training_budget_diagnostics.csv` and `trajectory_summary.csv` are small supporting numerical diagnostics, not additional experiments.

The old meaning of historical architecture D is not used: the new fixed D is **242–40**.

## Experimental protocol

MNIST: 55,000 train / 5,000 validation / 10,000 test; fixed split seed 42 and model seeds 0,1,2. Gaussian Glorot weights, zero biases, Adam learning rate .0012, batch size 60, cross-entropy. Every stage receives **10,000 optimizer updates**, with validation every **100**. Twelve iterative pruning rounds plus the dense stage yield **156 training stages**.

Each layer is pruned independently, removing `floor(rate × surviving weights)` with rate .2 for hidden layers and .1 for the output layer. Biases remain dense. Surviving weights and biases reset to the exact original initialization; each stage constructs fresh Adam. Masked gradients and post-update masking enforce zeros. **Final 10k weights determine the next mask.** An actual `best_validation.pt` is captured on strict validation-loss improvement, retaining the first exact tie, and used for reported selected-state performance. These are different saved states.

All **156** validation selections are fixed and reload-verified before any new-study test evaluation. Absolute validation tolerances are 1e-8 loss and 1e-12 accuracy. Every selected state is then tested on all 10,000 test images, at every observed stage 0–12. No test metric selects weights, masks, hyperparameters, architecture widths or training budget.

**The MNIST test set was examined in our previous study. This is an exploratory extension on that established test set, not an independent previously unseen confirmation of architecture D.** Training was not changed after inspecting its test results.

## Project structure

```text
README.md, requirements*.txt
final_experiment.ipynb
models.py, pruning.py, train.py, data.py, checkpoints.py  # validated core modules
study.py                    # fixed protocol, fresh IMP, snapshotting, verification/test gate
analysis.py                 # observed-point figure, aggregate tables and reports
execute_notebook.py         # optional IPython execution without kernel TCP sockets
configs/final_experiment.yaml
tests/
results/                    # essential CSVs, histories and verification
paper_assets/               # one main PDF/PNG figure, architecture table, caption
archive/                    # recovery references and preservation/provenance manifests
artifacts/                  # ignored large tensor checkpoints, retained separately
```

Core module filenames are retained to preserve the validated imports; reorganizing them into `src/` would add cosmetic changes. Pruning, training, data loading and atomic checkpoint helpers were copied unchanged from the approved implementation. The model registry adds D; the MLP implementation is unchanged. This is our PyTorch reimplementation of the paper's methodology, not the authors' original software.

## Insert the figure in LaTeX

With `graphicx` and `booktabs` in the preamble:

```tex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{paper_assets/main_results.pdf}
  \input{paper_assets/main_results_caption.tex}
  \label{fig:width-ratios}
\end{figure*}
```

The architecture table is `paper_assets/architectures.tex`. The PNG is a 600-dpi alternative; the PDF preserves vector text/lines.

## Environment setup

Python 3.12 is the tested interpreter. From the project root on Linux/macOS:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter lab final_experiment.ipynb
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyter lab final_experiment.ipynb
```

Linux CPU-only Torch can be installed first with `python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0`. Direct versions in `requirements.txt` match the installed study environment; `requirements-tested-linux.txt` records the historical complete dependency snapshot. If an exact wheel is unavailable on your platform, `requirements-portable-ranges.txt` offers the previously used compatible ranges. Different versions/platforms may change training numerics; native Windows/macOS and cross-platform bitwise training are not verified. The previous isolated dependency-install attempt failed to resolve the CPU wheel; no new successful independent install is claimed here.

## Default: reproduce the figure from saved results

Leave `MODE = "saved"` and `ENABLE_TRAINING = False` in the notebook and Run All. It loads and verifies final CSVs, recomputes the approved aggregates, and regenerates only the main figure. **It does not train, download MNIST, load checkpoint tensors, or evaluate test data.** The source measurements are never overwritten. `.gitattributes` preserves LF text bytes across checkouts so the SHA256 manifests are not invalidated by automatic line-ending conversion. The saved-data notebook works with the lightweight clean branch; large checkpoints are optional for this workflow.

Equivalent command:

```sh
python analysis.py plot
```

To save figure copies separately: `python analysis.py plot --output regenerated_figures`. To execute the actual notebook cells without Jupyter TCP sockets: `python execute_notebook.py --output final_experiment.executed.ipynb`.

## Explicit full reproduction

Do not run this merely to view existing results. It repeats all 156 stages, with no controls. Select a **new** run name so the verified main run is preserved:

```sh
python study.py run --run-dir artifacts/reproduction_01 --workers 3 --enable-training --download
python analysis.py finalize --run-dir artifacts/reproduction_01 --output reproductions/reproduction_01
```

The initial completed run uses `artifacts/main_study`; the notebook refuses that destination for a new reproduction. Repeating the same CLI run command resumes checksum-verified **complete stages**. An interrupted stage restarts from its prescribed masked original initialization with fresh Adam; Adam is not resumed halfway through a stage. Code, configuration, dataset and software versions are fingerprinted. A mismatch requires a new run directory. `--download` fetches the standard MNIST files, checking published MD5s and decompressed bytes. MNIST is unnecessary for saved-data mode.

Workers run independent architecture/seed trajectories with one Torch thread each. DataLoader workers remain zero for notebook/macOS/Windows compatibility. Up to four process workers are supported; the recorded execution uses three. Wall/summed CPU times are reported after completion in `results/verification.md`. Historical timing suggests roughly 3 CPU-hours for 156 stages, with wall time dependent on worker count and hardware; fresh timing takes precedence. Allow about 2 GB RAM per worker and a few GB of disk headroom for data, checkpoints, Python packages and Git history.

The notebook's full reproduction additionally requires `MODE="reproduce"` and `ENABLE_TRAINING=True`. Merely importing code or executing the default notebook never activates it.

## Submission and checkpoints

The intended submission branch is [`final-four-architecture-review`](https://github.com/pbn179/Advanced-topics-in-Deep-learning/tree/final-four-architecture-review). It includes every file needed for the default saved-results notebook. The complete large-artifact branch is [`final-four-architecture-checkpoints`](https://github.com/pbn179/Advanced-topics-in-Deep-learning/tree/final-four-architecture-checkpoints).

For a lightweight clone:

```sh
git clone --depth 1 --single-branch --branch final-four-architecture-review https://github.com/pbn179/Advanced-topics-in-Deep-learning.git lottery-ticket-research
cd lottery-ticket-research
```

To retrieve the exact saved artifacts into that clone, preserving the current clean branch:

```sh
git fetch --depth 1 origin final-four-architecture-checkpoints
git restore --source=FETCH_HEAD --worktree -- artifacts/main_study
```

Back up any unrelated local `artifacts/main_study` directory before restoring. This only retrieves files; it does not train or evaluate anything. Alternatively, open the checkpoint branch in GitHub and choose **Code → Download ZIP**, extracting it separately. `archive/new_checkpoint_artifacts.json` lists all required artifact SHA256 hashes. See `results/submission_verification.md` for the fresh remote-clone audit and test/notebook results.

### Artifact layout and historical recovery

Large tensor artifacts are not included in the clean branch's ordinary working tree. They are retained on the separate `final-four-architecture-checkpoints` branch and in the cloud workspace; see `archive/README.md` for exact recovery references and completeness. Essential CSVs, figure and notebook are directly accessible on the review branch. GitHub Releases or LFS can replace the separate checkpoint archive when their upload/authentication support is available. No generated ZIP or duplicate checkpoint tree is committed to main.

All existing historical files remain preserved. The cleanup branch is for review only: no main merge, history rewrite, force-push or irreversible deletion is authorized or performed.

## Tests and limitations

```sh
python -m pytest -q
```

Tests check all four parameter counts, survivor schedules, exact original resets, mask persistence, actual best-state capture, complete-stage resumption and corruption rejection, test-gate completeness, and full observed-point aggregation. Synthetic training examples are test fixtures, not new scientific experiments. New-run empirical verification additionally reloads every selected checkpoint and checks the final-state pruning chain.

The design has only three model seeds, one dataset/split and approximately equal dense sizes. Remaining counts and mask geometry diverge across architectures; identical rounds are not matched exact counts. Ten thousand updates is shorter than the original paper's 50,000. Minimum-loss iteration is retrospectively selected, **not early stopping** or a proof of convergence. Curves and stage-averaged summaries are descriptive, with dependent stages rather than extra independent replicates. The A/B/C seeds, split and hyperparameters match the earlier study, so old and new runs must not be pooled as additional independent replications. No statistically significant or universally optimal ratio is established by three seeds. No newly run controls or independent unseen test confirmation should be inferred.

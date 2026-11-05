# Lottery Ticket Hypothesis: hidden-layer width ratios

**Start with [final_experiment.ipynb](final_experiment.ipynb). Its default mode loads saved results, verifies checkpoints, and regenerates paper assets without training or reevaluating the test set.**

Research question: **How does the ratio between hidden-layer widths affect the accuracy and trainability of winning tickets when the total number of parameters is held approximately constant?**

Frankle and Carbin's Lottery Ticket Hypothesis proposes that dense networks contain sparse subnetworks that can train effectively from their original initialization. Appendix G.6 studies network size while retaining a 3:1 hidden-width ratio. We instead compare different hidden-layer allocations at approximately 200,000 dense parameters. The hypothesis is that allocation affects the accuracy and trainability of resulting tickets; the results do not establish a universally optimal ratio.

| Architecture | Hidden widths | Dense parameters, including biases |
|---|---|---:|
| A | 162–421 | 200,013 |
| B | 200–204 | 200,054 |
| C | 231–77 | 199,979 |

## Final findings

At observed pruning stage 12, B had the highest mean original-ticket held-out accuracy (98.293 ± 0.140%) and largest mean improvement over its own dense baseline (+0.443 ± 0.202 percentage points). Original-initialization tickets outperformed same-mask random controls in all 18 held-out pairs. At stage 12, mean ticket-minus-control advantages were A +0.573 ± 0.146 pp, B +0.873 ± 0.126 pp and C +0.843 ± 0.294 pp. C had the largest validation initialization advantage, but this ranking did not persist uniformly on test. Rankings depend on metric, sparsity and initialization condition.

These are **means ± sample standard deviations over three seeds**, not significance claims. See [final held-out findings](reports/stage4_test/findings.md), [Stage 3 findings](reports/stage3/findings.md), and [control interpretation](reports/stage4_controls/interpretation.md).

## Directory structure

```text
final_experiment.ipynb        consolidated executed notebook, saved-results mode by default
final_project.py             portable read-only analysis and paper-asset generation
reproduce_final.py           explicitly opt-in full reproduction and stage resumption
config_final.yaml            locked final protocol
models.py, pruning.py        tested model/initialization/masking/rewinding
train.py, data.py             frozen training, evaluation and MNIST loading
checkpoints.py               original atomic/checksummed checkpoint implementation
stage*.py, experiment.py      historical runners and analysis, retained for provenance
config*.yaml                 all historical experiment configurations
archive/notebooks/           original executed Stage 1–4B notebooks, unchanged
reports/                     original stage reports, CSVs, histories, figures and checkpoints
results/                     original early-stage experiment artifacts
paper_assets/                regenerated PDF/PNG figures and CSV/LaTeX tables
packaging/                   artifact manifests, preservation and release verification
requirements.txt             exact direct dependency versions used for this release
requirements-tested-linux.txt full historical Linux dependency snapshot
requirements-portable-ranges.txt historical compatible version ranges (fallback)
tests/                       unit and synthetic integration tests
06_Frankle2023_LotteryTicket.pdf supplied source paper (Frankle & Carbin, 2019)
```

The ZIP has one top-level `lottery_ticket_research_final/` directory. All checkpoints, masks, original initializations, saved starts/finals and the 27 reconstructed best-validation ticket states are included. No separate checkpoint download is needed. MNIST and virtual environments are excluded. Original notebooks have one canonical copy in `archive/notebooks/`; byte-identical report notebook copies are excluded from the ZIP and mapped in `packaging/archive_contents.json`. Caches, lock files, logs and duplicate live histories are excluded. Historical reports retain their original bytes and may reference such excluded operational files; they are historical audits, not the release manifest. No result/checkpoint is silently removed or regenerated.

## Installation (Python 3.12 recommended)

From the extracted project root on macOS/Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab final_experiment.ipynb
```

On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m jupyter lab final_experiment.ipynb
```

On Linux, CPU-only PyTorch can be installed first to avoid unnecessary CUDA dependencies:

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements.txt
```

The direct pins reflect the installed release environment; the complete Linux snapshot includes platform-specific transitive dependencies. If an exact version lacks a wheel on your platform/index, `python -m pip install -r requirements-portable-ranges.txt` provides the historical version ranges. Mode A checks stored measurements and hashes and does not require numerically identical training across platforms. Exact Mode B training results are not promised across software versions, CPU/GPU backends or operating systems. Linux CPU was tested here; native macOS/Windows were not. See `packaging/verification.md` for the actual clean-install outcome and tested environment.

Choose the project kernel in Jupyter and run from the project root. The notebook has no dependency on an existing `.venv`, external data paths or local machine directories.

## Mode A: saved-artifact reproduction (default)

Leave `MODE = "A"` and `ENABLE_FULL_REPRODUCTION = False` in the notebook, then Run All. This verifies the required CSVs and all 45 selected checkpoints, recomputes approved aggregates, displays the investigation, and writes only `paper_assets/`. It never calls training, downloads MNIST or evaluates a model on the test set. Missing artifacts produce errors naming the missing paths.

Regenerate all paper figures/tables without opening Jupyter:

```sh
python final_project.py --output paper_assets
python final_project.py --verify-all --output paper_assets
```

`--verify-all` verifies every file in the release data manifest rather than just the required analysis files and selected checkpoints. Paper assets may be regenerated; original `reports/`, `results/` and archived notebooks are protected from analysis writes. To use another figure directory: `python final_project.py --output regenerated_assets`.

Execute the notebook headlessly with a normal Jupyter kernel:

```sh
python -m jupyter nbconvert --to notebook --execute final_experiment.ipynb --output final_experiment.executed.ipynb --ExecutePreprocessor.timeout=1200
```

For restricted systems without TCP sockets, `python execute_final_notebook.py --output final_experiment.executed.ipynb` runs the actual cells using IPython in-process and captures their rich outputs. It refuses an enabled training mode.

## Locked scientific protocol

MNIST: 55,000 train / 5,000 validation / 10,000 held-out test; fixed split seed 42. Model seeds 0, 1, 2. Adam learning rate 0.0012, batch size 60, Gaussian Glorot initialization, ReLU hidden layers. Every original IMP stage trained for **10,000 fixed updates**, with validation every **100** updates. Twelve pruning rounds plus dense: prune floor(20% of surviving weights) independently per hidden layer and floor(10%) in the output layer; biases stay dense. Restore surviving weights and biases to original initialization and reset Adam after each round. Subsequent masks use final 10k trained weights, **not** validation-selected weights.

Stage 3 comprises 117 architecture/seed/stage observations. Stage 4A comprises 18 controls at stages 6/12, identical mask positions, fresh initialization seed `(seed + 1000003 + stage*1009) % 2**32`, and the same split, batch order, fresh Adam and budget. Controls do not affect future ticket masks. Stage 4B recovered 27 selected tickets at stages 0/6/12 and tested those plus the 18 saved selected controls only after all validation checks passed. Replay absolute tolerances were loss 1e-8 and accuracy 1e-12; maximum observed loss difference was 9.02e-17, with identical accuracies.

C seed 0 originated from Stage 2.5 with a 500-step primary grid and stored 100-step diagnostic traces. The approved Stage 3 results recomputed selection from its full 100-step grid without rerunning training or changing masks. Frozen source metrics and reports remain intact.

## Mode B: full experimental reproduction (optional; NOT run during packaging)

This trains **117 original stages + 18 same-mask controls**, 1,350,000 updates in total, and evaluates the 45 predefined selected states after a validation gate. Best-validation checkpoints are now captured during training for every stage, avoiding Stage 4B replay. It uses the unchanged tested model, pruning, resetting, data and training functions; only orchestration and snapshot capture are new.

Explicit command:

```sh
python reproduce_final.py --enable-full-reproduction --output reproductions/full_run --workers 1
```

Rerun the **same command** to resume checksum-verified completed stages. Interrupted individual stages restart from their prescribed masked initialization with fresh Adam; resumption is at complete-stage boundaries, not inside Adam updates. Changing configuration/code/software requires a new output directory. Workers 1–3 are supported; complete seed trajectories run independently. Default data-loading workers remain zero for Windows/macOS notebook compatibility. MNIST is automatically fetched using the published original-file checksums. No MNIST is needed for Mode A.

In the notebook, set **both** `MODE = "B"` and `ENABLE_FULL_REPRODUCTION = True`, then choose a relative directory under `reproductions/`. The notebook displays the new reproduction's tables and assets separately. Neither import nor ordinary Run All enables Mode B. Paths outside `reproductions/NAME` are rejected, including attempts to overwrite original reports/results.

Cost context: historical Stage 3 summed training+validation time was about **135 CPU minutes**, plus **18 minutes** for controls on the recorded CPU setup. Thus budget roughly **2.5–3 hours sequentially on a comparable CPU**, with substantial hardware and snapshot-I/O variation; up to three workers can reduce elapsed time but not total work. Each worker materializes about 220 MB of MNIST tensors plus Torch/model/optimizer overhead. Allow 2 GB RAM per worker, about 1 GB free storage for new runs/data, and more headroom for environment installation. These are estimates, not a newly timed reproduction benchmark. The new complete Mode B scientific sweep was intentionally not executed; its guard, checkpoint, resumption and synthetic integration behavior was tested.

## Result files and paper assets

- `reports/stage3/individual_results.csv`: 117 observed validation measurements; `summary_mean_std.csv`: three-seed means and sample SDs.
- `reports/stage3/common_budget_interpolated_*.csv`: interpolated validation estimates at common parameter budgets (including 16k/20k), **not observed checkpoints and not test results**.
- `reports/stage4_controls/paired_results.csv`: seed/architecture/stage paired validation differences; `paired_summary.csv`: sample SD of those paired differences.
- `reports/stage4_test/individual_results.csv`: 45 validation-selected states evaluated on 10,000 held-out examples; `summary.csv`, `paired_results.csv`, `paired_summary.csv`: condition and paired summaries. Original vs random dense-relative changes share the corresponding original dense baseline.
- `reports/stage4_test/replays/`: 27 recovered best states; selected controls remain under `reports/stage4_controls/runs/`.
- `paper_assets/`: architecture diagram; observed-parameter validation accuracy, selected-iteration and dense-relative curves; control accuracy/loss/trainability and paired gaps; held-out accuracy/dense-relative/paired comparisons. Each figure is PDF + 300-dpi PNG. Exact CSV tables accompany LaTeX tables (12 significant digits for readability).

Accuracy columns are fractions unless explicitly suffixed `_pp`; plotted accuracy is percent. `remaining_parameters` counts mask-active weights plus dense biases. `sparsity` is pruned **weights / total weights**, excluding biases; `parameter_sparsity` includes biases. Identical pruning rounds have different remaining counts across architectures. Bars show sample SD (ddof=1), not confidence intervals; seed trajectories are shown. Positive ticket-minus-control accuracy differences favor original initialization.

## Tests, integrity and release verification

```sh
python -m pytest -q
python final_project.py --verify-all --output paper_assets
```

Tests use synthetic examples for reversible training/checkpoint integration checks; they do not launch additional MNIST experiments. `packaging/original_artifact_checksums.json` freezes all original report/result/notebook/core bytes before packaging. `packaging/artifact_manifest.json` verifies the delivered data paths; `packaging/archive_contents.json` records excluded operational files and notebook aliases. Historical reports contain absolute provenance paths that are retained verbatim, but the final analysis and reproduction use paths relative to their delivered project. Archived milestone notebooks/runners are historical records, not the portable entry point; use the final notebook and new runner.

`packaging/verification.md` records the full test run, executed default notebook, extracted-archive check, dependency-install attempt and any limitations. `packaging/verification.json` provides machine-readable evidence. `packaging/release_manifest.json` additionally hashes every delivered file except itself; the ZIP SHA256 sidecar covers the whole archive. The archive is built with `python package_final.py`; it does not train or install dependencies. The separately printed ZIP SHA256 can verify the delivered download.

## Relation to Appendix G.1 and G.6; limitations

We retain G.1's split sizes, batch size, layer-wise survivor pruning, Gaussian Glorot/Adam setup, original initialization reset, retrospective validation selection, and 100-update validation grid. We use **10,000 updates instead of 50,000**, three architectures near 200k parameters, a fixed published split seed and three model seeds. The paper's G.6 changes size while retaining 3:1 widths; our primary intervention changes the width ratio at approximately fixed dense parameter count. We show sample SD rather than the paper's min/max bars. Test evaluation is performed only on the predefined selected models after the validation study, not every 100 updates. These are deliberate scope differences, not an exact replication of all Appendix G experiments.

Only three seeds and one dataset/split were studied. Dense counts are approximate, and counts/mask geometry diverge after pruning. Initialization, final accuracy and selected-iteration rankings need not agree. Random controls may be limited by the fixed 10k budget (16/18 minima occurred in its final 20%). Results support descriptive differences in accuracy/trainability and an original-initialization benefit in these observed comparisons, not statistical significance, asymptotic inferiority of random initialization, causal attribution to ratio alone, or a universally superior architecture. No test-based redesign or additional scientific experiment is part of this package.

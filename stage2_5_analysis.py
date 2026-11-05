"""Measured diagnostics for the bounded Stage 2.5 pilot."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from experiment import run_experiment, resume_experiment
from models import ARCHITECTURES, MLP, parameter_count


REPORT = Path('reports/stage2_5')


def architecture_table():
    return pd.DataFrame([
        dict(architecture=name, hidden1=ARCHITECTURES[name][0], hidden2=ARCHITECTURES[name][1],
             parameters=parameter_count(MLP(*ARCHITECTURES[name])))
        for name in ('A', 'B', 'C', 'LeNet300100')])


def run_benchmarks():
    """New networks: 500/1000 updates dense+one round; legacy: matched dense+one-round timing."""
    REPORT.mkdir(parents=True, exist_ok=True)
    frames = []
    for architecture in ('A', 'B', 'C', 'legacy_A', 'legacy_B', 'LeNet300100'):
        for iterations in (500, 1000):
            print(f'CPU benchmark: {architecture}, {iterations} updates', flush=True)
            run, frame = run_experiment(architecture=architecture, iterations=iterations,
                rounds=1,
                eval_interval=500, num_threads=1, preload_tensors=True,
                output_dir=REPORT / 'benchmarks', download=False, progress=False, save_plots=False)
            frame['run_directory'] = str(run)
            frame['setup'] = '200k' if architecture in ('A', 'B', 'C') else '266k matched pipeline'
            frames.append(frame)
            pd.concat(frames, ignore_index=True).to_csv(REPORT / 'benchmarks.csv', index=False)
    return pd.concat(frames, ignore_index=True)


def run_pilot(config):
    """Session manifest locates the same run after notebook/kernel interruption."""
    manifest = REPORT / 'pilot_location.json'
    if manifest.exists():
        run = Path(json.loads(manifest.read_text())['run_directory'])
        return resume_experiment(run)
    # Allocate a parent per notebook session; locate its sole run after a crash
    # even if execution failed before run_experiment could return its directory.
    parent = Path(config['output_dir'])
    parent.mkdir(parents=True, exist_ok=True)
    candidates = sorted(parent.glob('C-seed0-*'))
    if candidates:
        if len(candidates) != 1:
            raise ValueError('Multiple pilot runs found; choose an explicit resume directory')
        run = candidates[0]
        manifest.write_text(json.dumps(dict(run_directory=str(run)), indent=2))
        return resume_experiment(run)
    run, results = run_experiment(**config)
    manifest.write_text(json.dumps(dict(run_directory=str(run)), indent=2))
    return run, results


def convergence_and_sampling(run_dir):
    rows = []
    combined = []
    results = pd.read_csv(run_dir / 'results.csv')
    for record in results.to_dict('records'):
        history = pd.read_csv(run_dir / f"history_stage{record['stage']}.csv")
        fine = history.loc[history.validation_loss.idxmin()]
        coarse_history = history[(history.iteration % 500 == 0) | (history.iteration == record['iterations'])]
        coarse = coarse_history.loc[coarse_history.validation_loss.idxmin()]
        final = history.iloc[-1]
        late = coarse_history[coarse_history.iteration >= .8 * record['iterations']]
        early = coarse_history[coarse_history.iteration < .8 * record['iterations']]
        late_gain = early.validation_loss.min() - late.validation_loss.min()
        rows.append(dict(stage=record['stage'], remaining_weights=record['remaining_weights'],
            remaining_parameters=record['remaining_weights'] + record['parameters'] - record['total_weights'],
            remaining_percent=record['remaining_percent'], best_iteration_100=int(fine.iteration),
            best_iteration_500=int(coarse.iteration), loss_100=fine.validation_loss, loss_500=coarse.validation_loss,
            accuracy_100=fine.validation_accuracy, accuracy_500=coarse.validation_accuracy,
            coarse_minus_fine_loss=coarse.validation_loss - fine.validation_loss,
            coarse_minus_fine_accuracy_pp=100 * (coarse.validation_accuracy - fine.validation_accuracy),
            best_in_last_20pct_500=bool(coarse.iteration >= .8 * record['iterations']),
            best_at_final_500=bool(coarse.iteration == record['iterations']),
            loss_improvement_in_last_20pct=late_gain,
            final_loss=final.validation_loss, final_accuracy=final.validation_accuracy,
            primary_runtime_seconds=record['primary_runtime_seconds'],
            diagnostic_evaluation_seconds=record['diagnostic_evaluation_seconds']))
        history['stage'] = record['stage']
        history['remaining_percent'] = record['remaining_percent']
        combined.append(history)
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(REPORT / 'convergence_sampling.csv', index=False)
    history = pd.concat(combined, ignore_index=True)
    history.to_csv(REPORT / 'validation_history.csv', index=False)
    results.to_csv(REPORT / 'results.csv', index=False)
    return diagnostics, history


def cost_projections(results, benchmarks):
    """Use actual eval500 timings, not parameter-count scaling or diagnostic cost."""
    selected = benchmarks[benchmarks.iterations == 1000].copy()
    rates = selected.groupby('architecture').apply(
        lambda frame: frame.primary_runtime_seconds.sum() / frame.iterations.sum(), include_groups=False)
    matched_new_rates = rates[['A', 'B', 'C']].copy()
    # C includes every sparsity level in the long pilot, removing fine diagnostic eval time.
    rates['C'] = results.primary_runtime_seconds.sum() / results.iterations.sum()
    new_hours = 3 * 13 * 10000 * rates[['A', 'B', 'C']].sum() / 3600
    matched_old = 3 * 13 * 10000 * rates[['legacy_A', 'legacy_B', 'LeNet300100']].sum() / 3600
    # Previous Stage 2 projection used eval100, four CPU threads and per-image PIL transforms.
    prior = pd.read_csv('reports/stage2/cost_estimates.csv')
    previous_hours = float(prior[(prior.iterations_per_stage == 10000) &
        (~prior.with_controls) & (prior.training_stages == 117)].iloc[0].estimated_hours)
    rows = []
    for label, hours in [('200k eval500 optimized pipeline', new_hours),
                         ('266k eval500 matched optimized pipeline', matched_old),
                         ('266k previous Stage 2 eval100 pipeline', previous_hours)]:
        rows.append(dict(setup=label, device='cpu', architectures=3, seeds=3, stages=13,
                         iterations=10000, total_updates=1170000, estimated_hours=hours,
                         planning_hours_plus_25_percent=hours * 1.25))
    matched_new_hours = 3 * 13 * 10000 * matched_new_rates.sum() / 3600
    extra_validation_cost = 0.0
    for architecture in ('A', 'B'):
        group = selected[selected.architecture == architecture]
        seconds_per_evaluation = group.evaluation_seconds.sum() / (group.iterations / 500).sum()
        extra_validation_cost += seconds_per_evaluation
    extra_validation_cost += results.evaluation_seconds.sum() / (results.iterations / 100).sum()
    fine_hours = new_hours + 3 * 13 * 80 * extra_validation_cost / 3600
    for label, hours in [('200k eval500 matched short-probe projection', matched_new_hours),
                         ('200k eval100 sensitivity estimate', fine_hours)]:
        rows.append(dict(setup=label, device='cpu', architectures=3, seeds=3, stages=13,
                         iterations=10000, total_updates=1170000, estimated_hours=hours,
                         planning_hours_plus_25_percent=hours * 1.25))
    costs = pd.DataFrame(rows)
    costs.to_csv(REPORT / 'cost_estimates.csv', index=False)
    rates.rename('seconds_per_update').to_csv(REPORT / 'timing_rates.csv')
    (REPORT / 'hardware.json').write_text(json.dumps(dict(torch=torch.__version__,
        cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available(),
        cpu_threads=1, gpu_timings='Unavailable; no GPU measured'), indent=2))
    return costs


def save_figures(diagnostics, history):
    for metric, ylabel, name in [('validation_loss', 'Validation cross entropy', 'validation_loss'),
                                 ('validation_accuracy', 'Validation accuracy (fraction)', 'validation_accuracy')]:
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        for stage, group in history.groupby('stage'):
            ax.plot(group.iteration, group[metric], label=f'{stage}: {group.remaining_percent.iloc[0]:.1f}% weights')
        ax.set(xlabel='Training updates', ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(title='Pruning stage', fontsize=7, ncol=2)
        for ext in ('pdf', 'png'):
            fig.savefig(REPORT / f'{name}.{ext}', dpi=250)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for interval in (100, 500):
        axes[0].plot(diagnostics.remaining_parameters, 100 * diagnostics[f'accuracy_{interval}'],
                     marker='o', label=f'Validation every {interval}')
        axes[1].plot(diagnostics.remaining_percent, diagnostics[f'best_iteration_{interval}'],
                     marker='o', label=f'Validation every {interval}')
    axes[0].set(xlabel='Remaining parameters (including biases)', ylabel='Accuracy at minimum loss (%)')
    axes[1].set(xlabel='Remaining weights (%)', ylabel='Iteration of minimum validation loss')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ext in ('pdf', 'png'):
        fig.savefig(REPORT / f'accuracy_trainability.{ext}', dpi=250)
    plt.close(fig)


def write_assessment(diagnostics, costs, results):
    """State observed evidence and a bounded next step; never launch another pilot."""
    fine_late = int((diagnostics.best_iteration_100 >= 8000).sum())
    coarse_late = int(diagnostics.best_in_last_20pct_500.sum())
    final_count = int(diagnostics.best_at_final_500.sum())
    high = diagnostics[diagnostics.stage >= 9]
    loss_gap = diagnostics.coarse_minus_fine_loss.max()
    accuracy_gap = diagnostics.coarse_minus_fine_accuracy_pp.abs().max()
    dense, last = diagnostics.iloc[0], diagnostics.iloc[-1]
    dense_gap = 100 * (dense.accuracy_100 - dense.accuracy_500)
    iteration_gap = abs(int(dense.best_iteration_100 - dense.best_iteration_500))
    coarse_effect = 100 * (last.accuracy_500 - dense.accuracy_500)
    fine_effect = 100 * (last.accuracy_100 - dense.accuracy_100)
    effect_description = 'changes even the sign' if coarse_effect * fine_effect < 0 else 'changes the magnitude'
    hardware = json.loads((REPORT / 'hardware.json').read_text())
    gpu_status = ('available but not benchmarked' if hardware['cuda_available'] or hardware['mps_available']
                  else 'unavailable in this runtime')
    if max(fine_late, coarse_late) >= 3 or int(high.best_at_final_500.sum()) >= 2:
        recommendation = ('**B — investigate increasing the budget to 15,000–20,000 updates.** '
            'Repeated late minima make 10,000 insufficiently supported. First run fixed-mask '
            '15k/20k diagnostic probes at the flagged sparse C stages using saved masks and original '
            'initialization; those probes are not a replacement for an IMP run with a consistent longer '
            'budget. Then validate a common budget on limited A/B pilots before Stage 3.')
    else:
        recommendation = ('**C — run additional targeted A/B pilots before deciding.** '
            '10,000 updates looks promising for this C seed, but is not established for A/B. '
            'Propose A and B, seed 0 only, 12 pruning rounds at 10,000 updates and validation every '
            '100, downsampling the same traces to 500 for comparison. The 500-grid change is not yet '
            'scientifically neutral. These are two single-seed pilots, not the three-seed study. Extend flagged high-sparsity '
            'stages to 15k/20k in separate fixed-mask diagnostic probes if minima remain late.')
    new = costs.iloc[0]
    matched = costs.iloc[1]
    old = costs.iloc[2]
    text = f'''# Stage 2.5 assessment

{recommendation}

No Stage 3/full three-seed experiment has been started. The completed scientific pilot is C (231–77), seed 0, 12 pruning rounds / 13 training stages, 10,000 fixed updates per stage. All 55,000/5,000/10,000 MNIST examples are retained; the test set was not evaluated.

## Parameter counts and method

PyTorch verifies A=200,013, B=200,054, C=199,979 total parameters (including biases). The old LeNet-300-100 remains available as `LeNet300100` (266,610). The default main architecture names A/B/C now refer to the requested 200k models; legacy notebooks use explicit legacy names. Historical result files are unchanged.

The Glorot-normal/Adam 0.0012/batch-60 protocol and validated layer-wise 20%/20%/10% pruning, mask enforcement, initialization rewind and optimizer resets are preserved. No early stopping is used; pruning always uses the final fixed-budget ticket weights. New deviations from G.1 are 10,000 rather than 50,000 updates and primary evaluation every 500 rather than 100. Bias, preprocessing, rounding and split choices remain as documented in Stage 2.

## Convergence and validation frequency

- Primary (500-grid) minima in updates 8,000–10,000: **{coarse_late}/13**.
- Fine (100-grid) minima in updates 8,000–10,000: **{fine_late}/13**.
- Primary minima exactly at update 10,000: **{final_count}/13**.
- Final-stage retained weights: **{int(diagnostics.iloc[-1].remaining_weights):,}** ({diagnostics.iloc[-1].remaining_percent:.3f}%); sparsity is about {100-diagnostics.iloc[-1].remaining_percent:.2f}%.
- Largest extra selected loss from using the coarser grid: **{loss_gap:.6f}**.
- Largest absolute change in accuracy at the selected minimum: **{accuracy_gap:.3f} percentage points**.

**Sampling-frequency caution:** The dense 100-grid selection is {100*dense.accuracy_100:.2f}% at update {int(dense.best_iteration_100):,}; the 500-grid selection is {100*dense.accuracy_500:.2f}% at update {int(dense.best_iteration_500):,}. That is a **{abs(dense_gap):.2f} percentage-point** difference and **{iteration_gap:,} updates** in the reported trainability statistic. At stage 12, the ticket-minus-dense accuracy is {coarse_effect:+.2f} pp on the 500 grid but {fine_effect:+.2f} pp on the 100 grid. The coarser grid therefore {effect_description} of this descriptive single-seed comparison. This does not establish a population-level advantage, but it shows that 500-step evaluation cannot be assumed harmless for this research question. Keep the requested code default of 500; recommend retaining every-100 evaluation for scientific comparisons unless the follow-up establishes grid robustness.

These grid comparisons use the SAME trajectory and masks, not independently trained runs. The additional 100-grid evaluations never select the primary result, alter training duration, or feed into pruning. Low coarse-grid loss cannot be smaller than the fine-grid minimum. A coarser grid can shift the reported best iteration and accuracy without changing the trained network. These are descriptive single-seed differences, not significance tests. Earlier minima alone do not prove convergence; late validation fluctuations and transient minima must be interpreted alongside the full loss/accuracy traces. All per-stage observations are saved in `validation_history.csv` and `convergence_sampling.csv`.

## Measured CPU cost

C's full pilot training/evaluation runtime was **{results.runtime_seconds.sum()/60:.2f} minutes**, including fine diagnostic evaluations. Measured diagnostic-only evaluation time was **{results.diagnostic_evaluation_seconds.sum():.2f} seconds**. Removing that cost gives a primary-500-grid estimate of **{results.primary_runtime_seconds.sum()/60:.2f} minutes** for C. This subtraction is approximate and retains CSV/logging overhead.

For A/B/C × three seeds × 13 stages × 10,000 updates (**117 stage trainings; 1,170,000 updates**, no optional controls):

| Measurement basis | Estimated serial CPU hours | With 25% planning allowance |
|---|---:|---:|
| New 200k setup, eval500 | {new.estimated_hours:.2f} | {new.planning_hours_plus_25_percent:.2f} |
| Legacy 266k models, matched optimized pipeline / eval500 | {matched.estimated_hours:.2f} | {matched.planning_hours_plus_25_percent:.2f} |
| Previous Stage 2 266k pipeline / eval100 | {old.estimated_hours:.2f} | {old.planning_hours_plus_25_percent:.2f} |

The primary comparison uses measured new A/B 1,000-update dense/pruned timings and C's complete sparsity trajectory. The matched legacy probes use the same CPU thread count, pixel cache and 500 evaluation grid; they cover dense plus one pruning round and are less representative of mask-related overhead at higher sparsity. A directly matched new-200k short-probe estimate is also saved in the cost table to isolate this workload difference. Therefore these are planning estimates, not a controlled attribution of savings to model size alone. The historical comparison combines model-size, validation-frequency, tensor-caching and CPU-thread changes. Smaller parameter count alone is not used to predict speed. A/B behavior at high sparsity remains unmeasured. No sparse-kernel speedup is assumed. Initialization/loading/checkpoint/plot overhead and hardware load add uncertainty; the 25% allowance is not a statistical confidence interval.

The directly matched short-probe new-200k estimate is **{costs.iloc[3].estimated_hours:.2f} h**, versus **{matched.estimated_hours:.2f} h** for 266k: only about **{100*(1-costs.iloc[3].estimated_hours/matched.estimated_hours):.1f}%** faster in these measured CPU workloads. Parameter reduction alone did not deliver a 25% runtime improvement. Most of the historical speedup comes from the combined pipeline/thread/evaluation changes. Keeping every-100 validation instead is estimated at **{costs.iloc[4].estimated_hours:.2f} h**, or **{costs.iloc[4].planning_hours_plus_25_percent:.2f} h** with allowance. This sensitivity uses measured per-evaluation time for A/B probes and the C trace, adding 80 evaluations per stage; it is an extrapolation, not a completed full run.

GPU timing: **not measured**; GPU hardware is {gpu_status}. No GPU speed estimate is claimed.

## Validation, resumption and reproducibility

See `tests.txt` for the final passing test suite, including exact model counts, pruning/mask invariants through 13 rounds, original survivor restoration, optimizer reset, identical interrupted/resumed trajectories with and without controls, no retraining of committed stages, corruption/configuration rejection, and exact tensor-cache pixel equivalence. `stage2_5.ipynb` executed the benchmarks and the full authorized C pilot.

Completed stages commit checksummed original weights, split, masks, start/final states, validation histories and cumulative result rows atomically. Per-stage RNG/loader seeds reset explicitly and Adam is intentionally fresh, so those saved states suffice for exact same-environment stage-boundary resumption. A partial stage is rerun; completed stages are skipped. Automatic mid-stage Adam-state resume is not implemented. Code, versions, widths, split, device and CPU thread count must match. Do not use simultaneous writers to a run directory. Legacy Stage 1/2 results cannot be resumed/migrated and are left unchanged.

Scientific data sizes and pixel values are unchanged. The optional tensor cache materializes exactly equivalent float32 pixels once (~220 MB) and one CPU thread avoids overhead for these small dense matrices. Timing and parameter tables, configuration and device availability are saved alongside all figures and results in this directory. Native Windows/macOS and CUDA/MPS have not been tested.
'''
    (REPORT / 'assessment.md').write_text(text, encoding='utf-8')
    return text


def save_high_sparsity_figures(history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for stage, group in history[history.stage >= 9].groupby('stage'):
        label = f'Stage {stage}, {group.remaining_percent.iloc[0]:.1f}% weights'
        for ax, metric in zip(axes, ('validation_loss', 'validation_accuracy')):
            ax.plot(group.iteration, group[metric], label=label)
    for ax in axes:
        ax.axvline(8000, linestyle=':', color='gray', label='Last 20% begins')
        ax.set_xlabel('Training updates')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel('Validation cross entropy')
    axes[1].set_ylabel('Validation accuracy (fraction)')
    for ext in ('pdf', 'png'):
        fig.savefig(REPORT / f'high_sparsity_learning.{ext}', dpi=250)
    plt.close(fig)


def complete_legacy_pruning_benchmarks(benchmarks):
    """Balance old/new size comparisons on common dense+round1 workloads."""
    for architecture in ('legacy_A', 'legacy_B', 'LeNet300100'):
        for iterations in (500, 1000):
            selected = (benchmarks.architecture == architecture) & (benchmarks.iterations == iterations)
            if 1 in benchmarks.loc[selected, 'stage'].tolist():
                continue
            print(f'Balanced legacy timing: {architecture}, {iterations}', flush=True)
            run, frame = run_experiment(architecture=architecture, iterations=iterations,
                rounds=1, eval_interval=500, num_threads=1, preload_tensors=True,
                output_dir=REPORT / 'benchmarks', download=False, progress=False, save_plots=False)
            frame['run_directory'] = str(run)
            frame['setup'] = '266k matched pipeline'
            benchmarks = pd.concat([benchmarks[~selected], frame], ignore_index=True)
            benchmarks.to_csv(REPORT / 'benchmarks.csv', index=False)
    return benchmarks

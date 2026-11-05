"""Uniform 100-step analysis of A/B pilots and read-only reuse of C fine traces."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from checkpoints import load_completed, sha256
from experiment import run_experiment, resume_experiment
from models import ARCHITECTURES

REPORT = Path('reports/stage2_6')
PREVIOUS = Path('reports/stage2_5')


def preserve_previous_reports():
    """Snapshot every Stage 2.5 output before work; never write into that tree."""
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = REPORT / 'stage2_5_snapshot.json'
    observed = {str(path.relative_to(PREVIOUS)): sha256(path)
                for path in sorted(PREVIOUS.rglob('*')) if path.is_file()}
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        if observed != saved:
            raise ValueError('Stage 2.5 reports changed since the preservation snapshot')
    else:
        manifest.write_text(json.dumps(observed, indent=2), encoding='utf-8')
    return len(observed)


def verify_previous_reports():
    count = preserve_previous_reports()
    result = {'stage2_5_files_unchanged': True, 'files_verified': count}
    (REPORT / 'preservation_check.json').write_text(json.dumps(result, indent=2))
    return result


def validate_config(config, architecture, *, reused_c=False):
    expected = dict(architecture=architecture, hidden_widths=list(ARCHITECTURES[architecture]),
        seed=0, split_seed=42, rounds=12, iterations=10000, learning_rate=.0012, batch_size=60,
        control=False, num_threads=1, preload_tensors=True, device='cpu',
        dataset_sizes={'train': 55000, 'validation': 5000, 'test': 10000})
    expected['eval_interval'] = 500 if reused_c else 100
    expected['trace_interval'] = 100 if reused_c else None
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{architecture}: incompatible pilot configuration: {key}')


def run_ab_pilot(architecture, settings):
    """Run only A or B; restart uses the validated stage-boundary resume API."""
    if architecture not in ('A', 'B'):
        raise ValueError('Stage 2.6 may train only A and B; C must be reused')
    REPORT.mkdir(parents=True, exist_ok=True)
    parent = REPORT / 'pilots' / architecture
    parent.mkdir(parents=True, exist_ok=True)
    location = REPORT / f'{architecture}_location.json'
    candidates = sorted(parent.glob(f'{architecture}-seed0-*'))
    if location.exists():
        run = Path(json.loads(location.read_text(encoding='utf-8'))['run_directory'])
    elif len(candidates) == 1:
        run = candidates[0]
    elif candidates:
        raise ValueError('Ambiguous pilot directories; choose an explicit resume directory')
    else:
        run = None
    if run is not None:
        config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        # A stage may have failed before data initialization, so validate only
        # scientific settings here; full dataset/config checks follow collection.
        for key in ('seed', 'rounds', 'iterations', 'eval_interval', 'trace_interval',
                    'split_seed', 'learning_rate', 'control', 'num_threads', 'preload_tensors'):
            if config.get(key) != settings[key]:
                raise ValueError(f'Resume settings differ: {key}')
        run, results = resume_experiment(run)
    else:
        run, results = run_experiment(architecture=architecture, output_dir=parent, **settings)
    location.write_text(json.dumps({'run_directory': str(run.resolve())}, indent=2))
    return run, results


def history_metrics(history, iterations=10000):
    """Select loss minimum on the complete fine grid; measure late trends explicitly.

    Budget flags are descriptive: minimum in the last 20%, or a decreasing late
    trend ending within 2% of the best loss. They are not convergence proofs.
    """
    history = history.sort_values('iteration').copy()
    if history.iteration.tolist() != list(range(100, iterations + 1, 100)):
        raise ValueError('Expected a complete unique 100-update validation grid')
    numeric = history[['validation_loss', 'validation_accuracy']].to_numpy()
    if not np.isfinite(numeric).all() or (history.validation_loss < 0).any():
        raise ValueError('Non-finite or invalid validation metrics')
    if not history.validation_accuracy.between(0, 1).all():
        raise ValueError('Accuracy must be a fraction in [0,1]')
    best = history.loc[history.validation_loss.idxmin()]
    final = history.iloc[-1]
    late = history[history.iteration >= .8 * iterations]
    early = history[history.iteration < .8 * iterations]
    count = max(1, len(late) // 2)
    mean_change = float(late.validation_loss.iloc[-count:].mean() - late.validation_loss.iloc[:count].mean())
    slope = float(np.polyfit(late.iteration / 1000, late.validation_loss, 1)[0])
    slope = 0.0 if abs(slope) < 1e-12 else slope
    decreasing = slope < 0 and mean_change < -1e-12
    near_end = bool(best.iteration >= .8 * iterations)
    close_to_best = bool(final.validation_loss <= 1.02 * best.validation_loss)
    return dict(best_iteration=int(best.iteration), best_validation_loss=float(best.validation_loss),
        best_validation_accuracy=float(best.validation_accuracy), final_validation_loss=float(final.validation_loss),
        final_validation_accuracy=float(final.validation_accuracy), best_at_budget_end=bool(best.iteration == iterations),
        best_in_last_20pct=near_end, late_loss_slope_per_1000=slope,
        late_mean_loss_change=mean_change, late_loss_decreasing=decreasing,
        late_global_min_improvement=float(early.validation_loss.min() - late.validation_loss.min()),
        final_loss_within_2pct_of_best=close_to_best,
        budget_concern=near_end or (decreasing and close_to_best))


def recompute_results(records, histories, origin):
    """Return new result/history frames; never mutate historical frames or files."""
    rows, traces = [], []
    records = records.sort_values('stage')
    for source in records.to_dict('records'):
        stage = int(source['stage'])
        history = histories[stage].copy(deep=True)
        metrics = history_metrics(history, int(source['iterations']))
        row = dict(source)
        for key in ('best_iteration', 'best_validation_loss', 'best_validation_accuracy',
                    'accuracy_degradation_pp', 'eval_interval', 'trace_interval',
                    'diagnostic_evaluation_seconds'):
            row['source_' + key] = source.get(key)
        row.update(metrics)
        row.update(eval_interval=100, trace_interval=None, origin=origin,
            diagnostic_evaluation_seconds=0.0, primary_runtime_seconds=source['runtime_seconds'],
            remaining_parameters=int(source['remaining_weights'] + source['parameters'] - source['total_weights']))
        row['high_sparsity_concern'] = bool(stage >= 9 and metrics['budget_concern'])
        rows.append(row)
        if 'selection_eligible' in history:
            history['source_selection_eligible'] = history.selection_eligible
        history['selection_eligible'] = True  # all 100-grid observations now eligible
        for key in ('architecture', 'seed', 'stage', 'origin', 'remaining_percent', 'remaining_parameters'):
            history[key] = row[key]
        traces.append(history)
    results = pd.DataFrame(rows)
    dense = results.loc[results.stage == 0, 'best_validation_accuracy']
    if len(dense) != 1:
        raise ValueError('Exactly one dense baseline is required')
    results['accuracy_degradation_pp'] = 100 * (float(dense.iloc[0]) - results.best_validation_accuracy)
    return results, pd.concat(traces, ignore_index=True)


def source_c_run():
    location = json.loads((PREVIOUS / 'pilot_location.json').read_text(encoding='utf-8'))
    saved = Path(location['run_directory'])
    if saved.is_dir():
        return saved
    # Moving the project between macOS/Windows/cloud must not require editing
    # the preserved Stage 2.5 location manifest.
    basename = Path(str(location['run_directory']).replace('\\', '/')).name
    local = PREVIOUS / 'pilot' / basename
    if local.is_dir():
        return local
    raise FileNotFoundError('Saved C pilot is unavailable; restore its Stage 2.5 files')


def import_c():
    """Read/verify C, recompute all 13 stages, save new artifacts only in Stage 2.6."""
    REPORT.mkdir(parents=True, exist_ok=True)
    run = source_c_run()
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    validate_config(config, 'C', reused_c=True)
    last, committed, _ = load_completed(run, config)
    if last != 12:
        raise ValueError('C pilot must contain 13 committed stages')
    records = pd.DataFrame(committed)
    histories = {stage: pd.read_csv(run / f'history_stage{stage}.csv') for stage in range(13)}
    results, history = recompute_results(records, histories, 'reused_Stage2.5_fine_trace')
    target = REPORT / 'C_fine_grid'
    target.mkdir(parents=True, exist_ok=True)
    results.to_csv(target / 'results.csv', index=False)
    for stage, frame in history.groupby('stage'):
        frame.to_csv(target / f'history_stage{stage}.csv', index=False)
    provenance = dict(source_run_directory=str(run.resolve()), training_rerun=False,
        source_eval_interval=500, source_trace_interval=100, new_primary_eval_interval=100,
        split_sha256=config['split_sha256'], dataset_sizes=config['dataset_sizes'],
        source_sha256={name: sha256(run / name) for name in
                      ['config.json'] + [f'history_stage{stage}.csv' for stage in range(13)]},
        runtime_basis='Original measured wall time includes every-100 evaluations; not newly timed')
    (target / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return results, history


def collect_comparison():
    """Assemble 39 complete stages, with the same effective validation grid."""
    c_results, c_history = import_c()
    c_config = json.loads((source_c_run() / 'config.json').read_text())
    frames, histories, provenance = [c_results], [c_history], {'C': c_config}
    for architecture in ('A', 'B'):
        location = json.loads((REPORT / f'{architecture}_location.json').read_text())
        run = Path(location['run_directory'])
        config = json.loads((run / 'config.json').read_text())
        validate_config(config, architecture)
        for key in ('split_sha256', 'versions', 'source_sha256'):
            if config[key] != c_config[key]:
                raise ValueError(f'{architecture}/C comparability mismatch: {key}')
        last, committed, _ = load_completed(run, config)
        if last != 12:
            raise ValueError(f'{architecture}: incomplete pilot')
        recorded = pd.DataFrame(committed)
        source_histories = {stage: pd.read_csv(run / f'history_stage{stage}.csv') for stage in range(13)}
        normalized, history = recompute_results(recorded, source_histories, 'trained_Stage2.6_primary100')
        for key in ('best_validation_loss', 'best_validation_accuracy', 'best_iteration'):
            if not np.allclose(recorded[key], normalized[key], rtol=1e-10, atol=1e-12):
                raise ValueError(f'{architecture}: recorded/derived primary metrics differ: {key}')
        normalized['source_run_directory'] = str(run.resolve())
        frames.append(normalized)
        histories.append(history)
        provenance[architecture] = config
    c_results['source_run_directory'] = str(source_c_run().resolve())
    results = pd.concat(frames, ignore_index=True).sort_values(['architecture', 'stage']).reset_index(drop=True)
    history = pd.concat(histories, ignore_index=True).sort_values(['architecture', 'stage', 'iteration']).reset_index(drop=True)
    if len(results) != 39 or len(history) != 3900:
        raise ValueError('Expected 39 complete stages and 3900 validation observations')
    results.to_csv(REPORT / 'results.csv', index=False)
    history.to_csv(REPORT / 'validation_history.csv', index=False)
    results.to_csv(REPORT / 'stage_diagnostics.csv', index=False)
    summary, runtimes = [], []
    for architecture, group in results.groupby('architecture'):
        dense = group[group.stage == 0].iloc[0]
        final = group[group.stage == 12].iloc[0]
        summary.append(dict(architecture=architecture, seed=0, parameters=int(dense.parameters),
            dense_accuracy=dense.best_validation_accuracy, dense_best_iteration=int(dense.best_iteration),
            final_sparse_accuracy=final.best_validation_accuracy, final_sparse_best_iteration=int(final.best_iteration),
            final_remaining_weights=int(final.remaining_weights), final_remaining_parameters=int(final.remaining_parameters),
            final_remaining_percent=final.remaining_percent, final_accuracy_degradation_pp=final.accuracy_degradation_pp,
            minimum_best_iteration=int(group.best_iteration.min()), maximum_best_iteration=int(group.best_iteration.max()),
            late_minima=int(group.best_in_last_20pct.sum()), final_update_minima=int(group.best_at_budget_end.sum()),
            late_decreasing_stages=int(group.late_loss_decreasing.sum()),
            budget_concerns=int(group.budget_concern.sum()), high_sparsity_concerns=int(group.high_sparsity_concern.sum()),
            training_runtime_minutes=group.runtime_seconds.sum() / 60))
        runtimes.append(dict(architecture=architecture, device='cpu', stages=13, iterations_per_stage=10000,
            recorded_training_seconds=group.runtime_seconds.sum(), recorded_evaluation_seconds=group.evaluation_seconds.sum(),
            recorded_training_minutes=group.runtime_seconds.sum() / 60,
            newly_executed_training_seconds=0.0 if architecture == 'C' else group.runtime_seconds.sum(),
            provenance='Historical Stage 2.5 every-100 trajectory' if architecture == 'C' else 'New Stage 2.6 pilot'))
    summary, runtimes = pd.DataFrame(summary), pd.DataFrame(runtimes)
    summary.to_csv(REPORT / 'summary.csv', index=False)
    runtimes.to_csv(REPORT / 'runtime.csv', index=False)
    projections = pd.DataFrame([dict(architectures=3, seeds=3, stages_per_seed=13, updates_per_stage=10000,
        validation_interval=100, stage_trainings=117, total_updates=1170000,
        estimated_cpu_hours=3 * runtimes.recorded_training_seconds.sum() / 3600,
        planning_cpu_hours_plus_25_percent=3 * runtimes.recorded_training_seconds.sum() * 1.25 / 3600,
        basis='Three times the sum of the three measured seed-0 pilot runtimes; controls excluded')])
    projections['scenario'] = 'complete_three_seed_study_including_existing_seed0'
    remaining = projections.iloc[0].to_dict()
    remaining.update(seeds=2, stage_trainings=78, total_updates=780000,
        estimated_cpu_hours=2 * runtimes.recorded_training_seconds.sum() / 3600,
        planning_cpu_hours_plus_25_percent=2 * runtimes.recorded_training_seconds.sum() * 1.25 / 3600,
        scenario='new_work_if_protocol_unchanged_and_seed0_reused',
        basis='Seeds 1 and 2 only; reuse all validated seed-0 pilots if the final protocol stays unchanged')
    projections = pd.concat([projections, pd.DataFrame([remaining])], ignore_index=True)
    projections.to_csv(REPORT / 'cost_estimate.csv', index=False)
    (REPORT / 'comparison_config.json').write_text(json.dumps(dict(
        effective_eval_interval=100, common_budget=10000, seed=0, split_seed=42,
        split_sha256=c_config['split_sha256'], dataset_sizes=c_config['dataset_sizes'],
        versions=c_config['versions'], source_sha256=c_config['source_sha256'],
        pruning_rates={'hidden': .2, 'output': .1}, controls=False, test_evaluated=False,
        C_training_reused=True, A_B_training_new=True,
        screening_heuristics=dict(late_window_start=8000, high_sparsity_stage_start=9,
                                 decreasing_end_loss_relative_to_minimum_threshold=1.02)), indent=2))
    return results, history, summary, runtimes


def _save_figure(fig, name):
    directory = REPORT / 'figures'
    directory.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in ('pdf', 'png'):
            fig.savefig(directory / f'{name}.{suffix}', dpi=250)
    finally:
        plt.close(fig)


def save_figures(results, history):
    colors = {'A': 'C0', 'B': 'C1', 'C': 'C2'}
    choices = [('best_validation_accuracy', 'remaining_parameters', 100,
                'Validation accuracy at minimum loss (%)', 'Remaining parameters (including biases)',
                'accuracy_vs_remaining_parameters'),
               ('best_iteration', 'remaining_parameters', 1,
                'Iteration of minimum validation loss', 'Remaining parameters (including biases)',
                'best_iteration_vs_remaining_parameters'),
               ('accuracy_degradation_pp', 'sparsity', 1,
                'Accuracy degradation from own dense baseline (pp)', 'Weight sparsity (fraction)',
                'accuracy_degradation_vs_sparsity'),
               ('runtime_seconds', 'stage', 1,
                'Training + validation runtime (seconds)', 'Pruning stage', 'runtime_by_stage')]
    for metric, x, scale, ylabel, xlabel, name in choices:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
        for architecture, group in results.groupby('architecture'):
            group = group.sort_values(x)
            label = architecture + (' (historical timing)' if name == 'runtime_by_stage' and architecture == 'C' else '')
            ax.plot(group[x], scale * group[metric], marker='o', color=colors[architecture], label=label)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if 'remaining_parameters' in x:
            ax.set_xscale('log')
        ax.grid(alpha=.2)
        ax.legend(title='Architecture (seed 0)')
        _save_figure(fig, name)
    for architecture, group in history.groupby('architecture'):
        for metric, ylabel, name in [('validation_loss', 'Validation cross entropy', 'validation_loss'),
                                     ('validation_accuracy', 'Validation accuracy (fraction)', 'validation_accuracy')]:
            fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
            for stage, curve in group.groupby('stage'):
                ax.plot(curve.iteration, curve[metric], label=f'{stage}: {curve.remaining_percent.iloc[0]:.1f}%')
            ax.set(xlabel='Training updates', ylabel=ylabel, title=f'{architecture}: all stages, primary validation every 100')
            ax.grid(alpha=.2)
            ax.legend(title='Stage: weights remaining', ncol=2, fontsize=7)
            _save_figure(fig, f'{name}_{architecture}')
    for metric, ylabel, name in [('validation_loss', 'Validation cross entropy', 'high_sparsity_loss_curves'),
                                 ('validation_accuracy', 'Validation accuracy (fraction)', 'high_sparsity_accuracy_curves')]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
        for ax, architecture in zip(axes, ('A', 'B', 'C')):
            group = history[(history.architecture == architecture) & (history.stage >= 9)]
            for stage, curve in group.groupby('stage'):
                ax.plot(curve.iteration, curve[metric], label=f'{stage}: {curve.remaining_percent.iloc[0]:.1f}%')
            ax.axvline(8000, linestyle=':', color='gray')
            ax.set(title=architecture, xlabel='Training updates', ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(title='Stage: weights remaining', fontsize=7)
        _save_figure(fig, name)


def verify_actual_checkpoints():
    """Verify every real stage and read-only source preservation after both pilots."""
    import torch
    from unittest.mock import patch
    from models import MLP
    from pruning import make_masks, prune_masks
    torch.set_num_threads(1)
    checked = {}
    for architecture in ('A', 'B', 'C'):
        run = source_c_run() if architecture == 'C' else Path(json.loads(
            (REPORT / f'{architecture}_location.json').read_text())['run_directory'])
        config = json.loads((run / 'config.json').read_text())
        validate_config(config, architecture, reused_c=architecture == 'C')
        last, rows, _ = load_completed(run, config)
        assert last == 12 and len(rows) == 13
        before = {path.name: sha256(path) for path in run.glob('*.pt')}
        if architecture != 'C':
            with patch('experiment.train_stage', side_effect=AssertionError('Completed pilots must not retrain')):
                _, resumed = resume_experiment(run, progress=False, save_plots=False)
            assert len(resumed) == 13
        load = lambda name: torch.load(run / name, map_location='cpu', weights_only=True)
        initial = load('initial_weights.pt')
        previous = None
        for stage in range(13):
            start, final = load(f'stage{stage}_start.pt'), load(f'stage{stage}.pt')
            model = MLP(*ARCHITECTURES[architecture])
            if previous is None:
                masks = make_masks(model)
            else:
                model.load_state_dict(previous['state_dict'])
                masks = prune_masks(model, previous['masks'])
            for name, mask in masks.items():
                assert torch.equal(mask, start['masks'][name])
                assert torch.equal(mask, final['masks'][name])
                assert torch.equal(start['state_dict'][name + '.weight'][mask], initial[name + '.weight'][mask])
                assert torch.equal(start['state_dict'][name + '.bias'], initial[name + '.bias'])
                assert not start['state_dict'][name + '.weight'][~mask].count_nonzero()
                assert not final['state_dict'][name + '.weight'][~mask].count_nonzero()
            previous = final
        assert before == {path.name: sha256(path) for path in run.glob('*.pt')}
        checked[architecture] = dict(stages_verified=13, final_weight_pruning=True,
            original_weight_restoration=True, masks_enforced=True, checkpoints_unchanged=True,
            complete_resume_skips_training=True if architecture != 'C' else 'C not resumed or trained')
    checked['preservation'] = verify_previous_reports()
    (REPORT / 'checkpoint_verification.json').write_text(json.dumps(checked, indent=2))
    return checked


def write_assessment(results, history, summary, runtimes):
    """Report observations, budget decision and any precisely scoped follow-up."""
    concerns = results[results.budget_concern]
    high = results[results.high_sparsity_concern]
    if concerns.empty:
        recommendation = ('**Recommend 10,000 fixed updates per stage for all A/B/C in Stage 3, '
            'with primary validation every 100 updates, subject to your review.** '
            'All three seed-0 pilots covered their observed validation-loss minima before the final '
            '20% of the budget, and no stage met the predefined late-training concern screen. '
            'There is no pilot evidence requiring a 15k/20k increase. This is supporting evidence '
            'from one seed per architecture, not a guarantee of convergence for every future seed.')
        diagnostics = ('No targeted longer-training diagnostic is currently indicated by the screen. '
                       'If later seeds show late minima, investigate saved masks before changing the common budget.')
    else:
        labels = ', '.join(f"{row.architecture} stage {row.stage}" for row in concerns.itertuples())
        recommendation = ('**Do not finalize the 10,000-update Stage 3 budget yet.** '
            f'The late-training screen flagged {labels}. Run targeted longer-training diagnostics '
            'on these saved masks, then decide on a single common fixed budget for all architectures. '
            'Do not independently increase one architecture\'s Stage 3 budget.')
        diagnostics = ('For flagged masks, train from the saved original initialization using the exact '
            'same mask, seed and fresh Adam, extending to 15,000 and/or 20,000 updates with every-100 '
            'validation. Keep the mask fixed, do not prune again, and do not replace subsequent IMP '
            'masks. This diagnoses that ticket; a complete longer-budget IMP chain would train each '
            'parent longer and could generate different masks. These diagnostics are proposed only, '
            'not executed by Stage 2.6.')
    lines = ['| Model | Dense accuracy | Stage 12 accuracy | Degradation (pp) | Best-iteration range | Late minima | Runtime (min) |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in summary.itertuples():
        lines.append(f'| {row.architecture} | {100*row.dense_accuracy:.2f}% | {100*row.final_sparse_accuracy:.2f}% | '
                     f'{row.final_accuracy_degradation_pp:+.2f} | {row.minimum_best_iteration:,}–{row.maximum_best_iteration:,} | '
                     f'{row.late_minima}/13 | {row.training_runtime_minutes:.2f} |')
    detail = ['| Model | Stage | Weights remaining (%) | Minimum iteration | Loss at minimum | Accuracy at minimum | '
              'Late loss slope / 1k | Late mean change | Budget flag |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for row in results.itertuples():
        detail.append(f'| {row.architecture} | {row.stage} | {row.remaining_percent:.2f} | {row.best_iteration:,} | '
                      f'{row.best_validation_loss:.5f} | {100*row.best_validation_accuracy:.2f}% | '
                      f'{row.late_loss_slope_per_1000:+.5f} | {row.late_mean_loss_change:+.5f} | {row.budget_concern} |')
    costs = pd.read_csv(REPORT / 'cost_estimate.csv').iloc[0]
    remaining_cost = pd.read_csv(REPORT / 'cost_estimate.csv').iloc[1]
    preservation = json.loads((REPORT / 'preservation_check.json').read_text())
    tests = (REPORT / 'tests.txt').read_text().strip().splitlines()[-1]
    c_dense = results[(results.architecture == 'C') & (results.stage == 0)].iloc[0]
    c_final = results[(results.architecture == 'C') & (results.stage == 12)].iloc[0]
    text = f'''# Stage 2.6 assessment

{recommendation}

**Stage 3 has not been started.** A and B were executed here with seed 0, 13 stages each and 10,000 fixed updates per stage. C was reused from all 1,300 recorded Stage 2.5 fine-grid observations and not retrained. All 39 stages use primary selection every **100 updates**. No pruning, optimizer, initialization or architecture-specific budget changes were made.

## Comparable results

{chr(10).join(lines)}

Accuracy is selected at the iteration of **minimum validation loss**, not maximum accuracy. Positive degradation means worse than that architecture's own fine-grid dense baseline. The final column is measured training+validation time, excluding loading/initialization/checkpoint/plot overhead; C's time is the original historical measurement including every-100 evaluations. Seed-0 curves have no between-seed error bars and do not establish statistical significance or a superior width ratio.

C's corrected dense accuracy is **{100*c_dense.best_validation_accuracy:.2f}% at update {c_dense.best_iteration:,}**; stage 12 is **{100*c_final.best_validation_accuracy:.2f}%**, giving **{c_final.accuracy_degradation_pp:+.2f} pp** degradation. The original 500-grid selections remain in source provenance columns and all Stage 2.5 files are unchanged. No C training time was newly incurred here.

## Budget and late-training evidence

- Loss minima in updates 8,000–10,000: **{int(results.best_in_last_20pct.sum())}/39**.
- Minima exactly at update 10,000: **{int(results.best_at_budget_end.sum())}/39**.
- Late decreasing trends (both negative slope and negative mean change): **{int(results.late_loss_decreasing.sum())}/39**.
- Budget-concern flags: **{len(concerns)}/39**, including **{len(high)}** in highly pruned stages 9–12.

The screen is deliberately descriptive: flag a minimum in the last 20%, or a decreasing trend over updates 8,000–10,000 whose final loss is within 2% of the best observed loss. The slope is an ordinary linear fit (loss units per 1,000 updates). Mean change compares updates 9,100–10,000 with 8,000–8,900. These thresholds are heuristics, not formal convergence tests. A late decline well above an earlier global minimum can reflect fluctuation/overfitting recovery and is not by itself evidence that the budget is insufficient. Full per-stage losses and accuracies are in `validation_history.csv`; all derived measures and remaining parameter counts are in `results.csv` and `stage_diagnostics.csv`.

{diagnostics}

## Runtime

- Newly executed A+B training/evaluation: **{runtimes.newly_executed_training_seconds.sum()/60:.2f} minutes**.
- Total recorded seed-0 A+B+C training/evaluation (including reused C): **{runtimes.recorded_training_seconds.sum()/60:.2f} minutes**.
- Three architectures × three seeds × 13 stages × 10,000 updates: **117 stage trainings / 1,170,000 updates**. Projected serial CPU time: **{costs.estimated_cpu_hours:.2f} hours**, or **{costs.planning_cpu_hours_plus_25_percent:.2f} hours** with a 25% planning allowance. Optional random-reinitialization controls are not included and would add work.

If the reviewed Stage 3 protocol remains exactly the same, all three validated seed-0 pilots can be reused. Only seeds 1 and 2 would then remain: **78 stage trainings / 780,000 new updates**, estimated **{remaining_cost.estimated_cpu_hours:.2f} CPU hours**, or **{remaining_cost.planning_cpu_hours_plus_25_percent:.2f} hours** with allowance. This is conditional planning, not a launch or a claim that a new budget could reuse these masks.

The complete-study projection is three times the sum of the three measured seed-0 runtimes, rather than an assumption that parameter count predicts speed. Host load, seed-dependent variation and overhead add uncertainty; C was measured in the earlier stage. The allowance is not a statistical confidence interval. Runs use one CPU thread, exactly equivalent preloaded MNIST tensors, and dense matrix operations despite masking. GPU timings are unavailable and are not inferred. Native macOS/Windows/CUDA/MPS were not tested in this Linux CPU session.

## Verification and preservation

Final tests: **{tests}**. Every real stage checkpoint was checked against pruning its parent's final weights, original surviving initialization/biases and persistent zero masks. Completed A/B pilots were resumed with training forbidden to confirm no stages rerun. All **{preservation['files_verified']} Stage 2.5 files** passed the before/after checksum comparison. The six core training/pruning/data/checkpoint source hashes, software versions and fixed split hash agree between A/B and reused C. The held-out test set was not evaluated.

All results, live histories, masks/checkpoints, verification evidence and PDF/PNG figures are saved under `reports/stage2_6/`. `stage2_6.ipynb` executes the A/B pilots (or resumes their commits) and performs this same comparison. The original source C files are referenced through new provenance metadata, not modified.

## Methodological limits

The pilot budgets are uniformly 10,000 rather than G.1's 50,000 updates. Retrospective evaluation frequency is restored to G.1's 100. The new widths are the intended extension at approximately equal initial total parameter counts; biases stay unpruned. Because output weights are pruned more slowly and output-layer widths differ, the same pruning round leaves slightly different total remaining counts across architectures. Compare curves on actual remaining parameter counts; the stage-12 summary is not an exactly parameter-matched comparison. A fixed new split, zero biases, [0,1] pixels and floor rounding remain the documented choices. Scientific budget adequacy is judged from validation behavior, not a training-loss stopping rule. Earlier minima do not justify ending a stage early: final fixed-budget weights still determine the next mask. One model seed cannot rule out later minima in other seeds or guarantee the paper's lottery-ticket conclusions.

## Per-stage observations

{chr(10).join(detail)}
'''
    (REPORT / 'assessment.md').write_text(text, encoding='utf-8')
    return text

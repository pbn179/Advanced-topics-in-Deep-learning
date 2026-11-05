"""Read-only study assembly, seed variability, common-budget comparisons and controls planning."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from checkpoints import load_completed, sha256
from models import MLP, ARCHITECTURES, parameter_count
from pruning import make_masks, prune_masks
from stage2_6_analysis import recompute_results
from stage3_runner import ROOT, REPORT, CORE_FILES, seed0_runs, validate_run_config, preserve_previous

COMMON_BUDGETS = (16000, 20000, 30000, 50000, 100000, 150000)
EXPECTED_COUNTS = {'A': 200013, 'B': 200054, 'C': 199979}
METRICS = ('best_validation_accuracy', 'best_validation_loss', 'best_iteration',
           'accuracy_change_pp', 'accuracy_degradation_pp', 'training_loss',
           'final_validation_loss', 'final_validation_accuracy', 'runtime_seconds')


def study_runs():
    runs = {(architecture, 0): run for architecture, run in seed0_runs().items()}
    for architecture in ('A', 'B', 'C'):
        for seed in (1, 2):
            location = REPORT / 'runs' / f'{architecture}_seed{seed}' / 'location.json'
            if not location.exists():
                raise ValueError(f'Missing completed experiment: {architecture} seed {seed}')
            runs[architecture, seed] = Path(json.loads(location.read_text())['run_directory'])
    return runs


def validate_complete_results(results):
    keys = ['architecture', 'seed', 'stage']
    expected = set(itertools.product(('A', 'B', 'C'), (0, 1, 2), range(13)))
    if len(results) != 117 or results.duplicated(keys).any() or set(map(tuple, results[keys].to_numpy())) != expected:
        raise ValueError('Expected exactly 117 unique architecture/seed/stage observations')
    if not results.iterations.eq(10000).all() or not results.eval_interval.eq(100).all():
        raise ValueError('Training/validation budget differs from the locked protocol')
    for architecture, group in results.groupby('architecture'):
        if not group.parameters.eq(EXPECTED_COUNTS[architecture]).all():
            raise ValueError('Incorrect physical parameter count')
        h1, h2 = ARCHITECTURES[architecture]
        counts = [784 * h1, h1 * h2, h2 * 10]
        biases = h1 + h2 + 10
        for stage in range(13):
            if stage:
                counts = [count - int(np.floor(rate * count)) for count, rate in zip(counts, (.2, .2, .1))]
            selected = group[group.stage == stage]
            if not selected.remaining_weights.eq(sum(counts)).all():
                raise ValueError('Incorrect layer-wise survivor count')
            if not selected.remaining_parameters.eq(sum(counts) + biases).all():
                raise ValueError('Incorrect active parameter accounting')
    for (_, _), group in results.groupby(['architecture', 'seed']):
        dense = group[group.stage == 0].iloc[0].best_validation_accuracy
        if not np.allclose(group.accuracy_change_pp, 100 * (group.best_validation_accuracy - dense), atol=1e-10):
            raise ValueError('Accuracy change is not relative to the same seed\'s dense baseline')
    if not np.isfinite(results[list(METRICS)].to_numpy()).all():
        raise ValueError('Invalid experimental metrics')
    if not results.best_validation_accuracy.between(0, 1).all():
        raise ValueError('Invalid accuracy')
    return True


def aggregate_results(results):
    """Sample SD (ddof=1), not standard error or a confidence interval, for n=3."""
    grouped = results.groupby(['architecture', 'stage'], sort=True)
    if not grouped.seed.nunique().eq(3).all() or not grouped.size().eq(3).all():
        raise ValueError('Every aggregate must contain exactly three distinct seeds')
    for field in ('remaining_parameters', 'remaining_weights', 'parameters'):
        if not grouped[field].nunique().eq(1).all():
            raise ValueError('Remaining counts differ between seeds within an architecture/stage')
    summary = grouped[list(METRICS)].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    for field in ('remaining_parameters', 'remaining_weights', 'remaining_percent', 'sparsity', 'parameters'):
        summary[field] = grouped[field].first()
    summary['seed_count'] = grouped.seed.nunique()
    summary['late_minima_count'] = grouped.best_in_last_20pct.sum()
    summary['budget_concerns_count'] = grouped.budget_concern.sum()
    return summary.reset_index()


def common_budget_comparisons(results, budgets=COMMON_BUDGETS):
    """Interpolate within measured log-parameter curves; never extrapolate or call it new data."""
    groups = list(results.groupby(['architecture', 'seed']))
    lower = max(group.remaining_parameters.min() for _, group in groups)
    upper = min(group.remaining_parameters.max() for _, group in groups)
    if any(not lower <= budget <= upper for budget in budgets):
        raise ValueError('Common-budget interpolation would extrapolate beyond observed support')
    rows = []
    for (architecture, seed), group in groups:
        curve = group.sort_values('remaining_parameters')
        x = np.log(curve.remaining_parameters.to_numpy())
        for budget in budgets:
            rows.append(dict(architecture=architecture, seed=seed, parameter_budget=budget,
                validation_accuracy=np.interp(np.log(budget), x, curve.best_validation_accuracy),
                accuracy_change_pp=np.interp(np.log(budget), x, curve.accuracy_change_pp),
                best_iteration=np.interp(np.log(budget), x, curve.best_iteration),
                interpolated=True, interpolation='linear in log(active parameter count)',
                lower_observed_count=int(curve[curve.remaining_parameters <= budget].remaining_parameters.max()),
                upper_observed_count=int(curve[curve.remaining_parameters >= budget].remaining_parameters.min())))
    individual = pd.DataFrame(rows)
    grouped = individual.groupby(['architecture', 'parameter_budget'])
    summary = grouped[['validation_accuracy', 'accuracy_change_pp', 'best_iteration']].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary['seed_count'] = grouped.seed.nunique()
    summary['nonnegative_change_seeds'] = grouped.accuracy_change_pp.apply(lambda values: int((values >= -1e-12).sum()))
    return individual, summary.reset_index()


def pairwise_descriptions(individual):
    rows = []
    for budget, group in individual.groupby('parameter_budget'):
        for left, right in itertools.combinations(('A', 'B', 'C'), 2):
            first = group[group.architecture == left].set_index('seed')
            second = group[group.architecture == right].set_index('seed')
            if set(first.index) != {0, 1, 2} or set(second.index) != {0, 1, 2}:
                raise ValueError('Pairwise descriptions need all three seed indices')
            for metric, scale in [('validation_accuracy', 100), ('accuracy_change_pp', 1)]:
                differences = scale * (first[metric].sort_index() - second[metric].sort_index())
                rows.append(dict(parameter_budget=budget, comparison=f'{left}-{right}', metric=metric,
                    seed0_difference_pp=differences.loc[0], seed1_difference_pp=differences.loc[1],
                    seed2_difference_pp=differences.loc[2], mean_difference_pp=differences.mean(),
                    sample_std_difference_pp=differences.std(ddof=1),
                    positive_seeds=int((differences > 1e-12).sum()),
                    negative_seeds=int((differences < -1e-12).sum()),
                    inference='descriptive matched seed-index contrasts, no hypothesis test'))
    return pd.DataFrame(rows)


def collect_study():
    REPORT.mkdir(parents=True, exist_ok=True)
    frames, histories, provenance = [], [], []
    split_hash = None
    approved = pd.read_csv(ROOT / 'reports' / 'stage2_6' / 'results.csv')
    for (architecture, seed), run in sorted(study_runs().items()):
        config = json.loads((run / 'config.json').read_text())
        validate_run_config(config, architecture, seed)
        if config.get('dataset_sizes') != {'train': 55000, 'validation': 5000, 'test': 10000}:
            raise ValueError('Missing dataset readiness evidence')
        if split_hash is None:
            split_hash = config['split_sha256']
        if config['split_sha256'] != split_hash:
            raise ValueError('Dataset splits differ')
        last, committed, _ = load_completed(run, config)
        if last != 12 or len(committed) != 13:
            raise ValueError(f'Incomplete experiment: {architecture}/{seed}')
        source = pd.DataFrame(committed)
        raw = {stage: pd.read_csv(run / f'history_stage{stage}.csv') for stage in range(13)}
        frame, trace = recompute_results(source, raw, 'reused_seed0' if seed == 0 else 'new_Stage3')
        if seed == 0:
            check = approved[approved.architecture == architecture].sort_values('stage')
        else:
            check = source.sort_values('stage')
        for key in ('best_iteration', 'best_validation_loss', 'best_validation_accuracy', 'accuracy_degradation_pp'):
            if not np.allclose(frame[key], check[key], rtol=1e-10, atol=1e-12):
                raise ValueError(f'Inconsistent primary metrics: {architecture}/{seed}/{key}')
        frame['accuracy_change_pp'] = -frame.accuracy_degradation_pp
        frame['source_run_directory'] = str(run.resolve())
        frame['reused_seed0'] = seed == 0
        frames.append(frame)
        histories.append(trace)
        provenance.append(dict(architecture=architecture, seed=seed, run_directory=str(run.resolve()),
            reused_seed0=seed == 0, config_sha256=sha256(run / 'config.json'),
            original_eval_interval=config['eval_interval'], effective_eval_interval=100,
            checkpoints_verified=13, masks_retained_in_source=True))
    results = pd.concat(frames, ignore_index=True).sort_values(['architecture', 'seed', 'stage']).reset_index(drop=True)
    history = pd.concat(histories, ignore_index=True).sort_values(['architecture', 'seed', 'stage', 'iteration']).reset_index(drop=True)
    validate_complete_results(results)
    if len(history) != 11700 or not history.groupby(['architecture', 'seed', 'stage']).size().eq(100).all():
        raise ValueError('Expected 11,700 validation measurements')
    summary = aggregate_results(results)
    common, common_summary = common_budget_comparisons(results)
    pairs = pairwise_descriptions(common)
    results.to_csv(REPORT / 'individual_results.csv', index=False)
    history.to_csv(REPORT / 'validation_history.csv', index=False)
    summary.to_csv(REPORT / 'summary_mean_std.csv', index=False)
    common.to_csv(REPORT / 'common_budget_interpolated_individual.csv', index=False)
    common_summary.to_csv(REPORT / 'common_budget_interpolated_summary.csv', index=False)
    pairs.to_csv(REPORT / 'pairwise_descriptive_differences.csv', index=False)
    results[results.budget_concern].to_csv(REPORT / 'late_training_warnings.csv', index=False)
    pd.DataFrame(provenance).to_csv(REPORT / 'run_provenance.csv', index=False)
    write_runtime_and_control_estimates(results)
    return results, history, summary, common, common_summary, pairs


def write_runtime_and_control_estimates(results):
    runtime = results.groupby(['architecture', 'seed'], as_index=False).agg(
        stages=('stage', 'size'), training_validation_seconds=('runtime_seconds', 'sum'),
        evaluation_seconds=('evaluation_seconds', 'sum'))
    runtime['training_validation_minutes'] = runtime.training_validation_seconds / 60
    runtime['reused_seed0'] = runtime.seed == 0
    runtime.to_csv(REPORT / 'runtime.csv', index=False)
    rows = []
    for seeds, label in [((0,), 'Small initial same-mask control pilot'), ((0, 1, 2), 'Three-seed matched controls')]:
        stages = results[results.seed.isin(seeds) & results.stage.isin((6, 12))]
        seconds = stages.runtime_seconds.sum()
        rows.append(dict(study=label, architectures='A,B,C', seeds=','.join(map(str, seeds)),
            pruning_rounds='6,12', new_control_trainings=len(stages), updates_per_control=10000,
            new_updates=len(stages) * 10000, estimated_serial_cpu_minutes=seconds / 60,
            planning_minutes_plus_25_percent=seconds * 1.25 / 60,
            executed=False, ticket_retraining_required=False,
            basis='Matched ticket wall times; controls use same masks and fixed compute budget; timing approximate'))
    pd.DataFrame(rows).to_csv(REPORT / 'control_cost_estimate.csv', index=False)


def _save(fig, name):
    path = REPORT / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    try:
        for suffix in ('pdf', 'png'):
            fig.savefig(path / f'{name}.{suffix}', dpi=300)
    finally:
        plt.close(fig)


def save_figures(results, history, summary):
    colors = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73'}
    panels = [('best_validation_accuracy', 100, 'Validation accuracy at minimum loss (%)',
               'accuracy_vs_remaining_parameters'),
              ('best_iteration', 1, 'Iteration of minimum validation loss', 'trainability_vs_remaining_parameters'),
              ('accuracy_change_pp', 1, 'Accuracy change from own dense baseline (pp)', 'accuracy_change_vs_remaining_parameters')]
    for metric, scale, label, name in panels:
        fig, ax = plt.subplots(figsize=(7.4, 4.7), layout='constrained')
        for architecture in ('A', 'B', 'C'):
            group = summary[summary.architecture == architecture].sort_values('remaining_parameters')
            for seed in (0, 1, 2):
                individual = results[(results.architecture == architecture) & (results.seed == seed)].sort_values('remaining_parameters')
                ax.plot(individual.remaining_parameters, scale * individual[metric], color=colors[architecture],
                        alpha=.25, linewidth=.8, linestyle=(':', '--', '-.')[seed])
            mean, std = scale * group[metric + '_mean'], scale * group[metric + '_std']
            ax.plot(group.remaining_parameters, mean, color=colors[architecture], marker='o', markersize=3,
                    label=f'{architecture} ({ARCHITECTURES[architecture][0]}–{ARCHITECTURES[architecture][1]})')
            ax.fill_between(group.remaining_parameters, mean - std, mean + std, color=colors[architecture], alpha=.13)
        if metric == 'accuracy_change_pp':
            ax.axhline(0, color='gray', linewidth=.8)
        ax.set(xlabel='Active parameters remaining (including biases)', ylabel=label, xscale='log')
        ax.grid(alpha=.2)
        ax.legend(title='Mean ± sample SD; faint lines: seeds', fontsize=8)
        _save(fig, name)
    fig, ax = plt.subplots(figsize=(7.4, 4.7), layout='constrained')
    for architecture in ('A', 'B', 'C'):
        group = summary[summary.architecture == architecture].sort_values('stage')
        mean, std = group.best_validation_loss_mean, group.best_validation_loss_std
        ax.plot(group.stage, mean, marker='o', color=colors[architecture], label=architecture)
        ax.fill_between(group.stage, mean - std, mean + std, color=colors[architecture], alpha=.13)
    ax.set(xlabel='Pruning round (0 = dense; remaining counts differ)', ylabel='Minimum sampled validation loss')
    ax.grid(alpha=.2)
    ax.legend(title='Mean ± sample SD, n=3')
    _save(fig, 'validation_loss_vs_pruning_round')
    for stage in (6, 12):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for architecture in ('A', 'B', 'C'):
            rows = results[(results.architecture == architecture) & (results.stage == stage)].sort_values('seed')
            axes[0].plot(rows.seed, 100 * rows.best_validation_accuracy, marker='o', label=architecture, color=colors[architecture])
            axes[1].plot(rows.seed, rows.accuracy_change_pp, marker='o', label=architecture, color=colors[architecture])
        axes[0].set_ylabel('Validation accuracy at minimum loss (%)')
        axes[1].set_ylabel('Change from own dense baseline (pp)')
        axes[1].axhline(0, color='gray', linewidth=.8)
        for ax in axes:
            ax.set_xlabel('Model seed')
            ax.set_xticks([0, 1, 2])
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(f'Round {stage}: descriptive seed trajectories; parameter counts are not identical')
        _save(fig, f'individual_seeds_round{stage}')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
    for ax, architecture in zip(axes, ('A', 'B', 'C')):
        for seed, group in history[(history.architecture == architecture) & (history.stage == 12)].groupby('seed'):
            ax.plot(group.iteration, group.validation_loss, label=f'Seed {seed}')
        ax.axvline(8000, color='gray', linestyle=':')
        ax.set(title=architecture, xlabel='Training updates', ylabel='Round-12 validation loss')
        ax.grid(alpha=.2)
        ax.legend()
    _save(fig, 'high_sparsity_validation_loss_by_seed')


def verify_study():
    """Verify all real source checkpoints; never write into a reused run."""
    from unittest.mock import patch
    from data import set_seed
    from experiment import resume_experiment
    torch.set_num_threads(1)
    details = []
    for (architecture, seed), run in sorted(study_runs().items()):
        config = json.loads((run / 'config.json').read_text())
        validate_run_config(config, architecture, seed)
        last, rows, _ = load_completed(run, config)
        if last != 12 or len(rows) != 13:
            raise ValueError('Missing committed stage')
        load = lambda name: torch.load(run / name, map_location='cpu', weights_only=True)
        initial = load('initial_weights.pt')
        set_seed(seed)
        expected_model = MLP(*ARCHITECTURES[architecture])
        if parameter_count(expected_model) != EXPECTED_COUNTS[architecture]:
            raise ValueError('Incorrect PyTorch parameter count')
        if not all(torch.equal(initial[key], value) for key, value in expected_model.state_dict().items()):
            raise ValueError('Saved initialization does not match its recorded model seed')
        before = {path.name: sha256(path) for path in run.glob('*.pt')}
        previous = None
        for stage in range(13):
            start, final = load(f'stage{stage}_start.pt'), load(f'stage{stage}.pt')
            if previous is None:
                masks = make_masks(expected_model)
            else:
                expected_model.load_state_dict(previous['state_dict'])
                masks = prune_masks(expected_model, previous['masks'])
            for name, mask in masks.items():
                if not torch.equal(mask, start['masks'][name]) or not torch.equal(mask, final['masks'][name]):
                    raise ValueError('Pruning mask is inconsistent with final parent weights')
                if not torch.equal(start['state_dict'][name + '.weight'][mask], initial[name + '.weight'][mask]):
                    raise ValueError('Surviving weights were not restored to original initialization')
                if not torch.equal(start['state_dict'][name + '.bias'], initial[name + '.bias']):
                    raise ValueError('Bias reset mismatch')
                if start['state_dict'][name + '.weight'][~mask].count_nonzero() or final['state_dict'][name + '.weight'][~mask].count_nonzero():
                    raise ValueError('Pruned weights regrew')
                if previous is not None and (mask & ~previous['masks'][name]).any():
                    raise ValueError('Old masks were not preserved')
            previous = final
        if seed != 0:
            # A complete resume must be a read-only no-op with respect to model state.
            with patch('experiment.train_stage', side_effect=AssertionError('Completed experiments must not retrain')):
                _, resumed = resume_experiment(run, progress=False, save_plots=False)
            if len(resumed) != 13:
                raise ValueError('Completed resumption returned incomplete results')
        if before != {path.name: sha256(path) for path in run.glob('*.pt')}:
            raise ValueError('Verification modified a checkpoint')
        if list(run.glob('*reinitialized*.pt')):
            raise ValueError('Unexpected random-reinitialization controls')
        details.append(dict(architecture=architecture, seed=seed, stages_verified=13,
            initial_seed_reproduced_exactly=True, final_parent_magnitude_pruning=True,
            survivor_rewind_exact=True, masks_monotone_and_zero=True,
            effective_protocol_locked=True, reused=seed == 0))
    preserved = preserve_previous()
    execution = json.loads((REPORT / 'execution.json').read_text())
    result = dict(status='verified', combinations=9, stages=117, new_stages=78, reused_seed0_stages=39,
        prior_files_unchanged=preserved, test_set_evaluated=False, controls_executed=0,
        core_sha256={name: sha256(ROOT / name) for name in CORE_FILES},
        execution_workers=execution['workers'], execution_wall_seconds=execution['elapsed_seconds'],
        C_seed0_original_primary500_with_diagnostic100=True,
        C_seed0_uses_approved_fine100_metrics=True, source_configs_not_modified=True, details=details)
    (REPORT / 'verification.json').write_text(json.dumps(result, indent=2))
    return result


def write_reproducibility_report(results):
    import platform
    import sys
    import importlib.metadata
    payload = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
        packages={name: importlib.metadata.version(name) for name in
                  ('torch','torchvision','numpy','pandas','matplotlib','PyYAML','filelock')},
        cuda_available=torch.cuda.is_available(), mps_available=torch.backends.mps.is_available(),
        cpu_threads_per_worker=1, worker_isolation='multiprocessing spawn; independent RNG state',
        source_sha256={name: sha256(ROOT / name) for name in CORE_FILES},
        analysis_sha256={name: sha256(ROOT / name) for name in
                         ('stage3_runner.py','stage3_analysis.py','config_stage3.yaml')},
        stages=int(len(results)), validation_records=11700,
        test_set_evaluated=False, random_reinitialization_controls_executed=False)
    (REPORT / 'reproducibility.json').write_text(json.dumps(payload, indent=2))
    text = '''# Reproducibility

The protocol is sealed in `protocol.lock.json` and `config_stage3.yaml`. All nine combinations use the same MNIST 55,000/5,000 split (seed 42), Adam 0.0012, batch size 60, Gaussian Glorot weights/zero biases, 10,000 fixed updates per stage and 100-update evaluation grid. Model widths are A 162–421, B 200–204, C 231–77. Twelve survivor-pruning rounds use 20% hidden-layer and 10% output-layer rates with floor rounding; biases stay dense. Original surviving weights/biases and fresh Adam are restored for each round. Final trained weights, never selected-best weights, determine the next mask. There is no early stopping.

The six new runs use independent spawned processes (up to three workers, one PyTorch CPU thread each). Process isolation prevents shared random-number state. The saved initialization was reproduced bit for bit from each seed. Pixels use the already validated exact tensor cache; no examples are removed. Execution parallelism is recorded separately from the scientific protocol. Wall times include host contention and are not controlled architecture-speed benchmarks.

Seed 0 is reused from its original source checkpoints, masks and 100-step histories. C's original selector was every 500 but diagnostic evaluations were every 100 on the same fixed-budget trajectory. As approved in Stage 2.6, all 100-step observations select the reported C metrics; this does not alter any trained weights or masks. No source files are edited or seed-0 training repeated. See `run_provenance.csv`, `reused_seed0.json` and the complete previous-results checksum snapshot.

`individual_results.csv` contains all 117 stages; `validation_history.csv` contains all 11,700 evaluations. Accuracy is always at the minimum validation loss, rather than maximum sampled accuracy. Changes are computed against the same architecture AND seed's own dense baseline before calculating mean and sample SD (ddof=1) across seeds. Standard deviation bands are not confidence intervals; n=3 covers only initialization/shuffle variation on this fixed dataset split.

Actual active parameter counts include surviving weights and dense biases. Identical pruning rounds are not exactly parameter-matched across architectures. Optional common-budget comparisons interpolate each seed's observed curve linearly in log(active parameter count), strictly within common support (no extrapolation). These interpolated values are descriptive approximations, not separately trained networks. Their brackets are saved explicitly. Pairwise contrasts preserve individual seed indices and have no hypothesis tests or p-values.

Checkpoint resumption uses atomic completed-stage commits. Partial stages rerun from the preceding committed boundary; completed stages do not retrain. Preserve all source directories in `run_provenance.csv` when moving artifacts. Resume requires compatible code, software, device and thread settings; this is not mid-stage Adam-state resumption. Before/after hashes and direct tensor checks verify all stages and preserved prior outputs.

The experiment uses 10,000 updates per stage rather than the original paper's 50,000. Validation monitoring does not feed back into the training/pruning schedule. Late-training warnings are reported but do not modify the protocol. The 10,000-example test set has NOT been evaluated. Same-mask random-reinitialization controls have NOT been executed, so these are original-initialization IMP tickets, not demonstrated advantages over random reinitialization. Control costs and a limited follow-up proposal are provided separately.

Native macOS/Windows and accelerator results were not validated in this Linux CPU execution. Source code uses portable paths, no DataLoader worker subprocesses and no GPU requirement. Cross-platform bitwise numerical equality is not promised.
'''
    (REPORT / 'reproducibility.md').write_text(text, encoding='utf-8')


def describe_study(results, summary, common, common_summary, pairs):
    """Data-derived descriptive conclusions; do not infer significance from n=3."""
    high = common_summary[common_summary.parameter_budget.isin((16000,20000))]
    mean_leaders, individual_leaders = [], []
    for budget in (16000,20000):
        group = high[high.parameter_budget == budget]
        for metric in ('validation_accuracy','accuracy_change_pp','best_iteration'):
            maximum = group[metric + '_mean'].min() if metric == 'best_iteration' else group[metric + '_mean'].max()
            leaders = '/'.join(group.loc[np.isclose(group[metric + '_mean'],maximum,rtol=0,atol=1e-12),'architecture'])
            mean_leaders.append(dict(parameter_budget=budget,metric=metric,descriptive_mean_leader=leaders))
            for seed in (0,1,2):
                seed_group=common[(common.parameter_budget==budget)&(common.seed==seed)]
                maximum=seed_group[metric].min() if metric == 'best_iteration' else seed_group[metric].max()
                leaders='/'.join(seed_group.loc[np.isclose(seed_group[metric],maximum,rtol=0,atol=1e-12),'architecture'])
                individual_leaders.append(dict(parameter_budget=budget,seed=seed,metric=metric,descriptive_leader=leaders))
    pd.DataFrame(mean_leaders).to_csv(REPORT/'descriptive_mean_leaders.csv',index=False)
    pd.DataFrame(individual_leaders).to_csv(REPORT/'descriptive_individual_leaders.csv',index=False)
    def cell(mean, std, scale=1):
        return f'{scale*mean:.2f} ± {scale*std:.2f}'
    stage_table=['| Model | Stage | Active parameters | Accuracy (%), mean ± SD | Change (pp), mean ± SD | Best iteration, mean ± SD |',
                 '|---|---:|---:|---:|---:|---:|']
    for row in summary[summary.stage.isin((0,6,12))].itertuples():
        stage_table.append(f'| {row.architecture} | {row.stage} | {row.remaining_parameters:,} | '
            f'{cell(row.best_validation_accuracy_mean,row.best_validation_accuracy_std,100)} | '
            f'{cell(row.accuracy_change_pp_mean,row.accuracy_change_pp_std)} | '
            f'{cell(row.best_iteration_mean,row.best_iteration_std)} |')
    high_table=['| Budget (interpolated) | Model | Accuracy (%), mean ± SD | Change (pp), mean ± SD | Seeds retaining dense accuracy | Interpolated best iteration, mean ± SD |',
                '|---:|---|---:|---:|---:|---:|']
    for row in high.sort_values(['parameter_budget','architecture']).itertuples():
        high_table.append(f'| {row.parameter_budget:,} | {row.architecture} | '
            f'{cell(row.validation_accuracy_mean,row.validation_accuracy_std,100)} | '
            f'{cell(row.accuracy_change_pp_mean,row.accuracy_change_pp_std)} | '
            f'{row.nonnegative_change_seeds}/3 | {cell(row.best_iteration_mean,row.best_iteration_std)} |')
    interpretation=[]
    for budget in (16000,20000):
        group=high[high.parameter_budget==budget]
        gap=100*(group.validation_accuracy_mean.max()-group.validation_accuracy_mean.min())
        leaders=[record for record in mean_leaders if record['parameter_budget']==budget]
        accuracy_leader=next(record['descriptive_mean_leader'] for record in leaders if record['metric']=='validation_accuracy')
        retention_leader=next(record['descriptive_mean_leader'] for record in leaders if record['metric']=='accuracy_change_pp')
        fastest=next(record['descriptive_mean_leader'] for record in leaders if record['metric']=='best_iteration')
        retention_counts=', '.join(f'{row.architecture}: {row.nonnegative_change_seeds}/3' for row in group.itertuples())
        interpretation.append(f'- At **{budget:,} active parameters** (interpolated), the descriptive mean accuracy leader '
            f'is **{accuracy_leader}**; the mean dense-relative retention leader is **{retention_leader}**. '
            f'The largest mean accuracy separation is **{gap:.3f} pp**, approximately {gap*50:.1f} of 5,000 '
            f'validation classifications. Nonnegative dense-relative changes: {retention_counts}. '
            f'The smallest interpolated mean minimum-loss iteration is for **{fastest}**. '
            'These are descriptive comparisons, not new trained networks or held-out measurements.')
    consistency=[]
    for item in mean_leaders:
        records=[record for record in individual_leaders if record['parameter_budget']==item['parameter_budget'] and record['metric']==item['metric']]
        tally={architecture:sum(architecture in record['descriptive_leader'].split('/') for record in records) for architecture in ('A','B','C')}
        consistency.append(f"- {item['parameter_budget']:,}, {item['metric']}: individual-seed leaders/ties: "+', '.join(f'{key} {count}/3' for key,count in tally.items())+'.')
    warnings=results[results.budget_concern]
    execution=json.loads((REPORT/'execution.json').read_text())
    verify=json.loads((REPORT/'verification.json').read_text())
    controls=pd.read_csv(REPORT/'control_cost_estimate.csv')
    runtimes=pd.read_csv(REPORT/'runtime.csv')
    tests=(REPORT/'tests.txt').read_text().strip().splitlines()[-1]
    limits=[]
    if warnings.empty:
        limits.append('No stage triggered the prespecified late-budget concern screen. This supports adequacy of the '
                      '10,000-update window for the observed minima, not proof of universal convergence.')
    else:
        names=', '.join(f'{row.architecture}/seed {row.seed}/round {row.stage}' for row in warnings.itertuples())
        limits.append(f'Late-budget warnings: **{names}**. All runs still used the locked 10,000 updates. '
            'Do not reinterpret these stages as fully converged; warnings can affect trainability comparisons. '
            'Consider separately approved fixed-mask 15k/20k diagnostics, not architecture-specific changes to this completed study.')
    twenty=high[high.parameter_budget==20000].set_index('architecture')
    sixteen=high[high.parameter_budget==16000].set_index('architecture')
    dense=summary[summary.stage==0].set_index('architecture')
    retention_leaders={item['descriptive_mean_leader'] for item in mean_leaders if item['metric']=='accuracy_change_pp'}
    if len(retention_leaders)==1 and '/' not in next(iter(retention_leaders)):
        leader=next(iter(retention_leaders))
        retention_text=(f'**Dense-accuracy retention:** {leader} was the most consistent descriptive retention '
            f'candidate at the two shared high-sparsity budgets. Its mean changes were '
            f'{sixteen.loc[leader,"accuracy_change_pp_mean"]:+.3f} ± {sixteen.loc[leader,"accuracy_change_pp_std"]:.3f} pp '
            f'at 16k and {twenty.loc[leader,"accuracy_change_pp_mean"]:+.3f} ± {twenty.loc[leader,"accuracy_change_pp_std"]:.3f} pp '
            f'at 20k; it retained/exceeded its own dense accuracy in '
            f'{sixteen.loc[leader,"nonnegative_change_seeds"]}/3 and {twenty.loc[leader,"nonnegative_change_seeds"]}/3 seeds, respectively. '
            'This is a descriptive result for these seeds, not statistically established superiority.')
    else:
        retention_text='**Dense-accuracy retention:** no single architecture led the mean dense-relative change at both shared high-sparsity budgets.'
    maximum_gap=100*(twenty.validation_accuracy_mean.max()-twenty.validation_accuracy_mean.min())
    leader_counts={name:sum(name in item['descriptive_leader'].split('/') for item in individual_leaders
        if item['parameter_budget']==20000 and item['metric']=='validation_accuracy') for name in ('A','B','C')}
    seed_sd_min=100*twenty.validation_accuracy_std.min()
    seed_sd_max=100*twenty.validation_accuracy_std.max()
    spread_description=('smaller than each architecture\'s measured seed SD' if maximum_gap<seed_sd_min
                        else 'to be assessed alongside the measured seed SDs')
    ab16=pairs[(pairs.parameter_budget==16000)&(pairs.comparison=='A-B')&
               (pairs.metric=='validation_accuracy')].iloc[0]
    absolute_text=('**Absolute sparse accuracy:** at 20k active parameters the means were '+
        ', '.join(f'{name} {100*twenty.loc[name,"validation_accuracy_mean"]:.3f} ± {100*twenty.loc[name,"validation_accuracy_std"]:.3f}%' for name in ('A','B','C'))+
        f'. The largest separation was {maximum_gap:.3f} pp (about {maximum_gap*50:.1f} validation classifications), '
        f'{spread_description} ({seed_sd_min:.3f}–{seed_sd_max:.3f} pp). Individual-seed accuracy leaders/ties at 20k were '+
        ', '.join(f'{name} {leader_counts[name]}/3' for name in ('A','B','C'))+
        f'. At 16k, the A-minus-B accuracy contrast was {ab16.mean_difference_pp:+.3f} ± '
        f'{ab16.sample_std_difference_pp:.3f} pp; it was positive in {ab16.positive_seeds}/3 and '
        f'negative in {ab16.negative_seeds}/3 seed indices. Thus the descriptive ordering depends on '
        'parameter budget and seed. Dense means were '+
        ', '.join(f'{name} {100*dense.loc[name,"best_validation_accuracy_mean"]:.2f}%' for name in ('A','B','C'))+
        "; B's weaker dense baseline is important when interpreting its larger pruning gains.")
    at_twenty=common[common.parameter_budget==20000].pivot(index='seed',columns='architecture',values='best_iteration')
    c_later_a=int((at_twenty['C']>at_twenty['A']+1e-9).sum())
    c_later_b=int((at_twenty['C']>at_twenty['B']+1e-9).sum())
    ab_ties=int(np.isclose(at_twenty['A'],at_twenty['B'],rtol=0,atol=1e-9).sum())
    a_earlier=int((at_twenty['A']<at_twenty['B']-1e-9).sum())
    speed_text=('**Trainability:** at the same 20k budget, interpolated minimum-loss iterations were '+
        ', '.join(f'{name} {twenty.loc[name,"best_iteration_mean"]:.0f} ± {twenty.loc[name,"best_iteration_std"]:.0f}' for name in ('A','B','C'))+
        f'. C reached its minimum later than A in {c_later_a}/3 seeds and later than B in {c_later_b}/3 seeds. '
        f'A and B tied in {ab_ties}/3 seeds and A was earlier in {a_earlier}/3 at this budget. This is a more consistent '
        'descriptive learning-pattern difference than the absolute-accuracy ordering. It suggests the 3:1 '
        'reference reached its best sampled loss later in these pruned runs; it is not a population-level '
        'or wall-clock speed claim. All models still trained for 10,000 updates. Common-count numbers are '
        'interpolated approximations; observed actual-count curves remain primary.')
    findings=retention_text+'\n\n'+absolute_text+'\n\n'+speed_text+'\n\n'+(
        '**Implication:** the data favor examining B for robust dense-relative accuracy preservation and '
        'A/B for earlier minimum-loss iteration under this protocol, but do not justify declaring one '
        'hidden-layer ratio universally superior. The small seed count, shared validation set, reduced '
        'training budget, one late dense-B warning, interpolation and missing reinitialization controls '
        'limit stronger conclusions. In particular, the experiment has not established an advantage '
        'over random reinitialization or held-out test performance.')
    (REPORT/'findings.md').write_text('# Concise findings\n\n'+findings,encoding='utf-8')
    text=f'''# Stage 3 — hidden-layer ratio experiment

The main experiment is complete: **9 architecture–seed combinations, 117 training stages, 11,700 validation observations**. Exactly **78 new stages** (seeds 1 and 2) were executed; all **39 seed-0 stages were reused**, including their original masks/checkpoints. The held-out test set and random-reinitialization controls were not evaluated/run. Final tests: **{tests}**.

## Protocol

MNIST 55,000 train / 5,000 validation / 10,000 held-out test; split seed 42; A 162–421, B 200–204, C 231–77; seeds 0/1/2; Adam 0.0012; batch 60; Gaussian Glorot; 10,000 fixed updates; validation every 100; 12 layer-wise pruning rounds (20% hidden, 10% output survivors), original-initialization rewind and fresh Adam. No protocol was changed in response to accuracy or warnings. **10,000 replaces the original paper's 50,000 updates**, so this is a reduced-budget extension, not an exact replication.

## Observed accuracy and trainability

{chr(10).join(stage_table)}

These are observed points, with **mean ± sample SD across three seeds**, not confidence intervals. Changes are first computed relative to each model/seed's own dense accuracy. Equal rounds leave different active parameter counts, so this table does not establish a parameter-matched winner. Full per-seed values and all rounds are in `individual_results.csv`; all 39 mean/SD rows are in `summary_mean_std.csv`. Faint figure lines show individual seed trajectories.

## Accuracy at common high-sparsity parameter budgets

{chr(10).join(high_table)}

Each seed is interpolated linearly in log(active parameters), within its observed support only; these are descriptive estimates, not additional trained networks or exact-mask matches. Saved interpolation brackets document the approximation. Actual-count curves and individual observations remain the primary evidence.

{chr(10).join(interpretation)}

Consistency of those descriptive leaders across individual seed indices:

{chr(10).join(consistency)}

## Answer to the research question

{findings}

Observed accuracy and minimum-loss iteration vary among the hidden-layer ratios in this experiment, but the scope of that statement is descriptive: the tables/curves quantify the differences under this fixed split, optimizer and pruning schedule. Absolute accuracy leadership and preservation relative to the dense parent are different questions; a model with a weaker dense baseline can show a larger improvement after pruning without having the highest final accuracy. Use the retention means, worst individual changes and 0–3 seed-retention counts together, rather than selecting a winner from a single curve.

The consistency counts above show where ordering repeats and where it changes between seeds. There is **no statistically established superior ratio** from this study: n=3 provides limited precision, repeated rounds are correlated observations, all seeds share the same validation examples, and many comparisons are examined. No hypothesis tests or population-level significance claims are made. A descriptive leader is not proof of superiority. Practical relevance should be judged against the reported pp gaps, seed SDs and individual trajectories; the small-example-count translation above is a scale reference, not an uncertainty estimate. Remaining parameters are genuinely approximately matched initially (A 200,013; B 200,054; C 199,979), but output/bias allocations and slower output pruning make later counts differ.

Minimum-validation-loss iteration is a retrospective trainability proxy measured on a 100-step grid, not a wall-clock speed measurement. All executed stages still cost the full 10,000 updates, regardless of when the minimum was observed. Its mean, SD and individual variation show the ratio-dependent learning patterns. Dense masked matrix kernels do not become faster simply because fewer weights are active, and parallel host contention prevents treating timing differences here as controlled architecture-speed effects. Controls are still needed before attributing IMP-ticket behavior to the original initialization rather than the mask alone.

## Late-training warnings

- Minima in updates 8,000–10,000: **{int(results.best_in_last_20pct.sum())}/117**; new seeds: **{int(results[results.seed!=0].best_in_last_20pct.sum())}/78**.
- Minima exactly at update 10,000: **{int(results.best_at_budget_end.sum())}/117**.
- Budget-concern screen: **{len(warnings)}/117**. The screen flags a minimum in the last 20%, or a jointly decreasing late slope/mean whose final loss is within 2% of the global minimum. These are screening heuristics, not formal convergence tests.

{' '.join(limits)}

Detailed flags and late slopes are in `late_training_warnings.csv` and `individual_results.csv`. Earlier minima never terminated training: final fixed-budget weights determined every next mask.

## Runtime and verification

The six missing runs used three isolated spawned processes, each with one PyTorch CPU thread. Actual orchestration wall time: **{execution['elapsed_seconds']/60:.2f} minutes**. Summed new-run training/evaluation wall times: **{runtimes.loc[~runtimes.reused_seed0,'training_validation_minutes'].sum():.2f} minutes**; these overlap under parallel execution and are not CPU-seconds. Reused seed-0 timing is historical. Only CPU timing was measured; no GPU estimate is claimed.

All 117 real checkpoints were checked for final-parent magnitude pruning, correct active counts and initial seeds, exact survivor/bias reset, old-mask persistence and zero masked entries. Complete new-run resumption was tested with training forbidden. Configurations, code/software hashes and split identity agree; C/seed0 uses its approved every-100 fine observations without editing its original 500-primary/100-diagnostic metadata. **{verify['prior_files_unchanged']} prior report/result/notebook files** passed preservation checks. See `verification.json`, `run_provenance.csv`, `reproducibility.md/json`, `protocol.lock.json` and the checksum snapshot.

## Proposed small same-mask control study — not executed

Recommend starting with **A/B/C, seed 0, rounds 6 and 12**: **6 new reinitialization trainings / 60,000 updates**. Reuse the existing ticket measurements and saved masks; do not retrain tickets or run an entire IMP chain again. Each control should use fresh Gaussian Glorot weights, zero biases, the exact same fixed mask and batch order, fresh Adam, 10,000 updates and every-100 validation. Use the recorded derived seed convention `(seed + 1000003 + stage*1009) % 2**32`; controls never feed into pruning.

Estimated serial CPU cost for those six controls: **{controls.iloc[0].estimated_serial_cpu_minutes:.2f} minutes**, or **{controls.iloc[0].planning_minutes_plus_25_percent:.2f} minutes** with a 25% allowance. Extending the same two rounds to all three seeds would be **18 trainings / 180,000 updates**, approximately **{controls.iloc[1].estimated_serial_cpu_minutes:.2f} minutes** ({controls.iloc[1].planning_minutes_plus_25_percent:.2f} with allowance). These estimates use matched ticket wall times; contention, new initialization and overhead add uncertainty. They are estimates, not executions or confidence intervals. No random-reinitialization controls have been launched.
'''
    (REPORT/'discussion.md').write_text(text,encoding='utf-8')
    return text

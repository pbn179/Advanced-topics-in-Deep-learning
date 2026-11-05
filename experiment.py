"""Reproducible pilot runs and paired initialization controls (no implicit full sweep)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

from checkpoints import atomic_torch_save, commit_stage, load_completed, sha256, RESUME_FIELDS
from data import make_loaders, set_seed
from models import MLP, ARCHITECTURES, parameter_count
from pruning import apply_masks, make_masks, prune_masks, rewind_weights, weight_statistics
from train import choose_device, train_stage
from plot_results import plot_results


def _write_json(value, path):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def _write_csv(frame, path):
    temporary = Path(str(path) + '.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _validate(architecture, seed, rounds, iterations, eval_interval, learning_rate, split_seed):
    if architecture not in ARCHITECTURES:
        raise ValueError(f'Unknown architecture: {architecture}')
    for name, value, minimum in [('seed', seed, 0), ('split_seed', split_seed, 0),
                                  ('rounds', rounds, 0), ('iterations', iterations, 1),
                                  ('eval_interval', eval_interval, 1)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if seed >= 2**32 or split_seed >= 2**32:
        raise ValueError('Seeds must be below 2**32')
    if not 0 < learning_rate < float('inf'):
        raise ValueError('learning_rate must be finite and positive')


def run_experiment(architecture='C', seed=0, rounds=1, iterations=10000,
                   eval_interval=500, device='cpu', output_dir='results', data_dir='datasets',
                   control=False, split_seed=42, learning_rate=0.0012, progress=True,
                   save_plots=True, download=True, preload_tensors=False, num_threads=1,
                   trace_interval=None, resume_dir=None):
    """Train the dense parent, then tickets and optional same-mask controls.

    Only the ticket's final weights determine subsequent masks. Controls use
    fresh Gaussian Glorot values, zero biases, the SAME mask and batch order,
    a fresh Adam, and the same training budget. They never feed back into IMP.
    """
    _validate(architecture, seed, rounds, iterations, eval_interval, learning_rate, split_seed)
    actual_device = choose_device(device)
    if not isinstance(num_threads, int) or num_threads < 1:
        raise ValueError('num_threads must be a positive integer')
    if trace_interval is not None and (not isinstance(trace_interval, int) or trace_interval < 1):
        raise ValueError('trace_interval must be a positive integer')
    torch.set_num_threads(num_threads)
    set_seed(seed)
    model = MLP(*ARCHITECTURES[architecture])
    initial = _cpu_state(model)
    masks = make_masks(model)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    run = Path(resume_dir) if resume_dir is not None else Path(tempfile.mkdtemp(prefix=f'{architecture}-seed{seed}-', dir=output_dir))
    if resume_dir is not None and not (run / 'config.json').exists():
        raise ValueError('Resume directory must contain a run configuration')
    config = dict(architecture=architecture, seed=seed, rounds=rounds, iterations=iterations,
                  eval_interval=eval_interval, device=str(actual_device), requested_device=device,
                  split_seed=split_seed, learning_rate=learning_rate, batch_size=60,
                  checkpoint_schema=1, hidden_widths=list(ARCHITECTURES[architecture]),
                  num_threads=num_threads, preload_tensors=preload_tensors, trace_interval=trace_interval,
                  source_sha256={name: sha256(Path(__file__).parent / name) for name in
                                 ('experiment.py', 'train.py', 'pruning.py', 'models.py', 'data.py', 'checkpoints.py')},
                  control=control, data_dir=str(Path(data_dir).resolve()),
                  python=platform.python_version(), platform=platform.platform(),
                  versions={name: importlib.metadata.version(name) for name in
                            ('torch', 'torchvision', 'numpy', 'pandas', 'matplotlib')},
                  control_seed_rule='(seed + 1000003 + stage * 1009) % 2**32',
                  status='running')
    if resume_dir is not None:
        stored = json.loads((run / 'config.json').read_text(encoding='utf-8'))
        for key in RESUME_FIELDS:
            if key != 'split_sha256' and stored.get(key) != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
    else:
        _write_json(config, run / 'config.json')
        atomic_torch_save(initial, run / 'initial_weights.pt')
    rows = []
    started = time.perf_counter()
    resume_validated = resume_dir is None
    try:
        loaders = make_loaders(data_dir, seed=seed, split_seed=split_seed, download=download,
                               preload_tensors=preload_tensors)
        split = {name: loaders[name].dataset.indices for name in ('train', 'validation')}
        if resume_dir is None:
            atomic_torch_save(split, run / 'split.pt')
        config['split_sha256'] = hashlib.sha256(json.dumps(split).encode()).hexdigest()
        config['dataset_sizes'] = {name: len(loader.dataset) for name, loader in loaders.items()}
        last_stage = -1
        if resume_dir is not None:
            if config['split_sha256'] != stored.get('split_sha256'):
                raise ValueError('Resume configuration mismatch: split_sha256')
            last_stage, rows, checkpoint = load_completed(run, config)
            resume_validated = True
            initial = torch.load(run / 'initial_weights.pt', map_location='cpu', weights_only=True)
            if checkpoint is not None:
                model.load_state_dict(checkpoint['state_dict'])
                masks = checkpoint['masks']
            config['resumed_from_stage'] = last_stage
            if last_stage == rounds:
                if stored.get('status') != 'complete':
                    _write_csv(pd.DataFrame(rows), run / 'results.csv')
                    if save_plots:
                        plot_results(run / 'results.csv', run)
                    config.update(status='complete', elapsed_seconds=stored.get('elapsed_seconds'))
                    _write_json(config, run / 'config.json')
                return run, pd.DataFrame(rows)
        _write_json(config, run / 'config.json')
        dense_accuracy = rows[0]['best_validation_accuracy'] if rows else None
        for stage in range(last_stage + 1, rounds + 1):
            if stage:
                masks = prune_masks(model, masks)
                rewind_weights(model, initial, masks)
            conditions = ['dense'] if stage == 0 else ['ticket'] + (['reinitialized'] if control else [])
            for condition in conditions:
                initialization_seed = seed
                if condition == 'reinitialized':
                    initialization_seed = (seed + 1_000_003 + stage * 1009) % 2**32
                    set_seed(initialization_seed)
                    current = MLP(*ARCHITECTURES[architecture]).to(actual_device)
                    apply_masks(current, masks)
                else:
                    current = model
                # Each condition/stage starts with exactly the same data-order RNG.
                set_seed(seed)
                for offset, name in enumerate(('train', 'validation', 'test')):
                    loaders[name].generator.manual_seed(seed + offset)
                suffix = f'stage{stage}' + ('_reinitialized' if condition == 'reinitialized' else '')
                cpu_masks = {name: mask.detach().cpu().clone() for name, mask in masks.items()}
                atomic_torch_save({'state_dict': _cpu_state(current), 'masks': cpu_masks,
                            'initialization_seed': initialization_seed}, run / f'{suffix}_start.pt')
                if progress:
                    tqdm.write(f'{architecture}, seed {seed}, stage {stage}: {condition}')
                result = train_stage(current, loaders['train'], loaders['validation'], masks,
                                     iterations=iterations, learning_rate=learning_rate,
                                     eval_interval=eval_interval, device=actual_device, progress=progress,
                                     trace_interval=trace_interval,
                                     history_callback=lambda history: _write_csv(pd.DataFrame(history), run / f'{suffix}_live_history.csv'))
                history = pd.DataFrame(result.pop('history'))
                history_name = f'history_stage{stage}' + ('_reinitialized' if condition == 'reinitialized' else '')
                _write_csv(history, run / f'{history_name}.csv')
                if condition == 'dense':
                    dense_accuracy = result['best_validation_accuracy']
                row = dict(architecture=architecture, seed=seed, initialization_seed=initialization_seed,
                           split_seed=split_seed, stage=stage, condition=condition,
                           iterations=iterations, eval_interval=eval_interval,
                           parameters=parameter_count(current), **weight_statistics(current, masks), **result)
                row['accuracy_degradation_pp'] = 100 * (dense_accuracy - row['best_validation_accuracy'])
                row['trace_interval'] = trace_interval
                row['primary_runtime_seconds'] = row['runtime_seconds'] - row['diagnostic_evaluation_seconds']
                row['best_at_budget_end'] = row['best_iteration'] == iterations
                atomic_torch_save({'state_dict': _cpu_state(current), 'masks': cpu_masks}, run / f'{suffix}.pt')
                rows.append(row)
                _write_csv(pd.DataFrame(rows), run / 'results.csv')
            commit_stage(run, stage, rows, config)
        if save_plots:
            plot_results(run / 'results.csv', run)
        config.update(status='complete', elapsed_seconds=time.perf_counter() - started)
        _write_json(config, run / 'config.json')
    except Exception as exc:
        if resume_validated:
            config.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            _write_json(config, run / 'config.json')
        raise
    return run, pd.DataFrame(rows)


def resume_experiment(run_dir, *, progress=True, save_plots=True):
    """Resume only new stage-boundary checkpoints, with saved scientific settings."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    if config.get('checkpoint_schema') != 1:
        raise ValueError('Legacy Stage 1/2 runs have no resume commits; they are left unchanged')
    keys = ['architecture', 'seed', 'rounds', 'iterations', 'eval_interval', 'device',
            'control', 'split_seed', 'learning_rate', 'data_dir', 'preload_tensors',
            'num_threads', 'trace_interval']
    return run_experiment(**{key: config[key] for key in keys}, resume_dir=run_dir,
                          progress=progress, save_plots=save_plots, download=False)


def summarize_results(results):
    """Aggregate across independent model seeds, keeping conditions separate."""
    metrics = ['best_validation_accuracy', 'best_validation_loss', 'best_iteration',
               'accuracy_degradation_pp', 'runtime_seconds']
    grouped = results.groupby(['architecture', 'stage', 'condition'], sort=False)
    summary = grouped[metrics].agg(['mean', 'std', 'min', 'max'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary['seed_count'] = grouped['seed'].nunique()
    return summary.reset_index()


def paired_controls(results):
    """Positive accuracy gap favors original initialization; pair within seed/stage."""
    columns = ['architecture', 'seed', 'stage']
    ticket = results[results['condition'] == 'ticket']
    control = results[results['condition'] == 'reinitialized']
    pairs = ticket.merge(control, on=columns, suffixes=('_ticket', '_control'), validate='one_to_one')
    pairs['ticket_minus_control_pp'] = 100 * (
        pairs['best_validation_accuracy_ticket'] - pairs['best_validation_accuracy_control'])
    return pairs[columns + ['remaining_weights_ticket', 'ticket_minus_control_pp',
                           'best_iteration_ticket', 'best_iteration_control']]


def run_suite(architectures=('C',), seeds=(0, 1, 2), **kwargs):
    """Explicit bounded sweep, with incremental per-run and combined persistence."""
    architectures, seeds = list(architectures), list(seeds)
    if not architectures or not seeds or len(set(architectures)) != len(architectures) or len(set(seeds)) != len(seeds):
        raise ValueError('Provide nonempty, distinct architectures and seeds')
    for architecture in architectures:
        for seed in seeds:
            _validate(architecture, seed, kwargs.get('rounds', 1), kwargs.get('iterations', 10000),
                      kwargs.get('eval_interval', 500), kwargs.get('learning_rate', .0012), kwargs.get('split_seed', 42))
    kwargs = dict(kwargs)
    output_dir = Path(kwargs.pop('output_dir', 'results'))
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix='suite-', dir=output_dir))
    kwargs.pop('save_plots', None)
    rounds, control = kwargs.get('rounds', 1), kwargs.get('control', False)
    stages = len(architectures) * len(seeds) * (1 + rounds * (2 if control else 1))
    plan = dict(architectures=architectures, seeds=seeds, settings=kwargs,
                training_stages=stages, total_updates=stages * kwargs.get('iterations', 10000),
                status='running', runs=[])
    _write_json(plan, directory / 'suite.json')
    frames = []
    started = time.perf_counter()
    try:
        for architecture in architectures:
            for seed in seeds:
                run, frame = run_experiment(architecture=architecture, seed=seed,
                                            output_dir=directory, save_plots=False, **kwargs)
                frame['run_directory'] = run.name
                frames.append(frame)
                combined = pd.concat(frames, ignore_index=True)
                _write_csv(combined, directory / 'results.csv')
                _write_csv(summarize_results(combined), directory / 'summary.csv')
                _write_csv(paired_controls(combined), directory / 'paired_controls.csv')
                plan['runs'].append(run.name)
                _write_json(plan, directory / 'suite.json')
        plot_results(directory / 'results.csv', directory)
        plan.update(status='complete', elapsed_seconds=time.perf_counter() - started)
        _write_json(plan, directory / 'suite.json')
    except Exception as exc:
        plan.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        _write_json(plan, directory / 'suite.json')
        raise
    return directory, combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--resume', help='Resume a new-format run without repeating completed stages')
    parser.add_argument('--architecture', choices=ARCHITECTURES)
    parser.add_argument('--architectures', nargs='+', choices=ARCHITECTURES)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--seeds', nargs='+', type=int)
    for name in ('rounds', 'iterations', 'eval-interval'):
        parser.add_argument('--' + name, type=int)
    parser.add_argument('--device')
    parser.add_argument('--control', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--progress', action=argparse.BooleanOptionalAction, default=None)
    args = vars(parser.parse_args())
    if args.get('resume'):
        directory, results = resume_experiment(args['resume'], progress=args['progress'] is not False)
        print(results.to_string(index=False))
        print(f'Saved to {directory.resolve()}')
        return
    args.pop('resume')
    config = yaml.safe_load(Path(args.pop('config')).read_text(encoding='utf-8'))
    config.update({key: value for key, value in args.items() if value is not None})
    if 'architectures' in config or 'seeds' in config:
        if args['architecture'] is not None:
            config['architectures'] = [config.pop('architecture')]
        else:
            config.setdefault('architectures', [config.pop('architecture', 'C')])
        if args['seed'] is not None:
            config['seeds'] = [config.pop('seed')]
        else:
            config.setdefault('seeds', [config.pop('seed', 0)])
        directory, results = run_suite(**config)
    else:
        directory, results = run_experiment(**config)
    print(results.to_string(index=False))
    print(f'Saved to {directory.resolve()}')


if __name__ == '__main__':
    main()

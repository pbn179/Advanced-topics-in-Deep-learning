"""Run only the six missing Stage 3 combinations in isolated, resumable processes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

from filelock import FileLock
import yaml

from checkpoints import load_completed, sha256
from experiment import run_experiment, resume_experiment
from models import ARCHITECTURES
from stage2_6_analysis import source_c_run, history_metrics

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports' / 'stage3'
CORE_FILES = ('experiment.py', 'train.py', 'pruning.py', 'models.py', 'data.py', 'checkpoints.py')
EXPECTED_WIDTHS = {'A': (162, 421), 'B': (200, 204), 'C': (231, 77)}
NEW_RUNS = (('A', 1), ('B', 1), ('C', 1), ('A', 2), ('B', 2), ('C', 2))
SETTINGS = dict(rounds=12, iterations=10000, eval_interval=100, trace_interval=None,
    split_seed=42, learning_rate=.0012, control=False, device='cpu', num_threads=1,
    preload_tensors=True, data_dir=str(ROOT / 'datasets'), download=False,
    progress=False, save_plots=False)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def previous_files():
    paths = [path for path in (ROOT / 'reports').rglob('*')
             if path.is_file() and path.relative_to(ROOT / 'reports').parts[0] != 'stage3']
    if (ROOT / 'results').exists():
        paths += [path for path in (ROOT / 'results').rglob('*') if path.is_file()]
    paths += [ROOT / name for name in ('stage1.ipynb', 'stage2.ipynb', 'stage2_5.ipynb', 'stage2_6.ipynb')]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def preserve_previous():
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / 'previous_results_snapshot.json'
    current = previous_files()
    if path.exists():
        if json.loads(path.read_text()) != current:
            raise ValueError('Previously recorded reports/results changed')
    else:
        write_json(current, path)
    return len(current)


def seed0_runs():
    runs = {architecture: Path(json.loads((ROOT / 'reports' / 'stage2_6' /
            f'{architecture}_location.json').read_text())['run_directory']) for architecture in ('A', 'B')}
    runs['C'] = source_c_run()
    return runs


def validate_run_config(config, architecture, seed):
    if architecture not in EXPECTED_WIDTHS or seed not in (0, 1, 2):
        raise ValueError('Unexpected architecture/seed')
    expected = dict(architecture=architecture, seed=seed, hidden_widths=list(EXPECTED_WIDTHS[architecture]),
        rounds=12, iterations=10000, split_seed=42, batch_size=60, learning_rate=.0012,
        control=False, device='cpu', num_threads=1, preload_tensors=True)
    reused_c = architecture == 'C' and seed == 0
    expected.update(eval_interval=500 if reused_c else 100, trace_interval=100 if reused_c else None)
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{architecture}/{seed}: protocol mismatch: {key}')
    hashes = {name: sha256(ROOT / name) for name in CORE_FILES}
    if config.get('source_sha256') != hashes:
        raise ValueError('Validated core code changed; refusing a mixed-protocol study')
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'numpy', 'pandas', 'matplotlib')}
    if config.get('versions') != versions:
        raise ValueError('Software versions differ from the locked experiment')
    if 'dataset_sizes' in config and config['dataset_sizes'] != {'train': 55000, 'validation': 5000, 'test': 10000}:
        raise ValueError('Dataset sizes changed')


def initialize_study():
    """Verify immutable seed-0 references and save the protocol before training."""
    import pandas as pd
    preserve_previous()
    source = yaml.safe_load((ROOT / 'config_stage3.yaml').read_text())
    if {name: tuple(widths) for name, widths in source['architectures'].items()} != EXPECTED_WIDTHS:
        raise ValueError('Locked widths changed')
    expected_fields = dict(model_seeds=[0, 1, 2], new_model_seeds=[1, 2], batch_size=60,
                           hidden_pruning_rate=.2, output_pruning_rate=.1)
    expected_fields.update({key: SETTINGS[key] for key in ('rounds', 'iterations', 'eval_interval',
        'trace_interval', 'split_seed', 'learning_rate', 'control', 'device', 'num_threads', 'preload_tensors')})
    for key, value in expected_fields.items():
        if source[key] != value:
            raise ValueError(f'Locked protocol changed: {key}')
    if {name: ARCHITECTURES[name] for name in EXPECTED_WIDTHS} != EXPECTED_WIDTHS:
        raise ValueError('Model definitions do not match approved widths')
    lock = dict(protocol=source, core_sha256={name: sha256(ROOT / name) for name in CORE_FILES})
    path = REPORT / 'protocol.lock.json'
    if path.exists() and json.loads(path.read_text()) != lock:
        raise ValueError('Cannot replace an existing scientific protocol lock')
    if not path.exists():
        write_json(lock, path)
    references = {}
    for architecture, run in seed0_runs().items():
        config = json.loads((run / 'config.json').read_text())
        validate_run_config(config, architecture, 0)
        last, rows, _ = load_completed(run, config)
        if last != 12 or len(rows) != 13:
            raise ValueError('Seed 0 is incomplete')
        for stage in range(13):
            history_metrics(pd.read_csv(run / f'history_stage{stage}.csv'))
        references[architecture] = dict(run_directory=str(run.resolve()), seed=0,
            effective_eval_interval=100, checkpoint_commits=13,
            config_sha256=sha256(run / 'config.json'),
            provenance='Stage 2.5 fine trace' if architecture == 'C' else 'Stage 2.6 pilot',
            retrained=False)
    write_json(references, REPORT / 'reused_seed0.json')
    return lock


def run_missing(architecture, seed):
    """Top-level spawn worker: no RNG state is shared with other experiments."""
    if (architecture, seed) not in NEW_RUNS:
        raise ValueError('Only the six missing seed-1/seed-2 combinations may train')
    parent = REPORT / 'runs' / f'{architecture}_seed{seed}'
    parent.mkdir(parents=True, exist_ok=True)
    location = parent / 'location.json'
    candidates = sorted(parent.glob(f'{architecture}-seed{seed}-*'))
    if location.exists():
        run = Path(json.loads(location.read_text())['run_directory'])
    elif len(candidates) == 1:
        run = candidates[0]
    elif candidates:
        raise ValueError('Ambiguous checkpoints; refusing duplicate training')
    else:
        run = None
    started = time.perf_counter()
    if run is not None:
        config = json.loads((run / 'config.json').read_text())
        validate_run_config(config, architecture, seed)
        run, results = resume_experiment(run, progress=False, save_plots=False)
    else:
        run, results = run_experiment(architecture=architecture, seed=seed, output_dir=parent, **SETTINGS)
    validate_run_config(json.loads((run / 'config.json').read_text()), architecture, seed)
    if len(results) != 13:
        raise ValueError('Run did not complete all stages')
    record = dict(architecture=architecture, seed=seed, run_directory=str(run.resolve()),
                  stages=13, elapsed_this_call_seconds=time.perf_counter() - started)
    write_json(record, location)
    return record


def execute(workers=3):
    if workers not in (1, 2, 3):
        raise ValueError('Use 1–3 isolated one-thread workers')
    REPORT.mkdir(parents=True, exist_ok=True)
    with FileLock(str(REPORT / 'runner.lock'), timeout=0):
        initialize_study()
        existing_path = REPORT / 'execution.json'
        if existing_path.exists():
            previous = json.loads(existing_path.read_text())
            if previous.get('status') == 'complete':
                for architecture, seed in NEW_RUNS:
                    location = REPORT / 'runs' / f'{architecture}_seed{seed}' / 'location.json'
                    run = Path(json.loads(location.read_text())['run_directory'])
                    config = json.loads((run / 'config.json').read_text())
                    validate_run_config(config, architecture, seed)
                    last, rows, _ = load_completed(run, config)
                    if last != 12 or len(rows) != 13:
                        raise ValueError('A previously complete study is missing committed stages')
                print('All six new experiments are already complete; no training was repeated.', flush=True)
                return previous
        started = time.perf_counter()
        state = dict(status='running', workers=workers, process_start_method='spawn',
            new_combinations=[list(case) for case in NEW_RUNS], reused_seed0=['A', 'B', 'C'], completed=[])
        write_json(state, REPORT / 'execution.json')
        try:
            context = multiprocessing.get_context('spawn')
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = {pool.submit(run_missing, *case): case for case in NEW_RUNS}
                for future in as_completed(futures):
                    record = future.result()
                    state['completed'].append(record)
                    write_json(state, REPORT / 'execution.json')
                    print(f"Completed {record['architecture']} seed {record['seed']}: 13 stages", flush=True)
            state.update(status='complete', elapsed_seconds=time.perf_counter() - started)
            write_json(state, REPORT / 'execution.json')
            preserve_previous()
            return state
        except BaseException as exc:
            state.update(status='interrupted_or_failed', error=f'{type(exc).__name__}: {exc}',
                         elapsed_seconds=time.perf_counter() - started)
            write_json(state, REPORT / 'execution.json')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    execute(args.workers)

"""Atomic commits for fully completed IMP stages (never resume inside Adam training)."""
import hashlib
import json
from pathlib import Path

import torch


RESUME_FIELDS = (
    'checkpoint_schema', 'architecture', 'hidden_widths', 'seed', 'rounds', 'iterations',
    'eval_interval', 'trace_interval', 'device', 'split_seed', 'learning_rate', 'batch_size',
    'control', 'preload_tensors', 'num_threads', 'versions', 'source_sha256', 'split_sha256',
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def commit_stage(run, stage, rows, config):
    """Marker is written LAST: earlier checkpoints/history alone are not committed."""
    files = ['initial_weights.pt', 'split.pt', f'stage{stage}.pt', f'stage{stage}_start.pt',
             f'history_stage{stage}.csv']
    if stage > 0 and config['control']:
        files += [f'stage{stage}_reinitialized.pt', f'stage{stage}_reinitialized_start.pt',
                  f'history_stage{stage}_reinitialized.csv']
    record = dict(stage=stage, rows=rows, schema=1,
                  fingerprint={key: config[key] for key in RESUME_FIELDS},
                  sha256={name: sha256(run / name) for name in files})
    path = run / f'stage{stage}_complete.json'
    if path.exists():
        raise RuntimeError(f'Cannot overwrite a completed stage: {path}')
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(record, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_completed(run, config):
    """Validate the entire committed prefix; reject corrupt or incompatible data."""
    latest = None
    for stage in range(config['rounds'] + 1):
        marker = run / f'stage{stage}_complete.json'
        if not marker.exists():
            if any((run / f'stage{later}_complete.json').exists()
                   for later in range(stage + 1, config['rounds'] + 1)):
                raise ValueError('Noncontiguous stage commits; refusing ambiguous resumption')
            break
        record = json.loads(marker.read_text(encoding='utf-8'))
        if record['schema'] != 1 or record['stage'] != stage:
            raise ValueError('Unsupported or inconsistent stage checkpoint')
        for key in RESUME_FIELDS:
            if record['fingerprint'][key] != config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        for name, expected in record['sha256'].items():
            if not (run / name).is_file() or sha256(run / name) != expected:
                raise ValueError(f'Corrupt or missing completed-stage artifact: {name}')
        latest = record
    if latest is None:
        return -1, [], None
    checkpoint = torch.load(run / f"stage{latest['stage']}.pt", map_location='cpu', weights_only=True)
    return latest['stage'], latest['rows'], checkpoint

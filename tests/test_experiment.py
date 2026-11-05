"""Offline integration checks of IMP versus its paired initialization control."""
import json

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import experiment
from models import MLP


@pytest.fixture
def small_experiment(monkeypatch):
    generator = torch.Generator().manual_seed(88)
    dataset = TensorDataset(torch.randn(64, 1, 28, 28, generator=generator), torch.arange(64) % 10)

    def loaders(data_dir, seed, **kwargs):
        del data_dir, kwargs
        return {
            name: DataLoader(Subset(dataset, indices), batch_size=8, shuffle=name == 'train',
                             generator=torch.Generator().manual_seed(seed + offset))
            for offset, (name, indices) in enumerate([
                ('train', list(range(48))), ('validation', list(range(48, 56))), ('test', list(range(56, 64)))])
        }
    monkeypatch.setattr(experiment, 'make_loaders', loaders)
    monkeypatch.setitem(experiment.ARCHITECTURES, 'C', (8, 4))


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def test_controls_use_identical_masks_and_different_initial_weights(small_experiment, tmp_path):
    run, rows = experiment.run_experiment(output_dir=tmp_path, rounds=2, iterations=3,
                                        eval_interval=2, control=True, progress=False, save_plots=False)
    assert rows['condition'].tolist() == ['dense', 'ticket', 'reinitialized', 'ticket', 'reinitialized']
    assert rows['best_iteration'].isin([2, 3]).all()
    initial = load(run / 'initial_weights.pt')
    previous_masks = None
    for stage in (1, 2):
        ticket = load(run / f'stage{stage}_start.pt')
        control = load(run / f'stage{stage}_reinitialized_start.pt')
        for name, mask in ticket['masks'].items():
            assert torch.equal(mask, control['masks'][name])
            if previous_masks is not None:
                assert not (mask & ~previous_masks[name]).any()
            assert torch.equal(ticket['state_dict'][name + '.weight'][mask], initial[name + '.weight'][mask])
            assert not torch.equal(control['state_dict'][name + '.weight'][mask], initial[name + '.weight'][mask])
            for checkpoint in (ticket, control, load(run / f'stage{stage}.pt'),
                               load(run / f'stage{stage}_reinitialized.pt')):
                assert torch.count_nonzero(checkpoint['state_dict'][name + '.weight'][~mask]) == 0
            assert torch.count_nonzero(control['state_dict'][name + '.bias']) == 0
        previous_masks = ticket['masks']
    assert len(experiment.paired_controls(rows)) == 2
    metadata = json.loads((run / 'config.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata['dataset_sizes'] == {'train': 48, 'validation': 8, 'test': 8}


def test_control_cannot_change_ticket_trajectory_and_runs_do_not_overwrite(small_experiment, tmp_path):
    paths = []
    for control in (False, True):
        run, _ = experiment.run_experiment(output_dir=tmp_path, rounds=2, iterations=3,
                                           eval_interval=2, control=control, progress=False, save_plots=False)
        paths.append(run)
    assert paths[0] != paths[1]
    for stage in range(3):
        first, second = [load(path / f'stage{stage}.pt') for path in paths]
        assert all(torch.equal(first['state_dict'][key], second['state_dict'][key]) for key in first['state_dict'])
        pd.testing.assert_frame_equal(*[pd.read_csv(path / f'history_stage{stage}.csv') for path in paths])


def test_ticket_and_control_replay_same_batch_order(small_experiment, monkeypatch, tmp_path):
    original = experiment.train_stage
    batches = []

    def observed(model, train_loader, *args, **kwargs):
        # Observe a copy of the sampler sequence without advancing the real RNG.
        state = train_loader.generator.get_state()
        batches.append(list(iter(train_loader.sampler)))
        train_loader.generator.set_state(state)
        return original(model, train_loader, *args, **kwargs)
    monkeypatch.setattr(experiment, 'train_stage', observed)
    experiment.run_experiment(output_dir=tmp_path, rounds=1, iterations=2,
                              control=True, progress=False, save_plots=False)
    assert batches[0] == batches[1] == batches[2]


def test_suite_aggregates_seeds_without_mixing_conditions(small_experiment, tmp_path):
    directory, rows = experiment.run_suite(seeds=[1, 2, 3], output_dir=tmp_path, rounds=1,
                                         iterations=2, eval_interval=1, control=True, progress=False)
    assert len(rows) == 9
    summary = pd.read_csv(directory / 'summary.csv')
    assert len(summary) == 3
    assert summary['seed_count'].tolist() == [3, 3, 3]
    pairs = pd.read_csv(directory / 'paired_controls.csv')
    assert len(pairs) == 3
    for metric in ('accuracy', 'trainability', 'degradation', 'learning_curves'):
        for extension in ('pdf', 'png'):
            assert (directory / f'{metric}.{extension}').stat().st_size > 100
    plan = json.loads((directory / 'suite.json').read_text())
    assert plan['status'] == 'complete' and plan['total_updates'] == 18
    assert rows['run_directory'].nunique() == 3


def test_failed_run_is_marked_and_partial_results_retained(small_experiment, monkeypatch, tmp_path):
    original = experiment.train_stage
    calls = 0
    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(experiment, 'train_stage', fail_second)
    with pytest.raises(RuntimeError, match='simulated failure'):
        experiment.run_experiment(output_dir=tmp_path, iterations=2, progress=False)
    run = next(tmp_path.iterdir())
    assert json.loads((run / 'config.json').read_text())['status'] == 'failed'
    assert pd.read_csv(run / 'results.csv')['condition'].tolist() == ['dense']


@pytest.mark.parametrize('setting', [dict(rounds=-1), dict(iterations=0), dict(eval_interval=0), dict(seed=-1)])
def test_invalid_configuration_rejected_before_creating_run(tmp_path, setting):
    with pytest.raises(ValueError):
        experiment.run_experiment(output_dir=tmp_path, **setting)
    assert not list(tmp_path.iterdir())


def test_cost_projection_counts_baseline_and_optional_controls_correctly():
    from verification import estimate_cost
    evidence = pd.DataFrame([
        dict(architecture='C', stage=0, iterations=1000, runtime_seconds=10),
        dict(architecture='C', stage=1, iterations=1000, runtime_seconds=10),
    ])
    probes = [pd.DataFrame([dict(architecture=architecture, stage=0, iterations=500, runtime_seconds=5)])
              for architecture in ('A', 'B')]
    costs, _ = estimate_cost(evidence, probes, budgets=(50000,))
    assert costs['training_stages'].tolist() == [117, 225, 126, 243]
    assert costs['total_updates'].tolist() == [5850000, 11250000, 6300000, 12150000]
    assert costs.iloc[0]['estimated_hours'] == pytest.approx(5850000 * .01 / 3600)
    assert costs.iloc[0]['planning_hours_plus_25_percent'] == pytest.approx(5850000 * .0125 / 3600)


@pytest.mark.parametrize('control', [False, True])
def test_resume_matches_uninterrupted_and_skips_committed_stages(small_experiment, monkeypatch, tmp_path, control):
    settings = dict(output_dir=tmp_path, rounds=3, iterations=3, eval_interval=2,
                    control=control, progress=False, save_plots=False)
    full, _ = experiment.run_experiment(**settings)
    original = experiment.train_stage
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        # Stage 1's ticket/control are a single completed-stage transaction.
        if calls == (5 if control else 3):
            raise RuntimeError('interrupted')
        return original(*args, **kwargs)
    monkeypatch.setattr(experiment, 'train_stage', interrupt)
    with pytest.raises(RuntimeError, match='interrupted'):
        experiment.run_experiment(**settings)
    interrupted = next(path for path in tmp_path.iterdir() if path != full)
    completed_before = {path.name: path.read_bytes() for path in interrupted.glob('*_complete.json')}
    calls = 0
    def observed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)
    monkeypatch.setattr(experiment, 'train_stage', observed)
    resumed, rows = experiment.resume_experiment(interrupted, progress=False, save_plots=False)
    assert resumed == interrupted
    assert calls == (4 if control else 2)
    assert all((interrupted / name).read_bytes() == value for name, value in completed_before.items())
    for stage in range(4):
        for suffix in ([''] if stage == 0 or not control else ['', '_reinitialized']):
            first, second = [load(path / f'stage{stage}{suffix}.pt') for path in (full, resumed)]
            assert all(torch.equal(first['state_dict'][key], second['state_dict'][key]) for key in first['state_dict'])
            assert all(torch.equal(first['masks'][key], second['masks'][key]) for key in first['masks'])
            pd.testing.assert_frame_equal(*[pd.read_csv(path / f'history_stage{stage}{suffix}.csv') for path in (full, resumed)])
    assert len(rows) == (7 if control else 4)
    # A second resume is a no-op, including all stage training.
    calls = 0
    experiment.resume_experiment(interrupted, progress=False, save_plots=False)
    assert calls == 0


def test_resume_rejects_corrupt_checkpoint_without_changing_metadata(small_experiment, tmp_path):
    run, _ = experiment.run_experiment(output_dir=tmp_path, rounds=0, iterations=2, progress=False, save_plots=False)
    metadata = (run / 'config.json').read_bytes()
    (run / 'stage0.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt'):
        experiment.resume_experiment(run, progress=False)
    assert (run / 'config.json').read_bytes() == metadata


def test_resume_rejects_changed_training_budget(small_experiment, tmp_path):
    run, _ = experiment.run_experiment(output_dir=tmp_path, rounds=0, iterations=2, progress=False, save_plots=False)
    metadata = (run / 'config.json').read_bytes()
    with pytest.raises(ValueError, match='iterations'):
        experiment.run_experiment(resume_dir=run, rounds=0, iterations=3, progress=False, save_plots=False)
    assert (run / 'config.json').read_bytes() == metadata


def test_legacy_results_are_not_modified_by_resume(tmp_path):
    metadata = '{"architecture": "C", "seed": 0}'
    (tmp_path / 'config.json').write_text(metadata)
    with pytest.raises(ValueError, match='Legacy'):
        experiment.resume_experiment(tmp_path)
    assert (tmp_path / 'config.json').read_text() == metadata

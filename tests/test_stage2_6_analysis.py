"""Synthetic analytics tests: no scientific runs or MNIST downloads."""
import json

import numpy as np
import pandas as pd
import pytest

import stage2_6_analysis as analysis
from models import ARCHITECTURES


def history(losses=None, center=400):
    steps = np.arange(100, 1001, 100)
    losses = .2 + ((steps - center) / 1000) ** 2 if losses is None else np.asarray(losses)
    return pd.DataFrame(dict(iteration=steps, training_loss=.1, validation_loss=losses,
                             validation_accuracy=np.linspace(.7, .9, len(steps)),
                             selection_eligible=steps % 500 == 0))


def records(stages=(0, 1)):
    return pd.DataFrame([dict(architecture='C', seed=0, stage=stage, iterations=1000,
        parameters=1000, total_weights=990, remaining_weights=990 - stage * 10,
        remaining_percent=100 - stage, runtime_seconds=2.0,
        best_iteration=500, best_validation_loss=.4, best_validation_accuracy=.95,
        accuracy_degradation_pp=-.1, eval_interval=500, trace_interval=100,
        diagnostic_evaluation_seconds=.5) for stage in stages])


def config():
    return dict(architecture='C', hidden_widths=list(ARCHITECTURES['C']), seed=0, split_seed=42,
        rounds=12, iterations=10000, learning_rate=.0012, batch_size=60, control=False,
        num_threads=1, preload_tensors=True, device='cpu', eval_interval=500, trace_interval=100,
        dataset_sizes={'train': 55000, 'validation': 5000, 'test': 10000}, split_sha256='synthetic-test-only')


def test_fine_grid_selects_loss_not_accuracy_and_recomputes_own_dense_baseline():
    dense, ticket = history(center=400), history(center=700)
    dense.loc[dense.iteration == 400, 'validation_accuracy'] = .80
    ticket.loc[ticket.iteration == 700, 'validation_accuracy'] = .75
    source = records()
    before_records = source.copy(deep=True)
    before_dense = dense.copy(deep=True)
    result, traces = analysis.recompute_results(source, {0: dense, 1: ticket}, 'unit-test')
    assert result.best_iteration.tolist() == [400, 700]
    assert result.best_validation_accuracy.tolist() == [.80, .75]
    assert result.accuracy_degradation_pp.tolist() == pytest.approx([0, 5])
    assert result.eval_interval.eq(100).all()
    assert result.primary_runtime_seconds.tolist() == [2, 2]  # includes all fine-grid evaluation
    assert result.diagnostic_evaluation_seconds.eq(0).all()
    assert result.source_best_validation_accuracy.eq(.95).all()
    assert traces.selection_eligible.all()
    assert not traces.source_selection_eligible.all()
    pd.testing.assert_frame_equal(source, before_records)
    pd.testing.assert_frame_equal(dense, before_dense)


@pytest.mark.parametrize('corruption', ['gap', 'duplicate', 'nan', 'accuracy_range'])
def test_incomplete_or_invalid_fine_grid_is_rejected(corruption):
    frame = history()
    if corruption == 'gap':
        frame = frame.iloc[1:]
    elif corruption == 'duplicate':
        frame.loc[1, 'iteration'] = 100
    elif corruption == 'nan':
        frame.loc[1, 'validation_loss'] = np.nan
    else:
        frame.loc[1, 'validation_accuracy'] = 1.5
    with pytest.raises(ValueError):
        analysis.history_metrics(frame, 1000)


def test_late_global_improvement_is_flagged():
    result = analysis.history_metrics(history(np.linspace(1, .1, 10)), 1000)
    assert result['best_iteration'] == 1000
    assert result['best_at_budget_end'] and result['best_in_last_20pct']
    assert result['late_loss_decreasing'] and result['budget_concern']
    assert result['late_global_min_improvement'] > 0


def test_late_recovery_above_earlier_minimum_is_not_budget_failure():
    frame = history([1, .1, .3, .4, .5, .6, .7, .8, .7, .6])
    result = analysis.history_metrics(frame, 1000)
    assert result['late_loss_decreasing']
    assert not result['final_loss_within_2pct_of_best']
    assert not result['budget_concern']


def test_high_sparsity_flag_requires_sparse_stage():
    frame = history(np.linspace(1, .1, 10))
    result, _ = analysis.recompute_results(records((0, 9)), {0: frame, 9: frame}, 'unit-test')
    assert result.budget_concern.all()
    assert result.high_sparsity_concern.tolist() == [False, True]


def test_c_training_is_forbidden_in_ab_entrypoint(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('C must not be trained')
    monkeypatch.setattr(analysis, 'run_experiment', forbidden)
    with pytest.raises(ValueError, match='only A and B'):
        analysis.run_ab_pilot('C', {})


def test_source_preservation_detects_changes_without_overwriting(monkeypatch, tmp_path):
    previous, report = tmp_path / 'old', tmp_path / 'new'
    previous.mkdir()
    old = previous / 'history.csv'
    old.write_text('original')
    monkeypatch.setattr(analysis, 'PREVIOUS', previous)
    monkeypatch.setattr(analysis, 'REPORT', report)
    assert analysis.preserve_previous_reports() == 1
    assert analysis.verify_previous_reports()['stage2_5_files_unchanged']
    old.write_text('external change')
    with pytest.raises(ValueError, match='changed'):
        analysis.verify_previous_reports()
    assert old.read_text() == 'external change'


def test_import_c_is_read_only_and_never_trains_or_resumes(monkeypatch, tmp_path):
    source, target = tmp_path / 'source', tmp_path / 'new'
    source.mkdir()
    settings = config()
    (source / 'config.json').write_text(json.dumps(settings))
    source_records = records(tuple(range(13)))
    source_records['iterations'] = 10000
    for stage in range(13):
        steps = np.arange(100, 10001, 100)
        frame = pd.DataFrame(dict(iteration=steps, training_loss=.1,
            validation_loss=.1 + ((steps - 4100) / 10000) ** 2,
            validation_accuracy=.98, selection_eligible=steps % 500 == 0))
        frame.to_csv(source / f'history_stage{stage}.csv', index=False)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    monkeypatch.setattr(analysis, 'REPORT', target)
    monkeypatch.setattr(analysis, 'source_c_run', lambda: source)
    monkeypatch.setattr(analysis, 'load_completed', lambda *args: (12, source_records.to_dict('records'), None))
    def forbidden(*args, **kwargs):
        pytest.fail('Importing C must not train or resume a model')
    monkeypatch.setattr(analysis, 'run_experiment', forbidden)
    monkeypatch.setattr(analysis, 'resume_experiment', forbidden)
    results, history_frame = analysis.import_c()
    assert len(results) == 13 and len(history_frame) == 1300
    assert results.best_iteration.eq(4100).all()
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    provenance = json.loads((target / 'C_fine_grid' / 'provenance.json').read_text())
    assert provenance['training_rerun'] is False
    assert provenance['new_primary_eval_interval'] == 100


def test_primary_config_rejects_500_step_ab_results():
    settings = config()
    settings.update(architecture='A', hidden_widths=list(ARCHITECTURES['A']))
    with pytest.raises(ValueError, match='eval_interval'):
        analysis.validate_config(settings, 'A')
    settings.update(eval_interval=100, trace_interval=None)
    analysis.validate_config(settings, 'A')


def test_c_location_survives_project_move_without_editing_source_manifest(monkeypatch, tmp_path):
    previous = tmp_path / 'previous'
    local = previous / 'pilot' / 'C-seed0-portable'
    local.mkdir(parents=True)
    manifest = previous / 'pilot_location.json'
    manifest.write_text(json.dumps({'run_directory': 'Z:\\old-machine\\pilot\\C-seed0-portable'}))
    original = manifest.read_bytes()
    monkeypatch.setattr(analysis, 'PREVIOUS', previous)
    assert analysis.source_c_run() == local
    assert manifest.read_bytes() == original

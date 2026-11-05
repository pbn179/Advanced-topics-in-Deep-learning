"""Offline checks for completion, n=3 summaries, matched budgets and control planning."""
import json

import numpy as np
import pandas as pd
import pytest

import stage3_analysis as analysis
import stage3_runner as runner
from models import ARCHITECTURES


def complete_fixture():
    rows = []
    for architecture in ('A', 'B', 'C'):
        h1, h2 = ARCHITECTURES[architecture]
        for seed, baseline in enumerate((.90, .95, .98)):
            counts = [784*h1, h1*h2, h2*10]
            for stage in range(13):
                if stage:
                    counts = [n-int(np.floor(rate*n)) for n, rate in zip(counts, (.2,.2,.1))]
                accuracy = baseline + stage * .0001
                rows.append(dict(architecture=architecture, seed=seed, stage=stage,
                    iterations=10000, eval_interval=100, parameters=analysis.EXPECTED_COUNTS[architecture],
                    remaining_weights=sum(counts), remaining_parameters=sum(counts)+h1+h2+10,
                    remaining_percent=100*sum(counts)/(784*h1+h1*h2+h2*10),
                    sparsity=1-sum(counts)/(784*h1+h1*h2+h2*10),
                    best_iteration=2000, best_validation_accuracy=accuracy, best_validation_loss=.1,
                    accuracy_change_pp=100*(accuracy-baseline), accuracy_degradation_pp=-100*(accuracy-baseline),
                    training_loss=.2, final_validation_loss=.12, final_validation_accuracy=accuracy,
                    runtime_seconds=60., evaluation_seconds=10., best_in_last_20pct=False, budget_concern=False))
    return pd.DataFrame(rows)


def test_completion_requires_all_117_unique_observations_and_counts():
    frame = complete_fixture()
    assert analysis.validate_complete_results(frame)
    with pytest.raises(ValueError, match='117'):
        analysis.validate_complete_results(frame.iloc[1:])
    with pytest.raises(ValueError, match='117'):
        analysis.validate_complete_results(pd.concat([frame.iloc[:-1], frame.iloc[[0]]]))
    corrupted = frame.copy()
    corrupted.loc[corrupted.index[-1], 'remaining_weights'] += 1
    with pytest.raises(ValueError, match='survivor'):
        analysis.validate_complete_results(corrupted)


def test_changes_must_be_relative_to_the_same_seed_dense_model():
    frame = complete_fixture()
    frame.loc[(frame.seed==1)&(frame.stage==12), 'accuracy_change_pp'] = 5
    with pytest.raises(ValueError, match='same seed'):
        analysis.validate_complete_results(frame)


def test_aggregation_uses_sample_sd_and_within_seed_changes():
    frame = complete_fixture()
    selected = frame[(frame.architecture=='A')&(frame.stage==1)].copy()
    selected['best_validation_accuracy'] = [.90,.92,.94]
    selected['accuracy_change_pp'] = [-1,0,1]
    summary = analysis.aggregate_results(selected).iloc[0]
    assert summary.seed_count == 3
    assert summary.best_validation_accuracy_mean == pytest.approx(.92)
    assert summary.best_validation_accuracy_std == pytest.approx(.02)
    assert summary.accuracy_change_pp_std == pytest.approx(1)
    assert summary.best_validation_accuracy_std != pytest.approx(.02/np.sqrt(3))
    with pytest.raises(ValueError, match='three distinct'):
        analysis.aggregate_results(selected.iloc[:2])


def test_common_budget_interpolation_uses_actual_counts_and_no_extrapolation():
    frame = complete_fixture()
    individual, summary = analysis.common_budget_comparisons(frame, budgets=(16000,20000))
    assert len(individual)==18 and len(summary)==6
    assert individual.interpolated.all()
    assert individual.lower_observed_count.le(individual.parameter_budget).all()
    assert individual.upper_observed_count.ge(individual.parameter_budget).all()
    # Identical rounds have distinct remaining counts between architectures.
    assert frame[frame.stage==12].groupby('architecture').remaining_parameters.first().nunique()==3
    with pytest.raises(ValueError, match='extrapolate'):
        analysis.common_budget_comparisons(frame,budgets=(10000,))


def test_interpolation_preserves_observed_endpoint_values():
    frame=complete_fixture()
    single=frame[(frame.architecture=='A')&(frame.seed==0)]
    endpoint=int(single.iloc[-1].remaining_parameters)
    individual,_=analysis.common_budget_comparisons(single,budgets=(endpoint,))
    assert individual.iloc[0].validation_accuracy==pytest.approx(single.iloc[-1].best_validation_accuracy)
    assert individual.iloc[0].accuracy_change_pp==pytest.approx(single.iloc[-1].accuracy_change_pp)


def test_pairwise_contrasts_keep_individual_seeds_and_ties():
    rows=[]
    for architecture,values in [('A',[.90,.91,.93]),('B',[.91,.90,.93]),('C',[.90,.90,.90])]:
        for seed,value in enumerate(values):
            rows.append(dict(architecture=architecture,seed=seed,parameter_budget=20000,
                             validation_accuracy=value,accuracy_change_pp=100*(value-.90)))
    contrast=analysis.pairwise_descriptions(pd.DataFrame(rows))
    ab=contrast[(contrast.comparison=='A-B')&(contrast.metric=='validation_accuracy')].iloc[0]
    assert [ab.seed0_difference_pp,ab.seed1_difference_pp,ab.seed2_difference_pp]==pytest.approx([-1,1,0])
    assert ab.mean_difference_pp==pytest.approx(0)
    assert ab.sample_std_difference_pp==pytest.approx(1)
    assert ab.positive_seeds==1 and ab.negative_seeds==1


def test_control_study_is_estimated_not_executed(monkeypatch,tmp_path):
    monkeypatch.setattr(analysis,'REPORT',tmp_path)
    analysis.write_runtime_and_control_estimates(complete_fixture())
    controls=pd.read_csv(tmp_path/'control_cost_estimate.csv')
    assert controls.new_control_trainings.tolist()==[6,18]
    assert controls.new_updates.tolist()==[60000,180000]
    assert controls.estimated_serial_cpu_minutes.tolist()==pytest.approx([6,18])
    assert not controls.executed.any()
    assert not controls.ticket_retraining_required.any()


@pytest.mark.parametrize('architecture,seed',[('A',0),('B',0),('C',0),('D',1),('A',3)])
def test_runner_cannot_retrain_seed_zero_or_add_unapproved_experiments(architecture,seed):
    with pytest.raises(ValueError,match='six missing'):
        runner.run_missing(architecture,seed)


def test_protocol_config_rejects_any_changed_training_setting(monkeypatch):
    monkeypatch.setattr(runner,'sha256',lambda path:'unit')
    monkeypatch.setattr(runner.importlib.metadata,'version',lambda name:'unit')
    config=dict(architecture='A',seed=1,hidden_widths=[162,421],rounds=12,iterations=10000,
        split_seed=42,batch_size=60,learning_rate=.0012,control=False,device='cpu',num_threads=1,
        preload_tensors=True,eval_interval=100,trace_interval=None,
        source_sha256={name:'unit' for name in runner.CORE_FILES},
        versions={name:'unit' for name in ('torch','torchvision','numpy','pandas','matplotlib')})
    runner.validate_run_config(config,'A',1)
    for key,value in [('iterations',15000),('eval_interval',500),('learning_rate',.001),('control',True)]:
        bad=dict(config,**{key:value})
        with pytest.raises(ValueError,match='protocol mismatch'):
            runner.validate_run_config(bad,'A',1)
    c0=dict(config,architecture='C',seed=0,hidden_widths=[231,77],eval_interval=500,trace_interval=100)
    runner.validate_run_config(c0,'C',0)
    with pytest.raises(ValueError,match='protocol mismatch'):
        runner.validate_run_config(dict(c0,seed=1),'C',1)


def test_preservation_covers_all_old_reports_results_and_notebooks(monkeypatch,tmp_path):
    (tmp_path/'reports'/'old').mkdir(parents=True)
    (tmp_path/'reports'/'stage3').mkdir()
    (tmp_path/'results').mkdir()
    old=tmp_path/'reports'/'old'/'result.csv'
    old.write_text('old report')
    (tmp_path/'results'/'result.csv').write_text('old result')
    (tmp_path/'reports'/'stage3'/'new.csv').write_text('new data')
    for name in ('stage1.ipynb','stage2.ipynb','stage2_5.ipynb','stage2_6.ipynb'):
        (tmp_path/name).write_text('old notebook')
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner,'REPORT',tmp_path/'reports'/'stage3')
    assert runner.preserve_previous()==6
    (tmp_path/'reports'/'stage3'/'new.csv').write_text('more new data')
    assert runner.preserve_previous()==6
    old.write_text('changed')
    with pytest.raises(ValueError,match='changed'):
        runner.preserve_previous()


def test_completed_controller_is_idempotent_and_never_starts_workers(monkeypatch,tmp_path):
    monkeypatch.setattr(runner,'REPORT',tmp_path)
    monkeypatch.setattr(runner,'initialize_study',lambda:None)
    monkeypatch.setattr(runner,'validate_run_config',lambda *args:None)
    monkeypatch.setattr(runner,'load_completed',lambda *args:(12,[{}]*13,None))
    def forbidden(*args,**kwargs):
        pytest.fail('Completed study must not start more workers')
    monkeypatch.setattr(runner,'ProcessPoolExecutor',forbidden)
    saved=dict(status='complete',elapsed_seconds=1234,completed=[])
    (tmp_path/'execution.json').write_text(json.dumps(saved))
    before=(tmp_path/'execution.json').read_bytes()
    for architecture,seed in runner.NEW_RUNS:
        parent=tmp_path/'runs'/f'{architecture}_seed{seed}'
        run=parent/'run'
        run.mkdir(parents=True)
        (run/'config.json').write_text('{}')
        (parent/'location.json').write_text(json.dumps(dict(run_directory=str(run))))
    assert runner.execute()==saved
    assert (tmp_path/'execution.json').read_bytes()==before

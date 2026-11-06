"""Synthetic schema/aggregation fixtures; no fabricated scientific results."""
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
import nbformat
import torch
from models import ARCHITECTURES
import study
import analysis
from execute_notebook import require_saved


def measurements():
    rows=[]
    for a,seed,stage in study.CASES:
        h1,h2=ARCHITECTURES[a];weights=sum(study.expected_layer_counts(a,stage).values());total=784*h1+h1*h2+h2*10
        parameters=study.COUNTS[a];active=weights+h1+h2+10
        rows.append(dict(architecture=a,model_seed=seed,pruning_stage=stage,hidden1=h1,hidden2=h2,width_ratio=h1/h2,
            initial_parameters=parameters,active_parameters=active,active_weights=weights,total_weights=total,
            remaining_weight_percentage=100*weights/total,weight_sparsity=1-weights/total,
            parameter_sparsity=1-active/parameters,iteration_min_validation_loss=1000+100*stage,
            minimum_validation_loss=.08+.001*stage,validation_accuracy=.97+.001*seed,
            test_accuracy=.98+.001*seed-.00005*stage,test_loss=.08,test_accuracy_change_pp=-.005*stage,
            validation_accuracy_change_pp=0.,runtime_seconds=.1,test_examples=10000))
    return pd.DataFrame(rows)


def test_sample_sd_uses_three_seeds_at_52_actual_stage_positions():
    data=measurements();assert analysis.validate_individual(data)
    table=analysis.aggregate(data)
    assert len(table)==52 and table.seed_count.eq(3).all()
    assert table.test_accuracy_std.to_numpy()==pytest.approx([.001]*52)
    assert table.test_accuracy_change_pp_std.eq(0).all()
    assert len(table[table.pruning_stage==12].remaining_weight_percentage.unique())==4


def test_missing_duplicate_or_invented_stage_rejected():
    data=measurements()
    for bad in (data.iloc[:-1],pd.concat([data,data.iloc[[0]]]),pd.concat([data.iloc[:-1],data.iloc[[0]]])):
        with pytest.raises(ValueError):analysis.aggregate(bad)
    data.loc[0,'remaining_weight_percentage']=99
    with pytest.raises(ValueError):analysis.validate_individual(data)


def test_figure_requires_all_observed_points(tmp_path):
    with pytest.raises(ValueError):analysis.plot_main(pd.DataFrame(index=range(51)),tmp_path)


def test_headless_default_cannot_enable_training():
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('MODE="saved"\nENABLE_TRAINING=False')]);require_saved(n)
    n.cells[0].source='MODE="reproduce"\nENABLE_TRAINING=True'
    with pytest.raises(PermissionError):require_saved(n)


def test_report_writer_handles_complete_four_architecture_trajectories(tmp_path,monkeypatch):
    import json
    data=measurements();data['verified']=True
    data['validation_loss_absolute_difference']=0.;data['validation_accuracy_absolute_difference']=0.
    summary=analysis.aggregate(data)
    run=tmp_path/'synthetic_run';run.mkdir()
    results=tmp_path/'results';results.mkdir();assets=tmp_path/'paper_assets';assets.mkdir()
    (run/'protocol.json').write_text(json.dumps(dict(source_sha256={},versions={})))
    (run/'execution.json').write_text(json.dumps(dict(elapsed_seconds=1.,workers=3)))
    records=[dict(selected_checkpoint=f'synthetic_{a}_{s}_{p}.pt',selected_checkpoint_sha256='0'*64) for a,s,p in study.CASES]
    (run/'selection_manifest.json').write_text(json.dumps(dict(count=156,records=records)))
    history=[dict(architecture=a,model_seed=s,pruning_stage=p,iteration=i,
        validation_loss=.1+abs(5000-i)*1e-6,validation_accuracy=.98) for a,s,p in study.CASES for i in range(100,10001,100)]
    pd.DataFrame(history).to_csv(run/'validation_history.csv',index=False)
    monkeypatch.setattr(analysis,'approved_run_dir',lambda x:Path(x))
    monkeypatch.setattr(analysis,'preserve_previous',lambda:dict(status='synthetic fixture',checked=0))
    analysis.write_reports(data,summary,run,results,assets)
    proof=json.loads((results/'verification.json').read_text())
    assert proof['training_stages']==156 and proof['figure_observed_points_per_panel']==52
    assert proof['new_controls_run']==0
    assert (assets/'architectures.tex').exists()
    assert 'exploratory extension' in (results/'findings.md').read_text()
    trajectories=pd.read_csv(results/'trajectory_summary.csv')
    assert len(trajectories)==4
    assert trajectories.pruned_stages_with_nonnegative_mean_dense_change.eq(0).all()

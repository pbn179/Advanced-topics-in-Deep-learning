"""Release analysis and opt-in reproduction tests; no scientific training."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader,TensorDataset
import final_project as final
import reproduce_final as full
from models import MLP
from pruning import make_masks,apply_masks
from data import set_seed
from train import evaluate


def test_mode_b_requires_explicit_activation_before_any_io(tmp_path,monkeypatch):
    monkeypatch.setattr(full,'ROOT',tmp_path)
    with pytest.raises(PermissionError):full.full_reproduction('reproductions/never_started')
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('path',['reports/stage3','results/replay','/tmp/unsafe','reproductions/../reports','reproductions'])
def test_reproduction_rejects_original_or_ambiguous_output(path):
    with pytest.raises(ValueError):full.approved_output(path,enabled=True)


def test_analysis_cannot_overwrite_experimental_artifacts(tmp_path):
    for path in ('reports/stage3','results/anything','archive/notebooks','datasets','.'):
        with pytest.raises(ValueError):final.output_directory(tmp_path,path)
    assert final.output_directory(tmp_path,'paper_assets')==tmp_path/'paper_assets'


def test_missing_artifacts_fail_with_actionable_paths(tmp_path):
    with pytest.raises(FileNotFoundError,match='reports/stage3/individual_results.csv'):
        final.verify_artifacts(tmp_path)


def test_best_snapshot_and_committed_resume_use_frozen_training(tmp_path,monkeypatch):
    torch.set_num_threads(1);set_seed(3)
    dataset=TensorDataset(torch.randn(30,1,28,28),torch.randint(0,10,(30,)))
    loaders={'train':DataLoader(dataset,batch_size=6,shuffle=True,generator=torch.Generator().manual_seed(3)),
             'validation':DataLoader(dataset,batch_size=15,shuffle=False)}
    model=MLP(5,3);masks=make_masks(model);masks['fc1'][:,::2]=False;apply_masks(model,masks)
    row=full.train_one(model,masks,loaders,tmp_path/'stage',dict(architecture='synthetic',seed=3,stage=0),iterations=4,progress=False)
    best=torch.load(tmp_path/'stage/best_validation.pt',weights_only=True)
    restored=MLP(5,3);restored.load_state_dict(best['state_dict'])
    metric=evaluate(restored,loaders['validation'])
    assert metric['loss']==pytest.approx(row['best_validation_loss'],abs=1e-12)
    assert best['iteration']==row['best_iteration']
    assert not restored.fc1.weight.detach()[~masks['fc1']].count_nonzero()
    monkeypatch.setattr(full,'train_stage',lambda *a,**k:pytest.fail('Committed stage retrained'))
    assert full.train_one(model,masks,loaders,tmp_path/'stage',{},iterations=4,progress=False)==row
    (tmp_path/'stage/final.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='Corrupt'):full.completed(tmp_path/'stage')


def test_frozen_config_is_consistent():
    import yaml
    assert yaml.safe_load((full.ROOT/'config_final.yaml').read_text())==full.CONFIG


def test_headless_executor_refuses_enabled_training():
    import nbformat
    from execute_final_notebook import require_saved_mode
    n=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('MODE="A"\nENABLE_FULL_REPRODUCTION=False')])
    require_saved_mode(n)
    n.cells[0].source='MODE="B"\nENABLE_FULL_REPRODUCTION=True'
    with pytest.raises(PermissionError):require_saved_mode(n)


def test_new_reproduction_export_is_readable_without_original_machine_paths(tmp_path):
    # Synthetic schema fixture only: no real results are manufactured or published.
    rows=[];tested=[]
    for a in ('A','B','C'):
        for seed in (0,1,2):
            for stage in range(13):
                for condition in (['original_initialization','random_reinitialization'] if stage in (6,12) else ['original_initialization']):
                    row=dict(architecture=a,seed=seed,stage=stage,condition=condition,
                        best_validation_accuracy=.97+seed*.001,best_validation_loss=.1+seed*.001,
                        best_iteration=100+seed*100,remaining_parameters=200000-stage*1000,
                        parameters=200000,total_weights=199500,remaining_weights=199500-stage*1000,
                        sparsity=stage*1000/199500,iterations=10000,eval_interval=100)
                    rows.append(row)
                    if stage in (0,6,12):
                        tested.append(dict(**row,test_accuracy=.98+seed*.001,test_loss=.08,test_examples=10000))
                        p=tmp_path/'runs'/f'{a}_seed{seed}'/f'{"control" if condition=="random_reinitialization" else "ticket"}_stage{stage}'/'best_validation.pt'
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic checksum fixture, not model weights')
    full.export_saved_view(tmp_path,rows,tested)
    loaded=final.load_results(tmp_path)
    assert 'remaining_parameters' in loaded['main_summary']
    assert 'remaining_parameters' in loaded['test_summary']
    assert len(loaded['main'])==117
    assert len(loaded['test'])==45
    assert len(loaded['control_pairs'])==18
    assert final.verify_artifacts(tmp_path,check_all=True)['status']=='verified'

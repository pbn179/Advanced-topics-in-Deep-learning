"""Meaningful synthetic checks of the locked four-architecture workflow."""
import itertools
import json
from pathlib import Path
import pytest
import torch
from torch.utils.data import TensorDataset,DataLoader
from models import MLP,ARCHITECTURES,parameter_count
from data import set_seed
from pruning import make_masks,prune_masks,rewind_weights,apply_masks
from train import evaluate,train_stage
import study


def test_four_fixed_parameter_counts_and_budget():
    assert {a:parameter_count(MLP(*w)) for a,w in ARCHITECTURES.items()}==study.COUNTS
    assert (max(study.COUNTS.values())-min(study.COUNTS.values()))/200000<.0007
    assert study.config()==study.EXPECTED


@pytest.mark.parametrize('architecture',list(ARCHITECTURES))
def test_layerwise_survivors_original_restore_and_persistent_masks(architecture):
    set_seed(0);model=MLP(*ARCHITECTURES[architecture]);initial=study.cpu_state(model);masks=make_masks(model)
    for stage in range(13):
        if stage:
            with torch.no_grad():
                for p in model.parameters():p.add_(torch.randn_like(p)*.01)
            old={n:m.clone() for n,m in masks.items()}
            masks=prune_masks(model,masks,.2,.1)
            assert all(torch.all(m<=old[n]) for n,m in masks.items())
            rewind_weights(model,initial,masks)
        study.assert_counts(architecture,stage,masks);study.assert_rewind(model,initial,masks)


def test_best_state_is_actual_minimum_and_final_is_distinct(tmp_path):
    torch.set_num_threads(1);set_seed(4)
    model=MLP(5,3);masks=make_masks(model);masks['fc1'][:,::2]=False;apply_masks(model,masks)
    data=TensorDataset(torch.randn(32,1,28,28),torch.randint(0,10,(32,)))
    train=DataLoader(data,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(4))
    val=DataLoader(data,batch_size=16)
    saver=study.BestSaver(model,masks,tmp_path,dict(architecture='synthetic'))
    snapshots={}
    def callback(history):
        row=history[-1].copy();row['validation_loss']=1 if row['iteration']==2 else 2
        # Controlled callback losses test earlier best capture, independent of final training state.
        snapshots[row['iteration']]=study.cpu_state(model);saver([row])
    train_stage(model,train,val,masks,iterations=4,eval_interval=2,progress=False,history_callback=callback)
    best=torch.load(tmp_path/'best_validation.pt',weights_only=True)
    assert best['iteration']==2
    assert all(torch.equal(v,snapshots[2][n]) for n,v in best['state_dict'].items())
    assert any(not torch.equal(v,model.state_dict()[n]) for n,v in best['state_dict'].items())
    study.assert_masked(model,masks)


def test_short_training_commit_reload_and_skip(tmp_path,monkeypatch):
    torch.set_num_threads(1);set_seed(8);model=MLP(4,3);masks=make_masks(model)
    masks['fc1'][:,::2]=False;apply_masks(model,masks)
    data=TensorDataset(torch.randn(24,1,28,28),torch.randint(0,10,(24,)))
    loaders={'train':DataLoader(data,batch_size=6,shuffle=True,generator=torch.Generator().manual_seed(8)),
             'validation':DataLoader(data,batch_size=12)}
    result=study.train_one(model,masks,loaders,tmp_path/'stage',dict(architecture='synthetic'),iterations=4)
    saved=torch.load(tmp_path/'stage/best_validation.pt',weights_only=True);model.load_state_dict(saved['state_dict'])
    actual=evaluate(model,loaders['validation']);study.check_metrics(actual,result['best_validation_loss'],result['best_validation_accuracy'])
    monkeypatch.setattr(study,'train_stage',lambda *a,**k:pytest.fail('Committed stage retrained'))
    assert study.train_one(model,masks,loaders,tmp_path/'stage',{},iterations=4)==result
    (tmp_path/'stage/final.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError):study.completed(tmp_path/'stage')


def test_test_gate_requires_all_156_verified_states():
    rows=[dict(architecture=a,model_seed=s,pruning_stage=p,verified=True) for a,s,p in study.CASES]
    study.require_complete_gate(rows)
    for invalid in (rows[:-1],rows+[rows[0]],rows[:-1]+[rows[0]]):
        with pytest.raises(ValueError):study.require_complete_gate(invalid)
    rows[0]['verified']=False
    with pytest.raises(ValueError):study.require_complete_gate(rows)


def test_invalid_or_nonfinite_validation_is_rejected():
    for loss,acc in [(float('nan'),.98),(.1,float('inf')),(.2,.98),(.1,.9802)]:
        with pytest.raises(ValueError):study.check_metrics(dict(loss=loss,accuracy=acc),.1,.98)


def test_no_training_without_explicit_activation():
    with pytest.raises(PermissionError):study.execute()
    for p in ('results/main','artifacts/../results','/tmp/experiment','artifacts'):
        with pytest.raises(ValueError):study.approved_run_dir(p)


def test_final_trained_weights_determine_mask_not_selected_state():
    model=MLP(4,3);masks=make_masks(model)
    with torch.no_grad():
        for layer in (model.fc1,model.fc2,model.fc3):layer.weight.copy_(torch.arange(layer.weight.numel()).reshape(layer.weight.shape)+1)
    final_masks=prune_masks(model,masks)
    with torch.no_grad():
        for layer in (model.fc1,model.fc2,model.fc3):layer.weight.copy_(torch.flip(layer.weight,[0,1]))
    selected_masks=prune_masks(model,masks)
    assert any(not torch.equal(m,selected_masks[n]) for n,m in final_masks.items())


def test_incomplete_or_failed_gate_never_accesses_test_loader(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'make_loaders',lambda *a,**k:pytest.fail('Test data accessed before complete validation gate'))
    rows=[dict(architecture=a,model_seed=s,pruning_stage=p,verified=True) for a,s,p in study.CASES]
    with pytest.raises(ValueError):study.evaluate_fixed_test('artifacts/blocked',rows[:-1])
    rows[-1]['verified']=False
    with pytest.raises(ValueError):study.evaluate_fixed_test('artifacts/blocked',rows)

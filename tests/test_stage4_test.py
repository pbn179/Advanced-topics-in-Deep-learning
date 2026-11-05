"""Stage 4B safety gate, numerical matching and deterministic checkpoint recovery."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from data import set_seed
from models import MLP
from pruning import make_masks, apply_masks
from train import train_stage, evaluate
from stage4_test_runner import (TICKETS, CONTROLS, require_gate, verify_metrics,
                               assert_state, source, replay)
from stage4_controls_runner import cpu_state


def gate_records():
    return [dict(architecture=a,seed=s,stage=r,condition=c,verified=True)
            for cases,c in [(TICKETS,'original_initialization'),(CONTROLS,'random_reinitialization')]
            for a,s,r in cases]


def test_all_models_must_pass_before_test_access():
    records=gate_records();require_gate(records)
    for invalid in (records[:-1],records+[records[0]],records[:-1]+[records[0]]):
        with pytest.raises(ValueError): require_gate(invalid)
    records[0]['verified']=False
    with pytest.raises(ValueError): require_gate(records)


def test_validation_tolerance_is_absolute_and_accuracy_cannot_change_count():
    expected=dict(loss=.08,accuracy=.98)
    verify_metrics(dict(loss=.080000001,accuracy=.98),expected)
    with pytest.raises(ValueError): verify_metrics(dict(loss=.080001,accuracy=.98),expected)
    with pytest.raises(ValueError): verify_metrics(dict(loss=.08,accuracy=.9802),expected)


@pytest.mark.parametrize('case',[('D',0,6),('A',3,6),('B',1,1),('C',0,11)])
def test_replay_cannot_launch_unapproved_training(case):
    with pytest.raises(ValueError): source(*case)
    with pytest.raises(ValueError): replay(*case)


def test_mask_regrowth_is_rejected():
    model=MLP(4,3);masks=make_masks(model);masks['fc1'][0,0]=False
    apply_masks(model,masks);assert_state(model,masks)
    with torch.no_grad(): model.fc1.weight[0,0]=1
    with pytest.raises(ValueError): assert_state(model,masks)


def test_saved_start_replay_recovers_selected_weights_without_training_to_end(tmp_path):
    torch.set_num_threads(1);set_seed(7)
    data=TensorDataset(torch.randn(35,1,28,28),torch.randint(0,10,(35,)))
    model=MLP(5,3);masks=make_masks(model);masks['fc1'][:,::3]=False;apply_masks(model,masks)
    start=cpu_state(model);snapshots={}
    def loaders():
        return (DataLoader(data,batch_size=6,shuffle=True,generator=torch.Generator().manual_seed(7)),
                DataLoader(data,batch_size=16,shuffle=False,generator=torch.Generator().manual_seed(8)))
    train,val=loaders();set_seed(7)
    def save(history): snapshots[history[-1]['iteration']]=cpu_state(model)
    result=train_stage(model,train,val,masks,iterations=12,eval_interval=2,progress=False,history_callback=save)
    # Replay a previously recorded intermediate iteration, never choose using held-out data.
    selected=6;recovered=MLP(5,3);recovered.load_state_dict(start);train,val=loaders();set_seed(7)
    train_stage(recovered,train,val,masks,iterations=selected,eval_interval=2,progress=False)
    assert all(torch.equal(v,snapshots[selected][k]) for k,v in recovered.state_dict().items())
    row=result['history'][2];actual=evaluate(recovered,val)
    verify_metrics(actual,dict(loss=row['validation_loss'],accuracy=row['validation_accuracy']))
    assert_state(recovered,masks)


def test_nonfinite_validation_cannot_pass_gate():
    for value in (float('nan'),float('inf')):
        with pytest.raises(ValueError): verify_metrics(dict(loss=value,accuracy=.98),dict(loss=.08,accuracy=.98))


def test_dense_relative_changes_and_paired_sample_sd():
    import pandas as pd
    from stage4_test_analysis import comparisons,aggregate
    records=gate_records()
    for r in records:
        r.update(test_accuracy=.98+r['seed']*.001-r['stage']*.0001,
                 validation_accuracy=.97+r['seed']*.001,test_loss=.08,
                 remaining_parameters=200000 if r['stage']==0 else 15000,
                 remaining_weights=199400 if r['stage']==0 else 14400,
                 mask_sha256=f"{r['architecture']}-{r['seed']}-{r['stage']}",test_examples=10000)
        if r['condition']=='random_reinitialization': r['test_accuracy']-=.001*(r['seed']+1)
    individual,pairs=comparisons(pd.DataFrame(records))
    assert individual[individual.stage==0].test_accuracy_change_pp.eq(0).all()
    assert pairs.ticket_minus_control_test_accuracy_pp.to_numpy()==pytest.approx([.1,.1,.2,.2,.3,.3]*3)
    summary=aggregate(pairs,['architecture','stage'],['ticket_minus_control_test_accuracy_pp'])
    assert summary.ticket_minus_control_test_accuracy_pp_mean.to_numpy()==pytest.approx([.2]*6)
    assert summary.ticket_minus_control_test_accuracy_pp_std.to_numpy()==pytest.approx([.1]*6)
    with pytest.raises(ValueError): comparisons(pd.DataFrame(records[:-1]))

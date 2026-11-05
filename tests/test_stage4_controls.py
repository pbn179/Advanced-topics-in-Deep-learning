"""Synthetic/offline controls tests; no extra MNIST experiment or ticket replay."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader,Subset,TensorDataset

import stage4_controls_runner as controls
from data import set_seed
from models import MLP
from pruning import apply_masks
from train import train_stage,evaluate


def toy_loader(seed=2):
    generator=torch.Generator().manual_seed(55)
    dataset=TensorDataset(torch.randn(73,1,28,28,generator=generator),torch.arange(73)%10)
    subset=Subset(dataset,list(range(10,73)))
    loader=DataLoader(subset,batch_size=8,shuffle=True,num_workers=0,
                     generator=torch.Generator().manual_seed(seed),drop_last=False)
    return loader,DataLoader(subset,batch_size=20)


def fixed_masks(model):
    # Synthetic saved positions, not regenerated magnitude-pruning masks.
    masks={name:torch.ones_like(getattr(model,name).weight,dtype=torch.bool) for name in ('fc1','fc2','fc3')}
    for mask in masks.values():
        mask.flatten()[::4]=False
    return masks


def test_control_initialization_repeats_and_never_regenerates_masks(monkeypatch):
    monkeypatch.setitem(controls.ARCHITECTURES,'C',(8,4))
    model=MLP(8,4)
    masks=fixed_masks(model)
    before={name:mask.clone() for name,mask in masks.items()}
    first,fresh,seed=controls.initialize_control('C',2,12,masks)
    second,repeated,same_seed=controls.initialize_control('C',2,12,masks)
    assert seed==same_seed==(2+1000003+12*1009)%(2**32)
    assert all(torch.equal(fresh[name],repeated[name]) for name in fresh)
    for name,mask in masks.items():
        assert torch.equal(before[name],mask)
        assert torch.equal(getattr(first,name).weight[mask],fresh[name+'.weight'][mask])
        assert not getattr(first,name).weight[~mask].count_nonzero()
        assert not getattr(first,name).bias.count_nonzero()
    _,different,_=controls.initialize_control('C',1,12,masks)
    assert not torch.equal(fresh['fc1.weight'],different['fc1.weight'])


def test_actual_fetch_order_equals_exact_loader_replay_through_epoch_boundaries():
    original,_=toy_loader(2)
    audited,audit=controls.make_audited_loader(original)
    expected=controls.reference_batch_order(original.dataset.indices,2,iterations=23,batch_size=8)
    iterator=iter(audited)
    for _ in range(23):
        try:
            batch=next(iterator)
        except StopIteration:
            iterator=iter(audited)
            batch=next(iterator)
        assert batch[0].shape[0] in (7,8)
    assert audit.hexdigest()==expected['order_sha256']
    assert audit.examples==expected['examples']
    assert hashlib.sha256(audited.generator.get_state().numpy().tobytes()).hexdigest()==expected['final_generator_sha256']
    wrong=controls.reference_batch_order(original.dataset.indices,3,iterations=23,batch_size=8)
    assert wrong['order_sha256']!=audit.hexdigest()


def test_audit_preserves_returned_examples_and_labels():
    loader,_=toy_loader()
    audit=controls.AuditedDataset(loader.dataset)
    for index in (0,1,12,62):
        actual=audit[index]
        expected=loader.dataset[index]
        assert torch.equal(actual[0],expected[0]) and torch.equal(actual[1],expected[1])


def test_best_checkpoint_contains_earlier_selected_weights_not_final_alias(tmp_path):
    model=MLP(8,4)
    masks=fixed_masks(model)
    callback=controls.BestCheckpoint(model,masks,tmp_path)
    history=[]
    for iteration,loss in ((100,.8),(200,.3),(300,.3)):
        with torch.no_grad():
            model.fc1.weight.fill_(iteration)
        apply_masks(model,masks)
        history.append(dict(iteration=iteration,validation_loss=loss,validation_accuracy=.5,training_loss=1))
        callback(history)
    best=torch.load(tmp_path/'best_validation.pt',weights_only=True)
    assert best['iteration']==200 and best['validation_loss']==.3
    assert (best['state_dict']['fc1.weight'][masks['fc1']]==200).all()
    assert (model.fc1.weight[masks['fc1']]==300).all()
    assert callback.mask_checks==3
    assert not best['state_dict']['fc1.weight'][~masks['fc1']].count_nonzero()


def test_short_control_training_saves_reproducible_selected_state_and_fixed_mask(tmp_path):
    set_seed(10)
    model=MLP(8,4)
    masks=fixed_masks(model)
    apply_masks(model,masks)
    training,validation=toy_loader(2)
    callback=controls.BestCheckpoint(model,masks,tmp_path)
    result=train_stage(model,training,validation,masks,iterations=12,eval_interval=3,
                       progress=False,history_callback=callback)
    best=torch.load(tmp_path/'best_validation.pt',weights_only=True)
    assert best['iteration']==result['best_iteration']
    selected=MLP(8,4);selected.load_state_dict(best['state_dict'])
    metrics=evaluate(selected,validation)
    assert metrics['loss']==pytest.approx(result['best_validation_loss'],abs=1e-10)
    assert metrics['accuracy']==pytest.approx(result['best_validation_accuracy'],abs=1e-10)
    for name,mask in masks.items():
        assert torch.equal(mask,best['masks'][name])
        assert not getattr(model,name).weight[~mask].count_nonzero()
    assert callback.mask_checks==4


@pytest.mark.parametrize('case',[('A',0,0),('B',1,5),('C',2,11),('A',3,6),('D',0,12)])
def test_only_18_approved_cases_are_allowed(case):
    with pytest.raises(ValueError,match='18 approved'):
        controls.run_control(*case)


def paired_frames():
    keys=[]
    for seed in (0,1,2):
        keys.append(dict(architecture='A',seed=seed,stage=6,remaining_weights=100,remaining_parameters=110,
                         best_validation_loss=.1,best_iteration=2000))
    tickets=pd.DataFrame(keys);controls_frame=pd.DataFrame(keys)
    tickets['best_validation_accuracy']=[.90,.92,.94]
    controls_frame['best_validation_accuracy']=[.89,.91,.93]
    return tickets,controls_frame


def test_paired_difference_sd_is_not_independent_error_propagation():
    from stage4_controls_analysis import paired_results,aggregate_pairs
    tickets,control=paired_frames()
    pairs=paired_results(tickets,control)
    summary=aggregate_pairs(pairs).iloc[0]
    assert pairs.ticket_minus_control_accuracy_pp.tolist()==pytest.approx([1,1,1])
    assert summary.ticket_minus_control_accuracy_pp_mean==pytest.approx(1)
    assert summary.ticket_minus_control_accuracy_pp_std==pytest.approx(0,abs=1e-12)
    assert tickets.best_validation_accuracy.std()>0 and control.best_validation_accuracy.std()>0
    assert summary.ticket_accuracy_higher_count==3


def test_paired_analysis_reports_control_wins_and_ties_without_assuming_ticket_advantage():
    from stage4_controls_analysis import paired_results,aggregate_pairs
    ticket,control=paired_frames()
    control['best_validation_accuracy']=[.91,.92,.93]
    pairs=paired_results(ticket,control)
    row=aggregate_pairs(pairs).iloc[0]
    assert pairs.ticket_minus_control_accuracy_pp.tolist()==pytest.approx([-1,0,1])
    assert row.ticket_accuracy_higher_count==row.control_accuracy_higher_count==row.accuracy_tied_count==1
    assert row.ticket_minus_control_accuracy_pp_std==pytest.approx(1)


def test_pairing_rejects_missing_controls_duplicate_keys_and_count_mismatch():
    from stage4_controls_analysis import paired_results
    ticket,control=paired_frames()
    with pytest.raises(ValueError,match='Missing'):
        paired_results(ticket,control.iloc[:-1])
    with pytest.raises(ValueError,match='Duplicate'):
        paired_results(ticket,pd.concat([control,control.iloc[[0]]]))
    bad=control.copy();bad.loc[0,'remaining_parameters']=111
    with pytest.raises(ValueError,match='counts differ'):
        paired_results(ticket,bad)

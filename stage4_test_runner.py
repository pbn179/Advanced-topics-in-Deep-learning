"""Recover exactly 27 preselected ticket states, verify all 45 states, then test once.

No pruning or test-based selection occurs. Original scientific files are read-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import math
import json
import multiprocessing
from pathlib import Path
import time
import importlib.metadata

from filelock import FileLock
import pandas as pd
import torch
from tqdm.auto import tqdm
from checkpoints import atomic_torch_save, sha256
from data import make_loaders, set_seed
from models import ARCHITECTURES, MLP, parameter_count
from pruning import apply_masks, weight_statistics
from stage3_analysis import study_runs
from stage3_runner import validate_run_config, CORE_FILES
from stage4_controls_runner import (cpu_state, tensor_digest, write_json, case_key,
    verify_case_files, make_audited_loader, reference_batch_order)
from train import train_stage, evaluate

ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports'/'stage4_test'
TICKETS=tuple(itertools.product(('A','B','C'),(0,1,2),(0,6,12)))
CONTROLS=tuple(itertools.product(('A','B','C'),(0,1,2),(6,12)))
LOSS_ATOL=1e-8
ACCURACY_ATOL=1e-12


def preserve_previous():
    files=[p for p in (ROOT/'reports').rglob('*') if p.is_file()
           and p.relative_to(ROOT/'reports').parts[0]!='stage4_test']
    files += [p for p in (ROOT/'results').rglob('*') if p.is_file()]
    files += [p for p in ROOT.glob('*.ipynb') if p.name!='stage4_test.ipynb']
    current={str(p.relative_to(ROOT)):sha256(p) for p in sorted(files)}
    target=REPORT/'previous_results_snapshot.json'
    if target.exists() and json.loads(target.read_text())!=current:
        raise ValueError('Previous experiment artifacts changed')
    if not target.exists(): write_json(current,target)
    return len(current)


def verify_metrics(actual,expected):
    """Predeclared absolute tolerances; accuracy allows no changed classification count."""
    if not all(math.isfinite(float(v)) for pair in (actual,expected) for v in pair.values()):
        raise ValueError('Nonfinite validation metric')
    loss_delta=abs(float(actual['loss'])-float(expected['loss']))
    accuracy_delta=abs(float(actual['accuracy'])-float(expected['accuracy']))
    if loss_delta>LOSS_ATOL or accuracy_delta>ACCURACY_ATOL:
        raise ValueError(f'Validation mismatch: loss delta={loss_delta}, accuracy delta={accuracy_delta}')
    return dict(loss_absolute_difference=loss_delta,accuracy_absolute_difference=accuracy_delta)


def assert_state(model,masks):
    for name,mask in masks.items():
        if mask.dtype!=torch.bool or not torch.isfinite(getattr(model,name).weight).all():
            raise ValueError('Invalid mask or nonfinite weight')
        if getattr(model,name).weight.detach()[~mask].count_nonzero():
            raise ValueError('Masked weight regrew')
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Nonfinite model state')


def source(architecture,seed,stage):
    if (architecture,seed,stage) not in TICKETS: raise ValueError('Unapproved replay case')
    run=study_runs()[architecture,seed]
    config=json.loads((run/'config.json').read_text());validate_run_config(config,architecture,seed)
    marker=json.loads((run/f'stage{stage}_complete.json').read_text())
    for name,digest in marker['sha256'].items():
        if sha256(run/name)!=digest: raise ValueError('Source committed artifact changed')
    frame=pd.read_csv(ROOT/'reports/stage3/individual_results.csv')
    rows=frame[(frame.architecture==architecture)&(frame.seed==seed)&(frame.stage==stage)]
    if len(rows)!=1: raise ValueError('Missing unique preselected ticket')
    row=rows.iloc[0].to_dict()
    start=torch.load(run/f'stage{stage}_start.pt',weights_only=True,map_location='cpu')
    final=torch.load(run/f'stage{stage}.pt',weights_only=True,map_location='cpu')
    original=torch.load(run/'initial_weights.pt',weights_only=True,map_location='cpu')
    masks=start['masks']
    for name,mask in masks.items():
        if not torch.equal(mask,final['masks'][name]): raise ValueError('Source mask mismatch')
    expected={name:value.clone() for name,value in original.items()}
    for name,mask in masks.items(): expected[name+'.weight'].mul_(mask)
    if any(not torch.equal(expected[k],v) for k,v in start['state_dict'].items()):
        raise ValueError('Saved start does not equal masked original initialization')
    return run,config,row,start


def loaders_for(run,seed):
    loaders=make_loaders(ROOT/'datasets',seed=seed,split_seed=42,download=False,preload_tensors=True)
    split=torch.load(run/'split.pt',weights_only=True,map_location='cpu')
    for key in ('train','validation'):
        if list(loaders[key].dataset.indices)!=split[key]: raise ValueError('Dataset split mismatch')
    return loaders


def initialize():
    REPORT.mkdir(parents=True,exist_ok=True);preserve_previous()
    protocol=dict(ticket_cases=[list(c) for c in TICKETS],control_cases=[list(c) for c in CONTROLS],
        loss_absolute_tolerance=LOSS_ATOL,accuracy_absolute_tolerance=ACCURACY_ATOL,
        selection='Stage 3/4A recorded minimum validation loss, no new selection',
        replay_budget='Only until previously selected iteration, no mask generation',
        training=dict(learning_rate=.0012,batch_size=60,split_seed=42,optimizer='Adam',
                      validation_interval=100,original_budget=10000,device='cpu',num_threads=1),
        core_sha256={f:sha256(ROOT/f) for f in CORE_FILES},runner_sha256=sha256(Path(__file__)),
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pandas','matplotlib')})
    path=REPORT/'protocol.lock.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Replay protocol changed')
    if not path.exists(): write_json(protocol,path)
    for case in TICKETS: source(*case)
    return protocol


def replay(architecture,seed,stage):
    if (architecture,seed,stage) not in TICKETS: raise ValueError('Unapproved replay')
    torch.set_num_threads(1)
    lock=json.loads((REPORT/'protocol.lock.json').read_text())
    if sha256(Path(__file__))!=lock['runner_sha256']: raise ValueError('Runner changed')
    run,config,row,start=source(architecture,seed,stage)
    directory=REPORT/'replays'/case_key(architecture,seed,stage);directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'best_validation.pt'; marker=directory/'verified.json'
    if marker.exists():
        result=json.loads(marker.read_text())
        if sha256(checkpoint)!=result['checkpoint_sha256']: raise ValueError('Reconstructed state changed')
        return result
    target=int(row['best_iteration'])
    if target<=0 or target>10000 or target%100: raise ValueError('Invalid preselected iteration')
    model=MLP(*ARCHITECTURES[architecture]);model.load_state_dict(start['state_dict'],strict=True)
    masks=start['masks'];assert_state(model,masks)
    loaders=loaders_for(run,seed)
    train_loader,audit=make_audited_loader(loaders['train'])
    set_seed(seed)
    for offset,name in enumerate(('train','validation','test')): loaders[name].generator.manual_seed(seed+offset)
    checks=[]
    def mask_check(history):
        assert_state(model,masks);checks.append(int(history[-1]['iteration']))
    began=time.perf_counter()
    result=train_stage(model,train_loader,loaders['validation'],masks,iterations=target,
        learning_rate=.0012,eval_interval=100,device='cpu',progress=False,history_callback=mask_check)
    history=pd.DataFrame(result['history']);history.to_csv(directory/'replay_history.csv',index=False)
    original=pd.read_csv(run/f'history_stage{stage}.csv');original=original[original.iteration<=target]
    if history.iteration.tolist()!=original.iteration.tolist(): raise ValueError('Historical evaluation grid differs')
    for now,old in zip(history.to_dict('records'),original.to_dict('records')):
        verify_metrics(dict(loss=now['validation_loss'],accuracy=now['validation_accuracy']),
                       dict(loss=old['validation_loss'],accuracy=old['validation_accuracy']))
    reference=reference_batch_order(loaders['train'].dataset.indices,seed,iterations=target)
    generator_hash=__import__('hashlib').sha256(train_loader.generator.get_state().numpy().tobytes()).hexdigest()
    if audit.hexdigest()!=reference['order_sha256'] or audit.examples!=reference['examples'] or generator_hash!=reference['final_generator_sha256']:
        raise ValueError('Replay batch stream differs from original protocol')
    actual=evaluate(model,loaders['validation'])
    delta=verify_metrics(actual,dict(loss=row['best_validation_loss'],accuracy=row['best_validation_accuracy']))
    atomic_torch_save(dict(state_dict=cpu_state(model),masks=masks,iteration=target,
        validation_loss=actual['loss'],validation_accuracy=actual['accuracy']),checkpoint)
    reloaded=torch.load(checkpoint,weights_only=True,map_location='cpu')
    if tensor_digest(reloaded['state_dict'])!=tensor_digest(cpu_state(model)): raise ValueError('Serialization changed weights')
    record=dict(architecture=architecture,seed=seed,stage=stage,condition='original_initialization',
        iteration=target,validation_loss=actual['loss'],validation_accuracy=actual['accuracy'],**delta,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=sha256(checkpoint),
        state_sha256=tensor_digest(reloaded['state_dict']),mask_sha256=tensor_digest(masks),
        source_start_sha256=sha256(run/f'stage{stage}_start.pt'),original_sha256=sha256(run/'initial_weights.pt'),
        split_sha256=sha256(run/'split.pt'),history_prefix_verified=True,mask_checks=len(checks),
        original_initialization_restored=True,batch_order_reference=reference,
        batch_order_basis='Exact reconstruction of frozen loader protocol; historical fetch hashes unavailable',
        runtime_seconds=time.perf_counter()-began,verified=True)
    write_json(record,marker)
    return record


def verify_all():
    """No test loader is iterated until this complete 45-checkpoint gate passes."""
    torch.set_num_threads(1);records=[]
    for architecture,seed,stage in TICKETS+CONTROLS:
        control=(architecture,seed,stage) in CONTROLS and len(records)>=27
        run,config,row,start=source(architecture,seed,stage)
        if control:
            directory=ROOT/'reports/stage4_controls/runs'/case_key(architecture,seed,stage)
            result=verify_case_files(directory);path=directory/'best_validation.pt'
            expected=dict(loss=result['best_validation_loss'],accuracy=result['best_validation_accuracy'])
            iteration=int(result['best_iteration']);condition='random_reinitialization'
        else:
            directory=REPORT/'replays'/case_key(architecture,seed,stage)
            result=json.loads((directory/'verified.json').read_text());path=directory/'best_validation.pt'
            if sha256(path)!=result['checkpoint_sha256']: raise ValueError('Replay checkpoint checksum mismatch')
            expected=dict(loss=row['best_validation_loss'],accuracy=row['best_validation_accuracy'])
            iteration=int(row['best_iteration']);condition='original_initialization'
        state=torch.load(path,weights_only=True,map_location='cpu')
        if state['iteration']!=iteration: raise ValueError('Selected checkpoint iteration changed')
        if set(state['masks'])!=set(start['masks']) or any(not torch.equal(m,start['masks'][n]) for n,m in state['masks'].items()):
            raise ValueError('Evaluation mask differs from original')
        model=MLP(*ARCHITECTURES[architecture]);model.load_state_dict(state['state_dict'],strict=True)
        assert_state(model,state['masks']);loaders=loaders_for(run,seed)
        actual=evaluate(model,loaders['validation']);delta=verify_metrics(actual,expected)
        stats=weight_statistics(model,state['masks']);total=parameter_count(model)
        active=stats['remaining_weights']+total-stats['total_weights']
        if active!=row['remaining_parameters']: raise ValueError('Active parameter count mismatch')
        records.append(dict(architecture=architecture,seed=seed,stage=stage,condition=condition,
            selected_iteration=iteration,validation_loss=actual['loss'],validation_accuracy=actual['accuracy'],
            **delta,**stats,parameters=total,remaining_parameters=active,
            parameter_sparsity=1-active/total,checkpoint=str(path.relative_to(ROOT)),
            checkpoint_sha256=sha256(path),mask_sha256=tensor_digest(state['masks']),verified=True))
    require_gate(records)
    write_json(dict(status='verified',checkpoint_count=45,test_evaluation_started=False,
        loss_atol=LOSS_ATOL,accuracy_atol=ACCURACY_ATOL,records=records),REPORT/'validation_gate.json')
    pd.DataFrame(records).to_csv(REPORT/'checkpoint_verification.csv',index=False)
    return records


def require_gate(records):
    expected={(a,s,r,'original_initialization') for a,s,r in TICKETS}|{(a,s,r,'random_reinitialization') for a,s,r in CONTROLS}
    keys=[(r['architecture'],r['seed'],r['stage'],r['condition']) for r in records]
    if len(keys)!=45 or len(set(keys))!=45 or set(keys)!=expected or not all(r['verified'] for r in records):
        raise ValueError('All 45 approved checkpoints must pass validation before test access')


def evaluate_test(records):
    require_gate(records)
    # Recheck EVERY hash before the first held-out example is evaluated.
    for row in records:
        if sha256(ROOT/row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Gate checkpoint changed')
    output=REPORT/'test_evaluation.json'
    if output.exists():
        saved=json.loads(output.read_text())
        if saved['checkpoint_sha256']!=[r['checkpoint_sha256'] for r in records]: raise ValueError('Test checkpoint set changed')
        return pd.DataFrame(saved['results'])
    rows=[];began=time.perf_counter()
    loaders=make_loaders(ROOT/'datasets',download=False,preload_tensors=True,seed=0,split_seed=42)
    if len(loaders['test'].dataset)!=10000: raise ValueError('Expected full held-out set')
    for record in tqdm(records,desc='Final held-out evaluation',unit='model'):
        state=torch.load(ROOT/record['checkpoint'],weights_only=True,map_location='cpu')
        model=MLP(*ARCHITECTURES[record['architecture']]);model.load_state_dict(state['state_dict'],strict=True)
        assert_state(model,state['masks']);metrics=evaluate(model,loaders['test'])
        rows.append(dict(**record,test_accuracy=metrics['accuracy'],test_loss=metrics['loss'],test_examples=10000))
    write_json(dict(checkpoint_sha256=[r['checkpoint_sha256'] for r in records],results=rows,
        test_examples=10000,selection_changed=False,runtime_seconds=time.perf_counter()-began),output)
    return pd.DataFrame(rows)


def execute(workers=3):
    REPORT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(REPORT/'execution.lock'),timeout=0):
        initialize();began=time.perf_counter()
        try:
            write_json(dict(status='replaying',ticket_replays=27),REPORT/'execution.json')
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(replay,*case) for case in TICKETS]
                for task in tqdm(as_completed(futures),total=27,desc='Verify ticket replays',unit='ticket'): task.result()
            records=verify_all();preserve_previous()
            frame=evaluate_test(records);frame.to_csv(REPORT/'individual_results.csv',index=False)
            count=preserve_previous()
            state=dict(status='complete',ticket_replays=27,controls=18,test_models=45,
                previous_files_unchanged=count,runtime_seconds=time.perf_counter()-began)
            write_json(state,REPORT/'execution.json');return state
        except BaseException as exc:
            write_json(dict(status='failed',error=f'{type(exc).__name__}: {exc}'),REPORT/'execution.json');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=3)
    execute(parser.parse_args().workers)

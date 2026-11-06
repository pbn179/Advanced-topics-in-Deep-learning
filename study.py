"""Fresh four-architecture IMP study; fixed validation selection before all tests."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import gzip
import hashlib
import importlib.metadata
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import re
import time
import platform
from filelock import FileLock
import numpy as np
import pandas as pd
import torch
import yaml
from checkpoints import atomic_torch_save,sha256
from data import make_loaders,set_seed,split_indices
from models import ARCHITECTURES,MLP,parameter_count
from pruning import make_masks,prune_masks,rewind_weights,apply_masks,weight_statistics
from train import train_stage,evaluate

ROOT=Path(__file__).resolve().parent
CASES=tuple(itertools.product(('A','B','C','D'),(0,1,2),range(13)))
JOBS=tuple(itertools.product(('A','B','C','D'),(0,1,2)))
COUNTS={'A':200013,'B':200054,'C':199979,'D':200100}
SOURCE_FILES=('models.py','pruning.py','train.py','data.py','checkpoints.py','study.py')
LOSS_ATOL=1e-8
ACCURACY_ATOL=1e-12
EXPECTED=dict(architectures={'A':[162,421],'B':[200,204],'C':[231,77],'D':[242,40]},
 model_seeds=[0,1,2],split_seed=42,training_examples=55000,validation_examples=5000,test_examples=10000,
 initialization='gaussian_glorot',bias_initialization='zero',optimizer='Adam',learning_rate=.0012,batch_size=60,
 iterations=10000,validation_interval=100,pruning_rounds=12,hidden_pruning_rate=.2,output_pruning_rate=.1,
 biases_pruned=False,device='cpu',num_threads=1,controls=False,test_stages='all')


def write_json(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2),encoding='utf-8');temporary.replace(path)


def cpu_state(model):return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def state_digest(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((str(value.dtype),tuple(value.shape))).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def approved_run_dir(value):
    p=Path(value)
    if p.is_absolute():
        try:p=p.relative_to(ROOT)
        except ValueError:raise ValueError('Run directory must be under this project/artifacts')
    if len(p.parts)!=2 or p.parts[0]!='artifacts' or not re.fullmatch(r'[A-Za-z0-9_-]+',p.parts[1]):
        raise ValueError('Use artifacts/RUN_NAME, separate from previous results')
    result=(ROOT/p).resolve()
    if result.parent!=ROOT/'artifacts':raise ValueError('Run directory resolves outside artifacts')
    return result


def config():
    result=yaml.safe_load((ROOT/'configs/final_experiment.yaml').read_text())
    if result!=EXPECTED:raise ValueError('The predefined scientific protocol was changed')
    for a,widths in result['architectures'].items():
        if tuple(widths)!=ARCHITECTURES[a] or parameter_count(MLP(*widths))!=COUNTS[a]:raise ValueError('Architecture or parameter count mismatch')
    return result


def dataset_integrity():
    from torchvision.datasets import MNIST
    raw=ROOT/'datasets/MNIST/raw';record={}
    for name,md5 in MNIST.resources:
        compressed=raw/name;uncompressed=raw/name.removesuffix('.gz')
        if not compressed.exists() or not uncompressed.exists():raise FileNotFoundError(f'MNIST cache missing {name}; run study.py prepare --download')
        if hashlib.md5(compressed.read_bytes()).hexdigest()!=md5:raise ValueError('Published MNIST checksum mismatch')
        with gzip.open(compressed,'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest()!=sha256(uncompressed):raise ValueError('Extracted MNIST bytes differ from verified gzip')
        record[name]=dict(published_md5=md5,uncompressed_sha256=sha256(uncompressed))
    return record


def prepare(run_dir,download=False):
    run_dir=approved_run_dir(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);settings=config()
    if download:make_loaders(ROOT/'datasets',download=True,preload_tensors=False)
    data_hashes=dataset_integrity()
    protocol=dict(configuration=settings,source_sha256={n:sha256(ROOT/n) for n in SOURCE_FILES},
        config_sha256=sha256(ROOT/'configs/final_experiment.yaml'),
        versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pandas','matplotlib')},
        python=platform.python_version(),dataset_integrity=data_hashes,
        study_origin='fresh four-architecture run, no previous model/checkpoint reuse',
        interpretation='exploratory extension using the previously examined MNIST test set')
    lock=run_dir/'protocol.json'
    if lock.exists() and json.loads(lock.read_text())!=protocol:raise ValueError('Code, software, dataset or locked protocol changed; use a new run directory')
    if not lock.exists():write_json(protocol,lock)
    return protocol


def check_sources(run_dir):
    p=json.loads((run_dir/'protocol.json').read_text())
    if any(sha256(ROOT/n)!=h for n,h in p['source_sha256'].items()):raise ValueError('Scientific implementation changed after protocol lock')
    if sha256(ROOT/'configs/final_experiment.yaml')!=p['config_sha256']:raise ValueError('Configuration changed after protocol lock')
    return p


def assert_masked(model,masks):
    if set(masks)!=set(make_masks(model)):raise ValueError('Incorrect mask names')
    for name,mask in masks.items():
        weights=getattr(model,name).weight.detach()
        if mask.dtype!=torch.bool or mask.shape!=weights.shape or weights[~mask].count_nonzero():raise ValueError('Mask enforcement failed')
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):raise ValueError('Nonfinite selected weights')


def assert_rewind(model,initial,masks):
    expected={n:v.clone() for n,v in initial.items()}
    for name,mask in masks.items():expected[name+'.weight'].mul_(mask)
    if any(not torch.equal(value,expected[n]) for n,value in model.state_dict().items()):raise ValueError('Start is not exact masked original initialization')
    assert_masked(model,masks)


def expected_layer_counts(architecture,stage):
    h1,h2=ARCHITECTURES[architecture];counts=[784*h1,h1*h2,h2*10]
    for _ in range(stage):counts=[n-math.floor(n*r) for n,r in zip(counts,(.2,.2,.1))]
    return dict(zip(('fc1','fc2','fc3'),counts))


def assert_counts(architecture,stage,masks):
    actual={name:int(mask.sum()) for name,mask in masks.items()}
    if actual!=expected_layer_counts(architecture,stage):raise ValueError('Incorrect survivor counts')
    return actual


class BestSaver:
    """Capture actual min-loss weights while final training continues unchanged."""
    def __init__(self,model,masks,directory,metadata):
        self.model=model;self.masks=masks;self.directory=Path(directory);self.metadata=metadata
        self.loss=float('inf');self.mask_checks=0
    def __call__(self,history):
        assert_masked(self.model,self.masks);self.mask_checks+=1
        row=history[-1]
        pd.DataFrame(history).to_csv(self.directory/'history_live.csv',index=False)
        write_json(dict(**self.metadata,iteration=int(row['iteration']),mask_checks=self.mask_checks),self.directory.parent/'progress.json')
        if row['validation_loss']<self.loss:
            self.loss=row['validation_loss']
            atomic_torch_save(dict(state_dict=cpu_state(self.model),masks=self.masks,
                iteration=int(row['iteration']),validation_loss=float(row['validation_loss']),
                validation_accuracy=float(row['validation_accuracy'])),self.directory/'best_validation.pt')


def completed(directory):
    marker=directory/'complete.json'
    if not marker.exists():return None
    record=json.loads(marker.read_text())
    for name,digest in record['sha256'].items():
        if sha256(directory/name)!=digest:raise ValueError(f'Committed artifact changed: {directory/name}')
    return record['result']


def train_one(model,masks,loaders,directory,metadata,iterations=10000,progress=False):
    """Atomic complete-stage resumption; smaller budgets only in synthetic tests."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if completed(directory) is not None:return completed(directory)
    atomic_torch_save(dict(state_dict=cpu_state(model),masks=masks),directory/'start.pt')
    saver=BestSaver(model,masks,directory,metadata)
    result=train_stage(model,loaders['train'],loaders['validation'],masks,iterations=iterations,
        learning_rate=.0012,eval_interval=100,device='cpu',progress=progress,history_callback=saver)
    history=pd.DataFrame(result.pop('history'));history.to_csv(directory/'history.csv',index=False)
    atomic_torch_save(dict(state_dict=cpu_state(model),masks=masks,iteration=iterations),directory/'final.pt')
    assert_masked(model,masks)
    best=torch.load(directory/'best_validation.pt',weights_only=True,map_location='cpu')
    selected=history.loc[history.validation_loss.idxmin()]
    if best['iteration']!=int(selected.iteration) or result['best_iteration']!=best['iteration']:raise ValueError('Selected-state capture failed')
    stats=weight_statistics(model,masks);parameters=parameter_count(model)
    result=dict(**metadata,**result,**stats,initial_parameters=parameters,
        active_parameters=stats['remaining_weights']+parameters-stats['total_weights'],
        parameter_sparsity=1-(stats['remaining_weights']+parameters-stats['total_weights'])/parameters,
        iterations=iterations,validation_interval=100,mask_checks=saver.mask_checks,
        best_state_sha256=state_digest(best['state_dict']),mask_sha256=state_digest(masks))
    write_json(result,directory/'metrics.json')
    files=('start.pt','best_validation.pt','final.pt','history.csv','metrics.json')
    write_json(dict(result=result,sha256={n:sha256(directory/n) for n in files}),directory/'complete.json')
    return result


def train_seed(run_dir,architecture,seed):
    if (architecture,seed) not in JOBS:raise ValueError('Unapproved architecture/seed')
    run_dir=approved_run_dir(run_dir);check_sources(run_dir);torch.set_num_threads(1)
    directory=run_dir/'runs'/f'{architecture}_seed{seed}';directory.mkdir(parents=True,exist_ok=True)
    # Every model is freshly drawn here, never loaded from an earlier study.
    set_seed(seed);model=MLP(*ARCHITECTURES[architecture]);initial=cpu_state(model);masks=make_masks(model)
    initial_path=directory/'initial_weights.pt'
    if initial_path.exists():
        stored=torch.load(initial_path,weights_only=True,map_location='cpu')
        if state_digest(stored)!=state_digest(initial):raise ValueError('Fresh initialization differs on resumption')
    else:atomic_torch_save(initial,initial_path)
    loaders=make_loaders(ROOT/'datasets',batch_size=60,split_seed=42,seed=seed,download=False,preload_tensors=True)
    split={n:list(loaders[n].dataset.indices) for n in ('train','validation')}
    expected_train,expected_val=split_indices(60000,5000,42)
    if split!={'train':expected_train,'validation':expected_val}:raise ValueError('Incorrect original fixed split')
    split_path=directory/'split.pt'
    if split_path.exists() and torch.load(split_path,weights_only=True)!=split:raise ValueError('Split changed on resumption')
    if not split_path.exists():atomic_torch_save(split,split_path)
    for stage in range(13):
        stage_dir=directory/f'stage{stage}'
        cached=completed(stage_dir)
        if cached is not None:
            if (cached['architecture'],cached['model_seed'],cached['pruning_stage'])!=(architecture,seed,stage):raise ValueError('Cached identity mismatch')
            saved=torch.load(stage_dir/'final.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(saved['state_dict']);masks=saved['masks'];assert_counts(architecture,stage,masks);continue
        if stage:
            # model holds the previous FINAL state; never restore best weights here.
            masks=prune_masks(model,masks,hidden_rate=.2,output_rate=.1);rewind_weights(model,initial,masks)
        assert_rewind(model,initial,masks);counts=assert_counts(architecture,stage,masks)
        set_seed(seed)
        for offset,name in enumerate(('train','validation','test')):loaders[name].generator.manual_seed(seed+offset)
        h1,h2=ARCHITECTURES[architecture]
        metadata=dict(architecture=architecture,hidden1=h1,hidden2=h2,width_ratio=h1/h2,
            model_seed=seed,split_seed=42,pruning_stage=stage,
            original_initialization=str(initial_path.relative_to(ROOT)),initialization_sha256=sha256(initial_path),
            split_sha256=sha256(split_path),layer_active_weights=counts,
            previous_final_sha256=sha256(directory/f'stage{stage-1}'/'final.pt') if stage else None,
            protocol_sha256=sha256(run_dir/'protocol.json'),
            selected_checkpoint=str((stage_dir/'best_validation.pt').relative_to(ROOT)),
            final_checkpoint=str((stage_dir/'final.pt').relative_to(ROOT)))
        train_one(model,masks,loaders,stage_dir,metadata)
    write_json(dict(status='complete',architecture=architecture,model_seed=seed,stages=13),directory/'training_complete.json')
    return str(directory.relative_to(ROOT))


def check_metrics(actual,loss,accuracy):
    values=(actual['loss'],actual['accuracy'],loss,accuracy)
    if not all(math.isfinite(float(v)) for v in values):raise ValueError('Nonfinite validation metrics')
    ld=abs(actual['loss']-loss);ad=abs(actual['accuracy']-accuracy)
    if ld>LOSS_ATOL or ad>ACCURACY_ATOL:raise ValueError(f'Selected validation mismatch: loss={ld}, accuracy={ad}')
    return ld,ad


def require_complete_gate(records):
    keys=[(r['architecture'],r['model_seed'],r['pruning_stage']) for r in records]
    if len(keys)!=156 or len(set(keys))!=156 or set(keys)!=set(CASES) or not all(r['verified'] for r in records):
        raise ValueError('All 156 fixed selected states must verify before any test evaluation')


def verify_selected(run_dir):
    run_dir=approved_run_dir(run_dir);check_sources(run_dir);torch.set_num_threads(1)
    if dataset_integrity()!=json.loads((run_dir/'protocol.json').read_text())['dataset_integrity']:raise ValueError('Dataset changed')
    loaders=make_loaders(ROOT/'datasets',download=False,preload_tensors=True,seed=0,split_seed=42)
    records=[];histories=[]
    for architecture,seed in JOBS:
        directory=run_dir/'runs'/f'{architecture}_seed{seed}'
        initial=torch.load(directory/'initial_weights.pt',weights_only=True,map_location='cpu')
        set_seed(seed);fresh=MLP(*ARCHITECTURES[architecture])
        if state_digest(cpu_state(fresh))!=state_digest(initial):raise ValueError('Original initialization is not reproducible')
        previous=None
        for stage in range(13):
            d=directory/f'stage{stage}';row=completed(d)
            if row is None:raise ValueError('Missing completed training stage')
            if row['iterations']!=10000 or row['validation_interval']!=100 or row['mask_checks']!=100:raise ValueError('Mixed training protocol')
            if row['protocol_sha256']!=sha256(run_dir/'protocol.json') or row['initialization_sha256']!=sha256(directory/'initial_weights.pt'):raise ValueError('Provenance mismatch')
            start=torch.load(d/'start.pt',weights_only=True,map_location='cpu')
            final=torch.load(d/'final.pt',weights_only=True,map_location='cpu')
            best=torch.load(d/'best_validation.pt',weights_only=True,map_location='cpu')
            model=MLP(*ARCHITECTURES[architecture]);model.load_state_dict(start['state_dict'])
            masks=start['masks'];assert_rewind(model,initial,masks);assert_counts(architecture,stage,masks)
            if previous is not None:
                model.load_state_dict(previous['state_dict'])
                expected=prune_masks(model,previous['masks'],hidden_rate=.2,output_rate=.1)
                if any(not torch.equal(m,expected[n]) for n,m in masks.items()):raise ValueError('Mask did not come from previous final weights')
                if row['previous_final_sha256']!=sha256(directory/f'stage{stage-1}'/'final.pt'):raise ValueError('Final-state pruning chain mismatch')
            for saved in (best,final):
                if set(saved['masks'])!=set(masks) or any(not torch.equal(m,saved['masks'][n]) for n,m in masks.items()):raise ValueError('Start/best/final masks differ')
                model.load_state_dict(saved['state_dict']);assert_masked(model,masks)
            if final['iteration']!=10000:raise ValueError('Final state is not full budget')
            history=pd.read_csv(d/'history.csv')
            if history.iteration.tolist()!=list(range(100,10001,100)):raise ValueError('Invalid complete validation grid')
            selected=history.loc[history.validation_loss.idxmin()]
            if int(selected.iteration)!=best['iteration'] or best['iteration']!=row['best_iteration']:raise ValueError('Best checkpoint not selected by minimum validation loss')
            check_metrics(dict(loss=best['validation_loss'],accuracy=best['validation_accuracy']),float(selected.validation_loss),float(selected.validation_accuracy))
            model.load_state_dict(best['state_dict']);actual=evaluate(model,loaders['validation'])
            ld,ad=check_metrics(actual,row['best_validation_loss'],row['best_validation_accuracy'])
            if state_digest(best['state_dict'])!=row['best_state_sha256'] or state_digest(masks)!=row['mask_sha256']:raise ValueError('Tensor integrity mismatch')
            records.append(dict(**row,verified=True,selected_checkpoint_sha256=sha256(d/'best_validation.pt'),
                validation_loss_absolute_difference=ld,validation_accuracy_absolute_difference=ad,
                original_rewind_verified=True,mask_persistence_verified=True,final_state_pruning_verified=True))
            history['architecture']=architecture;history['model_seed']=seed;history['pruning_stage']=stage
            histories.append(history);previous=final
            write_json(dict(status='verifying',verified=len(records),total=156),run_dir/'gate_progress.json')
    require_complete_gate(records)
    write_json(dict(status='verified',count=156,loss_atol=LOSS_ATOL,accuracy_atol=ACCURACY_ATOL,
        checkpoint_selection_fixed=True,records=records),run_dir/'selection_manifest.json')
    pd.concat(histories,ignore_index=True).to_csv(run_dir/'validation_history.csv',index=False)
    return records


def evaluate_fixed_test(run_dir,records):
    run_dir=approved_run_dir(run_dir);require_complete_gate(records);torch.set_num_threads(1)
    # Check the entire checkpoint set before the first held-out forward pass.
    for r in records:
        if sha256(ROOT/r['selected_checkpoint'])!=r['selected_checkpoint_sha256']:raise ValueError('Fixed checkpoint changed after gate')
    loaders=make_loaders(ROOT/'datasets',download=False,preload_tensors=True,seed=0,split_seed=42)
    if len(loaders['test'].dataset)!=10000:raise ValueError('Expected complete test set')
    output=[]
    for row in records:
        path=ROOT/row['selected_checkpoint'];cache=path.parent/'test_evaluation.json'
        if cache.exists():
            tested=json.loads(cache.read_text())
            if tested['selected_checkpoint_sha256']!=row['selected_checkpoint_sha256']:raise ValueError('Test cache checkpoint mismatch')
        else:
            saved=torch.load(path,weights_only=True,map_location='cpu')
            model=MLP(*ARCHITECTURES[row['architecture']]);model.load_state_dict(saved['state_dict']);assert_masked(model,saved['masks'])
            result=evaluate(model,loaders['test'])
            tested=dict(**row,test_accuracy=result['accuracy'],test_loss=result['loss'],test_examples=10000)
            write_json(tested,cache)
        output.append(tested)
        write_json(dict(status='testing_fixed_states',evaluated=len(output),total=156),run_dir/'test_progress.json')
    require_complete_gate(output)
    write_json(dict(results=output,selection_changed=False,test_set_previously_examined=True),run_dir/'test_results.json')
    return output


def execute(run_dir='artifacts/main_study',workers=3,enable_training=False,download=False):
    if enable_training is not True:raise PermissionError('Training requires explicit --enable-training; the notebook defaults to saved results')
    if workers not in (1,2,3,4):raise ValueError('Use 1–4 independent single-threaded workers')
    run_dir=approved_run_dir(run_dir);run_dir.mkdir(parents=True,exist_ok=True)
    with FileLock(str(run_dir/'execution.lock'),timeout=0):
        prepare(run_dir,download);began=time.perf_counter()
        state=dict(status='training',total_stages=156,workers=workers,controls=0)
        write_json(state,run_dir/'execution.json')
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                pending={pool.submit(train_seed,str(run_dir),a,s) for a,s in JOBS}
                while pending:
                    done,pending=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                    for future in done:future.result()
                    state.update(completed_stages=len(list((run_dir/'runs').glob('*/stage*/complete.json'))),elapsed_seconds=time.perf_counter()-began)
                    write_json(state,run_dir/'execution.json')
            state.update(status='verifying_all_selected_states');write_json(state,run_dir/'execution.json')
            records=verify_selected(run_dir)
            state.update(status='testing_fixed_states');write_json(state,run_dir/'execution.json')
            results=evaluate_fixed_test(run_dir,records)
            pd.DataFrame(results).to_csv(run_dir/'all_measurements.csv',index=False)
            state.update(status='complete',completed_stages=156,verified_states=156,test_evaluations=156,elapsed_seconds=time.perf_counter()-began)
            write_json(state,run_dir/'execution.json');return state
        except BaseException as exc:
            state.update(status='failed_or_interrupted',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-began)
            write_json(state,run_dir/'execution.json');raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('prepare','run'))
    parser.add_argument('--run-dir',default='artifacts/main_study');parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--enable-training',action='store_true');parser.add_argument('--download',action='store_true')
    args=parser.parse_args()
    print(json.dumps(prepare(args.run_dir,args.download) if args.action=='prepare' else execute(args.run_dir,args.workers,args.enable_training,args.download),indent=2))

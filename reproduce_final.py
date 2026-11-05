"""Explicitly opt-in, resumable reproduction of the locked 117+18 training stages.

This is a future reproduction entry point. Packaging/default analysis never calls it.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import itertools
import json
import multiprocessing
from pathlib import Path
import time
from filelock import FileLock
import pandas as pd
import torch
import yaml
from checkpoints import atomic_torch_save,sha256
from data import make_loaders,set_seed
from models import ARCHITECTURES,MLP,parameter_count
from pruning import make_masks,prune_masks,rewind_weights,apply_masks,weight_statistics
from train import train_stage,evaluate

ROOT=Path(__file__).resolve().parent
CORE=('models.py','pruning.py','train.py','data.py','checkpoints.py')
CONFIG=dict(architectures={'A':[162,421],'B':[200,204],'C':[231,77]},model_seeds=[0,1,2],
    split_seed=42,iterations=10000,eval_interval=100,rounds=12,learning_rate=.0012,batch_size=60,
    hidden_pruning_rate=.2,output_pruning_rate=.1,initialization='gaussian_glorot',optimizer='Adam',
    control_stages=[6,12],test_stages=[0,6,12],device='cpu',num_threads=1,data_dir='datasets')


def write_json(value,path):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)


def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def approved_output(output,enabled=False):
    if enabled is not True:raise PermissionError('Full reproduction is disabled; explicitly pass enabled=True or --enable-full-reproduction')
    path=Path(output)
    if path.is_absolute() or '..' in path.parts or len(path.parts)<2 or path.parts[0]!='reproductions':
        raise ValueError('Use a separate relative output under reproductions/NAME')
    path=(ROOT/path).resolve()
    if ROOT/'reproductions' not in path.parents:raise ValueError('Unsafe reproduction output')
    return path


class SelectedCheckpoint:
    def __init__(self,model,masks,path):self.model=model;self.masks=masks;self.path=path;self.loss=float('inf')
    def __call__(self,history):
        row=history[-1]
        for name,mask in self.masks.items():
            if getattr(self.model,name).weight.detach()[~mask].count_nonzero():raise ValueError('Mask persistence failed')
        if row['validation_loss']<self.loss:
            self.loss=row['validation_loss']
            atomic_torch_save(dict(state_dict=state(self.model),masks=self.masks,iteration=int(row['iteration']),
                validation_loss=float(row['validation_loss']),validation_accuracy=float(row['validation_accuracy'])),self.path)


def completed(directory):
    path=directory/'complete.json'
    if not path.exists():return None
    record=json.loads(path.read_text())
    for name,h in record['sha256'].items():
        if sha256(directory/name)!=h:raise ValueError(f'Corrupt reproduction artifact: {directory/name}')
    return record['result']


def train_one(model,masks,loaders,directory,metadata,iterations=10000,progress=True):
    """One resumable commit; optional small iterations exist only for synthetic tests."""
    directory.mkdir(parents=True,exist_ok=True)
    cached=completed(directory)
    if cached is not None:return cached
    atomic_torch_save(dict(state_dict=state(model),masks=masks),directory/'start.pt')
    callback=SelectedCheckpoint(model,masks,directory/'best_validation.pt')
    result=train_stage(model,loaders['train'],loaders['validation'],masks,iterations=iterations,
        learning_rate=.0012,eval_interval=100,progress=progress,history_callback=callback)
    pd.DataFrame(result.pop('history')).to_csv(directory/'history.csv',index=False)
    atomic_torch_save(dict(state_dict=state(model),masks=masks),directory/'final.pt')
    stats=weight_statistics(model,masks);total=parameter_count(model)
    row=dict(**metadata,**result,**stats,iterations=iterations,eval_interval=100,split_seed=42,parameters=total,remaining_parameters=stats['remaining_weights']+total-stats['total_weights'])
    write_json(row,directory/'result.json')
    files=['start.pt','best_validation.pt','final.pt','history.csv','result.json']
    write_json(dict(result=row,sha256={f:sha256(directory/f) for f in files}),directory/'complete.json')
    return row


def run_seed(output,architecture,seed,progress=True):
    """Sequential IMP per seed; controls never feed into ticket states."""
    torch.set_num_threads(1);directory=Path(output)/'runs'/f'{architecture}_seed{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    loaders=make_loaders(ROOT/'datasets',seed=seed,split_seed=42,download=True,preload_tensors=True)
    split={k:list(loaders[k].dataset.indices) for k in ('train','validation')}
    set_seed(seed);model=MLP(*ARCHITECTURES[architecture]);initial=state(model);masks=make_masks(model)
    if (directory/'initial_weights.pt').exists():
        saved=torch.load(directory/'initial_weights.pt',weights_only=True,map_location='cpu')
        if any(not torch.equal(saved[k],v) for k,v in initial.items()):raise ValueError('Initialization differs on resume')
        if torch.load(directory/'split.pt',weights_only=True)!=split:raise ValueError('Split differs on resume')
    else:
        atomic_torch_save(initial,directory/'initial_weights.pt');atomic_torch_save(split,directory/'split.pt')
    rows=[]
    for stage in range(13):
        target=directory/f'ticket_stage{stage}'
        cached=completed(target)
        if cached is not None:
            final=torch.load(target/'final.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(final['state_dict']);masks=final['masks'];rows.append(cached);continue
        if stage:
            masks=prune_masks(model,masks,hidden_rate=.2,output_rate=.1)
            rewind_weights(model,initial,masks)
        set_seed(seed)
        for offset,key in enumerate(('train','validation','test')):loaders[key].generator.manual_seed(seed+offset)
        row=train_one(model,masks,loaders,target,dict(architecture=architecture,seed=seed,stage=stage,
            condition='original_initialization',initialization_seed=seed),progress=progress)
        rows.append(row)
    for stage in (6,12):
        target=directory/f'control_stage{stage}';cached=completed(target)
        if cached is not None:rows.append(cached);continue
        ticket=torch.load(directory/f'ticket_stage{stage}'/'final.pt',weights_only=True,map_location='cpu')
        masks={k:v.clone() for k,v in ticket['masks'].items()}
        chosen=(seed+1000003+stage*1009)%2**32
        set_seed(chosen);control=MLP(*ARCHITECTURES[architecture]);apply_masks(control,masks)
        set_seed(seed)
        for offset,key in enumerate(('train','validation','test')):loaders[key].generator.manual_seed(seed+offset)
        rows.append(train_one(control,masks,loaders,target,dict(architecture=architecture,seed=seed,stage=stage,
            condition='random_reinitialization',initialization_seed=chosen),progress=progress))
    return rows


def verify_and_test(output,rows):
    """All 45 selected states must validate before the first held-out forward pass."""
    output=Path(output);torch.set_num_threads(1);selected=[]
    loaders=make_loaders(ROOT/'datasets',download=False,preload_tensors=True,seed=0,split_seed=42)
    for row in rows:
        if row['stage'] not in (0,6,12):continue
        a,s,stage=row['architecture'],row['seed'],row['stage'];control=row['condition']=='random_reinitialization'
        run=output/'runs'/f'{a}_seed{s}';directory=run/f'{"control" if control else "ticket"}_stage{stage}'
        completed(directory)
        best=torch.load(directory/'best_validation.pt',weights_only=True,map_location='cpu')
        start=torch.load(directory/'start.pt',weights_only=True,map_location='cpu')
        reference=torch.load(run/f'ticket_stage{stage}'/'start.pt',weights_only=True,map_location='cpu')
        if any(not torch.equal(m,reference['masks'][k]) or not torch.equal(m,start['masks'][k]) for k,m in best['masks'].items()):raise ValueError('Selected masks differ')
        model=MLP(*ARCHITECTURES[a]);model.load_state_dict(best['state_dict'])
        for k,m in best['masks'].items():
            if getattr(model,k).weight.detach()[~m].count_nonzero():raise ValueError('Selected mask regrew')
        if best['iteration']!=row['best_iteration']:raise ValueError('Selected iteration mismatch')
        metric=evaluate(model,loaders['validation'])
        if abs(metric['loss']-row['best_validation_loss'])>1e-8 or abs(metric['accuracy']-row['best_validation_accuracy'])>1e-12:
            raise ValueError('Best checkpoint validation mismatch; test evaluation blocked')
        selected.append((row,directory/'best_validation.pt',best))
    expected={(a,s,r,'original_initialization') for a,s,r in itertools.product(('A','B','C'),(0,1,2),(0,6,12))}
    expected|={(a,s,r,'random_reinitialization') for a,s,r in itertools.product(('A','B','C'),(0,1,2),(6,12))}
    if len(selected)!=45 or {(r['architecture'],r['seed'],r['stage'],r['condition']) for r,_,_ in selected}!=expected:
        raise ValueError('Incomplete predefined selected-state set')
    hashes={str(p.relative_to(output)):sha256(p) for _,p,_ in selected}
    write_json(dict(verified_checkpoints=45,sha256=hashes),output/'validation_gate.json')
    cache=output/'test_results.json'
    if cache.exists():
        record=json.loads(cache.read_text())
        if record['sha256']!=hashes:raise ValueError('Cached test checkpoint set changed')
        return record['results']
    results=[]
    for row,_,best in selected:
        model=MLP(*ARCHITECTURES[row['architecture']]);model.load_state_dict(best['state_dict'])
        metric=evaluate(model,loaders['test'])
        results.append(dict(**row,test_accuracy=metric['accuracy'],test_loss=metric['loss'],test_examples=10000))
    write_json(dict(sha256=hashes,results=results),cache);return results


def full_reproduction(output='reproductions/full_run',enabled=False,workers=1,progress=True):
    output=approved_output(output,enabled)
    config=yaml.safe_load((ROOT/'config_final.yaml').read_text())
    if config!=CONFIG:raise ValueError('Final locked configuration changed')
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/'execution.lock'),timeout=0):
        protocol=dict(config=config,source_sha256={f:sha256(ROOT/f) for f in CORE+('reproduce_final.py',)},
            versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','pandas','matplotlib')})
        lock=output/'protocol.json'
        if lock.exists() and json.loads(lock.read_text())!=protocol:raise ValueError('Reproduction protocol/software changed; use a new output directory')
        if not lock.exists():write_json(protocol,lock)
        began=time.perf_counter();jobs=list(itertools.product(('A','B','C'),(0,1,2)));rows=[]
        # Cache/download once before parallel readers, avoiding concurrent MNIST extraction.
        make_loaders(ROOT/'datasets',download=True,preload_tensors=False)
        if workers==1:
            for a,s in jobs:rows.extend(run_seed(output,a,s,progress))
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures=[pool.submit(run_seed,str(output),a,s,False) for a,s in jobs]
                for future in futures:rows.extend(future.result())
        pd.DataFrame(rows).to_csv(output/'all_training_results.csv',index=False)
        tested=verify_and_test(output,rows);pd.DataFrame(tested).to_csv(output/'test_results.csv',index=False)
        write_json(dict(status='complete',training_stages=135,selected_test_models=45,
            elapsed_seconds=time.perf_counter()-began),output/'execution.json')
        export_saved_view(output,rows,tested)
        return output


def export_saved_view(output,rows,tested):
    """Make the new run readable by the same analysis API, without touching originals."""
    output=Path(output);all_rows=pd.DataFrame(rows)
    main=all_rows[all_rows.condition=='original_initialization'].copy()
    dense=main[main.stage==0][['architecture','seed','best_validation_accuracy']].rename(columns={'best_validation_accuracy':'dense_accuracy'})
    main=main.merge(dense,on=['architecture','seed'],validate='many_to_one')
    main['accuracy_change_pp']=100*(main.best_validation_accuracy-main.dense_accuracy)
    controls=all_rows[all_rows.condition=='random_reinitialization'].copy()
    ci=pd.concat([main[main.stage.isin([6,12])],controls],ignore_index=True)
    def summary(frame,keys,metrics):
        table=frame.groupby(keys)[metrics].agg(['mean','std'])
        table.columns=['_'.join(c) for c in table.columns]
        table=table.reset_index()
        if 'remaining_parameters' in frame:
            counts=frame.groupby(keys).remaining_parameters.first().reset_index()
            table=table.merge(counts,on=keys,validate='one_to_one')
        return table
    keys=['architecture','seed','stage']
    cp=main[main.stage.isin([6,12])].merge(controls,on=keys,suffixes=('_ticket','_control'),validate='one_to_one')
    cp['ticket_minus_control_accuracy_pp']=100*(cp.best_validation_accuracy_ticket-cp.best_validation_accuracy_control)
    test=pd.DataFrame(tested)
    test['validation_accuracy']=test.best_validation_accuracy
    test['validation_loss']=test.best_validation_loss
    test['selected_iteration']=test.best_iteration
    test['parameter_sparsity']=1-test.remaining_parameters/test.parameters
    dense=test[test.stage==0][['architecture','seed','test_accuracy']].rename(columns={'test_accuracy':'dense_test_accuracy'})
    test=test.merge(dense,on=['architecture','seed'],validate='many_to_one')
    test['test_accuracy_change_pp']=100*(test.test_accuracy-test.dense_test_accuracy)
    tp=test[(test.condition=='original_initialization')&test.stage.isin([6,12])].merge(test[test.condition=='random_reinitialization'],on=keys,suffixes=('_ticket','_control'),validate='one_to_one')
    tp['ticket_minus_control_test_accuracy_pp']=100*(tp.test_accuracy_ticket-tp.test_accuracy_control)
    tables={'stage3/individual_results':main,
        'stage3/summary_mean_std':summary(main,['architecture','stage'],['best_validation_accuracy','best_validation_loss','best_iteration','accuracy_change_pp']),
        'stage4_controls/individual_results':ci,'stage4_controls/paired_results':cp,
        'stage4_controls/paired_summary':summary(cp,['architecture','stage'],['ticket_minus_control_accuracy_pp']),
        'stage4_test/individual_results':test,
        'stage4_test/summary':summary(test,['architecture','stage','condition'],['test_accuracy','test_accuracy_change_pp']),
        'stage4_test/paired_results':tp,
        'stage4_test/paired_summary':summary(tp,['architecture','stage'],['ticket_minus_control_test_accuracy_pp'])}
    for name,frame in tables.items():
        path=output/'reports'/f'{name}.csv';path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False)
    records=[]
    for row in test.to_dict('records'):
        a,s,r=row['architecture'],row['seed'],row['stage'];control=row['condition']=='random_reinitialization'
        checkpoint=Path('runs')/f'{a}_seed{s}'/f'{"control" if control else "ticket"}_stage{r}'/'best_validation.pt'
        records.append(dict(architecture=a,seed=s,stage=r,condition=row['condition'],checkpoint=str(checkpoint),checkpoint_sha256=sha256(output/checkpoint),verified=True))
    write_json(dict(checkpoint_count=45,records=records),output/'reports/stage4_test/validation_gate.json')
    write_json(dict(status='verified',source='new explicitly enabled reproduction',test_based_selection=False),output/'reports/stage4_test/verification.json')
    (output/'packaging').mkdir(exist_ok=True)
    manifest={str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file() and p.name not in ('artifact_manifest.json','execution.lock')}
    write_json(manifest,output/'packaging/artifact_manifest.json')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--enable-full-reproduction',action='store_true',help='Required explicit permission to train')
    parser.add_argument('--output',default='reproductions/full_run');parser.add_argument('--workers',type=int,choices=(1,2,3),default=1)
    args=parser.parse_args();print(full_reproduction(args.output,args.enable_full_reproduction,args.workers))

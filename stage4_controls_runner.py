"""Exactly 18 fixed-mask reinitialization controls; no ticket training or mask generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
from pathlib import Path
import struct
import time

from filelock import FileLock
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from checkpoints import atomic_torch_save, sha256
from data import make_loaders, set_seed, split_indices
from models import ARCHITECTURES, MLP, parameter_count
from pruning import apply_masks, weight_statistics
from stage3_analysis import study_runs
from stage3_runner import CORE_FILES, validate_run_config
from train import train_stage

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'reports' / 'stage4_controls'
CASES = tuple(itertools.product(('A','B','C'), (0,1,2), (6,12)))


def write_json(value, path):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def tensor_digest(tensors):
    digest=hashlib.sha256()
    for name,value in sorted(tensors.items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str((str(array.dtype),array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialization_seed(seed, stage):
    return (seed+1000003+stage*1009) % 2**32


def case_key(architecture,seed,stage):
    return f'{architecture}_seed{seed}_round{stage}'


def previous_snapshot():
    REPORT.mkdir(parents=True,exist_ok=True)
    files=[path for path in (ROOT/'reports').rglob('*')
           if path.is_file() and path.relative_to(ROOT/'reports').parts[0]!='stage4_controls']
    files += [path for path in (ROOT/'results').rglob('*') if path.is_file()]
    files += [ROOT/name for name in ('stage1.ipynb','stage2.ipynb','stage2_5.ipynb','stage2_6.ipynb','stage3.ipynb')]
    current={str(path.relative_to(ROOT)):sha256(path) for path in sorted(files)}
    target=REPORT/'previous_results_snapshot.json'
    if target.exists():
        if current!=json.loads(target.read_text()):
            raise ValueError('Previous reports/results/notebooks changed')
    else:
        write_json(current,target)
    return len(current)


class IndexDataset(Dataset):
    def __init__(self,indices):
        self.indices=list(indices)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self,index):
        return self.indices[index]


class AuditedDataset(Dataset):
    """Record actual train-fetch order without changing the returned examples."""
    def __init__(self,dataset):
        self.dataset=dataset
        self.indices=list(dataset.indices)
        self.digest=hashlib.sha256()
        self.examples=0
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        self.digest.update(struct.pack('<q',int(self.indices[index])))
        self.examples+=1
        return self.dataset[index]
    def hexdigest(self):
        return self.digest.hexdigest()


def reference_batch_order(indices,seed,iterations=10000,batch_size=60):
    """Replay the exact DataLoader RNG/sampling protocol, without model training."""
    loader=DataLoader(IndexDataset(indices),batch_size=batch_size,shuffle=True,drop_last=False,
                      num_workers=0,generator=torch.Generator().manual_seed(seed))
    iterator=iter(loader)
    digest=hashlib.sha256()
    examples=0
    for _ in range(iterations):
        try:
            batch=next(iterator)
        except StopIteration:
            iterator=iter(loader)
            batch=next(iterator)
        values=batch.numpy().astype('<i8',copy=False)
        digest.update(values.tobytes())
        examples+=len(batch)
    return dict(seed=seed,iterations=iterations,batch_size=batch_size,examples=examples,
                order_sha256=digest.hexdigest(),
                final_generator_sha256=hashlib.sha256(loader.generator.get_state().numpy().tobytes()).hexdigest())


def make_audited_loader(original):
    if original.num_workers!=0 or original.drop_last:
        raise ValueError('Controls require the validated worker-free, non-dropping loader')
    audit=AuditedDataset(original.dataset)
    loader=DataLoader(audit,batch_size=original.batch_size,shuffle=True,drop_last=False,num_workers=0,
        generator=original.generator,worker_init_fn=original.worker_init_fn,pin_memory=False)
    return loader,audit


def initialize_control(architecture,seed,stage,masks):
    chosen_seed=initialization_seed(seed,stage)
    set_seed(chosen_seed)
    model=MLP(*ARCHITECTURES[architecture])
    fresh=cpu_state(model)
    apply_masks(model,masks)
    return model,fresh,chosen_seed


class BestCheckpoint:
    """Snapshot the actual current weights on strict validation-loss improvement."""
    def __init__(self,model,masks,directory):
        self.model=model
        self.masks=masks
        self.directory=Path(directory)
        self.best_loss=float('inf')
        self.mask_checks=0
    def __call__(self,history):
        row=history[-1]
        for name,mask in self.masks.items():
            if getattr(self.model,name).weight.detach()[~mask].count_nonzero():
                raise ValueError('A pruned control weight regrew')
        self.mask_checks+=1
        pd.DataFrame(history).to_csv(self.directory/'history_live.csv',index=False)
        if row['validation_loss']<self.best_loss:
            self.best_loss=row['validation_loss']
            atomic_torch_save(dict(state_dict=cpu_state(self.model),masks=self.masks,
                iteration=int(row['iteration']),validation_loss=float(row['validation_loss']),
                validation_accuracy=float(row['validation_accuracy'])),self.directory/'best_validation.pt')


def source_case(architecture,seed,stage):
    if (architecture,seed,stage) not in CASES:
        raise ValueError('Only the 18 approved architecture/seed/round controls are permitted')
    runs=study_runs()
    source=runs[architecture,seed]
    config=json.loads((source/'config.json').read_text())
    validate_run_config(config,architecture,seed)
    state=torch.load(source/f'stage{stage}.pt',map_location='cpu',weights_only=True)
    start=torch.load(source/f'stage{stage}_start.pt',map_location='cpu',weights_only=True)
    masks={name:value.clone() for name,value in state['masks'].items()}
    if not all(torch.equal(mask,start['masks'][name]) for name,mask in masks.items()):
        raise ValueError('Source ticket start/final masks differ')
    row=pd.read_csv(ROOT/'reports'/'stage3'/'individual_results.csv')
    row=row[(row.architecture==architecture)&(row.seed==seed)&(row.stage==stage)]
    if len(row)!=1:
        raise ValueError('Missing unique Stage 3 ticket measurement')
    row=row.iloc[0].to_dict()
    if sum(int(mask.sum()) for mask in masks.values())!=row['remaining_weights']:
        raise ValueError('Source mask count differs from the reported ticket')
    return source,config,masks,row


def audit_ticket_checkpoints():
    """Inspect actual Stage 3 artifacts and checkpoint keys; no training or test evaluation."""
    rows=[]
    measurements=pd.read_csv(ROOT/'reports'/'stage3'/'individual_results.csv')
    for (architecture,seed),run in sorted(study_runs().items()):
        candidate_files=[path.name for path in run.glob('*.pt') if 'best' in path.name.lower()]
        for row in measurements[(measurements.architecture==architecture)&(measurements.seed==seed)].itertuples():
            checkpoint=torch.load(run/f'stage{row.stage}.pt',map_location='cpu',weights_only=True)
            # The existing immutable engine writes this tensor state only AFTER
            # the complete train_stage call, and never restores best weights.
            if set(checkpoint)!= {'state_dict','masks'}:
                raise ValueError('Unexpected Stage 3 checkpoint schema; inspect before interpreting')
            selected_is_final=int(row.best_iteration)==10000
            rows.append(dict(architecture=architecture,seed=seed,stage=row.stage,
                selected_iteration=int(row.best_iteration),final_checkpoint_iteration=10000,
                best_named_checkpoint_files=';'.join(candidate_files),
                selected_state_saved=selected_is_final,
                final_state_saved=True,initial_and_start_states_saved=True,
                selected_test_without_replay=selected_is_final,
                final_weight_test_without_replay=True,
                final_checkpoint=str((run/f'stage{row.stage}.pt').resolve())))
    frame=pd.DataFrame(rows)
    frame.to_csv(REPORT/'stage3_checkpoint_audit.csv',index=False)
    count=int(frame.selected_state_saved.sum())
    text=f'''# Stage 3 minimum-loss checkpoint audit

Inspected all **{len(frame)} ticket stages** and all saved tensor filenames/checkpoint schemas. The immutable Stage 3 engine saved original initialization, each masked start and the **final 10,000-update state**. It did not save a best-validation tensor snapshot or restore a best state before saving. Best-validation metrics and histories were saved, but metrics alone cannot reconstruct weights.

Minimum-loss states available without replay: **{count}/{len(frame)}**. None of the current Stage 3 minima was at update 10,000, so its final checkpoints cannot stand in for the selected checkpoints.

**Final test evaluation at the reported minimum-validation-loss states requires deterministic training replay.** Saved starts, masks, model/data seeds and exact histories allow a separately authorized replay to each selected iteration, with fresh Adam and the original batch order; no pruning should be regenerated. Validation metrics should then be checked before any test evaluation. No such replay is performed here.

**Final-weight test evaluation is possible without retraining**, using the saved 10,000-update states, but that is a different estimand from the reported validation-selected results and must be labeled separately. Do not silently substitute final states for selected ones.

Stage 4 controls save actual `best_validation.pt` snapshots during training as well as their initial/start/final states. Selected-state control evaluation will therefore be possible without control retraining, but a paired selected-state ticket comparison still needs the missing ticket snapshots. The test set remains unevaluated.
'''
    (REPORT/'stage3_checkpoint_audit.md').write_text(text,encoding='utf-8')
    return frame


def initialize_study():
    torch.set_num_threads(1)
    previous_snapshot()
    protocol=dict(architectures={name:list(ARCHITECTURES[name]) for name in ('A','B','C')},
        model_seeds=[0,1,2],rounds=[6,12],controls=18,iterations=10000,eval_interval=100,
        batch_size=60,learning_rate=.0012,split_seed=42,optimizer='fresh Adam',
        initialization='fresh Gaussian Glorot, zero biases',
        control_seed_rule='(seed+1000003+stage*1009)%2**32',
        masks='exact saved Stage 3 positions; no regeneration',test_evaluated=False,
        core_sha256={name:sha256(ROOT/name) for name in CORE_FILES},
        runner_sha256=sha256(Path(__file__)),
        versions={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','pandas','matplotlib')})
    lock=REPORT/'protocol.lock.json'
    if lock.exists() and json.loads(lock.read_text())!=protocol:
        raise ValueError('Existing control protocol differs')
    if not lock.exists():
        write_json(protocol,lock)
    for case in CASES:
        source_case(*case)
    audit_ticket_checkpoints()
    references=REPORT/'batch_order_reference.json'
    if not references.exists():
        indices,_=split_indices(60000,5000,42)
        write_json({str(seed):reference_batch_order(indices,seed) for seed in (0,1,2)},references)
    return protocol


def verify_case_files(directory):
    complete=directory/'complete.json'
    if not complete.exists():
        raise ValueError('Control is not committed')
    record=json.loads(complete.read_text())
    for name,digest in record['sha256'].items():
        if sha256(directory/name)!=digest:
            raise ValueError(f'Control artifact changed: {name}')
    return record['result']


def run_control(architecture,seed,stage):
    if (architecture,seed,stage) not in CASES:
        raise ValueError('Only the 18 approved controls may train')
    torch.set_num_threads(1)
    protocol=json.loads((REPORT/'protocol.lock.json').read_text())
    if protocol['runner_sha256']!=sha256(Path(__file__)):
        raise ValueError('Control runner changed after protocol lock')
    source,source_config,masks,ticket=source_case(architecture,seed,stage)
    directory=REPORT/'runs'/case_key(architecture,seed,stage)
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():
        result=verify_case_files(directory)
        if result['mask_sha256']!=tensor_digest(masks):
            raise ValueError('Completed control no longer matches the source mask')
        return result
    model,fresh,chosen_seed=initialize_control(architecture,seed,stage,masks)
    original=torch.load(source/'initial_weights.pt',map_location='cpu',weights_only=True)
    for name,mask in masks.items():
        if torch.equal(fresh[name+'.weight'][mask],original[name+'.weight'][mask]):
            raise ValueError('Control initialization is not fresh')
    mask_hash=tensor_digest(masks)
    atomic_torch_save(dict(state_dict=fresh,initialization_seed=chosen_seed),directory/'fresh_initial.pt')
    atomic_torch_save(dict(state_dict=cpu_state(model),masks=masks),directory/'start.pt')
    metadata=dict(architecture=architecture,seed=seed,stage=stage,initialization_seed=chosen_seed,
        source_run_directory=str(source.resolve()),source_final_sha256=sha256(source/f'stage{stage}.pt'),
        source_start_sha256=sha256(source/f'stage{stage}_start.pt'),mask_sha256=mask_hash,
        iterations=10000,eval_interval=100,learning_rate=.0012,batch_size=60,split_seed=42,
        source_split_sha256=source_config['split_sha256'],num_threads=1,device='cpu',status='running')
    write_json(metadata,directory/'config.json')
    loaders=make_loaders(ROOT/'datasets',seed=seed,split_seed=42,download=False,preload_tensors=True)
    source_split=torch.load(source/'split.pt',map_location='cpu',weights_only=True)
    if list(loaders['train'].dataset.indices)!=source_split['train'] or list(loaders['validation'].dataset.indices)!=source_split['validation']:
        raise ValueError('Control and ticket splits differ')
    train_loader,audit=make_audited_loader(loaders['train'])
    set_seed(seed)  # restore ticket training RNG after drawing independent model weights
    for offset,name in enumerate(('train','validation','test')):
        loaders[name].generator.manual_seed(seed+offset)
    callback=BestCheckpoint(model,masks,directory)
    try:
        metrics=train_stage(model,train_loader,loaders['validation'],masks,iterations=10000,
            learning_rate=.0012,eval_interval=100,device='cpu',progress=False,history_callback=callback)
        history=pd.DataFrame(metrics.pop('history'))
        history.to_csv(directory/'history.csv',index=False)
        expected=json.loads((REPORT/'batch_order_reference.json').read_text())[str(seed)]
        final_rng=hashlib.sha256(train_loader.generator.get_state().numpy().tobytes()).hexdigest()
        if audit.hexdigest()!=expected['order_sha256'] or audit.examples!=expected['examples'] or final_rng!=expected['final_generator_sha256']:
            raise ValueError('Observed control batch order differs from the source-ticket loader replay')
        best=torch.load(directory/'best_validation.pt',map_location='cpu',weights_only=True)
        if best['iteration']!=metrics['best_iteration'] or best['validation_loss']!=metrics['best_validation_loss']:
            raise ValueError('Best checkpoint does not match selected metrics')
        if tensor_digest(masks)!=mask_hash:
            raise ValueError('A source mask was mutated')
        atomic_torch_save(dict(state_dict=cpu_state(model),masks=masks,iteration=10000),directory/'final.pt')
        result=dict(architecture=architecture,seed=seed,stage=stage,initialization_seed=chosen_seed,
            condition='random_reinitialization',parameters=parameter_count(model),**weight_statistics(model,masks),
            **metrics,iterations=10000,eval_interval=100,mask_sha256=mask_hash,
            batch_order_sha256=audit.hexdigest(),training_examples_seen=audit.examples,
            final_generator_sha256=final_rng,mask_checks_during_training=callback.mask_checks,
            source_run_directory=str(source.resolve()),control_directory=str(directory.resolve()),
            best_checkpoint_saved=True,test_evaluated=False)
        write_json(result,directory/'result.json')
        metadata.update(status='complete')
        write_json(metadata,directory/'config.json')
        names=('fresh_initial.pt','start.pt','best_validation.pt','final.pt','history.csv','result.json','config.json')
        write_json(dict(result=result,sha256={name:sha256(directory/name) for name in names}),directory/'complete.json')
        return result
    except Exception as exc:
        metadata.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(metadata,directory/'config.json')
        raise


def execute(workers=3):
    if workers not in (1,2,3):
        raise ValueError('Use 1–3 isolated one-thread workers')
    REPORT.mkdir(parents=True,exist_ok=True)
    with FileLock(str(REPORT/'runner.lock'),timeout=0):
        initialize_study()
        path=REPORT/'execution.json'
        if path.exists() and json.loads(path.read_text()).get('status')=='complete':
            for case in CASES:
                verify_case_files(REPORT/'runs'/case_key(*case))
            return json.loads(path.read_text())
        started=time.perf_counter()
        state=dict(status='running',workers=workers,start_method='spawn',cases=[list(case) for case in CASES],
                   original_tickets_retrained=0,test_evaluated=False,completed=[])
        write_json(state,path)
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(run_control,*case):case for case in CASES}
                for future in as_completed(futures):
                    result=future.result()
                    state['completed'].append({key:result[key] for key in ('architecture','seed','stage','control_directory')})
                    write_json(state,path)
                    print('Completed',case_key(result['architecture'],result['seed'],result['stage']),flush=True)
            state.update(status='complete',elapsed_seconds=time.perf_counter()-started)
            write_json(state,path)
            previous_snapshot()
            return state
        except BaseException as exc:
            state.update(status='interrupted_or_failed',error=f'{type(exc).__name__}: {exc}',elapsed_seconds=time.perf_counter()-started)
            write_json(state,path)
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    execute(parser.parse_args().workers)

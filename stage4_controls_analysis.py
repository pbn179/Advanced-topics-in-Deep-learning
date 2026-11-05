"""Paired original-ticket versus same-mask reinitialization control analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from data import make_loaders
from models import ARCHITECTURES,MLP
from pruning import weight_statistics
from stage2_6_analysis import history_metrics
from stage4_controls_runner import (ROOT,REPORT,CASES,case_key,verify_case_files,source_case,
    previous_snapshot,tensor_digest,initialization_seed,cpu_state)
from train import evaluate

KEYS=['architecture','seed','stage']
METRICS=['best_validation_accuracy','best_validation_loss','best_iteration','runtime_seconds']


def paired_results(tickets,controls):
    """Pair first, then summarize differences; never combine independent SDs."""
    if tickets.duplicated(KEYS).any() or controls.duplicated(KEYS).any():
        raise ValueError('Duplicate ticket/control pair keys')
    if set(map(tuple,tickets[KEYS].to_numpy()))!=set(map(tuple,controls[KEYS].to_numpy())):
        raise ValueError('Missing corresponding ticket/control observations')
    merged=tickets.merge(controls,on=KEYS,suffixes=('_ticket','_control'),validate='one_to_one')
    for field in ('remaining_weights','remaining_parameters'):
        if not merged[field+'_ticket'].eq(merged[field+'_control']).all():
            raise ValueError('Paired active parameter counts differ')
    merged['ticket_minus_control_accuracy_pp']=100*(merged.best_validation_accuracy_ticket-merged.best_validation_accuracy_control)
    merged['ticket_minus_control_loss']=merged.best_validation_loss_ticket-merged.best_validation_loss_control
    merged['ticket_minus_control_iteration']=merged.best_iteration_ticket-merged.best_iteration_control
    merged['ticket_accuracy_higher']=merged.ticket_minus_control_accuracy_pp>1e-12
    merged['control_accuracy_higher']=merged.ticket_minus_control_accuracy_pp < -1e-12
    merged['accuracy_tied']=merged.ticket_minus_control_accuracy_pp.abs()<=1e-12
    return merged


def aggregate_pairs(pairs):
    grouped=pairs.groupby(['architecture','stage'])
    if not grouped.seed.nunique().eq(3).all() or not grouped.size().eq(3).all():
        raise ValueError('Each paired aggregate needs three distinct model seeds')
    metrics=['ticket_minus_control_accuracy_pp','ticket_minus_control_loss','ticket_minus_control_iteration']
    summary=grouped[metrics].agg(['mean','std','min','max'])
    summary.columns=['_'.join(column) for column in summary.columns]
    summary['seed_count']=grouped.seed.nunique()
    for name in ('ticket_accuracy_higher','control_accuracy_higher','accuracy_tied'):
        summary[name+'_count']=grouped[name].sum()
    summary['remaining_parameters']=grouped.remaining_parameters_ticket.first()
    return summary.reset_index()


def collect_results():
    rows=[]
    for architecture,seed,stage in CASES:
        row=verify_case_files(REPORT/'runs'/case_key(architecture,seed,stage))
        rows.append(row)
    controls=pd.DataFrame(rows).sort_values(KEYS).reset_index(drop=True)
    if len(controls)!=18 or controls.duplicated(KEYS).any() or set(map(tuple,controls[KEYS].to_numpy()))!=set(CASES):
        raise ValueError('Expected exactly 18 approved control cases')
    controls['remaining_parameters']=controls.remaining_weights+controls.parameters-controls.total_weights
    controls['control_late_minimum']=controls.best_iteration>=8000
    controls['control_best_at_final']=controls.best_iteration==10000
    ticket=pd.read_csv(ROOT/'reports'/'stage3'/'individual_results.csv')
    ticket=ticket[ticket.stage.isin((6,12))].copy().sort_values(KEYS).reset_index(drop=True)
    pairs=paired_results(ticket,controls)
    summary=aggregate_pairs(pairs)
    ticket_long=ticket[KEYS+METRICS+['remaining_parameters','remaining_weights']].copy()
    ticket_long['condition']='original_initialization'
    control_long=controls[KEYS+METRICS+['remaining_parameters','remaining_weights']].copy()
    control_long['condition']='random_reinitialization'
    individual=pd.concat([ticket_long,control_long],ignore_index=True)
    grouped=individual.groupby(['architecture','stage','condition'])
    aggregate=grouped[METRICS].agg(['mean','std','min','max'])
    aggregate.columns=['_'.join(column) for column in aggregate.columns]
    aggregate['seed_count']=grouped.seed.nunique()
    aggregate['remaining_parameters']=grouped.remaining_parameters.first()
    aggregate=aggregate.reset_index()
    controls.to_csv(REPORT/'control_results.csv',index=False)
    ticket.to_csv(REPORT/'ticket_measurements.csv',index=False)
    pairs.to_csv(REPORT/'paired_results.csv',index=False)
    summary.to_csv(REPORT/'paired_summary.csv',index=False)
    individual.to_csv(REPORT/'individual_results.csv',index=False)
    aggregate.to_csv(REPORT/'condition_summary.csv',index=False)
    controls[controls.control_late_minimum].to_csv(REPORT/'control_late_minima.csv',index=False)
    histories=[]
    for architecture,seed,stage in CASES:
        source,_,_,_=source_case(architecture,seed,stage)
        for condition,path in [('original_initialization',source/f'history_stage{stage}.csv'),
            ('random_reinitialization',REPORT/'runs'/case_key(architecture,seed,stage)/'history.csv')]:
            frame=pd.read_csv(path)
            if len(frame)!=100 or frame.iteration.tolist()!=list(range(100,10001,100)):
                raise ValueError('Invalid validation trajectory')
            if 'selection_eligible' in frame:
                frame['source_selection_eligible']=frame.selection_eligible
            frame['selection_eligible']=True
            frame['architecture']=architecture;frame['seed']=seed;frame['stage']=stage
            frame['condition']=condition
            histories.append(frame)
    history=pd.concat(histories,ignore_index=True)
    history.to_csv(REPORT/'validation_history.csv',index=False)
    return controls,pairs,summary,individual,aggregate,history


def _save(fig,name):
    directory=REPORT/'figures'
    directory.mkdir(parents=True,exist_ok=True)
    try:
        for extension in ('png','pdf'):
            fig.savefig(directory/f'{name}.{extension}',dpi=300)
    finally:
        plt.close(fig)


def comparison_plot(individual,aggregate,metric,scale,ylabel,name):
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    condition_colors={'original_initialization':'#0072B2','random_reinitialization':'#D55E00'}
    offsets={'original_initialization':-.17,'random_reinitialization':.17}
    for ax,stage in zip(axes,(6,12)):
        for index,architecture in enumerate(('A','B','C')):
            for seed in (0,1,2):
                rows=individual[(individual.architecture==architecture)&(individual.stage==stage)&(individual.seed==seed)]
                values=rows.set_index('condition')[metric]
                jitter=(seed-1)*.045
                ax.plot([index-.17+jitter,index+.17+jitter],
                        scale*values.loc[['original_initialization','random_reinitialization']],
                        color='gray',alpha=.45,linewidth=.8,zorder=1)
            for condition in condition_colors:
                group=aggregate[(aggregate.architecture==architecture)&(aggregate.stage==stage)&(aggregate.condition==condition)].iloc[0]
                ax.errorbar(index+offsets[condition],scale*group[metric+'_mean'],
                    yerr=scale*group[metric+'_std'],fmt='o',capsize=4,color=condition_colors[condition],
                    label=('Ticket' if condition=='original_initialization' else 'Random reinitialization') if index==0 else None)
                rows=individual[(individual.architecture==architecture)&(individual.stage==stage)&(individual.condition==condition)].sort_values('seed')
                ax.scatter(index+offsets[condition]+np.array([-.045,0,.045]),scale*rows[metric],
                    color=condition_colors[condition],s=17,alpha=.75,zorder=3)
        labels=[f'{name}\n{int(aggregate[(aggregate.architecture==name)&(aggregate.stage==stage)].iloc[0].remaining_parameters):,}' for name in ('A','B','C')]
        ax.set(title=f'Pruning round {stage}',ylabel=ylabel,xticks=[0,1,2],xticklabels=labels,
               xlabel='Architecture and active parameter count')
        ax.grid(axis='y',alpha=.2)
        ax.legend(title='Mean ± sample SD, n=3; paired points',fontsize=8)
        if metric=='best_iteration':
            ax.axhline(10000,linestyle=':',color='gray',linewidth=.8)
    _save(fig,name)


def save_figures(pairs,summary,individual,aggregate,history):
    comparison_plot(individual,aggregate,'best_validation_accuracy',100,
        'Validation accuracy at minimum loss (%)','accuracy_ticket_vs_control')
    comparison_plot(individual,aggregate,'best_iteration',1,
        'Iteration of minimum validation loss','min_loss_iteration_ticket_vs_control')
    comparison_plot(individual,aggregate,'best_validation_loss',1,
        'Minimum validation cross entropy','minimum_loss_ticket_vs_control')
    fig,ax=plt.subplots(figsize=(8.5,4.6),layout='constrained')
    labels=[]
    for index,(architecture,stage) in enumerate((a,s) for a in ('A','B','C') for s in (6,12)):
        group=pairs[(pairs.architecture==architecture)&(pairs.stage==stage)].sort_values('seed')
        row=summary[(summary.architecture==architecture)&(summary.stage==stage)].iloc[0]
        ax.errorbar(index,row.ticket_minus_control_accuracy_pp_mean,yerr=row.ticket_minus_control_accuracy_pp_std,
            color='#0072B2',fmt='o',capsize=4)
        for seed,shift,marker in zip((0,1,2),(-.06,0,.06),('o','s','^')):
            ax.scatter(index+shift,group[group.seed==seed].iloc[0].ticket_minus_control_accuracy_pp,
                       marker=marker,color=('C0','C1','C2')[seed],label=f'Seed {seed}' if index==0 else None,s=25)
        labels.append(f'{architecture}, round {stage}')
    ax.axhline(0,color='gray',linewidth=.9)
    ax.set(xticks=range(6),xticklabels=labels,ylabel='Ticket − control accuracy (percentage points)')
    ax.tick_params(axis='x',rotation=20)
    ax.grid(axis='y',alpha=.2)
    ax.legend(title='Points: individual paired differences; mean ± SD')
    _save(fig,'paired_accuracy_differences')
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,stage in enumerate((6,12)):
        for column,architecture in enumerate(('A','B','C')):
            ax=axes[row,column]
            for condition,color in [('original_initialization','#0072B2'),('random_reinitialization','#D55E00')]:
                group=history[(history.architecture==architecture)&(history.stage==stage)&(history.condition==condition)]
                stats=group.groupby('iteration').validation_loss.agg(['mean','std'])
                ax.plot(stats.index,stats['mean'],color=color,linestyle='-' if condition=='original_initialization' else '--',
                        label='Ticket' if condition=='original_initialization' else 'Random reinit')
                ax.fill_between(stats.index,stats['mean']-stats['std'],stats['mean']+stats['std'],color=color,alpha=.12)
            ax.set(title=f'{architecture}, round {stage}',xlabel='Training updates',ylabel='Validation cross entropy')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    _save(fig,'validation_loss_trajectories')


def verify_controls():
    torch.set_num_threads(1)
    from data import set_seed
    references=json.loads((REPORT/'batch_order_reference.json').read_text())
    loaders=make_loaders(ROOT/'datasets',seed=0,split_seed=42,download=False,preload_tensors=True)
    checks=[]
    for architecture,seed,stage in CASES:
        directory=REPORT/'runs'/case_key(architecture,seed,stage)
        result=verify_case_files(directory)
        source,config,masks,ticket=source_case(architecture,seed,stage)
        load=lambda name:torch.load(directory/name,map_location='cpu',weights_only=True)
        fresh,start,best,final=[load(name) for name in ('fresh_initial.pt','start.pt','best_validation.pt','final.pt')]
        expected_seed=initialization_seed(seed,stage)
        set_seed(expected_seed)
        model=MLP(*ARCHITECTURES[architecture])
        expected=cpu_state(model)
        original=torch.load(source/'initial_weights.pt',map_location='cpu',weights_only=True)
        for name,value in expected.items():
            if not torch.equal(value,fresh['state_dict'][name]):
                raise ValueError('Fresh initialization is not reproducible')
        for name,mask in masks.items():
            if torch.equal(expected[name+'.weight'][mask],original[name+'.weight'][mask]):
                raise ValueError('Fresh control weights equal original ticket initialization')
            for state in (start,best,final):
                if not torch.equal(state['masks'][name],mask) or state['state_dict'][name+'.weight'][~mask].count_nonzero():
                    raise ValueError('Control mask differs or masked weights regrew')
            if not torch.equal(start['state_dict'][name+'.weight'][mask],expected[name+'.weight'][mask]):
                raise ValueError('Wrong active start weights')
            if not torch.equal(start['state_dict'][name+'.bias'],expected[name+'.bias']):
                raise ValueError('Wrong bias initialization')
        if tensor_digest(masks)!=result['mask_sha256']:
            raise ValueError('Wrong source mask')
        reference=references[str(seed)]
        if result['batch_order_sha256']!=reference['order_sha256'] or result['training_examples_seen']!=reference['examples']:
            raise ValueError('Control batch order mismatch')
        if result['final_generator_sha256']!=reference['final_generator_sha256']:
            raise ValueError('Control generator state mismatch')
        history=pd.read_csv(directory/'history.csv')
        selected=history_metrics(history)
        if best['iteration']!=selected['best_iteration'] or result['best_iteration']!=selected['best_iteration']:
            raise ValueError('Wrong selected checkpoint iteration')
        model.load_state_dict(best['state_dict'])
        metrics=evaluate(model,loaders['validation']) # verification only; NEVER the test loader
        if not np.isclose(metrics['loss'],result['best_validation_loss'],rtol=0,atol=1e-10) or not np.isclose(metrics['accuracy'],result['best_validation_accuracy'],rtol=0,atol=1e-10):
            raise ValueError('Selected-state validation metrics do not reproduce')
        stats=weight_statistics(model,masks)
        if stats['remaining_weights']!=ticket['remaining_weights']:
            raise ValueError('Active positions/counts do not match ticket')
        if result['mask_checks_during_training']!=100:
            raise ValueError('Missing in-training mask checks')
        checks.append(dict(architecture=architecture,seed=seed,stage=stage,mask_matches_exactly=True,
            fresh_initialization_reproduced=True,batch_order_matches_ticket_protocol=True,
            source_batch_order_basis='Reconstructed exact split/loader RNG protocol; original fetch hashes were not recorded',
            masked_start_best_final_zero=True,in_training_mask_checks=100,
            best_checkpoint_validation_reproduced=True,original_ticket_retrained=False))
    preserved=previous_snapshot()
    result=dict(status='verified',controls=18,original_tickets_retrained=0,test_evaluated=False,
        prior_files_unchanged=preserved,checks=checks,
        note='No pruning masks were regenerated; all controls use copies of saved source masks.')
    (REPORT/'verification.json').write_text(json.dumps(result,indent=2))
    return result


def write_report(controls,pairs,summary,aggregate):
    wins=int(pairs.ticket_accuracy_higher.sum())
    losses=int(pairs.control_accuracy_higher.sum())
    ties=int(pairs.accuracy_tied.sum())
    late=int(controls.control_late_minimum.sum())
    final=int(controls.control_best_at_final.sum())
    if losses:
        conclusion=(f'Original-initialization tickets had higher selected validation accuracy in **{wins}/18** '
            f'pairs, lower accuracy in **{losses}/18**, and tied in **{ties}/18**. '
            'Original initialization was beneficial in many observed cases, but it did not uniformly outperform '
            'random reinitialization. Individual outcomes and each architecture/round\'s three-seed results matter.')
    elif wins:
        conclusion=(f'Original-initialization tickets had higher selected validation accuracy in **{wins}/18** '
            f'pairs, lower accuracy in **{losses}/18**, and tied in **{ties}/18**. '
            'The direction consistently favored original initialization in the observed controls, '
            'but this is descriptive evidence under the locked finite training budget, not a general superiority theorem.')
    else:
        conclusion=(f'Original-initialization tickets were not consistently better: wins **{wins}/18**, '
            f'losses **{losses}/18**, ties **{ties}/18**. The observed data do not justify assuming '
            'original initialization improves accuracy under this protocol.')
    table=['| Model | Round | Active parameters | Ticket accuracy (%), mean ± SD | Control accuracy (%), mean ± SD | Ticket − control (pp), mean ± SD | Ticket wins/ties/losses |',
           '|---|---:|---:|---:|---:|---:|---|']
    learning=['| Model | Round | Ticket minimum loss, mean ± SD | Control minimum loss, mean ± SD | Ticket minimum iteration, mean ± SD | Control minimum iteration, mean ± SD |',
              '|---|---:|---:|---:|---:|---:|']
    for row in summary.sort_values(['stage','architecture']).itertuples():
        ticket=aggregate[(aggregate.architecture==row.architecture)&(aggregate.stage==row.stage)&(aggregate.condition=='original_initialization')].iloc[0]
        control=aggregate[(aggregate.architecture==row.architecture)&(aggregate.stage==row.stage)&(aggregate.condition=='random_reinitialization')].iloc[0]
        table.append(f'| {row.architecture} | {row.stage} | {row.remaining_parameters:,} | '
            f'{100*ticket.best_validation_accuracy_mean:.2f} ± {100*ticket.best_validation_accuracy_std:.2f} | '
            f'{100*control.best_validation_accuracy_mean:.2f} ± {100*control.best_validation_accuracy_std:.2f} | '
            f'{row.ticket_minus_control_accuracy_pp_mean:+.3f} ± {row.ticket_minus_control_accuracy_pp_std:.3f} | '
            f'{row.ticket_accuracy_higher_count}/{row.accuracy_tied_count}/{row.control_accuracy_higher_count} |')
        learning.append(f'| {row.architecture} | {row.stage} | '
            f'{ticket.best_validation_loss_mean:.4f} ± {ticket.best_validation_loss_std:.4f} | '
            f'{control.best_validation_loss_mean:.4f} ± {control.best_validation_loss_std:.4f} | '
            f'{ticket.best_iteration_mean:.0f} ± {ticket.best_iteration_std:.0f} | '
            f'{control.best_iteration_mean:.0f} ± {control.best_iteration_std:.0f} |')
    round_effects=[]
    for stage in (6,12):
        group=pairs[pairs.stage==stage]
        names=[]
        for architecture in ('A','B','C'):
            row=summary[(summary.architecture==architecture)&(summary.stage==stage)].iloc[0]
            names.append(f'{architecture} {row.ticket_minus_control_accuracy_pp_mean:+.3f} ± {row.ticket_minus_control_accuracy_pp_std:.3f} pp')
        round_effects.append(f'- Round {stage}: paired accuracy differences '+', '.join(names)+
            f'; ticket wins {int(group.ticket_accuracy_higher.sum())}/9, losses {int(group.control_accuracy_higher.sum())}/9, ties {int(group.accuracy_tied.sum())}/9.')
    r12=summary[summary.stage==12].sort_values('ticket_minus_control_accuracy_pp_mean',ascending=False)
    largest=r12.iloc[0]
    exception_rows=pairs[pairs.control_accuracy_higher]
    exceptions=', '.join(f'{row.architecture}/seed {row.seed}/round {row.stage} ({row.ticket_minus_control_accuracy_pp:+.3f} pp)' for row in exception_rows.itertuples()) or 'none'
    gap_by_round=pairs.pivot(index=['architecture','seed'],columns='stage',values='ticket_minus_control_accuracy_pp')
    gap_by_round['round12_minus_round6_advantage_pp']=gap_by_round[12]-gap_by_round[6]
    gap_by_round.reset_index().to_csv(REPORT/'benefit_change_with_sparsity.csv',index=False)
    grows=int((gap_by_round['round12_minus_round6_advantage_pp']>1e-12).sum())
    init_sensitivity=(f'The initialization benefit increased from round 6 to round 12 in **{grows}/9** '
        f'architecture–seed trajectories. At round 12, **{largest.architecture}** showed the largest '
        f'observed mean paired accuracy advantage, '
        f'**{largest.ticket_minus_control_accuracy_pp_mean:+.3f} ± {largest.ticket_minus_control_accuracy_pp_std:.3f} pp**. '
        'This identifies stronger initialization sensitivity for those saved masks, not the best absolute '
        'ticket architecture. In Stage 3, B had the strongest dense-relative preservation; here the size '
        'of the initialization advantage is a separate quantity. In particular, C can depend more on its '
        'original initialization while still having lower absolute ticket accuracy than B. The ratios '
        'therefore appear to differ in initialization sensitivity under the specified sparsity/budget, '
        'but active counts/mask geometry differ and only three seeds were tested. No universally optimal '
        'ratio or statistically established cross-architecture superiority follows.')
    iteration_faster=int((pairs.ticket_minus_control_iteration<0).sum())
    lower_loss=int((pairs.ticket_minus_control_loss<0).sum())
    execution=json.loads((REPORT/'execution.json').read_text())
    verification=json.loads((REPORT/'verification.json').read_text())
    tests=(REPORT/'tests.txt').read_text().strip().splitlines()[-1]
    counts=summary[summary.stage==12][['architecture','remaining_parameters']]
    final_counts=', '.join(f'{row.architecture} {row.remaining_parameters:,}' for row in counts.itertuples())
    text=f'''# Stage 4A — same-mask random-reinitialization controls

**Exactly 18 approved controls completed**, across A/B/C × seeds 0/1/2 × pruning rounds 6/12. No original tickets were retrained, no masks regenerated, and the held-out test set was not evaluated. Final tests: **{tests}**.

## Does original initialization help?

{conclusion}

Accuracy exceptions where random reinitialization was higher: **{exceptions}**.

{chr(10).join(table)}

Differences are **ticket minus its corresponding control**, calculated within architecture, model seed and exact mask before aggregation. SD is the **sample SD of the three paired differences**, not a combination of independent SDs or a confidence interval. Accuracy is at each run's own minimum validation loss, sampled every 100 updates. The original metrics are reused from Stage 3's approved fine grid.

{chr(10).join(round_effects)}

## Loss and trainability

{chr(10).join(learning)}

Tickets reached their selected minimum earlier in **{iteration_faster}/18** pairs and had lower minimum validation loss in **{lower_loss}/18**. These counts are descriptive, not 18 independent hypothesis-test replicates. The two pruning rounds share parent experiments, seeds and a fixed validation dataset.

Control minima in updates 8,000–10,000: **{late}/18**; exactly at update 10,000: **{final}/18**. Late control minima can indicate budget-limited learning. Consequently, accuracy gaps establish behavior at the specified **10,000-update budget**, not an asymptotic inability of randomly initialized networks to catch up. The budget was not changed for any architecture or outcome. The original paper's 50,000-update setting remains a documented deviation.

## What this adds to the width-ratio research question

{init_sensitivity}

The control isolates initialization **within each saved mask**: architecture, active positions, data split, exact batch order, optimizer settings and budget are matched. It separates the behavior of the original-initialization IMP ticket from what a fresh initialization can do with the same selected connectivity. The paired tables show how initialization sensitivity differs among the tested width allocations; they must not be read as evidence that one ratio is universally best.

Stage 3's dense-relative preservation and Stage 4's initialization advantage answer different questions. A ratio may preserve its dense accuracy well yet have a small initialization advantage, or show a large ticket/control gap because its random control learns slowly. Compare absolute ticket accuracy, paired gaps, minimum-loss iteration and individual seeds together. At round 12, active counts are {final_counts}; identical rounds are **not exactly parameter-matched across architectures**. Thus cross-architecture differences in control benefit combine allocation and the resulting mask/sparsity geometry. Within each pair the positions and count are exactly identical.

Only three model seeds and one predetermined fresh initialization per mask were used. Mean ± SD and individual paired observations describe these runs; there are no population-level significance claims or post-hoc seed searches. The fixed validation split, many correlated comparisons, finite budget and unevaluated test set limit stronger generalization. This is evidence about initialization under this protocol, not a claim that every ticket beats every random initialization or that any ratio is statistically superior.

## Verification and preservation

- Exact source masks were copied, not regenerated. Start, selected-best and final control states match all saved active positions and keep masked weights zero; masks were also checked at all 100 training validation events.
- Full fresh Gaussian Glorot states were reproduced bit for bit from the predetermined seed rule `(model_seed + 1000003 + round*1009) % 2**32`, with zero biases, and differ from the corresponding original initialization on surviving weights.
- Each control's **observed** training fetch stream and final sampler RNG state match replay of the source ticket's exact split/loader protocol for all 10,000 updates, including partial batches and epoch reshuffles. Original Stage 3 fetch hashes were not recorded, so this is verified reconstruction from the frozen code, versions, split and seeds; no ticket training was replayed.
- A fresh Adam is created for each run. All original source checkpoints/masks and **{verification['prior_files_unchanged']} prior files** remained unchanged by checksum comparison.
- Actual saved control `best_validation.pt` states reproduce the selected validation loss/accuracy when reloaded. This verification used validation only, never the held-out test loader.

See `verification.json`, `protocol.lock.json`, `batch_order_reference.json` and each run's committed artifacts. All tables/figures and full histories are under `reports/stage4_controls/`.

## Can final test evaluation proceed without retraining?

**Not at Stage 3's reported minimum-loss ticket states.** The audit inspected all 117 source stages: only original/start/final 10,000-update tensors were saved, and no selected minimum occurred at update 10,000. Thus **0/117 selected ticket states are currently available** without replay. Histories and metrics cannot substitute for weights.

Saved starts/masks/seeds allow a separately authorized deterministic training replay to each selected iteration, followed by validation-metric reproduction. That would be ticket retraining/replay and is **not performed here**. Do not regenerate masks or silently replace the reported states.

The saved final 10,000-update ticket checkpoints can be evaluated without retraining **if a distinct final-weight test estimand is explicitly chosen**. It would not be the validation-selected evaluation currently reported. Controls now have actual minimum-loss snapshots, so their selected test evaluation would not require control retraining; paired selected-state ticket testing still needs the missing ticket states. See `stage3_checkpoint_audit.csv/md` for the complete audit.

## Runtime and scope

Actual orchestration wall time: **{execution['elapsed_seconds']/60:.2f} minutes**, with three isolated one-thread CPU worker processes. Summed control training/validation wall times: **{controls.runtime_seconds.sum()/60:.2f} minutes**, overlapping under parallel execution and including audit/checkpoint callback overhead. No GPU timings are claimed. Only these 18 control trainings were executed; unit/integration tests use synthetic data and are not extra MNIST experiments. No additional diagnostics, ticket replays or test-set evaluations were launched.
'''
    (REPORT/'discussion.md').write_text(text,encoding='utf-8')
    short=f'''# Short interpretation

{conclusion}

'''+chr(10).join(round_effects)+f'''

{init_sensitivity}

Original tickets selected their minimum loss earlier in {iteration_faster}/18 pairs. {late}/18 controls selected minima in the last 20% of their budget, so the comparison concerns fixed-budget accuracy/trainability, not guaranteed asymptotic superiority.

The evidence quantifies initialization sensitivity for each ratio, but does not by itself establish a universally superior hidden-layer allocation. Active counts differ across architectures, and n=3 supports descriptive paired results rather than strong significance claims.

Stage 3 did not save minimum-loss weight snapshots (0/117 available); validation-selected ticket test evaluation needs separately authorized deterministic replay. Final 10k weights are available for a different, explicitly labeled final-weight evaluation. Control best snapshots were saved. No test evaluation or ticket replay was performed.
'''
    (REPORT/'interpretation.md').write_text(short,encoding='utf-8')
    return text

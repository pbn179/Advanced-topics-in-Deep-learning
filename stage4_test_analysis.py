"""Descriptive held-out comparisons at observed pruning stages, never interpolation."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from checkpoints import sha256
from stage4_test_runner import ROOT, REPORT, TICKETS, CONTROLS, require_gate, preserve_previous

KEYS=['architecture','seed','stage']
CONDITIONS={'original_initialization':'Original initialization','random_reinitialization':'Random reinitialization'}
COLORS={'A':'#0072B2','B':'#D55E00','C':'#009E73'}


def comparisons(frame):
    expected={(a,s,r,'original_initialization') for a,s,r in TICKETS}|{(a,s,r,'random_reinitialization') for a,s,r in CONTROLS}
    keys=KEYS+['condition']
    if len(frame)!=45 or frame.duplicated(keys).any() or set(map(tuple,frame[keys].to_numpy()))!=expected:
        raise ValueError('Expected exactly the 45 approved evaluated states')
    if not frame.test_examples.eq(10000).all() or not frame.test_accuracy.between(0,1).all():
        raise ValueError('Invalid held-out measurement')
    dense=frame[(frame.stage==0)&(frame.condition=='original_initialization')][['architecture','seed','test_accuracy','validation_accuracy']]
    dense=dense.rename(columns={'test_accuracy':'dense_test_accuracy','validation_accuracy':'dense_validation_accuracy'})
    individual=frame.merge(dense,on=['architecture','seed'],validate='many_to_one')
    individual['test_accuracy_change_pp']=100*(individual.test_accuracy-individual.dense_test_accuracy)
    individual['validation_accuracy_change_pp']=100*(individual.validation_accuracy-individual.dense_validation_accuracy)
    ticket=individual[(individual.stage!=0)&(individual.condition=='original_initialization')]
    control=individual[individual.condition=='random_reinitialization']
    pairs=ticket.merge(control,on=KEYS,suffixes=('_ticket','_control'),validate='one_to_one')
    for name in ('remaining_parameters','remaining_weights','mask_sha256'):
        if not pairs[name+'_ticket'].eq(pairs[name+'_control']).all(): raise ValueError('Pair mismatch')
    pairs['ticket_minus_control_test_accuracy_pp']=100*(pairs.test_accuracy_ticket-pairs.test_accuracy_control)
    pairs['ticket_minus_control_validation_accuracy_pp']=100*(pairs.validation_accuracy_ticket-pairs.validation_accuracy_control)
    pairs['ticket_minus_control_test_loss']=pairs.test_loss_ticket-pairs.test_loss_control
    pairs['ticket_accuracy_higher']=pairs.ticket_minus_control_test_accuracy_pp>1e-12
    pairs['control_accuracy_higher']=pairs.ticket_minus_control_test_accuracy_pp < -1e-12
    pairs['accuracy_tied']=pairs.ticket_minus_control_test_accuracy_pp.abs()<=1e-12
    return individual,pairs


def aggregate(frame,keys,metrics):
    grouped=frame.groupby(keys,sort=True)
    if not grouped.seed.nunique().eq(3).all() or not grouped.size().eq(3).all():
        raise ValueError('Every mean/sample SD needs exactly three seeds')
    table=grouped[metrics].agg(['mean','std','min','max'])
    table.columns=['_'.join(x) for x in table.columns]
    table['seed_count']=3
    return table.reset_index()


def save_figure(fig,name):
    path=REPORT/'figures';path.mkdir(parents=True,exist_ok=True)
    for extension in ('png','pdf'): fig.savefig(path/f'{name}.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def stage_comparison(individual,metric,scale,ylabel,name,stages=(0,6,12)):
    fig,axes=plt.subplots(1,len(stages),figsize=(4*len(stages),4.3),layout='constrained',squeeze=False)
    for ax,stage in zip(axes[0],stages):
        subset=individual[individual.stage==stage]
        for i,a in enumerate(('A','B','C')):
            group=subset[subset.architecture==a]
            for condition,offset,marker in [('original_initialization',-.12,'o'),('random_reinitialization',.12,'s')]:
                observations=group[group.condition==condition].sort_values('seed')
                if observations.empty: continue
                values=scale*observations[metric].to_numpy();x=i+offset
                ax.scatter(x+np.linspace(-.035,.035,len(values)),values,c=COLORS[a],marker=marker,s=22,alpha=.6)
                ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),color=COLORS[a],marker=marker,
                    markersize=7,capsize=4,linewidth=1.5,label=CONDITIONS[condition] if i==0 else None)
            if len(group)==6:
                for seed in (0,1,2):
                    pair=group[group.seed==seed]
                    values=[float(pair[pair.condition==c][metric].iloc[0])*scale for c in CONDITIONS]
                    ax.plot([i-.12,i+.12],values,color='.65',lw=.6,zorder=0)
        counts=subset.groupby('architecture').remaining_parameters.first()
        ax.set_xticks(range(3),[f'{a}\n{int(counts[a]):,} params' for a in ('A','B','C')])
        ax.set_title(f'Observed pruning stage {stage}');ax.grid(axis='y',alpha=.2)
        if metric.endswith('change_pp'): ax.axhline(0,color='.45',lw=.8)
        if stage==stages[-1]: ax.legend(fontsize=8,loc='best')
    axes[0,0].set_ylabel(ylabel)
    fig.suptitle('Individual seeds and mean ± sample SD (n=3)',fontsize=11)
    save_figure(fig,name)


def paired_plot(pairs):
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    markers=('o','s','^')
    for ax,stage in zip(axes,(6,12)):
        for i,a in enumerate(('A','B','C')):
            group=pairs[(pairs.architecture==a)&(pairs.stage==stage)].sort_values('seed')
            values=group.ticket_minus_control_test_accuracy_pp.to_numpy()
            for seed,value in enumerate(values):
                ax.scatter(i+(seed-1)*.06,value,c=COLORS[a],marker=markers[seed],s=35,label=f'Seed {seed}' if i==0 else None)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),color=COLORS[a],marker='D',capsize=5,lw=1.5)
        ax.axhline(0,color='.45',lw=1);ax.set_xticks(range(3),('A','B','C'));ax.grid(axis='y',alpha=.2)
        ax.set_title(f'Observed pruning stage {stage}')
    axes[0].set_ylabel('Ticket − control test accuracy (percentage points)')
    axes[1].legend(fontsize=8)
    fig.suptitle('Paired seed differences; diamonds show mean ± sample SD',fontsize=11)
    save_figure(fig,'paired_test_accuracy_differences')


def remaining_plot(individual):
    fig,ax=plt.subplots(figsize=(7.5,4.8),layout='constrained')
    for a in ('A','B','C'):
        for condition,style,marker in [('original_initialization','-','o'),('random_reinitialization','--','s')]:
            group=individual[(individual.architecture==a)&(individual.condition==condition)]
            for seed,observations in group.groupby('seed'):
                observations=observations.sort_values('remaining_parameters')
                ax.plot(observations.remaining_parameters,100*observations.test_accuracy,color=COLORS[a],ls=style,alpha=.18,lw=.8)
            means=group.groupby('stage').agg(x=('remaining_parameters','first'),mean=('test_accuracy','mean'),sd=('test_accuracy','std')).sort_values('x')
            ax.errorbar(means.x,100*means['mean'],yerr=100*means.sd,color=COLORS[a],ls=style,marker=marker,
                capsize=4,label=f'{a}: {CONDITIONS[condition]}')
    ax.set_xscale('log');ax.set_xlabel('Actual active parameters, including dense biases (log scale)')
    ax.set_ylabel('Held-out test accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    ax.set_title('Observed stages only; lines connect measurements, no interpolated test estimates',fontsize=10)
    save_figure(fig,'test_accuracy_vs_remaining_parameters')


def write_reports(individual,pairs,summary,paired_summary):
    execution=json.loads((REPORT/'execution.json').read_text())
    gate=json.loads((REPORT/'validation_gate.json').read_text())
    accuracy=[]
    for row in summary.itertuples():
        accuracy.append(f'| {row.architecture} | {row.stage} | {CONDITIONS[row.condition]} | {100*row.test_accuracy_mean:.3f} ± {100*row.test_accuracy_std:.3f} | {row.test_accuracy_change_pp_mean:+.3f} ± {row.test_accuracy_change_pp_std:.3f} |')
    differences=[]
    for row in paired_summary.itertuples():
        differences.append(f'| {row.architecture} | {row.stage} | {row.ticket_minus_control_test_accuracy_pp_mean:+.3f} ± {row.ticket_minus_control_test_accuracy_pp_std:.3f} | {row.ticket_accuracy_higher_count}/3 |')
    tickets=individual[individual.condition=='original_initialization']
    observed=[]
    for stage in (6,12):
        sub=summary[(summary.stage==stage)&(summary.condition=='original_initialization')]
        best=sub.loc[sub.test_accuracy_change_pp_mean.idxmax()]
        observed.append(f'At stage {stage}, {best.architecture} had the highest mean dense-relative test accuracy change ({best.test_accuracy_change_pp_mean:+.3f} pp).')
    high=paired_summary[paired_summary.stage==12]
    largest=high.loc[high.ticket_minus_control_test_accuracy_pp_mean.idxmax()]
    wins=int(pairs.ticket_accuracy_higher.sum());losses=int(pairs.control_accuracy_higher.sum());ties=int(pairs.accuracy_tied.sum())
    highpairs=pairs[pairs.stage==12]
    sensitivity=[]
    for seed,group in highpairs.groupby('seed'):
        winner=group.loc[group.ticket_minus_control_test_accuracy_pp.idxmax()]
        sensitivity.append(f'seed {seed}: {winner.architecture}')
    b=tickets[(tickets.architecture=='B')&(tickets.stage==12)].sort_values('seed')
    bt='; '.join(f'seed {int(r.seed)}: {r.test_accuracy_change_pp:+.3f} pp' for r in b.itertuples())
    b_wins=[]
    for stage in (6,12):
        group=tickets[tickets.stage==stage]
        leaders=group.loc[group.groupby('seed').test_accuracy_change_pp.idxmax()]
        b_wins.append(int(leaders.architecture.eq('B').sum()))
    counts=individual.groupby(['architecture','stage'])[['remaining_parameters','sparsity','parameter_sparsity']].first().reset_index()
    count_lines=[f'| {r.architecture} | {r.stage} | {int(r.remaining_parameters):,} | {100*r.sparsity:.3f}% | {100*r.parameter_sparsity:.3f}% |' for r in counts.itertuples()]
    gap=lambda a,stage: paired_summary[(paired_summary.architecture==a)&(paired_summary.stage==stage)].iloc[0]
    c6,c12,b12=gap('C',6),gap('C',12),gap('B',12)
    text=f'''# Final held-out research findings

All 27 preselected ticket states were reconstructed only to their recorded minimum-validation-loss iteration. All 27 tickets and 18 saved control states passed validation before any held-out evaluation. The 45 selected states were evaluated on all 10,000 MNIST test images; no test-based selection or hyperparameter changes were made.

## Test accuracy and preservation of each seed's dense baseline

Mean ± sample standard deviation across three model seeds. Accuracy is percent; dense-relative changes are percentage points. Random controls use the corresponding original architecture/seed dense baseline, not a separately trained random dense model.

| Architecture | Observed stage | Initialization | Test accuracy (%) | Change from dense (pp) |
|---|---:|---|---:|---:|
{chr(10).join(accuracy)}

## Same-mask paired initialization comparison

Positive ticket-minus-control differences favor original initialization. Differences are paired before computing sample SD.

| Architecture | Observed stage | Ticket − control test accuracy (pp) | Ticket wins |
|---|---:|---:|---:|
{chr(10).join(differences)}

Original tickets had higher test accuracy in **{wins}/18** pairs, lower accuracy in **{losses}/18**, with **{ties}/18** ties. At stage 12 they won **{int(highpairs.ticket_accuracy_higher.sum())}/9** pairs. The greatest mean stage-12 initialization advantage was **{largest.architecture}: {largest.ticket_minus_control_test_accuracy_pp_mean:+.3f} ± {largest.ticket_minus_control_test_accuracy_pp_std:.3f} pp**. Largest per-seed high-sparsity gaps: {', '.join(sensitivity)}.

## Relation to Stage 3 and Stage 4A

{' '.join(observed)} B's stage-12 individual dense-relative changes were {bt}. Assessing preservation relative to dense and assessing absolute accuracy are different comparisons; the per-seed and aggregate tables report both. B led dense-relative preservation in {b_wins[0]}/3 seeds at stage 6 and {b_wins[1]}/3 at stage 12. All nine stage-12 original tickets exceeded their own dense test accuracy, so these are improvements rather than degradation on this held-out set. B's larger relative gain also reflects its somewhat lower dense baseline; both absolute accuracy and change should be considered.

C's claim to the largest initialization sensitivity **does not hold uniformly on test**. At stage 6 C has the largest mean gap ({c6.ticket_minus_control_test_accuracy_pp_mean:+.3f} pp). At stage 12, B's mean gap ({b12.ticket_minus_control_test_accuracy_pp_mean:+.3f} ± {b12.ticket_minus_control_test_accuracy_pp_std:.3f} pp) is slightly larger than C's ({c12.ticket_minus_control_test_accuracy_pp_mean:+.3f} ± {c12.ticket_minus_control_test_accuracy_pp_std:.3f} pp), reversing the Stage 4A validation mean ordering. This small B–C difference is not evidence of statistical superiority. A larger ticket–control advantage measures initialization sensitivity, not architectural superiority: B has the highest original-ticket mean accuracy at stage 12, while A has the highest random-control mean accuracy there. The ranking therefore depends on initialization condition as well as sparsity and metric.

`validation_test_comparison.csv` and the paired CSV preserve both validation and held-out measurements, allowing direct comparison without treating agreement as a statistical test. Conclusions depend on sparsity level, initialization condition, and whether the metric is absolute accuracy, dense-relative change, or paired initialization benefit. They are descriptive: three model seeds do not establish statistical significance, and all seeds share one dataset split and one test set.

## Active parameters and sparsity

Counts are identical between each ticket and its paired control. These are mask-active positions, not counts of numerically nonzero weights.

| Architecture | Observed stage | Active parameters (including biases) | Weight sparsity | Parameter sparsity |
|---|---:|---:|---:|---:|
{chr(10).join(count_lines)}

## Scope and limitations

Only **observed pruning stages 0, 6 and 12** were evaluated. The prior Stage 3 **16k/20k validation estimates were interpolations** between observed points; they are neither measured models nor test-set results. Actual active parameter counts differ between architectures at the same pruning stage. Counts include unpruned biases; weight sparsity excludes biases, while parameter sparsity includes them. These denominators are separately recorded in CSVs.

All original experiments used **10,000 fixed updates per stage rather than the original paper's 50,000**. Replays stopped only at preselected iterations to recover missing states; they did not change the original training budget, prune models again, or select new checkpoints. Controls were also trained for 10,000 updates. In Stage 4A, 16/18 control minima occurred within the last 20% of that budget, so random reinitialization may have been training-budget limited. These results do not establish asymptotic inferiority of random initialization or a universally superior hidden-layer ratio.

Experimental execution is complete. No further training experiments are automatically launched.
'''
    (REPORT/'findings.md').write_text(text)
    deltas=pd.DataFrame(gate['records'])
    previous=preserve_previous()
    proof=dict(status='verified',reconstructed_tickets=27,verified_controls=18,evaluated_models=45,
        test_examples_per_model=10000,prior_files_unchanged=previous,
        max_validation_loss_difference=float(deltas.loss_absolute_difference.max()),
        max_validation_accuracy_difference=float(deltas.accuracy_absolute_difference.max()),
        loss_absolute_tolerance=gate['loss_atol'],accuracy_absolute_tolerance=gate['accuracy_atol'],
        all_masks_exact=True,all_weights_finite=True,all_masked_weights_zero=True,
        all_ticket_starts_equal_masked_original=True,all_replay_history_prefixes_verified=True,
        checkpoints={r['checkpoint']:r['checkpoint_sha256'] for r in gate['records']},
        original_checkpoints_modified=False,pruning_regenerated=False,test_based_selection=False,
        runtime_seconds=execution['runtime_seconds'])
    (REPORT/'verification.json').write_text(json.dumps(proof,indent=2))
    report=f'''# Verification and reproducibility

- Exactly 27 ticket replays: A/B/C × seeds 0/1/2 × observed stages 0/6/12.
- Exactly 18 saved Stage 4A controls: A/B/C × seeds 0/1/2 × stages 6/12.
- All **45** states passed validation before test evaluation; no failed checkpoint was tested.
- Predeclared absolute tolerance: validation loss **{gate['loss_atol']}**, accuracy **{gate['accuracy_atol']}**; no relative tolerance. With 5,000 validation examples this accuracy tolerance permits no changed total classification count.
- Maximum observed loss discrepancy **{proof['max_validation_loss_difference']:.3g}**; maximum accuracy discrepancy **{proof['max_validation_accuracy_difference']:.3g}**.
- Every replay's entire 100-step validation-history prefix matched historical losses/accuracies within these tolerances, including C seed 0's original fine diagnostic trace. Its old 500-step primary eligibility flags do not alter the recovered 100-step selected iteration.
- Starts equal the saved original initialization with the exact saved mask applied, including restored biases. Source start/final masks match; original commit checksums and frozen source/software versions were checked.
- All reconstructed checkpoint tensors were checked for finite values, masked zeros, and lossless save/reload. Mask persistence was checked at every replay validation point and again after loading selected states.
- Actual replay sample-fetch hashes, sample counts and final DataLoader-generator states matched index-only reconstruction of the original seed/split/loader protocol. Historical Stage 3 fetch hashes were not recorded; the claim is protocol reconstruction, not comparison to an unavailable historical trace.
- Fresh Adam, learning rate .0012, batch size 60, split seed 42, CPU/one Torch thread, exact saved split, unchanged Gaussian-Glorot original starts; original core source SHA256 and versions are in `protocol.lock.json`.
- Test data uses the original [0,1] pixel transform, no augmentation, all 10,000 examples, and the existing example-weighted evaluation code. Test loss is reported as an additional measurement; it did not select weights.
- **{previous}** previous results/report/notebook files retain their SHA256 hashes. `previous_results_snapshot.json` is the preservation manifest. New recovered states live only under `reports/stage4_test/replays/`.
- All new selected-state checksums are in `validation_gate.json` and `verification.json`. Complete replay markers are written after verification; cached complete replays are checksum-checked and skipped. Cached test results are reused for the same checkpoint set, without reselection.
- Wall-clock replay/verification/test execution: **{execution['runtime_seconds']/60:.2f} minutes**, up to three replay worker processes. Package versions match the historical run; cross-platform bitwise replay is not promised. The notebook and spawn-safe runner use pathlib and workers=0 for data loading for Windows/macOS portability.

Reproduce or validate artifacts with `.venv/bin/python stage4_test_runner.py --workers 3`, then `.venv/bin/python stage4_test_analysis.py`, or execute `stage4_test.ipynb`. Completed checkpoints/results are reused. Run `.venv/bin/python -m pytest -q` for unit and synthetic integration tests; test results are saved in `tests.txt`. No synthetic test is an additional MNIST training experiment.
'''
    (REPORT/'reproducibility.md').write_text(report)


def analyze():
    execution=json.loads((REPORT/'execution.json').read_text())
    if execution['status']!='complete': raise ValueError('Final evaluation incomplete')
    records=json.loads((REPORT/'validation_gate.json').read_text())['records'];require_gate(records)
    for row in records:
        if sha256(ROOT/row['checkpoint'])!=row['checkpoint_sha256']: raise ValueError('Selected checkpoint changed')
    raw=pd.DataFrame(json.loads((REPORT/'test_evaluation.json').read_text())['results'])
    individual,pairs=comparisons(raw)
    metrics=['test_accuracy','test_loss','test_accuracy_change_pp','validation_accuracy','validation_accuracy_change_pp','selected_iteration']
    summary=aggregate(individual,['architecture','stage','condition'],metrics)
    counts=individual.groupby(['architecture','stage','condition'])[['remaining_parameters','remaining_weights','sparsity','parameter_sparsity','parameters']].first().reset_index()
    summary=summary.merge(counts,on=['architecture','stage','condition'],validate='one_to_one')
    paired_summary=aggregate(pairs,['architecture','stage'],['ticket_minus_control_test_accuracy_pp','ticket_minus_control_validation_accuracy_pp','ticket_minus_control_test_loss'])
    wins=pairs.groupby(['architecture','stage'])[['ticket_accuracy_higher','control_accuracy_higher','accuracy_tied']].sum().add_suffix('_count').reset_index()
    paired_summary=paired_summary.merge(wins,on=['architecture','stage'],validate='one_to_one')
    individual.to_csv(REPORT/'individual_results.csv',index=False)
    summary.to_csv(REPORT/'summary.csv',index=False)
    pairs.to_csv(REPORT/'paired_results.csv',index=False)
    paired_summary.to_csv(REPORT/'paired_summary.csv',index=False)
    individual[KEYS+['condition','validation_accuracy','test_accuracy','validation_accuracy_change_pp','test_accuracy_change_pp','remaining_parameters','sparsity']].to_csv(REPORT/'validation_test_comparison.csv',index=False)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    stage_comparison(individual,'test_accuracy',100,'Held-out test accuracy (%)','test_accuracy_by_observed_stage')
    stage_comparison(individual,'test_accuracy_change_pp',1,'Test accuracy change from dense (pp)','dense_relative_test_accuracy',stages=(6,12))
    paired_plot(pairs);remaining_plot(individual)
    write_reports(individual,pairs,summary,paired_summary)
    return individual,pairs,summary,paired_summary

if __name__=='__main__': analyze()

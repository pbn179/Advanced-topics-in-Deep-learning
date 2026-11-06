"""One observed-stage research figure and descriptive tables from the new study."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
import numpy as np
import pandas as pd
from study import ROOT,CASES,COUNTS,ARCHITECTURES,approved_run_dir,sha256

KEYS=['architecture','model_seed','pruning_stage']
COLORS={'A':'#0072B2','B':'#D55E00','C':'#009E73','D':'#CC79A7'}
MARKERS={'A':'o','B':'s','C':'^','D':'D'}
STYLES={'A':'-','B':'--','C':'-.','D':':'}
METRICS=['iteration_min_validation_loss','minimum_validation_loss','validation_accuracy',
    'test_accuracy','test_loss','test_accuracy_change_pp','runtime_seconds']


def validate_individual(frame):
    if len(frame)!=156 or frame.duplicated(KEYS).any() or set(map(tuple,frame[KEYS].to_numpy()))!=set(CASES):
        raise ValueError('Exactly 156 observed architecture/seed/stage measurements are required')
    if not frame.initial_parameters.eq(frame.architecture.map(COUNTS)).all():raise ValueError('Initial counts differ from fixed architectures')
    if not np.isfinite(frame[METRICS].to_numpy()).all():raise ValueError('Nonfinite measurement')
    if not frame.test_examples.eq(10000).all() or not frame.test_accuracy.between(0,1).all():raise ValueError('Invalid held-out observation')
    if not frame.iteration_min_validation_loss.between(100,10000).all() or not (frame.iteration_min_validation_loss%100).eq(0).all():raise ValueError('Selected iteration off the validation grid')
    if not np.allclose(frame.remaining_weight_percentage,100*frame.active_weights/frame.total_weights,atol=1e-10,rtol=0):raise ValueError('Remaining weights use incorrect denominator')
    if not np.allclose(frame.parameter_sparsity,1-frame.active_parameters/frame.initial_parameters,atol=1e-12,rtol=0):raise ValueError('Incorrect parameter sparsity')
    for a,(h1,h2) in ARCHITECTURES.items():
        group=frame[frame.architecture==a]
        if not group.hidden1.eq(h1).all() or not group.hidden2.eq(h2).all():raise ValueError('Widths differ from predefined registry')
    return True


def individual_table(run_dir):
    run_dir=approved_run_dir(run_dir)
    status=json.loads((run_dir/'execution.json').read_text())
    if status['status']!='complete':raise ValueError('Training, validation verification and all-stage test evaluation are not complete')
    raw=pd.read_csv(run_dir/'all_measurements.csv')
    frame=raw.rename(columns={'best_iteration':'iteration_min_validation_loss','best_validation_loss':'minimum_validation_loss',
        'best_validation_accuracy':'validation_accuracy','remaining_weights':'active_weights','remaining_percent':'remaining_weight_percentage','sparsity':'weight_sparsity'})
    frame['width_ratio_label']=frame.apply(lambda r:f'{int(r.hidden1)}:{int(r.hidden2)}',axis=1)
    dense=frame[frame.pruning_stage==0][['architecture','model_seed','test_accuracy','validation_accuracy']].rename(columns={'test_accuracy':'dense_test_accuracy','validation_accuracy':'dense_validation_accuracy'})
    frame=frame.merge(dense,on=['architecture','model_seed'],validate='many_to_one')
    frame['test_accuracy_change_pp']=100*(frame.test_accuracy-frame.dense_test_accuracy)
    frame['validation_accuracy_change_pp']=100*(frame.validation_accuracy-frame.dense_validation_accuracy)
    order=KEYS+['hidden1','hidden2','width_ratio','width_ratio_label','initial_parameters','active_parameters','total_weights','active_weights',
        'remaining_weight_percentage','weight_sparsity','parameter_sparsity','iteration_min_validation_loss','minimum_validation_loss',
        'validation_accuracy','test_accuracy','test_loss','test_accuracy_change_pp','validation_accuracy_change_pp','test_examples',
        'selected_checkpoint','selected_checkpoint_sha256','final_checkpoint','original_initialization','mask_sha256',
        'runtime_seconds','evaluation_seconds','iterations','validation_interval','mask_checks','verified',
        'validation_loss_absolute_difference','validation_accuracy_absolute_difference','initialization_sha256',
        'split_seed','protocol_sha256','original_rewind_verified','mask_persistence_verified','final_state_pruning_verified']
    frame=frame[order].sort_values(KEYS).reset_index(drop=True);validate_individual(frame)
    return frame


def aggregate(frame):
    validate_individual(frame);group=frame.groupby(['architecture','pruning_stage'],sort=True)
    if not group.size().eq(3).all() or not group.model_seed.nunique().eq(3).all():raise ValueError('Every point needs exactly three seeds')
    table=group[METRICS+['validation_accuracy_change_pp']].agg(['mean','std','min','max'])
    table.columns=['_'.join(c) for c in table.columns]
    for name in ('hidden1','hidden2','width_ratio','initial_parameters','active_parameters','active_weights','total_weights','remaining_weight_percentage','weight_sparsity','parameter_sparsity'):
        if not group[name].nunique().eq(1).all():raise ValueError('Counts/config differ between seeds')
        table[name]=group[name].first()
    table['seed_count']=3
    return table.reset_index()


def plot_main(summary,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if len(summary)!=52:raise ValueError('All 13 observed stages for each of four architectures must be plotted')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,
        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.25),sharex=True)
    for a in ARCHITECTURES:
        selected=summary[summary.architecture==a].sort_values('remaining_weight_percentage',ascending=False)
        if len(selected)!=13 or set(selected.pruning_stage)!=set(range(13)):raise ValueError('Missing observed stage in figure')
        for ax,metric,scale in [(axes[0],'iteration_min_validation_loss',1),(axes[1],'test_accuracy',100)]:
            ax.errorbar(selected.remaining_weight_percentage,scale*selected[metric+'_mean'],
                yerr=scale*selected[metric+'_std'],color=COLORS[a],marker=MARKERS[a],ls=STYLES[a],
                markersize=3,linewidth=1.15,elinewidth=.6,capsize=1.8,capthick=.6)
    low=float(summary.remaining_weight_percentage.min())
    for ax in axes:
        ax.set_xscale('log');ax.set_xlim(104,low*.95)
        ax.xaxis.set_major_locator(FixedLocator([100,50,25,10,7]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}%'))
        ax.xaxis.set_minor_formatter(NullFormatter());ax.set_xlabel('Percent of Weights Remaining')
        ax.grid(axis='both',color='.87',linewidth=.5)
    upper=float((summary.iteration_min_validation_loss_mean+summary.iteration_min_validation_loss_std).max())
    axes[0].set_ylim(0,max(10500,math.ceil(upper/500)*500))
    axes[0].yaxis.set_major_locator(FixedLocator([0,2000,4000,6000,8000,10000]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x/1000:g}k'))
    axes[0].axhline(10000,color='.4',ls=':',lw=.7,zorder=0)
    axes[0].text(.98,10000,'10k budget',transform=axes[0].get_yaxis_transform(),ha='right',va='bottom',fontsize=6.5,color='.4')
    axes[0].set_ylabel('Early-Stop Iteration (Val.)')
    axes[1].set_ylabel('Accuracy at Early-Stop (Test)%')
    axes[0].set_title('(a)',loc='left',fontsize=9,pad=4);axes[1].set_title('(b)',loc='left',fontsize=9,pad=4)
    handles=[Line2D([],[],color=COLORS[a],marker=MARKERS[a],ls=STYLES[a],markersize=3.5,lw=1.2,
        label=f'{a}: {ARCHITECTURES[a][0]}–{ARCHITECTURES[a][1]}') for a in ARCHITECTURES]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=4,frameon=False,handlelength=2.0,columnspacing=1.5)
    fig.subplots_adjust(top=.83,bottom=.17,left=.085,right=.985,wspace=.32)
    for ext in ('pdf','png'):fig.savefig(output/f'main_results.{ext}',dpi=600,facecolor='white',bbox_inches='tight',pad_inches=.035)
    plt.close(fig)


def preserve_previous(historical_root=None):
    """Optional packaging-time audit; ordinary analysis never needs an old checkout."""
    report=ROOT/'archive/preservation_verification.json'
    if historical_root is None:
        if report.exists():return json.loads(report.read_text())
        return dict(status='Historical checksum manifest retained; old checkout is not needed by the workflow',checked=0)
    old=Path(historical_root)
    manifest=json.loads((ROOT/'archive/previous_artifact_checksums.json').read_text())
    for name,h in manifest.items():
        if not (old/name).is_file() or sha256(old/name)!=h:raise ValueError(f'Old experiment artifact changed: {name}')
    result=dict(status='unchanged',checked=len(manifest),basis='Packaging-time audit of the original study; historical recovery references are in archive/README.md')
    report.write_text(json.dumps(result,indent=2))
    return result


def write_reports(frame,summary,run_dir,results_dir,assets_dir):
    run_dir=approved_run_dir(run_dir);results_dir=Path(results_dir);assets_dir=Path(assets_dir)
    protocol=json.loads((run_dir/'protocol.json').read_text());execution=json.loads((run_dir/'execution.json').read_text())
    gate=json.loads((run_dir/'selection_manifest.json').read_text())
    if gate['count']!=156 or len(gate['records'])!=156 or not frame.verified.eq(True).all():raise ValueError('Incomplete verification gate')
    preservation=preserve_previous()
    warnings=[]
    history=pd.read_csv(run_dir/'validation_history.csv')
    for key,group in history.groupby(KEYS):
        selected=group.loc[group.validation_loss.idxmin()];late=group[group.iteration>=8000]
        slope=float(np.polyfit(late.iteration,late.validation_loss,1)[0]*1000)
        warnings.append(dict(zip(KEYS,key),best_iteration=int(selected.iteration),best_in_last_20pct=bool(selected.iteration>=8000),late_loss_slope_per_1000=slope,late_loss_decreasing=slope<0))
    late=pd.DataFrame(warnings);late.to_csv(results_dir/'training_budget_diagnostics.csv',index=False)
    frame_history=history.sort_values(KEYS+['iteration']);frame_history.to_csv(results_dir/'validation_history.csv',index=False)
    trajectory=[]
    for a in ARCHITECTURES:
        group=summary[summary.architecture==a].sort_values('pruning_stage')
        dense=group[group.pruning_stage==0].iloc[0];sparse=group[group.pruning_stage==12].iloc[0]
        trajectory.append(dict(architecture=a,dense_test_accuracy_mean=dense.test_accuracy_mean,
            sparsest_test_accuracy_mean=sparse.test_accuracy_mean,sparsest_test_accuracy_std=sparse.test_accuracy_std,
            sparsest_dense_change_pp_mean=sparse.test_accuracy_change_pp_mean,sparsest_dense_change_pp_std=sparse.test_accuracy_change_pp_std,
            sparsest_selected_iteration_mean=sparse.iteration_min_validation_loss_mean,
            sparsest_selected_iteration_std=sparse.iteration_min_validation_loss_std,
            lowest_stage_mean_test_accuracy=group.test_accuracy_mean.min(),highest_stage_mean_test_accuracy=group.test_accuracy_mean.max(),
            pruned_stages_mean_test_accuracy=group[group.pruning_stage>0].test_accuracy_mean.mean(),
            sparsest_active_parameters=int(sparse.active_parameters),sparsest_remaining_weight_percentage=sparse.remaining_weight_percentage,
            pruned_stages_with_nonnegative_mean_dense_change=int((group[group.pruning_stage>0].test_accuracy_change_pp_mean>=-1e-12).sum())))
    trajectory=pd.DataFrame(trajectory);trajectory.to_csv(results_dir/'trajectory_summary.csv',index=False)
    lines=[]
    for row in trajectory.itertuples():
        lines.append(f'| {row.architecture} | {100*row.dense_test_accuracy_mean:.3f} | {100*row.sparsest_test_accuracy_mean:.3f} ± {100*row.sparsest_test_accuracy_std:.3f} | {row.sparsest_dense_change_pp_mean:+.3f} ± {row.sparsest_dense_change_pp_std:.3f} | {row.sparsest_selected_iteration_mean:.0f} ± {row.sparsest_selected_iteration_std:.0f} | {row.sparsest_active_parameters:,} |')
    trajectory_lines=[f'| {r.architecture} | {100*r.lowest_stage_mean_test_accuracy:.3f}–{100*r.highest_stage_mean_test_accuracy:.3f} | {100*r.pruned_stages_mean_test_accuracy:.3f} | {r.pruned_stages_with_nonnegative_mean_dense_change}/12 |' for r in trajectory.itertuples()]
    stage_spreads=summary.groupby('pruning_stage').test_accuracy_mean.agg(lambda x:100*(x.max()-x.min()))
    leader_counts={a:0 for a in ARCHITECTURES}
    for stage,group in summary.groupby('pruning_stage'):
        best=group.test_accuracy_mean.max()
        for a in group.loc[np.isclose(group.test_accuracy_mean,best,atol=1e-12,rtol=0),'architecture']:leader_counts[a]+=1
    leader_text=', '.join(f'{a}: {n}/13' for a,n in leader_counts.items())
    summary_text=f'''# Four-architecture exploratory findings

This new study freshly trained **156 stages**: four fixed architectures × three model seeds × 13 observed stages. Every stage received 10,000 Adam updates with 100-update validation. All 156 actual best-validation states passed reload validation before all-stage test evaluation. No historical A/B/C checkpoint or random-control measurement was substituted, and no new random-reinitialization controls were run.

The main figure shows complete observed trajectories of minimum-validation-loss iteration and held-out test accuracy against each architecture's actual remaining-weight percentage. Lines join observed means; bars are **sample SD across three seeds**, not confidence intervals. Identical pruning rounds do not have exactly equal remaining counts.

| Architecture | Dense mean test (%) | Stage 12 test (%) | Stage 12 change from own dense (pp) | Stage 12 selected iteration | Stage 12 active parameters |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

Across the full trajectories, the observed between-architecture mean test-accuracy spread ranged from **{stage_spreads.min():.3f} to {stage_spreads.max():.3f} percentage points**, depending on stage. This describes magnitude, not statistical significance or equal-count comparison.

| Architecture | Range of observed stage-mean test accuracy (%) | Mean across 12 pruned stage means (%) | Pruned stages with nonnegative mean dense-relative change |
|---|---:|---:|---:|
{chr(10).join(trajectory_lines)}

Across the **13 observed stage means**, counts of highest mean test accuracy were {leader_text} (ties count for each tied architecture). This is a descriptive trajectory summary, not 13 independent statistical replications. `trajectory_summary.csv` also reports the range of stage means and the arithmetic mean across the 12 pruned-stage means, which weights pruning rounds equally rather than matching parameter counts. Interpret these alongside the full curves and individual-seed CSV, not as proof of architectural superiority from a selected point.

Width allocation can show different observed accuracy and minimum-loss-iteration trajectories. Their practical size is expressed directly in percentage-point changes and the three-seed variability; no p-values or statistical-significance claims are made. A later selected minimum does not prove failure to converge, and an earlier minimum is not actual early stopping. All models continued to 10,000 updates. {int(late.best_in_last_20pct.sum())}/156 stages selected minima in the last 20% of the budget; diagnostic slopes are recorded without changing the protocol.

**This is an exploratory extension using the established, previously examined MNIST test set.** It is not an independent unseen confirmation of architecture D. Architectures and protocol were fixed before training and were not adjusted after test inspection. Only three model seeds, one data split, one dataset and approximately matched dense sizes were studied. A/B/C reuse the prior predefined seeds and split, so old and new runs must not be pooled as additional independent replications. Surviving parameter counts and mask geometry differ. The budget is **10,000 rather than the original paper's 50,000 updates**. No universally superior ratio is statistically established by this design.
'''
    (results_dir/'findings.md').write_text(summary_text,encoding='utf-8')
    caption='''Four fixed approximately 200k-parameter MNIST MLPs across 12 iterative pruning rounds plus the dense stage. (a) Iteration of minimum validation loss; the dotted line marks the fixed 10,000-update training budget. (b) Test accuracy of the corresponding validation-selected state. The axis wording follows the original paper: 'Early-Stop' denotes retrospective minimum-validation-loss selection, not actual early termination; every model received all 10,000 updates. Every architecture has all 13 observed stages, aggregated over model seeds 0, 1 and 2; error bars show sample standard deviation. The reversed logarithmic x-axis uses actual percentages of original mask-active weights remaining, excluding biases. Lines join observed points only. Checkpoint selection was fixed and verified before test evaluation. This is an exploratory extension using the previously examined MNIST test set, not an independent unseen confirmation.'''
    (assets_dir/'main_results_caption.txt').write_text(caption+'\n')
    (assets_dir/'main_results_caption.tex').write_text('\\caption{'+caption.replace('%','\\%')+'}\n')
    architectures=pd.DataFrame([dict(Architecture=a,Hidden1=w[0],Hidden2=w[1],Ratio={'A':'1:2.60','B':'1:1.02','C':'3:1','D':'6.05:1'}[a],Parameters=COUNTS[a]) for a,w in ARCHITECTURES.items()])
    architectures.to_csv(assets_dir/'architectures.csv',index=False)
    (assets_dir/'architectures.tex').write_text(architectures.to_latex(index=False,escape=True,float_format=lambda x:f'{x:.3f}'))
    proof=dict(status='verified',training_stages=156,selected_checkpoints=156,test_evaluations=156,
        original_parameter_counts=COUNTS,all_starts_exactly_original=True,all_masks_enforced=True,
        all_next_masks_from_final_states=True,all_protocols_identical=True,failed_checkpoints_tested=0,
        validation_loss_absolute_tolerance=1e-8,validation_accuracy_absolute_tolerance=1e-12,
        max_validation_loss_discrepancy=float(frame.validation_loss_absolute_difference.max()),
        max_validation_accuracy_discrepancy=float(frame.validation_accuracy_absolute_difference.max()),
        original_artifact_preservation=preservation,new_controls_run=0,test_set_previously_examined=True,
        training_wall_seconds=execution['elapsed_seconds'],source_sha256=protocol['source_sha256'],
        versions=protocol['versions'],figure_observed_points_per_panel=52,
        late_selected_minima=int(late.best_in_last_20pct.sum()),
        selected_checkpoint_sha256={r['selected_checkpoint']:r['selected_checkpoint_sha256'] for r in gate['records']})
    (results_dir/'verification.json').write_text(json.dumps(proof,indent=2))
    (results_dir/'protocol.json').write_text(json.dumps(protocol,indent=2))
    verification=f'''# Verification and reproducibility

- Four fixed counts including biases: A 200,013; B 200,054; C 199,979; D 200,100. Dense spread is 0.0605% of 200k.
- **156/156** training stages, best-state captures, complete 100-point histories and selected-state test evaluations verified. The new runner drew fresh original initializations; previous checkpoints were not inputs.
- All starts, including biases, equal exact original initialization with masks applied. Layer counts follow floor(20% hidden / 10% output survivor pruning); masks stay monotone and start/best/final masks match. Masked zeros checked at all 100 validation points, and unchanged training code enforces masks after every optimizer step.
- Next masks were independently recomputed from each previous **final 10k state** and compared exactly. Best states were selected by strict validation-loss improvement, retaining the earliest exact tie, and never used for subsequent pruning.
- Every actual best state was reloaded and validation metrics reproduced before test access. Absolute tolerances: loss 1e-8, accuracy 1e-12 (no changed correct-example count allowed). Max loss discrepancy {proof['max_validation_loss_discrepancy']:.3g}; max accuracy discrepancy {proof['max_validation_accuracy_discrepancy']:.3g}.
- SHA256 commits protect initializations, starts, best/final states, histories, metrics and chain provenance. Dataset gzip MD5s match the published MNIST values; extracted bytes match decompression. Fixed split indices and loader seeds are reused per stage; each train call creates fresh Adam.
- All 156 selections were fixed and passed the gate before any new-study test forward pass. Complete test set: 10,000 examples. No checkpoint selection or hyperparameter adjustment used test results. The test set was previously examined in the old study, so this is explicitly exploratory.
- Uniform 10k updates/100-update validation, learning rate .0012, batch 60, CPU single Torch thread per worker, Gaussian Glorot/zero biases, three seeds, split 42. No random-control runs and no old/new measurement mixing.
- The main plot has **52 actual points per panel** (4 × 13), each averaging three seeds; original weights exclude biases in the x denominator. Sample SD is ddof=1. No interpolation, extra measured points, or axis extension toward 1%.
- Previous artifact preservation: {preservation}. Immutable historical artifacts are documented under `archive/`; no main-branch merge or force-push is performed.
- Training/validation/gate/test wall time: {execution['elapsed_seconds']/60:.2f} minutes with up to {execution['workers']} workers. Summed stage training+validation time: {frame.runtime_seconds.sum()/60:.2f} minutes. CPU/software versions are pinned in protocol JSON.

Tests: `python -m pytest -q` (see `tests.txt`). Notebook default execution and figure/CSV integrity checks are reported in `notebook_verification.json` after execution. The full-reproduction path is explicitly enabled and uses a distinct run directory; cached complete stages are checksum-verified and skipped, while interrupted stages restart from masked original initialization with fresh Adam.

Original numerical equality across native Windows/macOS/GPU or different package versions is not promised; the notebook and filesystem paths are portable, and worker-free DataLoaders/spawn-safe orchestration avoid notebook multiprocessing assumptions. Native Windows/macOS execution was not tested here.
'''
    (results_dir/'verification.md').write_text(verification,encoding='utf-8')


def finalize(run_dir='artifacts/main_study',output=None):
    run_dir=approved_run_dir(run_dir)
    target=ROOT if output is None else Path(output).resolve()
    if output is not None and (target==ROOT or ROOT/'archive' in target.parents):raise ValueError('Reproduction outputs must be separate')
    results=target/'results';assets=target/'paper_assets';results.mkdir(parents=True,exist_ok=True);assets.mkdir(parents=True,exist_ok=True)
    frame=individual_table(run_dir);summary=aggregate(frame)
    frame.to_csv(results/'individual_results.csv',index=False);summary.to_csv(results/'aggregated_results.csv',index=False)
    plot_main(summary,assets);write_reports(frame,summary,run_dir,results,assets)
    # Stable essential-result checksums, not generated PDF creation timestamps.
    manifest={str(p.relative_to(target)):sha256(p) for p in (results/'individual_results.csv',results/'aggregated_results.csv',results/'validation_history.csv')}
    (results/'data_manifest.json').write_text(json.dumps(manifest,indent=2))
    return frame,summary


def reproduce_saved(root=ROOT,output=None):
    root=Path(root);results=root/'results';manifest=json.loads((results/'data_manifest.json').read_text())
    for name,h in manifest.items():
        if sha256(root/name)!=h:raise ValueError(f'Saved-result checksum mismatch: {name}')
    frame=pd.read_csv(results/'individual_results.csv');summary=pd.read_csv(results/'aggregated_results.csv')
    recomputed=aggregate(frame)
    for name in recomputed.columns:
        if pd.api.types.is_numeric_dtype(recomputed[name]):
            if not np.allclose(recomputed[name],summary[name],atol=1e-10,rtol=0):raise ValueError(f'Summary mismatch: {name}')
        elif not recomputed[name].eq(summary[name]).all():raise ValueError('Summary key mismatch')
    destination=root/'paper_assets' if output is None else Path(output)
    if destination.resolve()==results.resolve() or results.resolve() in destination.resolve().parents:raise ValueError('Do not write figures over result CSVs')
    plot_main(summary,destination)
    return frame,summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('finalize','plot'))
    parser.add_argument('--run-dir',default='artifacts/main_study');parser.add_argument('--output')
    args=parser.parse_args()
    result=finalize(args.run_dir,args.output) if args.action=='finalize' else reproduce_saved(output=args.output)
    print(f'Individual rows: {len(result[0])}; aggregate rows: {len(result[1])}; observed points per panel: 52')

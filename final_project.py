"""Portable saved-artifact analysis. Never trains or writes experimental reports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
LABELS={'A':'A (162–421)','B':'B (200–204)','C':'C (231–77)'}
COLORS={'A':'#0072B2','B':'#D55E00','C':'#009E73'}
FILES={'main':'reports/stage3/individual_results.csv','main_summary':'reports/stage3/summary_mean_std.csv',
       'control_individual':'reports/stage4_controls/individual_results.csv',
       'control_pairs':'reports/stage4_controls/paired_results.csv',
       'control_summary':'reports/stage4_controls/paired_summary.csv',
       'test':'reports/stage4_test/individual_results.csv','test_summary':'reports/stage4_test/summary.csv',
       'test_pairs':'reports/stage4_test/paired_results.csv','test_paired_summary':'reports/stage4_test/paired_summary.csv'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_artifacts(root=ROOT,check_all=False):
    root=Path(root);manifest=root/'packaging/artifact_manifest.json'
    required=list(FILES.values())+['reports/stage4_test/validation_gate.json','reports/stage4_test/verification.json']
    missing=[n for n in required if not (root/n).is_file()]
    if missing:raise FileNotFoundError('Missing research artifacts:\n'+'\n'.join(missing))
    gate=json.loads((root/'reports/stage4_test/validation_gate.json').read_text())
    if gate.get('checkpoint_count')!=45 or len(gate.get('records',[]))!=45 or not all(r['verified'] for r in gate['records']):
        raise ValueError('Expected 45 previously verified selected checkpoints')
    paths=set(required)
    for r in gate['records']:
        p=root/r['checkpoint']
        if not p.is_file():raise FileNotFoundError(f'Missing selected checkpoint: {r["checkpoint"]}')
        if digest(p)!=r['checkpoint_sha256']:raise ValueError(f'Checkpoint checksum mismatch: {r["checkpoint"]}')
    if not manifest.is_file():raise FileNotFoundError('Missing packaging/artifact_manifest.json; build the release manifest first')
    hashes=json.loads(manifest.read_text())
    if check_all:paths=set(hashes)
    for name in sorted(paths):
        if name not in hashes:raise ValueError(f'Missing release checksum: {name}')
        if not (root/name).is_file():raise FileNotFoundError(f'Missing packaged artifact: {name}')
        if digest(root/name)!=hashes[name]:raise ValueError(f'Packaged artifact checksum mismatch: {name}')
    return dict(selected_checkpoints=45,manifest_files_checked=len(paths),status='verified',training_performed=False)


def load_results(root=ROOT,verify=True):
    root=Path(root)
    if verify:verify_artifacts(root)
    results={k:pd.read_csv(root/name) for k,name in FILES.items()}
    main=results['main']
    expected={(a,s,r) for a in LABELS for s in (0,1,2) for r in range(13)}
    keys=['architecture','seed','stage']
    if len(main)!=117 or main.duplicated(keys).any() or set(map(tuple,main[keys].to_numpy()))!=expected:
        raise ValueError('Expected 117 unique main experiment measurements')
    if not main.iterations.eq(10000).all() or not main.eval_interval.eq(100).all():raise ValueError('Main protocol mismatch')
    if len(results['test'])!=45 or len(results['test_pairs'])!=18:raise ValueError('Incomplete held-out measurements')
    # Independently recompute the approved aggregates, allowing CSV-rounding only.
    grouped=main.groupby(['architecture','stage'])
    summary=results['main_summary'].set_index(['architecture','stage'])
    for metric in ('best_validation_accuracy','best_validation_loss','best_iteration','accuracy_change_pp'):
        for stat in ('mean','std'):
            actual=getattr(grouped[metric],stat)().sort_index()
            if not np.allclose(actual,summary.loc[actual.index,metric+'_'+stat],atol=1e-10,rtol=0):
                raise ValueError('Approved Stage 3 aggregate mismatch')
    test=results['test'];grouped=test.groupby(['architecture','stage','condition'])
    summary=results['test_summary'].set_index(['architecture','stage','condition'])
    for metric in ('test_accuracy','test_accuracy_change_pp'):
        for stat in ('mean','std'):
            actual=getattr(grouped[metric],stat)().sort_index()
            if not np.allclose(actual,summary.loc[actual.index,metric+'_'+stat],atol=1e-10,rtol=0):
                raise ValueError('Approved Stage 4B aggregate mismatch')
    for prefix in ('control','test'):
        frame=results[prefix+'_pairs'];metric='ticket_minus_control_'+('accuracy_pp' if prefix=='control' else 'test_accuracy_pp')
        g=frame.groupby(['architecture','stage'])[metric]
        sm=results[prefix+'_summary' if prefix=='control' else 'test_paired_summary'].set_index(['architecture','stage'])
        for stat in ('mean','std'):
            actual=getattr(g,stat)().sort_index()
            if not np.allclose(actual,sm.loc[actual.index,metric+'_'+stat],atol=1e-10,rtol=0):raise ValueError('Approved paired aggregate mismatch')
    return results


def output_directory(root,output):
    root=Path(root).resolve();path=Path(output)
    if not path.is_absolute():path=root/path
    path=path.resolve()
    protected=[root/'reports',root/'results',root/'archive',root/'datasets']
    if path==root or any(path==p or p in path.parents for p in protected):
        raise ValueError('Output must be separate from original research artifacts')
    path.mkdir(parents=True,exist_ok=True);return path


def savefig(fig,out,name):
    for extension in ('png','pdf'):fig.savefig(out/f'{name}.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def curve(frame,metric,scale,ylabel,out,name):
    fig,ax=plt.subplots(figsize=(7,4.5),layout='constrained')
    for a in LABELS:
        group=frame[frame.architecture==a]
        for _,seed in group.groupby('seed'):
            seed=seed.sort_values('remaining_parameters')
            ax.plot(seed.remaining_parameters,scale*seed[metric],color=COLORS[a],alpha=.22,lw=.8)
        g=group.groupby('stage').agg(x=('remaining_parameters','first'),mean=(metric,'mean'),sd=(metric,'std')).sort_values('x')
        ax.errorbar(g.x,scale*g['mean'],yerr=scale*g.sd,color=COLORS[a],marker='o',markersize=3,capsize=3,label=LABELS[a])
    ax.set_xscale('log');ax.set_xlabel('Actual active parameters (including biases; log scale)')
    ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend(fontsize=9)
    ax.set_title('Individual seeds and mean ± sample SD (n=3)',fontsize=11)
    savefig(fig,out,name)


def paired_figure(frame,metric,out,name,ylabel):
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,stage in zip(axes,(6,12)):
        for i,a in enumerate(LABELS):
            values=frame[(frame.architecture==a)&(frame.stage==stage)].sort_values('seed')[metric].to_numpy()
            for seed,value in enumerate(values):ax.scatter(i+(seed-1)*.06,value,color=COLORS[a],marker=('o','s','^')[seed],s=25,label=f'Seed {seed}' if i==0 else None)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),color=COLORS[a],marker='D',capsize=4)
        ax.axhline(0,color='.5',lw=.8);ax.set_xticks(range(3),list(LABELS));ax.set_title(f'Observed stage {stage}');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel(ylabel);axes[1].legend(fontsize=8)
    fig.suptitle('Paired differences and mean ± sample SD (n=3)',fontsize=11);savefig(fig,out,name)


def condition_figure(frame,metric,scale,ylabel,out,name,stages=(6,12)):
    fig,axes=plt.subplots(1,len(stages),figsize=(4.2*len(stages),4.4),layout='constrained',squeeze=False)
    for ax,stage in zip(axes[0],stages):
        for i,a in enumerate(LABELS):
            group=frame[(frame.architecture==a)&(frame.stage==stage)]
            for cond,offset,marker,label in [('original_initialization',-.12,'o','Original initialization'),('random_reinitialization',.12,'s','Random reinitialization')]:
                values=scale*group[group.condition==cond].sort_values('seed')[metric].to_numpy()
                if not len(values):continue
                ax.scatter(i+offset+np.linspace(-.03,.03,len(values)),values,color=COLORS[a],marker=marker,s=18,alpha=.5)
                ax.errorbar(i+offset,values.mean(),yerr=values.std(ddof=1),color=COLORS[a],marker=marker,capsize=3,label=label if i==0 else None)
            if len(group)==6:
                for seed in (0,1,2):
                    pair=group[group.seed==seed]
                    vals=[float(pair[pair.condition==c][metric].iloc[0])*scale for c in ('original_initialization','random_reinitialization')]
                    ax.plot([i-.12,i+.12],vals,color='.6',alpha=.5,lw=.7)
        counts=frame[frame.stage==stage].groupby('architecture').remaining_parameters.first()
        ax.set_xticks(range(3),[f'{a}\n{int(counts[a]):,} params' for a in LABELS]);ax.set_title(f'Observed stage {stage}');ax.grid(axis='y',alpha=.2)
    axes[0,0].set_ylabel(ylabel);axes[0,-1].legend(fontsize=8)
    fig.suptitle('Individual seeds and mean ± sample SD (n=3)',fontsize=11);savefig(fig,out,name)


def architecture_figure(out):
    fig,axes=plt.subplots(1,3,figsize=(10,3.2),layout='constrained')
    from models import MLP,ARCHITECTURES,parameter_count
    rows=[]
    for ax,a in zip(axes,LABELS):
        h1,h2=ARCHITECTURES[a];count=parameter_count(MLP(h1,h2))
        for i,width in enumerate((784,h1,h2,10)):
            ax.text(i,0,str(width),ha='center',va='center',bbox=dict(boxstyle='round,pad=.45',fc='white',ec=COLORS[a]))
            if i<3:ax.annotate('',xy=(i+.75,0),xytext=(i+.25,0),arrowprops=dict(arrowstyle='->',color='.4'))
        ax.set(xlim=(-.4,3.4),ylim=(-.5,.5));ax.axis('off');ax.set_title(f'{LABELS[a]}\n{count:,} parameters',fontsize=11)
        rows.append(dict(architecture=a,hidden1=h1,hidden2=h2,hidden_width_ratio=h1/h2,parameters=count))
    fig.suptitle('MNIST MLPs: ReLU hidden layers, ten-class linear output',fontsize=11)
    savefig(fig,out,'architecture_comparison');return pd.DataFrame(rows)


def generate_assets(root=ROOT,output='paper_assets',verify=True):
    results=load_results(root,verify);out=output_directory(root,output)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    architecture=architecture_figure(out)
    main=results['main']
    curve(main,'best_validation_accuracy',100,'Validation accuracy at minimum loss (%)',out,'validation_accuracy_vs_parameters')
    curve(main,'best_iteration',1,'Iteration of minimum validation loss',out,'minimum_loss_iteration_vs_parameters')
    curve(main,'accuracy_change_pp',1,'Validation accuracy change from dense (pp)',out,'dense_relative_validation_accuracy')
    control=results['control_individual']
    for metric,scale,ylabel,name in [('best_validation_accuracy',100,'Selected validation accuracy (%)','control_validation_accuracy'),('best_validation_loss',1,'Minimum validation loss','control_minimum_loss'),('best_iteration',1,'Iteration of minimum validation loss','control_trainability')]:
        condition_figure(control,metric,scale,ylabel,out,name)
    paired_figure(results['control_pairs'],'ticket_minus_control_accuracy_pp',out,'paired_validation_initialization_advantage','Ticket − control validation accuracy (pp)')
    condition_figure(results['test'],'test_accuracy',100,'Held-out test accuracy (%)',out,'held_out_test_accuracy',stages=(0,6,12))
    condition_figure(results['test'],'test_accuracy_change_pp',1,'Test accuracy change from dense (pp)',out,'dense_relative_test_accuracy')
    paired_figure(results['test_pairs'],'ticket_minus_control_test_accuracy_pp',out,'paired_test_initialization_advantage','Ticket − control test accuracy (pp)')
    # Compact paper tables accompany the complete machine-readable supplements.
    main_paper=results['main_summary'][['architecture','stage','remaining_parameters','best_validation_accuracy_mean','best_validation_accuracy_std','best_iteration_mean','best_iteration_std','accuracy_change_pp_mean','accuracy_change_pp_std']].copy()
    main_paper[['best_validation_accuracy_mean','best_validation_accuracy_std']]*=100
    main_paper.columns=['Architecture','Stage','Parameters','Accuracy mean (%)','Accuracy SD (%)','Iteration mean','Iteration SD','Dense change mean (pp)','Dense change SD (pp)']
    test_paper=results['test_summary'][['architecture','stage','condition','remaining_parameters','test_accuracy_mean','test_accuracy_std','test_accuracy_change_pp_mean','test_accuracy_change_pp_std']].copy()
    test_paper[['test_accuracy_mean','test_accuracy_std']]*=100
    test_paper.columns=['Architecture','Stage','Initialization','Parameters','Accuracy mean (%)','Accuracy SD (%)','Dense change mean (pp)','Dense change SD (pp)']
    cg=control.groupby(['architecture','stage','condition']).agg(parameters=('remaining_parameters','first'),accuracy_mean=('best_validation_accuracy','mean'),accuracy_sd=('best_validation_accuracy','std')).reset_index()
    cg[['accuracy_mean','accuracy_sd']]*=100
    cg.columns=['Architecture','Stage','Initialization','Parameters','Accuracy mean (%)','Accuracy SD (%)']
    vp=results['control_summary'][['architecture','stage','ticket_minus_control_accuracy_pp_mean','ticket_minus_control_accuracy_pp_std']]
    tp=results['test_paired_summary'][['architecture','stage','ticket_minus_control_test_accuracy_pp_mean','ticket_minus_control_test_accuracy_pp_std']]
    paired_paper=vp.merge(tp,on=['architecture','stage'],validate='one_to_one')
    paired_paper.columns=['Architecture','Stage','Validation gap mean (pp)','Validation gap SD (pp)','Test gap mean (pp)','Test gap SD (pp)']
    tables=dict(results,architectures=architecture,main_paper=main_paper,test_paper=test_paper,control_accuracy_paper=cg,paired_initialization_paper=paired_paper)
    for name,table in tables.items():
        # Exact machine-readable measurements; .tex uses 12 significant digits.
        table=table.copy()
        for column in table.columns:
            if ('directory' in column or column=='checkpoint') and table[column].dtype==object:
                def relative_provenance(value):
                    if not isinstance(value,str):return value
                    for prefix in ('reports/','results/'):
                        marker='/'+prefix
                        if marker in value:return prefix+value.split(marker,1)[1]
                    return value
                table[column]=table[column].map(relative_provenance)
        table.to_csv(out/f'{name}.csv',index=False)
        portable=table.drop(columns=[c for c in table.columns if 'directory' in c or c=='checkpoint'],errors='ignore')
        (out/f'{name}.tex').write_text(portable.to_latex(index=False,escape=True,float_format=lambda x:format(x,'.12g')),encoding='utf-8')
    (out/'README.md').write_text('''# Paper assets

Generated solely from frozen Stage 3, Stage 4A and Stage 4B CSVs. No training or test evaluation occurs. Bars show sample SD across three model seeds, not confidence intervals. Thin lines/dots show individual seeds. Initialization advantages are paired before aggregation. All test points are observed stages 0/6/12; the earlier 16k/20k validation estimates are not test measurements. Active counts include dense biases; weight sparsity in CSVs excludes biases and parameter sparsity includes them. See the project README for limitations.

CSV files retain the exact stored double-precision measurements (subject to standard CSV float round trips). LaTeX tables use 12 significant digits for readability; CSVs are authoritative for numerical reuse. Copied provenance path columns are normalized to project-relative report/result paths; numerical measurements are unchanged. Original reports retain their historical metadata.\n''')
    return results,dict(output=str(out.relative_to(Path(root).resolve())) if out.is_relative_to(Path(root).resolve()) else str(out),figures=11,tables=len(tables),training_performed=False,approved_aggregates_reproduced=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='paper_assets');parser.add_argument('--verify-all',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify_artifacts(check_all=args.verify_all),indent=2))
    print(json.dumps(generate_assets(output=args.output)[1],indent=2))

if __name__=='__main__':main()

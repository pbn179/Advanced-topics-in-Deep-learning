"""Paper-style comparison from frozen CSVs; never trains or evaluates models.

Defaults use the delivered project's report paths; --main/--controls/--test can
instead point to the three individual_results.csv files exported separately.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,FixedLocator
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
KEYS=['architecture','seed','stage']
CONDITIONS=('original_initialization','random_reinitialization')
WIDTHS={'A':'162–421','B':'200–204','C':'231–77'}
COLORS={'A':'#0072B2','B':'#D55E00','C':'#009E73'}


def exact_cases(frame,expected,keys):
    if frame.duplicated(keys).any() or len(frame)!=len(expected) or set(map(tuple,frame[keys].to_numpy()))!=expected:
        raise ValueError('Missing, duplicate or unexpected architecture/seed/stage cases')


def prepare(main_path,control_path,test_path):
    main=pd.read_csv(main_path);controls=pd.read_csv(control_path);test=pd.read_csv(test_path)
    exact_cases(main,set(itertools.product(WIDTHS,(0,1,2),range(13))),KEYS)
    expected_controls={(*case,c) for case in itertools.product(WIDTHS,(0,1,2),(6,12)) for c in CONDITIONS}
    exact_cases(controls,expected_controls,KEYS+['condition'])
    expected_test={(*case,'original_initialization') for case in itertools.product(WIDTHS,(0,1,2),(0,6,12))}
    expected_test|={(*case,'random_reinitialization') for case in itertools.product(WIDTHS,(0,1,2),(6,12))}
    exact_cases(test,expected_test,KEYS+['condition'])
    if not main.iterations.eq(10000).all() or not main.eval_interval.eq(100).all():raise ValueError('Unexpected original training protocol')
    if not test.test_examples.eq(10000).all() or not test.verified.eq(True).all():raise ValueError('Unverified or incomplete held-out results')
    lookup=main[KEYS+['remaining_weights','total_weights','remaining_percent','best_iteration','best_validation_loss','best_validation_accuracy']]
    paired=controls.merge(lookup,on=KEYS,suffixes=('','_source'),validate='many_to_one')
    if not paired.remaining_weights.eq(paired.remaining_weights_source).all():raise ValueError('Control/ticket active-weight counts differ')
    original=paired[paired.condition=='original_initialization']
    for field in ('best_iteration','best_validation_loss','best_validation_accuracy'):
        if not np.allclose(original[field],original[field+'_source'],atol=1e-10,rtol=0):raise ValueError('Original validation measurements differ between supplied files')
    main=main.copy();main['condition']='original_initialization'
    random=paired[paired.condition=='random_reinitialization'].copy()
    cols=KEYS+['condition','remaining_weights','total_weights','remaining_percent','best_iteration','best_validation_loss','best_validation_accuracy']
    validation=pd.concat([main[cols],random[cols]],ignore_index=True)
    expected_percent=100*validation.remaining_weights/validation.total_weights
    if not np.allclose(expected_percent,validation.remaining_percent,atol=1e-10,rtol=0):raise ValueError('Incorrect weight-percent denominator')
    observed=validation.merge(test,on=KEYS+['condition'],suffixes=('_validation','_test'),validate='one_to_one')
    for field in ('remaining_weights','total_weights','remaining_percent'):
        if not np.allclose(observed[field+'_validation'],observed[field+'_test'],atol=1e-10,rtol=0):raise ValueError('Held-out active-weight counts differ')
    for a,b in [('best_iteration','selected_iteration'),('best_validation_loss','validation_loss'),('best_validation_accuracy','validation_accuracy')]:
        if not np.allclose(observed[a],observed[b],atol=1e-10,rtol=0):raise ValueError('Held-out checkpoint selection does not match validation')
    hashes=test[test.stage.isin((6,12))].pivot(index=KEYS,columns='condition',values='mask_sha256')
    if not hashes.original_initialization.eq(hashes.random_reinitialization).all():raise ValueError('Held-out pair masks differ')
    return validation,test


def aggregate(frame,metric,scale):
    group=frame.groupby(['architecture','condition','stage'],sort=True)
    if not group.seed.nunique().eq(3).all() or not group.size().eq(3).all():raise ValueError('Every plotted point requires three seeds')
    if not group.remaining_percent.nunique().eq(1).all():raise ValueError('Weight percentages differ across seeds')
    table=group.agg(weights_remaining_percent=('remaining_percent','first'),mean=(metric,'mean'),sample_sd=(metric,'std'),seed_count=('seed','nunique')).reset_index()
    table[['mean','sample_sd']]*=scale
    table['metric']=metric
    return table


def panel(ax,summary):
    for architecture in WIDTHS:
        for condition in CONDITIONS:
            subset=summary[(summary.architecture==architecture)&(summary.condition==condition)].sort_values('weights_remaining_percent',ascending=False)
            control=condition=='random_reinitialization'
            ax.errorbar(subset.weights_remaining_percent,subset['mean'],yerr=subset.sample_sd,
                color=COLORS[architecture],linestyle=':' if control else '-',marker='s' if control else 'o',
                markersize=3.8,linewidth=1.5,elinewidth=.9,capsize=2.5,zorder=3 if control else 2)
    ax.set_xscale('log');ax.set_xlim(108,6)
    ax.xaxis.set_major_locator(FixedLocator([100,50,25,12.5,6.25]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel('Percent of weights remaining (log scale)')
    ax.grid(True,which='major',color='.84',linewidth=.7)


def draw(iterations,accuracy,output,name,test):
    fig,axes=plt.subplots(1,2,figsize=(11.6,4.8))
    panel(axes[0],iterations);panel(axes[1],accuracy)
    axes[0].set_title('(a) Validation-selected training iteration',fontsize=11)
    axes[0].set_ylabel('Iteration of minimum validation loss')
    axes[0].set_ylim(0,11000)
    axes[0].yaxis.set_major_locator(FixedLocator([0,2000,4000,6000,8000,10000]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x/1000:g}K'))
    axes[0].axhline(10000,color='.45',linestyle='--',lw=.7,zorder=0)
    axes[0].text(98,10200,'Fixed budget: 10K updates',color='.4',fontsize=8,va='bottom')
    axes[1].set_title('(b) Held-out test accuracy' if test else '(b) Validation accuracy',fontsize=11)
    axes[1].set_ylabel(('Test' if test else 'Validation')+' accuracy at selected checkpoint (%)')
    handles=[]
    for a in WIDTHS:
        for c in CONDITIONS:
            control=c=='random_reinitialization'
            handles.append(Line2D([],[],color=COLORS[a],ls=':' if control else '-',marker='s' if control else 'o',markersize=4,label=a+' reinit' if control else f'{a} ({WIDTHS[a]})'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.99),ncol=6,frameon=False,fontsize=9)
    caption='Mean ± sample SD across 3 seeds. Original tickets: rounds 0–12; reinitialization controls: rounds 6 and 12.'
    caption+='\nTest observations: rounds 0, 6 and 12. Lines connect observed means only.' if test else '\nValidation observations every 100 updates; no new training or checkpoint selection.'
    fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=8.5,color='.25')
    fig.subplots_adjust(top=.80,bottom=.23,left=.075,right=.985,wspace=.29)
    for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)


def make_figures(main_path,control_path,test_path,output):
    output=Path(output).resolve()
    if output==ROOT or any((ROOT/name).resolve()==output or (ROOT/name).resolve() in output.parents for name in ('reports','results','archive','datasets')):
        raise ValueError('Do not overwrite original scientific artifacts')
    validation,test=prepare(main_path,control_path,test_path)
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    iterations=aggregate(validation,'best_iteration',1)
    val_accuracy=aggregate(validation,'best_validation_accuracy',100)
    test_accuracy=aggregate(test,'test_accuracy',100)
    draw(iterations,test_accuracy,output,'paper_style_test_comparison',test=True)
    draw(iterations,val_accuracy,output,'paper_style_validation_comparison',test=False)
    pd.concat([iterations,val_accuracy,test_accuracy],ignore_index=True).to_csv(output/'plotted_means_and_sample_sd.csv',index=False)
    validation.to_csv(output/'validation_observations.csv',index=False)
    test.to_csv(output/'test_observations.csv',index=False)
    (output/'README.md').write_text('''# Paper-style model comparison

`paper_style_test_comparison.pdf/png` mirrors the requested two-panel layout: validation-loss-selected iteration (left) and test accuracy of the selected state (right). `paper_style_validation_comparison.pdf/png` uses validation accuracy on the right, providing all 13 observed original-ticket stages.

A = 162–421, B = 200–204, C = 231–77. Solid circles = original initialization; dotted squares = same-mask random reinitialization. Bars show mean ± sample SD (ddof=1) across model seeds 0, 1, 2, not confidence intervals or the original paper's min/max range.

The x-axis uses the actual percent of mask-active **weights** remaining, excluding biases from both numerator and denominator. It is logarithmic and decreases left-to-right. Each architecture uses its own actual counts. This differs from earlier plots of active parameters including biases.

Original-ticket validation curves have all 13 observed stages (0–12). Reinitialization controls exist only at stages 6 and 12. Test results exist only for original stages 0/6/12 and control stages 6/12. Markers identify measurements; connecting lines are visual guides, not measurements or fitted estimates at intermediate sparsities. No interpolation to 16k/20k budgets or extrapolation to 1% weights is used.

Checkpoints were selected by minimum validation loss; training originally continued for 10,000 fixed updates. 'Selected iteration' is not actual early stopping. No training, pruning, checkpoint selection, or test evaluation occurred while making these figures. A 10K budget line appears on the left; a mean-plus-SD bar may extend above the training budget because it is a variability display, not a trained iteration.

Only three seeds were studied. Controls may have been limited by the 10K budget. These figures do not establish statistical significance or a universally superior ratio. Source measurements and plotted summaries are included as CSVs.

Regenerate from the project root:

```sh
python plot_paper_style_comparison.py
```

Or supply independently copied CSVs using `--main PATH --controls PATH --test PATH --output PATH`.
''',encoding='utf-8')
    record=dict(source_sha256={name:hashlib.sha256(Path(path).read_bytes()).hexdigest() for name,path in [('main',main_path),('controls',control_path),('test',test_path)]},
        original_validation_observations=117,random_control_validation_observations=18,test_observations=45,
        plotted_iteration_points=len(iterations),plotted_validation_accuracy_points=len(val_accuracy),plotted_test_accuracy_points=len(test_accuracy),
        sample_sd_ddof=1,mask_hash_pairs_verified=18,training_performed=False,test_reevaluated=False)
    (output/'verification.json').write_text(json.dumps(record,indent=2))
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main',type=Path,default=ROOT/'reports/stage3/individual_results.csv')
    parser.add_argument('--controls',type=Path,default=ROOT/'reports/stage4_controls/individual_results.csv')
    parser.add_argument('--test',type=Path,default=ROOT/'reports/stage4_test/individual_results.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'paper_assets/paper_style_comparison')
    args=parser.parse_args();print(json.dumps(make_figures(args.main,args.controls,args.test,args.output),indent=2))

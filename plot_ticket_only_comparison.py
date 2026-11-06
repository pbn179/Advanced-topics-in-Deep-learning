"""Original-initialization curves only; read saved CSVs, never train or test."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullFormatter
import pandas as pd
from plot_paper_style_comparison import ROOT,WIDTHS,COLORS,aggregate,exact_cases
import itertools


def make_figures():
    source=ROOT/'reports/stage3/individual_results.csv'
    heldout=ROOT/'reports/stage4_test/individual_results.csv'
    main=pd.read_csv(source);test=pd.read_csv(heldout)
    main['condition']='original_initialization'
    test=test[test.condition=='original_initialization'].copy()
    exact_cases(main,set(itertools.product(WIDTHS,(0,1,2),range(13))),['architecture','seed','stage'])
    exact_cases(test,set(itertools.product(WIDTHS,(0,1,2),(0,6,12))),['architecture','seed','stage'])
    assert test.verified.eq(True).all() and test.test_examples.eq(10000).all()
    iteration=aggregate(main,'best_iteration',1)
    validation=aggregate(main,'best_validation_accuracy',100)
    accuracy=aggregate(test,'test_accuracy',100)
    assert len(iteration)==39 and len(accuracy)==9
    output=ROOT/'paper_assets/ticket_only_comparison';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    for right,name,is_test in [(accuracy,'paper_style_test_tickets_only',True),(validation,'paper_style_validation_tickets_only',False)]:
        fig,axes=plt.subplots(1,2,figsize=(11.6,4.8))
        for ax,frame in zip(axes,(iteration,right)):
            assert frame.condition.eq('original_initialization').all()
            for a in WIDTHS:
                group=frame[frame.architecture==a].sort_values('weights_remaining_percent',ascending=False)
                ax.errorbar(group.weights_remaining_percent,group['mean'],yerr=group.sample_sd,
                    color=COLORS[a],marker='o',markersize=3.8,lw=1.5,elinewidth=.9,capsize=2.5)
            ax.set_xscale('log');ax.set_xlim(108,0.9)
            ax.xaxis.set_major_locator(FixedLocator([100,51.4,26.5,13.7,7.1,3.7,1.9,1.0]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'100' if x==100 else f'{x:.1f}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('Percent of Weights Remaining');ax.grid(color='.84',lw=.7)
        axes[0].set_title('(a)',loc='left',fontsize=11)
        axes[0].set_ylabel('Early-Stop Iteration (Val.)')
        axes[0].set_ylim(0,11000)
        axes[0].yaxis.set_major_locator(FixedLocator([0,2000,4000,6000,8000,10000]))
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x/1000:g}K'))
        axes[0].axhline(10000,color='.45',ls='--',lw=.7)
        axes[0].text(98,10200,'Fixed budget: 10K updates',color='.4',fontsize=8,va='bottom')
        axes[1].set_title('(b)',loc='left',fontsize=11)
        axes[1].set_ylabel('Accuracy at Early-Stop (Test)%' if is_test else 'Accuracy at Early-Stop (Val.)%')
        handles=[Line2D([],[],color=COLORS[a],marker='o',lw=1.5,label=f'{a} ({WIDTHS[a]})') for a in WIDTHS]
        fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),ncol=3,frameon=False,fontsize=10)
        caption='Original-initialization tickets only. Mean ± sample SD across 3 seeds; validation iterations: rounds 0–12.'
        caption+='\nTest accuracy: rounds 0, 6 and 12. Lines connect observed means only.' if is_test else '\nValidation accuracy: all 13 observed stages. Lines connect observed means only.'
        caption+='\nEarly-stop = retrospective minimum validation loss. No observations below approximately 7% weights remaining.'
        fig.text(.5,.015,caption,ha='center',va='bottom',fontsize=8.5,color='.25')
        fig.subplots_adjust(top=.80,bottom=.23,left=.075,right=.985,wspace=.29)
        for ext in ('png','pdf'):fig.savefig(output/f'{name}.{ext}',dpi=300,facecolor='white')
        plt.close(fig)
    pd.concat([iteration,validation,accuracy],ignore_index=True).to_csv(output/'plotted_means_and_sample_sd.csv',index=False)
    (output/'README.md').write_text('''# Original-ticket-only report figures

Use `paper_style_test_tickets_only.pdf` for the main report: minimum-validation-loss iteration on the left (all 13 observed original-ticket stages), held-out test accuracy at the selected checkpoint on the right (predefined rounds 0, 6, 12). `paper_style_validation_tickets_only.pdf` is the companion with validation accuracy for all 13 stages on the right. Both have 300-dpi PNG copies.

Only original-initialization A (162–421), B (200–204), C (231–77) curves are shown. No random-reinitialization points, lines or legend entries appear. Bars are mean ± sample SD across seeds 0/1/2, not confidence intervals. Connecting lines are guides between observations. The x-axis is actual percent of mask-active weights remaining, excluding biases, on a descending logarithmic scale. Different architectures retain slightly different counts.

The requested y-axis labels are `Early-Stop Iteration (Val.)` and `Accuracy at Early-Stop (Test)%` (validation companion: `Accuracy at Early-Stop (Val.)%`). Here 'early-stop' means retrospective minimum validation loss, not actual early stopping. The paper's logarithmic x-axis tick labels are 100, 51.4, 26.5, 13.7, 7.1, 3.7, 1.9, 1.0. Actual measured x coordinates are unchanged: the last observed stage retains approximately 7% of weights, so the lower part of the displayed range has no data. No interpolation or extrapolation into that range is performed. Original training used 10,000 fixed updates per stage. No training, new selection, pruning or test evaluation was performed. Source scientific results remain unchanged. Previous figure versions remain available in Git history. Regenerate with `python plot_ticket_only_comparison.py` from the project root.
''',encoding='utf-8')
    record=dict(original_ticket_validation_points=39,original_ticket_test_points=9,seeds_per_point=3,
        reinitialization_curves=0,training_performed=False,test_reevaluated=False,
        x_tick_labels=['100','51.4','26.5','13.7','7.1','3.7','1.9','1.0'],actual_observation_coordinates_unchanged=True,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source,heldout)})
    (output/'verification.json').write_text(json.dumps(record,indent=2))
    files=[ROOT/'plot_ticket_only_comparison.py']+sorted(p for p in output.iterdir() if p.is_file() and p.name!='supplement_manifest.json')
    (output/'supplement_manifest.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2))
    print(json.dumps(record,indent=2))

if __name__=='__main__':make_figures()

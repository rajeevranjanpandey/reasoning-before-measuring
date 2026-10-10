"""Generate complete-study figures exclusively from audited final evidence."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from submission_gate import scientific_inputs
from build_postreview_results import LABELS

ROOT=Path(__file__).resolve().parents[1]


def main():
    issues=scientific_inputs(ROOT)
    if issues:raise RuntimeError(str(issues))
    out=ROOT/'paper/postreview'
    a=json.loads((ROOT/'results/revision/closed_loop_analysis.json').read_text())
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(8.8,3.4))
    for i,row in enumerate(a['contrasts']):
        point=100*row['difference'];low,high=np.array(row['ci9875'])*100
        ax.errorbar(point,i,xerr=[[point-low],[high-point]],fmt='o',capsize=4,
                    color='#0072B2' if row['model']=='qwen' else '#D55E00')
    ax.set_yticks(range(4),[x['model'].capitalize()+' assisted − '+LABELS[x['reference']] for x in a['contrasts']])
    ax.axvline(0,color='#555555',lw=.8)
    ax.axvline(3,color='#555555',lw=.8,ls='--',label='+3 percentage points')
    ax.set_xlabel('Balanced accuracy difference (percentage points)')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'closed_loop_contrasts.{ext}',dpi=300)
    plt.close(fig)
    with (out/'closed_loop_contrasts.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['model','reference','difference_pp','lower_98.75_pp','upper_98.75_pp'])
        for x in a['contrasts']:w.writerow([x['model'],x['reference'],100*x['difference'],*[100*v for v in x['ci9875']]])
    # Descriptive class diagnostics with the same identity/seed denominators.
    records=[json.loads(line) for line in (ROOT/'results/revision/closed_loop.jsonl').open()]
    policies=['forward','rounded_tool','qwen_assisted','granite_assisted','full']
    matrix=np.array([[100*np.mean([r['correct'] for r in records if r['policy']==p and r['label']==c]) for c in range(21)] for p in policies])
    with (out/'closed_loop_class_recall.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['policy','budget',*[f'class_{c}_recall_percent' for c in range(21)]])
        for p,row in zip(policies,matrix):w.writerow([p,41 if p=='full' else 3,*row])
    fig,ax=plt.subplots(figsize=(11,3.4))
    im=ax.imshow(matrix,aspect='auto',vmin=0,vmax=100,cmap='viridis')
    ax.hlines(np.arange(len(policies)-1)+.5,-.5,20.5,colors='white',linewidth=.3,zorder=3)
    ax.vlines(np.arange(20)+.5,-.5,4.5,colors='white',linewidth=.3,zorder=3)
    ax.set_yticks(range(5),[LABELS[p] for p in policies]);ax.set_xticks(range(21),['N',*range(1,21)])
    ax.set_xlabel('Operating class');fig.colorbar(im,ax=ax,label='Recall (%)',pad=.02)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'closed_loop_class_recall.{ext}',dpi=300)
    plt.close(fig)
    theta=np.arange(21)*2*np.pi/21
    fig,axes=plt.subplots(1,2,figsize=(10,5),subplot_kw={'projection':'polar'})
    for ax,model,label in zip(axes,['qwen_assisted','granite_assisted'],['A','B']):
        for p,color in [(model,'#D55E00'),('forward','#009E73'),('rounded_tool','#0072B2')]:
            row=matrix[policies.index(p)]
            ax.plot(np.r_[theta,theta[0]],np.r_[row,row[0]],lw=1.3,label=LABELS[p],color=color)
        ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
        ax.set_xticks(theta,['N',*map(str,range(1,21))],fontsize=8)
        ax.set_ylim(0,100);ax.set_yticks([25,50,75,100]);ax.tick_params(axis='y',labelsize=7)
        ax.set_title(label+'  '+LABELS[model],pad=22,fontsize=11)
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),fontsize=8,frameon=False)
    fig.tight_layout()
    for ext in ('png','pdf','svg'):fig.savefig(out/f'closed_loop_class_spider.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    (out/'supplementary_class_diagnostics.md').write_text('Supplementary Figure S5. Class recall in the selected-predictor study on testing runs 71–90. All rows use twenty identities per class. Model estimates average five decoding seeds within identities. The full row uses 41 sensors, all other rows three. Colors share the 0–100% scale. Thin white dividers delineate classes and policies. These are descriptive estimates without classwise tests. Supplementary Figure S6. Spider comparison of the same class recalls. Every spoke uses the common linear 0–100% scale and retains dataset class order. Connecting lines provide visual comparison, without area scoring or class-adjacency interpretation. Exact values appear in closed_loop_class_recall.csv. Visualization code for both figures was prepared with ChatGPT/Codex from recorded audited results.\n')
    print('Complete-study contrast and class diagnostic figures generated')


if __name__=='__main__':main()

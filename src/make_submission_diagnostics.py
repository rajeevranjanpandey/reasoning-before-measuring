"""Presentation and descriptive diagnostics from the completed pilot only.

Does not read incomplete confirmation outcomes or change the frozen analysis.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/paper01_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'svg.fonttype':'none', 'pdf.fonttype':42})
    analysis = json.loads((ROOT/'results/analysis.json').read_text())
    assert analysis['primary_pilot_complete']
    lookup = {(s['policy'],s['budget']):s for s in analysis['summaries']}
    fig, axes = plt.subplots(1,2,figsize=(9,4.35),sharey=True)
    panels = [('A  Adaptive and validation-selected policies',
               [('forward','Forward selection','#0072B2','o'),('tool','Entropy tool','#D55E00','s'),
                ('random','Random','#009E73','^')]),
              ('B  Other static policies',
               [('fixed','Ascending ID / fixed plans','#0072B2','o'),('mi','Mutual information','#D55E00','s'),
                ('importance','Tree importance','#009E73','^')])]
    for ax,(title,policies) in zip(axes,panels):
        full=lookup['full',41]
        ax.axhspan(full['ci95_low']*100,full['ci95_high']*100,color='.9',zorder=0)
        ax.axhline(full['balanced_accuracy']*100,ls='--',lw=1,color='.3')
        for policy,label,color,marker in policies:
            rows=[lookup[policy,b] for b in (1,3,5)]
            mid=np.array([r['balanced_accuracy'] for r in rows])*100
            low=np.array([r['ci95_low'] for r in rows])*100
            high=np.array([r['ci95_high'] for r in rows])*100
            ax.errorbar([1,3,5],mid,yerr=[mid-low,high-mid],color=color,marker=marker,
                        capsize=4,lw=1.5,markersize=5,label=label)
        ax.set(xlabel='Acquired sensors',xticks=[1,3,5],ylim=(0,50),xlim=(.75,5.25))
        ax.set_title(title,fontsize=10,pad=10)
        ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='upper left',bbox_to_anchor=(0,-.18),frameon=False,fontsize=9)
    axes[0].set_ylabel('Balanced accuracy (%)')
    fig.text(.5,.025,'Error bars: 95% run-family intervals. Dashed line and gray band: 41-sensor reference.',
             ha='center',fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,top=.88,bottom=.34,wspace=.13)
    for ext in ('png','svg','pdf'):
        fig.savefig(ROOT/'paper'/f'accuracy_by_budget.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)
    # Verify all identities and form matrices by averaging decoding seeds within each snapshot.
    policies=['forward','tool','qwen_assisted','granite_assisted']
    names=['Forward selection','Entropy tool','Qwen assisted','Granite assisted']
    records=collections.defaultdict(dict)
    sources=['classical_Testing_1_20.jsonl','qwen_pilot.jsonl','granite_pilot.jsonl']
    for filename in sources:
        for line in (ROOT/'results'/filename).read_text().splitlines():
            r=json.loads(line)
            p=r['model']+'_'+r['policy'] if filename.endswith('_pilot.jsonl') else r['policy']
            if p not in policies or r['budget']!=3:continue
            key=(r['run'],r['label'],r['seed'])
            assert key not in records[p]
            assert r['correct']==int(r['prediction']==r['label'])
            records[p][key]=r['prediction']
    matrices={}
    for p in policies:
        seeds=[17,29,43] if p.endswith('_assisted') else [0]
        expected={(r,c,s) for r in range(1,21) for c in range(21) for s in seeds}
        assert set(records[p])==expected
        matrix=np.zeros((21,21))
        for (run,true,seed),prediction in records[p].items():
            matrix[true,prediction]+=100/(20*len(seeds))
        assert np.allclose(matrix.sum(1),100)
        assert np.isclose(np.trace(matrix)/21,lookup[p,3]['balanced_accuracy']*100)
        matrices[p]=matrix
    spider=json.loads((ROOT/'results/class_recall_spider.json').read_text())
    for p in policies:assert np.allclose(np.diag(matrices[p]),spider['recall_percent'][p])
    supp=ROOT/'paper/supplementary';supp.mkdir(exist_ok=True)
    with PdfPages(supp/'confusion_matrices.pdf') as pdf:
        for number,(p,name) in enumerate(zip(policies,names),1):
            fig=plt.figure(figsize=(8.27,11.69))
            ax=fig.add_axes([.10,.30,.76,.55])
            im=ax.imshow(matrices[p],vmin=0,vmax=100,cmap='Blues',aspect='equal')
            labels=['N']+list(map(str,range(1,21)))
            ax.set(xticks=range(21),xticklabels=labels,yticks=range(21),yticklabels=labels,
                   xlabel='Predicted operating class',ylabel='True operating class')
            ax.tick_params(labelsize=8,length=0,pad=4)
            for i in range(21):
                for j in range(21):
                    v=matrices[p][i,j]
                    if v>0:ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=6.6,
                                   color='white' if v>=55 else 'black')
            ax.set_xticks(np.arange(-.5,21,1),minor=True)
            ax.set_yticks(np.arange(-.5,21,1),minor=True)
            ax.grid(which='minor',color='#e3e7eb',linewidth=.35)
            ax.tick_params(which='minor',length=0)
            cax=fig.add_axes([.885,.335,.022,.475])
            fig.colorbar(im,cax=cax,label='Within-class prediction share (%)')
            fig.text(.1,.945,'Supplementary confusion analysis',fontsize=15,weight='bold')
            fig.text(.1,.910,f'Figure S{number}. {name} at three sensors',fontsize=12)
            caption=(
                'Rows represent the true class and columns the predicted class. N denotes normal\n'
                'operation. IDs 1–20 follow the dataset fault numbering. Every row totals 100%.\n'
                'The diagonal is class recall, matching the main paper’s heatmap and spider chart.\n\n'
                'All panels use the same 420 pilot snapshots in 20 run families. Model results\n'
                'average three decoding seeds within each snapshot before averaging run families.\n'
                'These are descriptive point estimates, without cellwise significance tests or\n'
                'confidence intervals. Blank cells are observed zeroes. Printed percentages are\n'
                'rounded to whole numbers, so displayed row sums can differ from 100%.\n\n'
                'Exact values: results/confusion_matrices.csv. No confirmation outcomes are used.\n'
                'Generated from recorded predictions with code assisted by ChatGPT/Codex.')
            fig.text(.1,.255,caption,fontsize=9.5,va='top',linespacing=1.45)
            fig.text(.5,.045,str(number),ha='center',fontsize=9)
            pdf.savefig(fig)
            fig.savefig(supp/f'confusion_{p}.png',dpi=180)
            fig.savefig(supp/f'confusion_{p}.svg')
            plt.close(fig)
    with (ROOT/'results/confusion_matrices.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['policy','true_class','predicted_class','percent'])
        for p,matrix in matrices.items():
            for i in range(21):
                for j in range(21):w.writerow([p,i,j,float(matrix[i,j])])
    result={'scope':'Completed pilot only. Post-pilot descriptive analysis. No new inference trials.',
            'source_sha256':{n:hashlib.sha256((ROOT/'results'/n).read_bytes()).hexdigest() for n in sources},
            'checks':['exact unique identities','row sums','diagonal matches class recall','balanced accuracy agreement'],
            'normal_prediction_percent':{p:matrices[p][0].tolist() for p in policies},
            'budget_graph_intervals':'Existing 20000 paired run-family bootstrap replicates, pointwise 95%, not simultaneous across curves.'}
    (ROOT/'results/submission_diagnostics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'budget_graph':'95% intervals added','supplement_pages':4,'checks':result['checks']}))


if __name__=='__main__':main()

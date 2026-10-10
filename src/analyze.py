"""Run-family clustered summaries; refuses to call incomplete SLM cells results."""
import csv
import json
from collections import defaultdict
import os
os.environ.setdefault('MPLCONFIGDIR',str(__import__('pathlib').Path('/tmp/industrial_diagnosis_mpl')))
import numpy as np
from experiment import ROOT, SEED

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def run_values(records):
    cells=defaultdict(list)
    for r in records:cells[(r['run'],r['label'])].append(r['correct'])
    assert len(cells)==420, len(cells)
    return np.array([np.mean([np.mean(cells[(run,label)]) for label in range(21)]) for run in range(1,21)])

def main():
    rng=np.random.default_rng(SEED)
    resamples=rng.integers(0,20,size=(20000,20))
    groups=defaultdict(list)
    for row in rows(ROOT/'results/classical_Testing_1_20.jsonl'):
        groups[(row['policy'],row['budget'])].append(row)
    development=[];complete=True
    for model in ('qwen','granite'):
        dev=rows(ROOT/'results'/f'{model}_development.jsonl')
        for variant in ('raw','assisted'):
            cell=[r for r in dev if r['policy']==variant]
            if cell:
                development.append(dict(model=model,variant=variant,n=len(cell),
                    fallback_episodes=sum(r['fallback_count']>0 for r in cell),
                    repair_episodes=sum(any(len(a['attempts'])>1 for a in r['actions']) for r in cell),
                    median_seconds=float(np.median([r['seconds'] for r in cell])),
                    p95_seconds=float(np.quantile([r['seconds'] for r in cell],.95)),
                    max_worker_rss_bytes=max(a.get('rss_peak_bytes',0) for r in cell for z in r['actions'] for a in z['attempts'])))
        for row in rows(ROOT/'results'/f'{model}_fixed_Testing_1_20.jsonl'):
            groups[(model+'_fixed',row['budget'])].append(row)
        pilot=rows(ROOT/'results'/f'{model}_pilot.jsonl')
        if len(pilot)!=2520:
            complete=False;continue
        keys=[(r['policy'],r['seed'],r['run'],r['label']) for r in pilot]
        expected={(v,s,r,c) for v in ('raw','assisted') for s in (17,29,43) for r in range(1,21) for c in range(21)}
        assert len(set(keys))==2520 and set(keys)==expected, (model,'invalid paired identities')
        for row in pilot:groups[(model+'_'+row['policy'],row['budget'])].append(row)
    summaries=[];values={}
    for (policy,budget),records in groups.items():
        v=run_values(records);values[(policy,budget)]=v
        low,high=np.quantile(v[resamples].mean(1),[.025,.975])
        summaries.append(dict(policy=policy,budget=budget,episodes=420,seed_replicates=len(records)//420,
            balanced_accuracy=float(v.mean()),ci95_low=float(low),ci95_high=float(high),
            median_seconds=float(np.median([r['seconds'] for r in records])),
            p95_seconds=float(np.quantile([r['seconds'] for r in records],.95))))
    contrasts=[];decision='PENDING: both full SLM pilot cells are required'
    if complete:
        training=json.loads((ROOT/'results/training_summary.json').read_text())
        for model in ('qwen','granite'):
            for comparator in (training['champion'],model+'_fixed'):
                delta=values[(model+'_assisted',3)]-values[(comparator,3)]
                low,high=np.quantile(delta[resamples].mean(1),[.00625,.99375])
                contrasts.append(dict(model=model,comparator=comparator,difference=float(delta.mean()),
                    simultaneous_low=float(low),simultaneous_high=float(high)))
        positive=all(c['difference']>=.03 and c['simultaneous_low']>0 for c in contrasts)
        # Explicitly test seed direction; seeds remain repeated measurements.
        seed_consistent=True
        for model in ('qwen','granite'):
            records=groups[(model+'_assisted',3)]
            for seed in (17,29,43):
                val=run_values([r for r in records if r['seed']==seed]).mean()
                for comparator in (training['champion'],model+'_fixed'):
                    seed_consistent &= val>values[(comparator,3)].mean()
        if positive and seed_consistent:decision='GO for independent confirmation, subject to resource and mechanism review'
        elif any(c['simultaneous_high']<.03 for c in contrasts):decision='STOP positive-effect hypothesis in this tested configuration; no equivalence conclusion'
        else:decision='INCONCLUSIVE; independent confirmation sample size must be frozen using pilot variance'
    descriptive=[]
    if complete:
        for model in ('qwen','granite'):
            for comparator in ('tool', model+'_raw'):
                delta=values[(model+'_assisted',3)]-values[(comparator,3)]
                low,high=np.quantile(delta[resamples].mean(1),[.025,.975])
                descriptive.append(dict(model=model,comparator=comparator,difference=float(delta.mean()),
                    ci95_low=float(low),ci95_high=float(high),status='post-pilot descriptive; not multiplicity adjusted'))
    result=dict(primary_pilot_complete=complete,decision=decision,bootstrap_replicates=20000,
        bootstrap_unit='simulationRun families with all 21 classes retained',summaries=summaries,
        development=development,contrasts=contrasts,descriptive_contrasts=descriptive)
    (ROOT/'results/analysis.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'results/summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(7,4.2))
    budget_names={'fixed':'Ascending ID / fixed plans','mi':'Mutual information','importance':'Tree importance',
                  'forward':'Forward selection','random':'Random','tool':'Entropy tool'}
    for policy in budget_names:
        series=sorted([s for s in summaries if s['policy']==policy],key=lambda s:s['budget'])
        ax.plot([s['budget'] for s in series],[100*s['balanced_accuracy'] for s in series],marker='o',label=budget_names[policy])
    full=next(s for s in summaries if s['policy']=='full')
    ax.axhline(100*full['balanced_accuracy'],ls='--',color='black',lw=1,label='41-sensor reference')
    ax.set(xlabel='Acquired sensors',ylabel='Balanced accuracy (%)',xticks=[1,3,5],ylim=(0,55))
    ax.spines[['top','right']].set_visible(False);ax.legend(ncol=2,fontsize=8,loc='upper left')
    fig.tight_layout()
    for ext in ('svg','png'):fig.savefig(ROOT/'paper'/f'accuracy_by_budget.{ext}',dpi=180)
    plt.close(fig)
    if complete:
        labels=[c['model'].title()+' vs '+c['comparator'].replace('_',' ') for c in contrasts]
        fig,ax=plt.subplots(figsize=(7,3.5))
        for i,c in enumerate(contrasts):
            mid=100*c['difference'];lo=100*c['simultaneous_low'];hi=100*c['simultaneous_high']
            ax.errorbar(mid,i,xerr=[[max(0,mid-lo)],[max(0,hi-mid)]],fmt='o',color='#0072B2',capsize=4)
        ax.axvline(0,color='black',lw=1);ax.axvline(3,color='#D55E00',ls='--',lw=1,label='3-point target')
        ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Assisted policy difference (percentage points)')
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ('svg','png'):fig.savefig(ROOT/'paper'/f'paired_contrasts.{ext}',dpi=180)
        plt.close(fig)
        # Separate timing scopes and units rather than imply a matched compute frontier.
        timing={row['policy']:row for row in summaries if row['budget']==3}
        fig,axes=plt.subplots(2,1,figsize=(7.4,5.7))
        panels=[
            (['qwen_raw','qwen_assisted','granite_raw','granite_assisted'],
             ['Qwen raw','Qwen assisted','Granite raw','Granite assisted'],1,
             'A  Repeated model episodes','Elapsed time (seconds)'),
            (['forward','tool','qwen_fixed','granite_fixed'],
             ['Forward checkpoint','Entropy tool checkpoint','Qwen fixed-plan replay','Granite fixed-plan replay'],1000,
             'B  Reference measurements','Elapsed time (milliseconds)')]
        chart_rows=[]
        for ax,(policies,names,scale,title,xlabel) in zip(axes,panels):
            for i,(policy,name) in enumerate(zip(policies,names)):
                row=timing[policy];med=scale*row['median_seconds'];tail=scale*row['p95_seconds']
                ax.plot([med,tail],[i,i],color='#9AA0A6',lw=2,zorder=1)
                ax.scatter(med,i,s=42,color='#0072B2',label='Median' if i==0 else None,zorder=3)
                ax.scatter(tail,i,s=40,marker='D',color='#D55E00',label='95th percentile' if i==0 else None,zorder=3)
                ax.annotate(f'{med:.2f}',(med,i),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8,color='#00547F')
                ax.annotate(f'{tail:.2f}',(tail,i),xytext=(0,-13),textcoords='offset points',ha='center',fontsize=8,color='#A74700')
                chart_rows.append(dict(policy=policy,median_seconds=row['median_seconds'],p95_seconds=row['p95_seconds'],display_unit='seconds' if scale==1 else 'milliseconds'))
            ax.set(yticks=range(4),yticklabels=names,xlabel=xlabel,ylim=(3.55,-.55),xlim=(0,max(timing[k]['p95_seconds']*scale for k in policies)*1.12))
            ax.set_title(title,loc='left',fontsize=11,pad=12)
            ax.spines[['top','right','left']].set_visible(False);ax.grid(axis='x',alpha=.2);ax.tick_params(axis='y',length=0)
        axes[0].legend(loc='upper right',fontsize=8,frameon=False)
        fig.tight_layout(h_pad=2.2)
        for ext in ('svg','png'):fig.savefig(ROOT/'paper'/f'accuracy_latency.{ext}',dpi=220)
        plt.close(fig)
        (ROOT/'results/runtime_chart.json').write_text(json.dumps({'statistic':'median and 95th percentile of recorded elapsed time','scope':'descriptive; timing scopes differ; no speedup ratios','rows':chart_rows},indent=2))
    print(json.dumps({'decision':decision,'summaries':len(summaries),'development_cells':len(development)}))

if __name__=='__main__':main()

"""Descriptive scientific figures from audited episode records."""
import collections
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/industrial_diagnosis_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import ROOT

def save(fig,name):
    fig.tight_layout()
    for ext in ('svg','png'):fig.savefig(ROOT/'paper'/f'{name}.{ext}',dpi=200)
    plt.close(fig)

def main():
    audit=json.loads((ROOT/'results/pilot_audit.json').read_text())
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    labels=[];frequencies=[];totals=collections.Counter()
    for model,item in audit['models'].items():
        for variant,c in item['cells'].items():
            labels.append(model.title()+' '+variant)
            counts={int(k):v for k,v in c['sensor_selection_counts'].items()}
            totals.update(counts)
            frequencies.append({k:100*v/c['episodes'] for k,v in counts.items()})
    sensors=sorted(sorted(totals,key=lambda k:(-totals[k],k))[:12])
    matrix=np.array([[row.get(k,0) for k in sensors] for row in frequencies])
    coverage=sum(totals[k] for k in sensors)/sum(totals.values())
    fig,ax=plt.subplots(figsize=(8.5,3.3))
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set(xticks=range(len(sensors)),xticklabels=sensors,yticks=range(4),yticklabels=labels,
           xlabel='Sensor ID (12 most frequently acquired channels)')
    for i in range(4):
        for j in range(len(sensors)):
            label='<1' if 0<matrix[i,j]<1 else f'{matrix[i,j]:.0f}'
            ax.text(j,i,label,ha='center',va='center',fontsize=9,
                    color='white' if matrix[i,j]>=60 else 'black')
    # Thin cell boundaries retain the underlying colour scale.
    ax.hlines(np.arange(matrix.shape[0]-1)+0.5, -0.5, matrix.shape[1]-0.5,
              colors='white', linewidth=0.3, zorder=3)
    ax.vlines(np.arange(matrix.shape[1]-1)+0.5, -0.5, matrix.shape[0]-0.5,
              colors='white', linewidth=0.3, zorder=3)
    fig.colorbar(im,ax=ax,label='Episodes selecting sensor (%)',shrink=.9)
    save(fig,'sensor_selection')
    chosen=['forward','tool','qwen_fixed','qwen_raw','qwen_assisted','granite_raw','granite_assisted','full']
    names=['Forward (B=3)','Entropy tool (B=3)','Fixed plans (B=3)','Qwen raw (B=3)',
           'Qwen assisted (B=3)','Granite raw (B=3)','Granite assisted (B=3)','All sensors (B=41)']
    outcomes=collections.defaultdict(lambda:collections.defaultdict(list))
    for file in ('classical_Testing_1_20','qwen_fixed_Testing_1_20','qwen_pilot','granite_pilot'):
        for line in (ROOT/'results'/f'{file}.jsonl').open():
            r=json.loads(line)
            if r['budget'] not in (3,41):continue
            if file.endswith('_pilot'):policy=r['model']+'_'+r['policy']
            elif file.startswith('qwen_fixed'):policy='qwen_fixed'
            else:policy=r['policy']
            if policy in chosen:outcomes[policy][r['label']].append(r['correct'])
    matrix=np.array([[100*np.mean(outcomes[p][c]) for c in range(21)] for p in chosen])
    assert np.isfinite(matrix).all()
    fig,ax=plt.subplots(figsize=(8.5,4.5))
    im=ax.imshow(matrix,cmap='viridis',vmin=0,vmax=100,aspect='auto')
    ax.set(xticks=range(21),xticklabels=range(21),yticks=range(len(names)),yticklabels=names,
           xlabel='Operating class ID (0 = normal; 1–20 = documented fault classes)')
    ax.tick_params(axis='x',labelsize=8)
    # Thin cell boundaries retain the underlying colour scale.
    ax.hlines(np.arange(matrix.shape[0]-1)+0.5, -0.5, matrix.shape[1]-0.5,
              colors='white', linewidth=0.3, zorder=3)
    ax.vlines(np.arange(matrix.shape[1]-1)+0.5, -0.5, matrix.shape[0]-0.5,
              colors='white', linewidth=0.3, zorder=3)
    fig.colorbar(im,ax=ax,label='Class recall (%)',shrink=.9)
    save(fig,'class_recall')
    (ROOT/'results/diagnostic_figures.json').write_text(json.dumps({'sensor_ids':sensors,
        'selected_sensor_coverage':coverage,'class_policy_order':chosen,'class_recall_percent':matrix.tolist(),
        'scope':'Post-pilot descriptive visualizations; no new independent test cases'},indent=2))
    print(json.dumps({'diagnostic_figures':2,'sensor_coverage':coverage}))

if __name__=='__main__':main()

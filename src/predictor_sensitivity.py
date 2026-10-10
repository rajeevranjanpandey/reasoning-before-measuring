"""Post-pilot replay sensitivity with one fixed alternate classifier.

Uses original acquisition trajectories, not a rerun of closed-loop acquisition.
Only the ExtraTrees maximum depth changes; no test-based tuning is performed.
"""
import collections
import copy
import hashlib
import json
import time
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from experiment import ROOT,SEED,load

def main():
    assert json.loads((ROOT/'results/analysis.json').read_text())['primary_pilot_complete']
    start=time.perf_counter()
    package=joblib.load(ROOT/'results/training.joblib')
    p=copy.copy(package['predictor'])
    x,y,r=load('Training');tr=r<=350;x,y=x[tr],y[tr]
    np.testing.assert_allclose(p.donors,(x-p.mean)/p.scale)
    n,d=x.shape;rng=np.random.default_rng(SEED);masks=[]
    for i in range(16):
        mask=np.zeros((n,d),dtype=np.float32)
        if i%6:
            ix=np.argsort(rng.random((n,d)),axis=1)[:,:i%6]
            np.put_along_axis(mask,ix,1,axis=1)
        masks.append(mask)
    masks.append(np.ones((n,d),dtype=np.float32))
    features=np.concatenate([np.column_stack((p.donors*m,m)) for m in masks]).astype('float32')
    p.model=ExtraTreesClassifier(n_estimators=64,max_depth=None,min_samples_leaf=3,random_state=SEED,n_jobs=1)
    p.model.fit(features,np.tile(y,len(masks)))
    del features,masks
    fit_seconds=time.perf_counter()-start
    joblib.dump(p,ROOT/'results/sensitivity_predictor.joblib')
    x,y,r=load('Testing');keep=r<=20;x,y,r=x[keep],y[keep],r[keep]
    source={(int(run),int(label)):row for row,label,run in zip(x,y,r)}
    cells=collections.defaultdict(list)
    for name in ('classical_Testing_1_20','qwen_fixed_Testing_1_20','granite_fixed_Testing_1_20','qwen_pilot','granite_pilot'):
        for line in (ROOT/'results'/f'{name}.jsonl').open():
            row=json.loads(line)
            if row['budget'] not in (3,41):continue
            if name.endswith('_pilot'):
                policy=row['model']+'_'+row['policy'];sensors=[a['sensor'] for a in row['actions']]
            elif name.startswith(('qwen_fixed','granite_fixed')):
                policy=name.split('_')[0]+'_fixed';sensors=row['trajectory']
            else:policy=row['policy'];sensors=row['trajectory']
            obs={j-1:float(source[(row['run'],row['label'])][j-1]) for j in sensors}
            pred=int(p.predict(obs).argmax())
            cells[(policy,row['budget'])].append({'run':row['run'],'label':row['label'],'seed':row['seed'],
                'prediction':pred,'correct':int(pred==row['label']),'original_correct':row['correct']})
    rng=np.random.default_rng(SEED);resamples=rng.integers(0,20,size=(20000,20))
    summaries=[];values={}
    for (policy,budget),records in cells.items():
        grouped=collections.defaultdict(list)
        for record in records:grouped[(record['run'],record['label'])].append(record['correct'])
        assert len(grouped)==420
        val=np.array([np.mean([np.mean(grouped[(run,label)]) for label in range(21)]) for run in range(1,21)])
        values[(policy,budget)]=val;lo,hi=np.quantile(val[resamples].mean(1),[.025,.975])
        summaries.append({'policy':policy,'budget':budget,'accuracy':float(val.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    contrasts=[]
    for model in ('qwen','granite'):
        for comparator in ('forward',model+'_fixed','tool'):
            delta=values[(model+'_assisted',3)]-values[(comparator,3)]
            lo,hi=np.quantile(delta[resamples].mean(1),[.025,.975])
            contrasts.append({'model':model,'comparator':comparator,'difference':float(delta.mean()),'ci95_low':float(lo),'ci95_high':float(hi)})
    output={'status':'post-pilot replay sensitivity; no new closed-loop policies or independent confirmation',
            'change':'maximum tree depth 14 to unlimited; all other training choices retained',
            'training_runs':[1,350],'test_runs':[1,20],'fit_seconds':fit_seconds,
            'summaries':summaries,'contrasts':contrasts,
            'limitations':'Original acquisition paths used original predictor probabilities and tools. Replay cannot establish how policies would adapt to the alternate predictor.'}
    (ROOT/'results/predictor_sensitivity.json').write_text(json.dumps(output,indent=2))
    with (ROOT/'results/predictor_sensitivity_records.jsonl').open('w') as out:
        for (policy,budget),records in cells.items():
            for record in records:out.write(json.dumps(dict(policy=policy,budget=budget,**record))+'\n')
    print(json.dumps({'sensitivity_complete':True,'fit_seconds':fit_seconds,'policies':len(summaries)}))

if __name__=='__main__':main()

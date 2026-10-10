"""Matched post-review mechanism controls on training development identities.

Uses the original predictor; never changes or reads confirmation outcomes.
"""
import datetime
import json
import os
import time
import numpy as np
import joblib
from experiment import ROOT, SEED, load, state_view
from agents import LocalModel, parse_action, system_prompt

OUT=ROOT/'results/revision'
ARMS=('original','order','remap','anonymous','no_scores','shuffled_scores')

def transform(obs, scores, metadata, arm, index, seed):
    rng=np.random.default_rng(np.random.SeedSequence([20260926,index,seed]))
    mapping={j:j for j in range(41)}
    if arm=='remap':mapping=dict(enumerate(map(int,rng.permutation(41))))
    order=list(range(41))
    if arm=='order':order=list(map(int,rng.permutation(41)))
    meta={mapping[j]:('Process measurement' if arm=='anonymous' else metadata[j]) for j in order}
    shown_obs={mapping[j]:v for j,v in obs.items()}
    legal=[mapping[j] for j in order if j not in obs]
    shown_scores=None
    if arm!='no_scores':
        keys=[j for j in order if j not in obs]
        values=[scores[j] for j in keys]
        if arm=='shuffled_scores':values=list(rng.permutation(values))
        shown_scores={mapping[j]:float(v) for j,v in zip(keys,values)}
    return mapping,meta,shown_obs,legal,shown_scores

def main():
    OUT.mkdir(exist_ok=True)
    pkg=joblib.load(ROOT/'results/training.joblib');p=pkg['predictor']
    x,y,r=load('Training');ix=np.flatnonzero((r>=401)&(r<=420))
    ix=np.random.default_rng(SEED).choice(ix,20,replace=False)
    expected={(a,s,i) for a in ARMS for s in (17,29,43,61,79) for i in range(20)}
    for model in ('qwen','granite'):
        path=OUT/f'{model}_controls.jsonl';done=set()
        if path.exists():
            for line in path.open():
                row=json.loads(line);key=(row['arm'],row['seed'],row['index'])
                if key in done or key not in expected:raise RuntimeError('Unexpected control identity')
                done.add(key)
        endpoint=LocalModel(model,True)
        try:
            (OUT/f'{model}_controls_runtime.json').write_text(json.dumps(endpoint.info,indent=2))
            warm=endpoint.call([{'role':'user','content':'Return exactly {"sensor":1}.'}],17)
            if 'error' in warm:raise RuntimeError(str(warm))
            with path.open('a') as f:
                # Fixed randomized cell order reduces systematic timing-order effects.
                keys=sorted(expected);np.random.default_rng(20260926).shuffle(keys)
                for arm,seed,index in keys:
                    if (arm,seed,index) in done:continue
                    row=x[ix[index]];obs={};actions=[];start=time.perf_counter()
                    for step in range(3):
                        actual_legal=[j for j in range(41) if j not in obs]
                        scores=p.scores(obs,actual_legal)
                        mapping,meta,shown_obs,legal,shown_scores=transform(obs,scores,pkg['metadata'],arm,index,seed)
                        # Compute probabilities on physical IDs, then serialize displayed IDs.
                        state=state_view(obs,actual_legal,3-step,p,pkg['metadata'])
                        state['observed']={str(j+1):float(v) for j,v in shown_obs.items()}
                        state['legal_sensors']=[j+1 for j in legal]
                        if shown_scores is not None:state['expected_entropy_reduction']={str(j+1):round(v,5) for j,v in shown_scores.items()}
                        messages=[{'role':'system','content':system_prompt(meta)}, {'role':'user','content':json.dumps(state,separators=(',',':'))}]
                        attempts=[];chosen=None
                        for attempt in range(2):
                            result=endpoint.call(messages,seed+101*step+1009*attempt,legal_sensors=[j+1 for j in legal])
                            attempts.append(result);chosen=parse_action(result.get('text',''),legal)
                            if chosen is not None:break
                            if 'error' in result:
                                endpoint.close();endpoint=LocalModel(model,True)
                            messages += [{'role':'assistant','content':result.get('text','')},{'role':'user','content':'Invalid response. Return exactly one JSON object with key sensor and a legal integer.'}]
                        fallback=chosen is None
                        actual=next(j for j in pkg['rankings']['mi'] if j in actual_legal) if fallback else {v:k for k,v in mapping.items()}[chosen]
                        assert actual in actual_legal
                        obs[actual]=float(row[actual])
                        actions.append(dict(sensor=actual+1,displayed_sensor=mapping[actual]+1,value=obs[actual],state=state,
                            mapping=mapping,attempts=attempts,fallback=fallback))
                    prob=p.predict(obs)
                    record=dict(model=model,arm=arm,seed=seed,index=index,run=int(r[ix[index]]),label=int(y[ix[index]]),
                        actions=actions,probabilities=prob.tolist(),correct=int(prob.argmax()==y[ix[index]]),seconds=time.perf_counter()-start)
                    f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno());done.add((arm,seed,index))
                    (OUT/'controls_status.json').write_text(json.dumps(dict(model=model,completed=len(done),expected=len(expected),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())))
            assert done==expected
        finally:endpoint.close()
    audit(pkg,x,y,r,ix)

def audit(pkg,x,y,r,ix):
    report={'scope':'post-review development controls; not held-out confirmation; descriptive only','cells':[]}
    for model in ('qwen','granite'):
        records=[json.loads(line) for line in (OUT/f'{model}_controls.jsonl').open()]
        for record in records:
            obs={};idx=ix[record['index']]
            assert record['run']==int(r[idx]) and record['label']==int(y[idx])
            for action in record['actions']:
                scores=pkg['predictor'].scores(obs,[j for j in range(41) if j not in obs])
                mapping,meta,shown,legal,sc=transform(obs,scores,pkg['metadata'],record['arm'],record['index'],record['seed'])
                assert action['state']['observed']=={str(j+1):float(v) for j,v in shown.items()}
                assert action['state']['legal_sensors']==[j+1 for j in legal]
                assert action['state'].get('expected_entropy_reduction')==(None if sc is None else {str(j+1):round(v,5) for j,v in sc.items()})
                j=action['sensor']-1
                assert j not in obs and action['value']==float(x[idx,j]) and mapping[j]+1==action['displayed_sensor']
                parsed=[parse_action(z.get('text',''),legal) for z in action['attempts']]
                selected=next((z for z in parsed if z is not None),None)
                assert action['fallback']==(selected is None)
                if selected is not None:assert selected==mapping[j]
                else:assert j==next(k for k in pkg['rankings']['mi'] if k not in obs)
                np.testing.assert_allclose(action['state']['diagnosis_probabilities'],np.round(pkg['predictor'].predict(obs),5),rtol=0,atol=1e-12)
                obs[j]=action['value']
            prob=pkg['predictor'].predict(obs)
            np.testing.assert_allclose(prob,record['probabilities'],rtol=0,atol=1e-12)
            assert record['correct']==int(prob.argmax()==record['label'])
        for arm in ARMS:
            rows=[a for a in records if a['arm']==arm]
            original={(a['seed'],a['index']):a for a in records if a['arm']=='original'}
            changed=[tuple(z['sensor'] for z in a['actions']) != tuple(z['sensor'] for z in original[(a['seed'],a['index'])]['actions']) for a in rows]
            report['cells'].append(dict(model=model,arm=arm,n=len(rows),fraction_paths_changed=float(np.mean(changed)),
                accuracy=float(np.mean([a['correct'] for a in rows])),metric_note='development sample is not class-balanced; accuracy is not balanced accuracy',
                accuracy_by_seed={str(s):float(np.mean([a['correct'] for a in rows if a['seed']==s])) for s in (17,29,43,61,79)}))
    report['audit_passed']=True
    (OUT/'controls_analysis.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()

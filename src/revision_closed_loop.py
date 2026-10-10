"""Independent B=3 evaluation with a validation-selected predictor on test runs 71-90."""
import json
import os
import time
import numpy as np
import joblib
from experiment import ROOT, load, state_view
from agents import LocalModel, parse_action, system_prompt
from revision_diagnostics import metrics

OUT=ROOT/'results/revision'

def main():
    p=joblib.load(OUT/'selected_predictor.joblib')
    pkg=joblib.load(ROOT/'results/training.joblib')
    x,y,r=load('Training');take=(r>=351)&(r<=400);xv,yv=x[take],y[take]
    # Re-select forward order for this predictor using validation only.
    order=[]
    for _ in range(3):
        candidates=[(metrics(p.batch(xv,order+[j]),yv)['balanced_accuracy'],-j) for j in range(41) if j not in order]
        order.append(-max(candidates)[1])
    (OUT/'closed_loop_setup.json').write_text(json.dumps(dict(forward_order=[j+1 for j in order],test_runs=[71,90],budget=3),indent=2))
    x,y,r=load('Testing');take=(r>=71)&(r<=90);x,y,r=x[take],y[take],r[take]
    identities={(int(run),int(label)) for run,label in zip(r,y)}
    assert len(identities)==420
    path=OUT/'closed_loop.jsonl';done=set()
    if path.exists():
        for line in path.open():
            a=json.loads(line);key=(a['policy'],a['seed'],a['run'],a['label'])
            if key in done:raise RuntimeError('Duplicate closed-loop record')
            done.add(key)
    with path.open('a') as f:
        for policy in ('forward','rounded_tool','full','qwen_raw','qwen_assisted','granite_raw','granite_assisted'):
            endpoint=None
            try:
                if '_' in policy and policy!='rounded_tool':
                    endpoint=LocalModel(policy.split('_')[0],True)
                    (OUT/f'{policy}_runtime.json').write_text(json.dumps(endpoint.info,indent=2))
                    warm=endpoint.call([{'role':'user','content':'Return exactly {"sensor":1}.'}],17)
                    if 'error' in warm:raise RuntimeError(str(warm))
                for seed in ((17,29,43,61,79) if endpoint else (0,)):
                    for row,label,run in zip(x,y,r):
                        key=(policy,seed,int(run),int(label))
                        if key in done:continue
                        start=time.perf_counter();obs={};actions=[]
                        for step in range(41 if policy=='full' else 3):
                            legal=[j for j in range(41) if j not in obs]
                            attempts=[];state=None;fallback=False
                            if policy=='full':j=step
                            elif policy=='forward':j=order[step]
                            elif policy=='rounded_tool':
                                scores=p.scores(obs,legal);j=max(legal,key=lambda j:(round(scores[j],5),-j))
                            else:
                                scores=p.scores(obs,legal) if policy.endswith('assisted') else None
                                state=state_view(obs,legal,3-step,p,pkg['metadata'],scores)
                                messages=[{'role':'system','content':system_prompt(pkg['metadata'])},{'role':'user','content':json.dumps(state,separators=(',',':'))}]
                                j=None
                                for attempt in range(2):
                                    result=endpoint.call(messages,seed+step*101+attempt*1009,legal_sensors=[k+1 for k in legal]);attempts.append(result)
                                    j=parse_action(result.get('text',''),legal)
                                    if j is not None:break
                                    if 'error' in result:
                                        endpoint.close();endpoint=LocalModel(policy.split('_')[0],True)
                                    messages += [{'role':'assistant','content':result.get('text','')},{'role':'user','content':'Invalid response. Return exactly one JSON object with key sensor and a legal integer.'}]
                                if j is None:j=next(j for j in pkg['rankings']['mi'] if j in legal);fallback=True
                            obs[j]=float(row[j]);actions.append(dict(sensor=j+1,value=obs[j],state=state,attempts=attempts,fallback=fallback))
                        prob=p.predict(obs)
                        a=dict(policy=policy,seed=seed,run=int(run),label=int(label),actions=actions,probabilities=prob.tolist(),
                            prediction=int(prob.argmax()),correct=int(prob.argmax()==label),seconds=time.perf_counter()-start)
                        f.write(json.dumps(a)+'\n');f.flush();os.fsync(f.fileno());done.add(key)
                        (OUT/'closed_loop_status.json').write_text(json.dumps(dict(completed=len(done),expected=9660,policy=policy)))
            finally:
                if endpoint:endpoint.close()
    expected={(pol,s,run,c) for pol in ('forward','rounded_tool','full','qwen_raw','qwen_assisted','granite_raw','granite_assisted') for s in ((17,29,43,61,79) if pol.startswith(('qwen','granite')) else (0,)) for run,c in identities}
    assert done==expected
    records=[json.loads(line) for line in path.open()]
    source={(int(run),int(label)):row for run,label,row in zip(r,y,x)}
    for a in records:
        obs={};row=source[(a['run'],a['label'])]
        for step,action in enumerate(a['actions']):
            j=action['sensor']-1;assert j not in obs and action['value']==float(row[j])
            legal=[k for k in range(41) if k not in obs]
            if action['state'] is not None:
                scores=p.scores(obs,legal) if a['policy'].endswith('assisted') else None
                assert action['state']==state_view(obs,legal,3-step,p,pkg['metadata'],scores)
                parsed=[parse_action(z.get('text',''),legal) for z in action['attempts']]
                selected=next((z for z in parsed if z is not None),None)
                assert action['fallback']==(selected is None)
                if selected is not None:assert j==selected
                else:assert j==next(k for k in pkg['rankings']['mi'] if k in legal)
            elif a['policy']=='forward':assert j==order[step]
            elif a['policy']=='full':assert j==step
            elif a['policy']=='rounded_tool':
                scores=p.scores(obs,legal);assert j==max(legal,key=lambda k:(round(scores[k],5),-k))
            obs[j]=action['value']
        prob=p.predict(obs);np.testing.assert_allclose(prob,a['probabilities'],rtol=0,atol=1e-12)
        assert a['prediction']==int(prob.argmax()) and a['correct']==int(prob.argmax()==a['label'])
    rng=np.random.default_rng(20260926);bs=rng.integers(0,20,size=(20000,20));ss=rng.integers(0,5,size=(20000,5))
    summary=[];values={}
    for policy in sorted({a['policy'] for a in records}):
        rows=[a for a in records if a['policy']==policy]
        v=np.array([[np.mean([a['correct'] for a in rows if a['run']==run and a['seed']==(seed if policy.startswith(('qwen','granite')) else 0)]) for seed in (17,29,43,61,79)] for run in range(71,91)])
        values[policy]=v
        summary.append(dict(policy=policy,metrics=metrics(np.array([a['probabilities'] for a in rows]),np.array([a['label'] for a in rows])),
            ci95=np.quantile(v[bs[:,:,None],ss[:,None,:]].mean((1,2)),[.025,.975]).tolist(),median_seconds=float(np.median([a['seconds'] for a in rows])),
            accuracy_by_seed={str(s):float(np.mean([a['correct'] for a in rows if a['seed']==s])) for s in sorted({a['seed'] for a in rows})}))
    contrasts=[]
    for model in ('qwen','granite'):
        for ref in ('forward','rounded_tool'):
            delta=values[model+'_assisted']-values[ref]
            contrasts.append(dict(model=model,reference=ref,difference=float(delta.mean()),ci9875=np.quantile(delta[bs[:,:,None],ss[:,None,:]].mean((1,2)),[.00625,.99375]).tolist()))
    (OUT/'closed_loop_analysis.json').write_text(json.dumps(dict(audit_passed=True,scope='post-review independent test runs 71-90; crossed run-family and decoding-seed percentile bootstrap; five seeds and 20 families provide limited approximation; conditional on fitted predictor',summaries=summary,contrasts=contrasts),indent=2))

if __name__=='__main__':main()

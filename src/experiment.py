"""Paired, frozen-snapshot acquisition experiment. Policies never receive hidden rows."""
import hashlib
import json
import re
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260920

def load(partition):
    arrays = [np.load(ROOT/'data'/f'TEP_{kind}_{partition}_snapshot.npz')['values']
              for kind in ('FaultFree', 'Faulty')]
    a = np.concatenate(arrays)
    a = a[np.lexsort((a[:,0], a[:,1]))]
    return a[:,3:44], a[:,0].astype(int), a[:,1].astype(int)

def entropy(p):
    return -(p * np.log(np.maximum(p, 1e-12))).sum(axis=-1)

class Predictor:
    def fit(self, x, y):
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-8)
        self.donors = (x-self.mean)/self.scale
        rng = np.random.default_rng(SEED)
        n, d = x.shape
        masks = []
        for i in range(16):
            mask = np.zeros((n,d), dtype=np.float32)
            if i % 6:
                ix = np.argsort(rng.random((n,d)), axis=1)[:, :i%6]
                np.put_along_axis(mask, ix, 1, axis=1)
            masks.append(mask)
        masks.append(np.ones((n,d), dtype=np.float32))
        features = np.concatenate([np.column_stack((self.donors*m,m)) for m in masks]).astype('float32')
        self.model = ExtraTreesClassifier(n_estimators=64,max_depth=14,min_samples_leaf=3,
                                         random_state=SEED,n_jobs=1)
        self.model.fit(features, np.tile(y,len(masks)))
        self.empty_donors = rng.choice(n,8,replace=False)
        return self

    def predict(self, observed):
        z = np.zeros((1,41)); m = np.zeros((1,41))
        for j,v in observed.items():
            z[0,j] = (v-self.mean[j])/self.scale[j]; m[0,j] = 1
        return self.model.predict_proba(np.column_stack((z,m)))[0]

    def batch(self, x, selected):
        z = np.zeros_like(x); m = np.zeros_like(x)
        z[:,selected] = ((x-self.mean)/self.scale)[:,selected]; m[:,selected] = 1
        return self.model.predict_proba(np.column_stack((z,m)))

    def scores(self, observed, legal):
        # Distances use only acquired features. No query row or label is accepted.
        z = np.zeros(41); m = np.zeros(41)
        for j,v in observed.items():
            z[j]=(v-self.mean[j])/self.scale[j]; m[j]=1
        if observed:
            ix=list(observed)
            dist=((self.donors[:,ix]-z[ix])**2).sum(1)
            donor_ix=np.argsort(dist,kind='stable')[:8]
        else:
            donor_ix=self.empty_donors
        features=np.tile(np.r_[z,m],(len(legal)*8,1))
        for k,j in enumerate(legal):
            features[k*8:(k+1)*8,j]=self.donors[donor_ix,j]
            features[k*8:(k+1)*8,41+j]=1
        after=self.model.predict_proba(features).reshape(len(legal),8,21)
        score=entropy(self.predict(observed))-entropy(after).mean(1)
        return {int(j):float(s) for j,s in zip(legal,score)}

def sensor_metadata():
    text=(ROOT/'provenance/teprob.f').read_text()
    result={}
    for line in text.splitlines():
        match=re.match(r'C\s+XMEAS\((\d+)\)\s+(.+)',line)
        if match:
            j=int(match[1]); description=match[2].strip()
            if 23<=j<=28: description='Reactor feed '+description+' mole percent'
            elif 29<=j<=36: description='Purge gas '+description+' mole percent'
            elif 37<=j<=41: description='Product '+description+' mole percent'
            result[j-1]=description
    assert len(result)==41
    return result

def train():
    t=time.perf_counter()
    x,y,r=load('Training')
    tr=r<=350; va=(r>=351)&(r<=400)
    predictor=Predictor().fit(x[tr],y[tr])
    fit_seconds=time.perf_counter()-t
    mi=mutual_info_classif(x[tr],y[tr],random_state=SEED)
    rankings={
        'fixed':list(range(41)),
        'mi':np.argsort(-mi,kind='stable').tolist(),
        'importance':np.argsort(-predictor.model.feature_importances_[:41],kind='stable').tolist()
    }
    chosen=[]
    for _ in range(5):
        candidates=[]
        for j in range(41):
            if j not in chosen:
                pred=predictor.batch(x[va],chosen+[j]).argmax(1)
                candidates.append((balanced_accuracy_score(y[va],pred),-j))
        best=max(candidates); chosen.append(-best[1])
        print('forward',chosen,best[0],flush=True)
    rankings['forward']=chosen+[j for j in range(41) if j not in chosen]
    validation={}
    for name,rank in rankings.items():
        validation[name]=float(balanced_accuracy_score(y[va],predictor.batch(x[va],rank[:3]).argmax(1)))
    preds=[]
    for row in x[va]:
        obs={}
        for _ in range(3):
            legal=[j for j in range(41) if j not in obs]
            scores=predictor.scores(obs,legal)
            j=max(legal,key=lambda j:(scores[j],-j)); obs[j]=row[j]
        preds.append(int(predictor.predict(obs).argmax()))
    validation['tool']=float(balanced_accuracy_score(y[va],preds))
    champion=max(validation,key=validation.get)
    package=dict(predictor=predictor,rankings=rankings,validation=validation,champion=champion,
                 fit_seconds=fit_seconds,setup_seconds=time.perf_counter()-t,metadata=sensor_metadata())
    joblib.dump(package,ROOT/'results/training.joblib')
    summary={k:v for k,v in package.items() if k not in ('predictor','metadata')}
    (ROOT/'results/training_summary.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)

def state_view(observed,legal,remaining,predictor,metadata,scores=None):
    state=dict(observed={str(j+1):float(v) for j,v in observed.items()},
               legal_sensors=[j+1 for j in legal],remaining_budget=remaining,
               diagnosis_probabilities=np.round(predictor.predict(observed),5).tolist())
    if scores is not None:
        state['expected_entropy_reduction']={str(j+1):round(v,5) for j,v in scores.items()}
    return state

def evaluate_classical(partition='Testing',lo=1,hi=20,budgets=(1,3,5)):
    package=joblib.load(ROOT/'results/training.joblib'); p=package['predictor']
    x,y,r=load(partition); take=(r>=lo)&(r<=hi); x,y,r=x[take],y[take],r[take]
    output=ROOT/'results'/f'classical_{partition}_{lo}_{hi}.jsonl'
    with output.open('w') as f:
        for policy in list(package['rankings'])+['random','tool','full']:
            seeds=[17,29,43] if policy=='random' else [0]
            for seed in seeds:
                for row,label,run in zip(x,y,r):
                    rng=np.random.default_rng(np.random.SeedSequence([SEED,seed,int(label),int(run)]))
                    obs={}; trajectory=[]; start=time.perf_counter()
                    maximum=41 if policy=='full' else max(budgets)
                    for step in range(maximum):
                        legal=[j for j in range(41) if j not in obs]
                        if policy=='tool':
                            scores=p.scores(obs,legal); j=max(legal,key=lambda j:(scores[j],-j))
                        elif policy=='random': j=int(rng.choice(legal))
                        elif policy=='full': j=step
                        else: j=package['rankings'][policy][step]
                        obs[j]=float(row[j]); trajectory.append(j+1)
                        if (policy!='full' and step+1 in budgets) or (policy=='full' and step==40):
                            prob=p.predict(obs)
                            record=dict(policy=policy,seed=seed,label=int(label),run=int(run),budget=step+1,
                                        prediction=int(prob.argmax()),correct=int(prob.argmax()==label),
                                        seconds=time.perf_counter()-start,trajectory=trajectory.copy(),
                                        log_loss=float(-np.log(max(prob[label],1e-12))))
                            f.write(json.dumps(record)+'\n')
            f.flush(); print('finished',policy,flush=True)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','classical'])
    args=parser.parse_args()
    if args.mode=='train': train()
    else: evaluate_classical()

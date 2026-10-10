"""Post-review validation diagnostics. Never reads publisher testing outcomes.

This is a new exploratory analysis, not an amendment to the frozen confirmation.
"""
import copy
import datetime
import json
import time
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from experiment import ROOT, SEED, load

OUT = ROOT / 'results/revision'

def metrics(prob, y):
    correct = prob.argmax(1) == y
    confidence = prob.max(1)
    ece = 0.0
    for i in range(10):
        take = (confidence >= i / 10) & (confidence < (i + 1) / 10 if i < 9 else confidence <= 1)
        if take.any():
            ece += take.mean() * abs(correct[take].mean() - confidence[take].mean())
    return dict(balanced_accuracy=float(np.mean([correct[y == c].mean() for c in range(21)])),
                log_loss=float(-np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)).mean()),
                brier=float(((prob - np.eye(21)[y]) ** 2).sum(1).mean()), ece10=float(ece))

def main():
    OUT.mkdir(exist_ok=True)
    x, y, r = load('Training')
    xt, yt = x[r <= 350], y[r <= 350]
    xv, yv = x[(r >= 351) & (r <= 400)], y[(r >= 351) & (r <= 400)]
    base = joblib.load(ROOT / 'results/training.joblib')['predictor']
    rng = np.random.default_rng(SEED)
    masks = []
    for i in range(16):
        mask = np.zeros_like(xt, dtype=np.float32)
        if i % 6:
            ids = np.argsort(rng.random(xt.shape), axis=1)[:, :i % 6]
            np.put_along_axis(mask, ids, 1, axis=1)
        masks.append(mask)
    masks.append(np.ones_like(xt, dtype=np.float32))
    features = np.concatenate([np.column_stack((base.donors*m, m)) for m in masks]).astype('float32')
    labels = np.tile(yt, len(masks))
    del masks
    # Same fixed validation masks for every candidate, independent of test outcomes.
    rng = np.random.default_rng(20260926)
    subsets = [rng.choice(41, 3, replace=False).tolist() for _ in range(12)]
    report = dict(status='validation-only post-review diagnostics', selection='maximum mean balanced accuracy across 12 frozen three-sensor masks; candidate order breaks ties',
                  validation_masks=subsets, candidates=[], calibration_scope='uncalibrated probabilities; descriptive validation diagnostics, no calibration fitted')
    best = None
    for name, depth, leaf in [('original',14,3), ('depth24',24,3), ('unlimited',None,3), ('unlimited_leaf1',None,1)]:
        start = time.perf_counter()
        p = copy.copy(base)
        if name != 'original':
            p.model = ExtraTreesClassifier(n_estimators=64,max_depth=depth,min_samples_leaf=leaf,random_state=SEED,n_jobs=1)
            p.model.fit(features,labels)
        masked = [metrics(p.batch(xv, ids), yv) for ids in subsets]
        score = float(np.mean([a['balanced_accuracy'] for a in masked]))
        item = dict(name=name, depth=depth, min_samples_leaf=leaf, validation_mask_mean_accuracy=score,
                    validation_mask_metrics=masked, full=metrics(p.batch(xv,list(range(41))),yv), elapsed_seconds=time.perf_counter()-start)
        report['candidates'].append(item)
        if best is None or score > best[0]:
            best = (score, name)
            joblib.dump(p,OUT/'selected_predictor.joblib')
        (OUT/'predictor_diagnostics_partial.json').write_text(json.dumps(report,indent=2))
        print(name,score,flush=True)
    del features
    start=time.perf_counter()
    full = ExtraTreesClassifier(n_estimators=64,max_depth=None,min_samples_leaf=3,random_state=SEED,n_jobs=1)
    full.fit(xt,yt)
    report['dedicated_full_observation'] = dict(metrics=metrics(full.predict_proba(xv),yv),fit_seconds=time.perf_counter()-start,
        interpretation='validation diagnostic only; different training objective, not an acquisition comparator')
    report['selected']=best[1]
    report['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    report['test_outcomes_read']=False
    (OUT/'predictor_diagnostics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['dedicated_full_observation']),flush=True)

if __name__ == '__main__': main()

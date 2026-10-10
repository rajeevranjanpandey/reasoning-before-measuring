"""Validate complete paired pilot records and derive descriptive mechanisms.

This post-run audit does not select policies, tune prompts, or change endpoints.
"""
import collections
import hashlib
import json
import numpy as np
import joblib
from experiment import ROOT, load

def main():
    x, y, runs = load('Testing')
    source = {(int(r), int(label)): row for row, label, r in zip(x, y, runs) if r <= 20}
    package = joblib.load(ROOT / 'results/training.joblib')
    p = package['predictor']
    audit = {'complete': True, 'models': {}, 'scope': 'Paired pilot only; no independent confirmation'}
    audit['references']={}
    for name in ('classical_Testing_1_20','qwen_fixed_Testing_1_20','granite_fixed_Testing_1_20'):
        path=ROOT/'results'/f'{name}.jsonl'
        keys=set();groups=collections.Counter()
        for line in path.open():
            r=json.loads(line)
            key=(r['policy'],r['seed'],r['budget'],r['run'],r['label'])
            assert key not in keys
            keys.add(key);groups[(r['policy'],r['seed'],r['budget'])]+=1
            sensors=r['trajectory']
            assert len(sensors)==r['budget'] and len(set(sensors))==len(sensors)
            assert all(1<=j<=41 for j in sensors)
            row=source[(r['run'],r['label'])]
            pred=int(p.predict({j-1:float(row[j-1]) for j in sensors}).argmax())
            assert pred==r['prediction'] and r['correct']==int(pred==r['label'])
        assert all(n==420 for n in groups.values()),(name,groups)
        if name.startswith('classical'):
            expected_groups={(policy,0,b) for policy in ('fixed','mi','importance','forward','tool') for b in (1,3,5)}
            expected_groups|={('random',seed,b) for seed in (17,29,43) for b in (1,3,5)}
            expected_groups.add(('full',0,41))
        else:expected_groups={('fixed_plan',0,b) for b in (1,3,5)}
        assert set(groups)==expected_groups,(name,'missing reference cell')
        audit['references'][name]={'records':len(keys),'cells':len(groups),'all_predictions_reproduced':True,
                                   'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    expected = {(v, seed, run, label) for v in ('raw','assisted')
                for seed in (17,29,43) for run in range(1,21) for label in range(21)}
    for model in ('qwen','granite'):
        path = ROOT / 'results' / f'{model}_pilot.jsonl'
        seen = set(); cells = collections.defaultdict(list)
        for line in path.open():
            r = json.loads(line)
            key = (r['policy'],r['seed'],r['run'],r['label'])
            assert key not in seen, ('duplicate', model, key)
            assert key in expected, ('unexpected', model, key)
            seen.add(key)
            assert r['model'] == model and r['stage'] == 'pilot' and r['budget'] == 3
            row = source[(r['run'],r['label'])]
            observed = {}; tool_agreement = []; prefix_correct = []
            for step, a in enumerate(r['actions']):
                j = a['sensor'] - 1
                assert 0 <= j < 41 and j not in observed
                assert a['value'] == float(row[j])
                state = a['state']
                allowed = {'observed','legal_sensors','remaining_budget','diagnosis_probabilities'}
                if r['policy'] == 'assisted': allowed.add('expected_entropy_reduction')
                assert set(state) == allowed
                assert state['observed'] == {str(k+1):v for k,v in observed.items()}
                assert state['legal_sensors'] == [k+1 for k in range(41) if k not in observed]
                assert state['remaining_budget'] == 3-step
                serialized=json.dumps(state,separators=(',',':'))
                for attempt in a['attempts']:
                    if 'prompt' in attempt:assert serialized in attempt['prompt']
                np.testing.assert_allclose(state['diagnosis_probabilities'], np.round(p.predict(observed),5), atol=0, rtol=0)
                if r['policy'] == 'assisted':
                    scores = state['expected_entropy_reduction']
                    assert set(map(int,scores)) == set(state['legal_sensors'])
                    best = max(map(int,scores), key=lambda k:(scores[str(k)],-k))
                    tool_agreement.append(int(a['sensor'] == best))
                observed[j] = a['value']
                prefix_correct.append(int(p.predict(observed).argmax() == r['label']))
                assert len(a['attempts']) in (1,2)
            assert len(observed) == 3
            assert r['fallback_count'] == sum(a['fallback'] for a in r['actions'])
            pred = int(p.predict(observed).argmax())
            assert r['prediction'] == pred and r['correct'] == int(pred == r['label'])
            np.testing.assert_allclose(r['log_loss'], -np.log(max(p.predict(observed)[r['label']],1e-12)),rtol=1e-10)
            cells[r['policy']].append((r,tool_agreement,prefix_correct))
        assert seen == expected, (model, len(seen), len(expected))
        summary = {}
        for variant, entries in cells.items():
            frequencies = collections.Counter(a['sensor'] for r,_,_ in entries for a in r['actions'])
            trajectories = collections.Counter(tuple(a['sensor'] for a in r['actions']) for r,_,_ in entries)
            attempts = [a for r,_,_ in entries for action in r['actions'] for a in action['attempts']]
            summary[variant] = {
                'episodes': len(entries), 'fallback_episodes':sum(r['fallback_count']>0 for r,_,_ in entries),
                'repair_episodes':sum(any(len(a['attempts'])>1 for a in r['actions']) for r,_,_ in entries),
                'accuracy_by_seed':{str(seed):float(np.mean([r['correct'] for r,_,_ in entries if r['seed']==seed])) for seed in (17,29,43)},
                'class_recall':{str(c):float(np.mean([r['correct'] for r,_,_ in entries if r['label']==c])) for c in range(21)},
                'distinct_trajectories':len(trajectories),
                'most_common_trajectories':[{'sensors':list(t),'count':n} for t,n in trajectories.most_common(10)],
                'sensor_selection_counts':dict(sorted(frequencies.items())),
                'rounded_tool_top_choice_agreement':float(np.mean([v for _,agree,_ in entries for v in agree])) if variant=='assisted' else None,
                'prefix_accuracy':np.mean([prefix for _,_,prefix in entries],axis=0).tolist(),
                'prefix_interpretation':'Descriptive replay of B=3 trajectories, not independent B=1/B=2 policy trials',
                'mean_seconds':float(np.mean([r['seconds'] for r,_,_ in entries])),
                'median_seconds':float(np.median([r['seconds'] for r,_,_ in entries])),
                'p95_seconds':float(np.quantile([r['seconds'] for r,_,_ in entries],.95)),
                'max_worker_rss_bytes':max(a.get('rss_peak_bytes',0) for a in attempts),
                'prompt_tokens_total':sum(a.get('usage',{}).get('prompt_tokens',0) for a in attempts),
                'completion_tokens_total':sum(a.get('usage',{}).get('completion_tokens',0) for a in attempts),
                'error_attempts':sum('error' in a for a in attempts)
            }
        audit['models'][model] = {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(seen),'cells':summary}
    (ROOT / 'results/pilot_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps({'audit_passed':True,'verified_episodes':5040,'verified_actions':15120}))

if __name__ == '__main__':
    main()

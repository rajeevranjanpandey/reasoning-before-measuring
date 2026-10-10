"""Fail-closed audit and analysis of the separately frozen confirmation run."""
import collections
import json
import time
import numpy as np
import joblib
from confirmation import ROOT, OUT, verify_plan, key, write_json
from experiment import load, state_view, SEED
from agents import parse_action


def main():
    plan = verify_plan()
    expected_n = plan['clusters'] * 21
    # Require all identities before exposing any interim effect estimates.
    model_keys = {(v, b, s, r, c) for v in plan['variants'] for b in plan['budgets']
                  for s in plan['seeds'] for r in range(21, 71) for c in range(21)}
    for model in plan['models']:
        path = OUT / f'{model}.jsonl'
        seen = set()
        with path.open() as f:
            for line in f:
                k = key(json.loads(line))
                assert k not in seen, ('duplicate', model, k)
                seen.add(k)
        assert seen == model_keys, (model, 'incomplete or unexpected identities', len(seen))
    package = joblib.load(ROOT / 'results/training.joblib'); p = package['predictor']
    x, y, run = load('Testing')
    lookup = {(int(r), int(c)): row for row, c, r in zip(x, y, run) if 21 <= r <= 70}
    groups = {}; checked = 0; actions_checked = 0
    reference_expected = set()
    for policy in list(package['rankings']) + ['random', 'tool', 'rounded_tool', 'full', 'qwen_fixed_plan', 'granite_fixed_plan']:
        for b in ([41] if policy == 'full' else plan['budgets']):
            for s in (plan['seeds'] if policy == 'random' else [0]):
                reference_expected |= {(policy, b, s, r, c) for r in range(21, 71) for c in range(21)}
    refs_seen = set()
    for name in ['references'] + plan['models']:
        with (OUT / f'{name}.jsonl').open() as f:
            for line in f:
                rec = json.loads(line); row = lookup[(rec['run'], rec['label'])]
                if name == 'references':
                    assert key(rec) not in refs_seen
                    refs_seen.add(key(rec))
                obs = {}
                reference_rng = np.random.default_rng(np.random.SeedSequence(
                    [SEED, rec['seed'], rec['label'], rec['run']]))
                assert len(rec['actions']) == rec['budget']
                for step, action in enumerate(rec['actions']):
                    j = action['sensor'] - 1
                    assert 0 <= j < 41 and j not in obs
                    assert action['value'] == float(row[j])
                    if name == 'references':
                        legal = [k for k in range(41) if k not in obs]
                        policy = rec['policy']
                        if policy in ('tool', 'rounded_tool'):
                            scores = p.scores(obs, legal)
                            if policy == 'rounded_tool': scores = {k: round(v, 5) for k, v in scores.items()}
                            expected_j = max(legal, key=lambda k: (scores[k], -k))
                        elif policy == 'random': expected_j = int(reference_rng.choice(legal))
                        elif policy == 'full': expected_j = step
                        elif policy.endswith('_fixed_plan'):
                            saved = json.loads((ROOT / ('results/' + policy.split('_')[0] + '_fixed_plan.json')).read_text())
                            expected_j = saved['ranking'][step]
                        else: expected_j = package['rankings'][policy][step]
                        assert j == expected_j, ('reference selection mismatch', key(rec), step)
                    if name != 'references':
                        legal = [k for k in range(41) if k not in obs]
                        scores = p.scores(obs, legal) if rec['policy'] == 'assisted' else None
                        expected = state_view(obs, legal, rec['budget'] - step, p, package['metadata'], scores)
                        assert action['state'] == expected, ('state mismatch', name, key(rec), step)
                        attempts = action['attempts']
                        assert 1 <= len(attempts) <= 2
                        if action['fallback']:
                            assert all(parse_action(a.get('text', ''), legal) is None for a in attempts)
                            assert j == next(k for k in package['rankings']['mi'] if k in legal)
                        else:
                            assert parse_action(attempts[-1].get('text', ''), legal) == j
                        if len(attempts) == 2:
                            assert parse_action(attempts[0].get('text', ''), legal) is None
                    obs[j] = action['value']; actions_checked += 1
                prob = p.predict(obs)
                np.testing.assert_allclose(prob, rec['probabilities'], rtol=0, atol=1e-12)
                assert int(prob.argmax()) == rec['prediction']
                assert int(prob.argmax() == rec['label']) == rec['correct']
                assert rec['seconds'] > 0
                target = np.zeros(21); target[rec['label']] = 1
                assert abs(float(((prob - target)**2).sum()) - rec['brier']) < 1e-12
                assert abs(float(-np.log(max(prob[rec['label']], 1e-12))) - rec['log_loss']) < 1e-12
                policy = name + '_' + rec['policy'] if name != 'references' else rec['policy']
                group = groups.setdefault((policy, rec['budget']), dict(
                    correctness=np.zeros((50, 21)), counts=np.zeros((50, 21)), seconds=[],
                    logloss=[], brier=[], confidence=[], correct=[], paths=collections.Counter(),
                    fallbacks=0, repairs=0, tool_agree=0, tool_actions=0, prompt_tokens=0,
                    completion_tokens=0, worker_rss_peak_bytes=0, byseed=collections.defaultdict(list)))
                i, c = rec['run'] - 21, rec['label']
                group['correctness'][i, c] += rec['correct']; group['counts'][i, c] += 1
                for field, source in [('seconds', 'seconds'), ('logloss', 'log_loss'), ('brier', 'brier')]:
                    group[field].append(rec[source])
                group['confidence'].append(float(prob.max())); group['correct'].append(rec['correct'])
                group['paths'][tuple(a['sensor'] for a in rec['actions'])] += 1
                group['byseed'][rec['seed']].append(rec['correct'])
                if name != 'references':
                    assert sum(a['fallback'] for a in rec['actions']) == rec['fallback_count']
                    group['fallbacks'] += int(rec['fallback_count'] > 0)
                    for action in rec['actions']:
                        group['repairs'] += int(len(action['attempts']) > 1)
                        for attempt in action['attempts']:
                            usage = attempt.get('usage', {})
                            group['prompt_tokens'] += usage.get('prompt_tokens', 0)
                            group['completion_tokens'] += usage.get('completion_tokens', 0)
                            group['worker_rss_peak_bytes'] = max(group['worker_rss_peak_bytes'], attempt.get('rss_peak_bytes', 0))
                        if rec['policy'] == 'assisted':
                            scores = action['state']['expected_entropy_reduction']
                            best = max(scores, key=lambda k: (scores[k], -int(k)))
                            group['tool_agree'] += int(int(best) == action['sensor'])
                            group['tool_actions'] += 1
                checked += 1
                if checked % 1000 == 0: print('audited', checked, flush=True)
    assert refs_seen == reference_expected
    rng = np.random.default_rng(20260926)
    indices = rng.integers(0, 50, (20000, 50))
    matrices = {}; summaries = []
    for (policy, budget), g in groups.items():
        assert np.all(g['counts'] == (3 if policy == 'random' or policy.endswith(('_raw', '_assisted')) else 1))
        matrix = g['correctness'] / g['counts']; matrices[(policy, budget)] = matrix
        clusters = matrix.mean(axis=1); boot = clusters[indices].mean(axis=1)
        confidence = np.asarray(g['confidence']); correct = np.asarray(g['correct'])
        bins = np.minimum((confidence * 10).astype(int), 9)
        reliability = []; ece = 0.
        for b in range(10):
            sel = bins == b
            if sel.any():
                acc = float(correct[sel].mean()); conf = float(confidence[sel].mean())
                ece += sel.mean() * abs(acc - conf)
                reliability.append(dict(bin=b, count=int(sel.sum()), confidence=conf, accuracy=acc))
        summaries.append(dict(policy=policy, budget=budget, independent_clusters=50,
            snapshot_identities=expected_n, episodes=len(g['seconds']), balanced_accuracy=float(matrix.mean()),
            ci95=np.quantile(boot, [.025, .975]).tolist(), per_class_recall=matrix.mean(axis=0).tolist(),
            mean_seconds=float(np.mean(g['seconds'])), median_seconds=float(np.median(g['seconds'])),
            p95_seconds=float(np.quantile(g['seconds'], .95)), log_loss=float(np.mean(g['logloss'])),
            brier=float(np.mean(g['brier'])), top_label_ece_10_bins=float(ece), reliability=reliability,
            unique_paths=len(g['paths']), most_frequent_paths=[dict(path=list(k), n=n) for k,n in g['paths'].most_common(10)],
            fallback_episodes=g['fallbacks'], repaired_actions=g['repairs'],
            rounded_tool_agreement=g['tool_agree']/g['tool_actions'] if g['tool_actions'] else None,
            prompt_tokens=g['prompt_tokens'], completion_tokens=g['completion_tokens'],
            worker_rss_peak_bytes=g['worker_rss_peak_bytes'],
            seed_accuracy={str(s):float(np.mean(v)) for s,v in g['byseed'].items()}))
    contrasts = []
    for model in plan['models']:
        for ref in ['forward', model + '_fixed_plan']:
            diff = (matrices[(model + '_assisted', 3)] - matrices[(ref, 3)]).mean(axis=1)
            boot = diff[indices].mean(axis=1); low, high = np.quantile(boot, [.00625, .99375])
            contrasts.append(dict(model=model, reference=ref, difference=float(diff.mean()),
                                  simultaneous_interval=[float(low), float(high)],
                                  excludes_three_point_benefit=bool(high < .03)))
    write_json(OUT / 'audit.json', dict(passed=True, checked_records=checked, checked_acquisitions=actions_checked,
               expected_model_records=37800, expected_reference_records=len(reference_expected),
               checks='Complete identity sets; source values; budget and unique legal actions; independently reconstructed states; parsed actions, repairs and fallback; re-evaluated probabilities, labels, log loss and Brier score.'))
    write_json(OUT / 'analysis.json', dict(confirmation_complete=True, clusters=50, summaries=summaries,
               primary_contrasts=contrasts, bootstrap_replicates=20000,
               qualification='Confirmation of this frozen configuration only; no equivalence, universal model or cross-dataset claim.'))
    print('Complete confirmation audit and analysis saved', flush=True)


if __name__ == '__main__': main()

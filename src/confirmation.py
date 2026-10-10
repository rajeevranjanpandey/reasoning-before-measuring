"""Post-pilot, separately frozen confirmation. Never modifies pilot outcomes.

Run freeze before run. Budget trials start from an empty state independently.
Only analyze a complete, audited dataset; status never reports interim effects.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'provenance/confirmation_plan_20260926.json'
OUT = ROOT / 'results/confirmation'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(s) for s in f if s.strip()]


def write_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def key(r):
    return r['policy'], r['budget'], r['seed'], r['run'], r['label']


def freeze():
    if PLAN.exists():
        raise RuntimeError('Plan already exists; do not replace a frozen plan')
    if OUT.exists() and any(OUT.glob('*.jsonl')):
        raise RuntimeError('Outcome files already exist')
    classical = read(ROOT / 'results/classical_Testing_1_20.jsonl')
    forward = {r: statistics.mean(z['correct'] for z in classical
               if z['run'] == r and z['policy'] == 'forward' and z['budget'] == 3)
               for r in range(1, 21)}
    precision = []
    for model in ('qwen', 'granite'):
        rows = read(ROOT / f'results/{model}_pilot.jsonl')
        fixed = read(ROOT / f'results/{model}_fixed_Testing_1_20.jsonl')
        for ref in ('forward', 'fixed'):
            differences = []
            for run in range(1, 21):
                a = statistics.mean(z['correct'] for z in rows
                                    if z['run'] == run and z['policy'] == 'assisted')
                b = forward[run] if ref == 'forward' else statistics.mean(
                    z['correct'] for z in fixed if z['run'] == run and z['budget'] == 3)
                differences.append(a - b)
            sd = statistics.stdev(differences)
            precision.append(dict(model=model, reference=ref, pilot_cluster_sd=sd,
                                  normal_approx_n=math.ceil((2.498 * sd / .02) ** 2)))
    files = [ROOT / 'src/experiment.py', ROOT / 'src/agents.py', Path(__file__),
             ROOT / 'results/training.joblib']
    files += [ROOT / f'results/{m}_fixed_plan.json' for m in ('qwen', 'granite')]
    files += list((ROOT / 'data').glob('*Testing_snapshot.npz'))
    plan = dict(
        frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        status='Locally frozen after pilot inspection and before new outcome evaluation; not public preregistration',
        purpose='Confirm the bounded pilot comparison and measure budget dependence; no universal SLM or state-of-the-art claim',
        partition='Testing', first_run=21, last_run=70, clusters=50, classes=list(range(21)),
        budgets=[1, 3, 5], primary_budget=3, seeds=[17, 29, 43],
        models=['qwen', 'granite'], variants=['raw', 'assisted'],
        expected_model_episodes=37800,
        precision=dict(target_half_width=.02, simultaneous_z=2.498, estimates=precision,
                       rationale='Maximum normal approximation requires 42 clusters; 50 supplies about 19 percent inflation. This is a planning approximation, not a coverage or power guarantee. Zero pilot variance is not taken as zero population variance.'),
        frozen_components='Original predictor, scaling, training donors, rankings, model artifacts, prompt, grammar, generation settings, repair and fallback, and one-time plans. No tuning on confirmation outcomes.',
        model_hashes=json.loads((ROOT / 'provenance/model_checksums.json').read_text()),
        primary_contrasts='Each assisted model minus forward selection and minus its frozen fixed plan, at B=3. Average correctness across seeds within run/class; bootstrap 50 run families keeping all classes together, 20000 replicates, seed 20260926. Bonferroni 98.75 percent marginal percentile intervals across four contrasts.',
        decision='Report estimate and interval relative to zero and +3 percentage points. A useful benefit is supported only if all four differences are at least +3 points and simultaneous lower bounds exceed zero, without seed reversals. An upper bound below +3 excludes the target for that contrast in this configuration. No equivalence or noninferiority claim is prespecified.',
        secondary='Raw policies and B=1/5 are descriptive, with unadjusted 95 percent cluster intervals. Report per-class recall, paths, tool-score agreement, failures, log loss, multiclass Brier score and top-label ECE in ten fixed equal-width bins. Calibration describes existing probabilities; no recalibration on test data.',
        timing='Independent empty-state trials at every budget; each includes all selection work and one final prediction. No intermediate diagnostic checkpoints except those required by the policy. Record load separately, mean/median/p95 episode time and per-call usage. Sequential on one machine; timing differences remain observational, not randomized hardware causal effects.',
        precision_control='Additional tool-only policy ranks five-decimal scores exactly as serialized to the assisted model, with the same smallest-ID tie breaking.',
        scope='TEP confirmation, budget dependence, score-rounding control, and probability diagnostics. No unrelated dataset, learned/non-myopic reproduction, unavailable-sensor or semantic-intervention evidence is claimed.',
        stopping='Fixed sample size. No effect-based early stopping or interim effects. Stop on integrity failure; resume exact missing identities only. Process failures remain recorded and use the original repair/fallback policy.',
        files={str(p.relative_to(ROOT)): digest(p) for p in files})
    write_json(PLAN, plan)
    print(json.dumps(plan, indent=2))


def verify_plan():
    plan = json.loads(PLAN.read_text())
    for name, expected in plan['files'].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError('Frozen input changed: ' + name)
    return plan


def progress(phase, **fields):
    write_json(OUT / 'status.json', dict(updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               phase=phase, **fields))
    print(json.dumps(dict(phase=phase, **fields)), flush=True)


def metrics(prob, label):
    import numpy as np
    target = np.zeros(21); target[label] = 1
    return dict(prediction=int(prob.argmax()), correct=int(prob.argmax() == label),
                probabilities=prob.tolist(), log_loss=float(-np.log(max(prob[label], 1e-12))),
                brier=float(((prob - target) ** 2).sum()))


def emit(f, record):
    f.write(json.dumps(record, separators=(',', ':')) + '\n')
    f.flush()
    os.fsync(f.fileno())


def reference_run(plan, package, data):
    import numpy as np
    from experiment import SEED
    p = package['predictor']; x, y, runs = data
    path = OUT / 'references.jsonl'
    rows = read(path); completed = {key(r) for r in rows}
    if len(completed) != len(rows):
        raise RuntimeError('Duplicate reference records')
    fixed = {m: json.loads((ROOT / f'results/{m}_fixed_plan.json').read_text())['ranking']
             for m in plan['models']}
    policies = list(package['rankings']) + ['random', 'tool', 'rounded_tool', 'full']
    policies += [m + '_fixed_plan' for m in plan['models']]
    with path.open('a') as f:
        for policy in policies:
            budgets = [41] if policy == 'full' else plan['budgets']
            seeds = plan['seeds'] if policy == 'random' else [0]
            for budget in budgets:
                for seed in seeds:
                    for row, label, run in zip(x, y, runs):
                        identity = (policy, budget, seed, int(run), int(label))
                        if identity in completed:
                            continue
                        start = time.perf_counter(); obs = {}; actions = []
                        rng = np.random.default_rng(np.random.SeedSequence([SEED, seed, int(label), int(run)]))
                        for step in range(budget):
                            legal = [j for j in range(41) if j not in obs]
                            if policy in ('tool', 'rounded_tool'):
                                scores = p.scores(obs, legal)
                                if policy == 'rounded_tool':
                                    scores = {j: round(s, 5) for j, s in scores.items()}
                                j = max(legal, key=lambda j: (scores[j], -j))
                            elif policy == 'random': j = int(rng.choice(legal))
                            elif policy == 'full': j = step
                            elif policy.endswith('_fixed_plan'): j = fixed[policy.split('_')[0]][step]
                            else: j = package['rankings'][policy][step]
                            obs[j] = float(row[j]); actions.append(dict(sensor=j + 1, value=obs[j]))
                        prob = p.predict(obs)
                        seconds = time.perf_counter() - start
                        record = dict(policy=policy, budget=budget, seed=seed, run=int(run), label=int(label),
                                      actions=actions, seconds=seconds, **metrics(prob, int(label)))
                        emit(f, record); completed.add(identity)
            progress('references', completed=len(completed), latest_policy=policy)


def model_run(plan, package, data, model, cells):
    from agents import LocalModel, parse_action, system_prompt
    from experiment import state_view
    p = package['predictor']; x, y, runs = data
    path = OUT / f'{model}.jsonl'
    rows = read(path); completed = {key(r) for r in rows}
    if len(completed) != len(rows):
        raise RuntimeError('Duplicate model records')
    expected = {(v, b, s, int(r), int(y0)) for v, b in cells for s in plan['seeds'] for y0, r in zip(y, runs)}
    if expected.issubset(completed):
        return
    endpoint = LocalModel(model, True)
    session = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    write_json(OUT / f'{model}_{session}_runtime.json', endpoint.info)
    system = system_prompt(package['metadata'])
    try:
        warm = endpoint.call([{'role': 'user', 'content': 'Return exactly {"sensor":1}.'}], 17)
        write_json(OUT / f'{model}_{session}_warmup.json', warm)
        if 'error' in warm: raise RuntimeError('Warmup failed')
        with path.open('a') as f:
            for variant, budget in cells:
                for seed in plan['seeds']:
                    for row, label, run in zip(x, y, runs):
                        identity = (variant, budget, seed, int(run), int(label))
                        if identity in completed: continue
                        start = time.perf_counter(); obs = {}; actions = []; fallbacks = 0
                        for step in range(budget):
                            legal = [j for j in range(41) if j not in obs]
                            t = time.perf_counter()
                            scores = p.scores(obs, legal) if variant == 'assisted' else None
                            tool_seconds = time.perf_counter() - t
                            state = state_view(obs, legal, budget - step, p, package['metadata'], scores)
                            messages = [{'role': 'system', 'content': system},
                                        {'role': 'user', 'content': json.dumps(state, separators=(',', ':'))}]
                            attempts = []; j = None
                            for attempt in range(2):
                                result = endpoint.call(messages, seed + step * 101 + attempt * 1009,
                                                       legal_sensors=[a + 1 for a in legal])
                                attempts.append(result)
                                j = parse_action(result.get('text', ''), legal)
                                if j is not None: break
                                if 'error' in result:
                                    endpoint.close(); endpoint = LocalModel(model, True)
                                messages += [{'role': 'assistant', 'content': result.get('text', '')},
                                             {'role': 'user', 'content': 'Invalid response. Return exactly one JSON object with key sensor and a legal integer.'}]
                            fallback = j is None
                            if fallback:
                                j = next(a for a in package['rankings']['mi'] if a in legal); fallbacks += 1
                            obs[j] = float(row[j])
                            actions.append(dict(sensor=j + 1, value=obs[j], state=state, attempts=attempts,
                                                fallback=fallback, tool_seconds=tool_seconds))
                        prob = p.predict(obs); seconds = time.perf_counter() - start
                        record = dict(model=model, policy=variant, budget=budget, seed=seed, run=int(run),
                                      label=int(label), session=session, actions=actions, fallback_count=fallbacks,
                                      seconds=seconds, **metrics(prob, int(label)))
                        emit(f, record); completed.add(identity)
                        if len(completed) % 10 == 0:
                            progress('models', model=model, completed_for_model=len(completed),
                                     expected_for_model=18900, budget=budget, variant=variant,
                                     latest_episode_seconds=round(seconds, 3))
    finally:
        endpoint.close()


def run():
    import joblib
    from experiment import load
    plan = verify_plan(); OUT.mkdir(exist_ok=True)
    with (OUT / 'run.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, expected in plan['model_hashes'].items():
            if digest(ROOT / 'models' / name) != expected['sha256']:
                raise RuntimeError('Pinned model checksum failed: ' + name)
        package = joblib.load(ROOT / 'results/training.joblib')
        x, y, r = load('Testing'); take = (r >= plan['first_run']) & (r <= plan['last_run'])
        data = x[take], y[take], r[take]
        assert len(data[0]) == 1050
        progress('starting', expected_model_episodes=plan['expected_model_episodes'])
        reference_run(plan, package, data)
        # Complete primary comparison first, but never inspect effects or change the plan.
        for cells in [[('assisted', 3)], [('raw', 3)], [('raw', 1), ('assisted', 1), ('raw', 5), ('assisted', 5)]]:
            for model in plan['models']:
                model_run(plan, package, data, model, cells)
        progress('execution_complete_pending_audit', expected_model_episodes=37800)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('mode', choices=['freeze', 'run'])
    args = parser.parse_args()
    if args.mode == 'freeze': freeze()
    else:
        try: run()
        except BaseException as exc:
            OUT.mkdir(exist_ok=True)
            progress('stopped', error=repr(exc)); raise

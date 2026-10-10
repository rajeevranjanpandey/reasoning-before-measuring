from pathlib import Path
from collections import Counter
import json, hashlib

ROOT = Path(__file__).parent
OUT = ROOT / 'review'

def records(relative):
    with (ROOT / relative).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def audit():
    result = {'studies': {}, 'input_sha256': {}}
    for study, name in [('pilot', 'results/qwen_pilot.jsonl'), ('confirmation', 'results/confirmation/qwen.jsonl'), ('closed_loop', 'results/revision/closed_loop.jsonl')]:
        pairs = {}; counts = Counter(); paths = Counter(); ascending = 0
        for row in records(name):
            policy = row['policy']
            if study == 'closed_loop':
                if not policy.startswith('qwen_'): continue
                policy = policy.split('_', 1)[1]
            key = (row['run'], row['label'], row.get('budget', 3), row['seed'])
            path = tuple(a['sensor'] for a in row['actions'])
            counts[policy] += 1; paths[str(path)] += 1
            ascending += path == tuple(range(1, key[2] + 1))
            assert policy not in pairs.setdefault(key, {}), (study, key)
            pairs[key][policy] = (path, row['prediction'])
        assert all(set(x) == {'raw', 'assisted'} for x in pairs.values())
        mismatch = sum(x['raw'] != x['assisted'] for x in pairs.values())
        assert mismatch == 0
        assert ascending == sum(counts.values())
        result['studies'][study] = {'episodes': dict(counts), 'paired_identities': len(pairs), 'path_prediction_mismatches': mismatch, 'ascending_id_episodes': ascending, 'paths': dict(paths)}
        result['input_sha256'][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    agreement = {}; chance = []
    for model in ['qwen', 'granite']:
        n = matches = ties = 0
        for row in records('results/' + model + '_pilot.jsonl'):
            if row['policy'] != 'assisted': continue
            for action in row['actions']:
                scores = action['state']['expected_entropy_reduction']
                best = min(scores, key=lambda k: (-scores[k], int(k)))
                n += 1; matches += int(action['sensor'] == int(best))
                ties += sum(v == scores[best] for v in scores.values()) > 1
                if model == 'granite': chance.append(1 / len(scores))
        agreement[model] = {'decisions': n, 'matches': matches, 'agreement_percent': 100 * matches / n, 'tied_maximum_decisions': ties, 'tie_rule': 'lowest sensor ID among highest rounded scores'}
    agreement['uniform_legal_choice_expected_percent'] = 100 * sum(chance) / len(chance)
    result['pilot_tool_agreement'] = agreement
    combinations = Counter()
    for row in records('results/confirmation/references.jsonl'):
        combinations[(row['policy'], row['budget'], row['seed'])] += 1
    assert len(combinations) == 34 and set(combinations.values()) == {1050}
    result['confirmation_references'] = {'records': sum(combinations.values()), 'combinations': len(combinations), 'snapshots_per_combination': 1050, 'breakdown': {str(k): v for k, v in sorted(combinations.items())}}
    development = {}
    for file in sorted((ROOT / 'results').glob('*development/*.jsonl')):
        development[str(file.relative_to(ROOT))] = sum(1 for x in file.open() if x.strip())
    result['excluded_development_files'] = development
    result['archived_parser_and_runtime_check_episodes'] = sum(development.values())
    final_development = {name: sum(1 for _ in records('results/' + name)) for name in ['qwen_development.jsonl', 'granite_development.jsonl']}
    assert sum(final_development.values()) == 80
    result['excluded_final_development_checks'] = final_development
    result['excluded_final_development_episodes'] = 80
    import numpy as np
    runs = {}
    for row in records('results/revision/closed_loop.jsonl'):
        policy = row['policy']; key = (row['run'], row['label'], row['seed'])
        assert row['correct'] == int(row['prediction'] == row['label'])
        assert key not in runs.setdefault(policy, {}), (policy, key)
        runs[policy][key] = row['correct']
    permutations = []
    for policy in ['qwen_assisted', 'granite_assisted']:
        for reference in ['forward', 'rounded_tool']:
            differences = []
            for run in range(71,91):
                m = sum(runs[policy][(run,c,k)] for c in range(21) for k in [17,29,43,61,79])
                r = 5 * sum(runs[reference][(run,c,0)] for c in range(21))
                differences.append(m-r)
            sums = np.array([0], dtype=np.int32)
            for difference in differences: sums = np.concatenate((sums+difference,sums-difference))
            assert len(sums) == 2**20
            p = float(np.mean(np.abs(sums) >= abs(sum(differences))))
            permutations.append({'policy':policy,'reference':reference,'family_count':20,'integer_family_differences_out_of_105':differences,'observed_difference_percentage_points':100*sum(differences)/(20*105),'exact_two_sided_sign_flip_p':p,'bonferroni_four_contrasts_p':min(1,4*p),'sign_configurations':len(sums)})
    result['post_hoc_descriptive_sign_flip'] = {'qualification':'Conditional on fitted predictor and recorded seeds. Family-sign exchangeability under a null is an assumption, not established by the evaluation. Not a new primary analysis; does not include seed or predictor refitting uncertainty.','comparisons':permutations}
    (OUT / 'episode_evidence_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__': audit()

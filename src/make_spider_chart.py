"""Compare all class recalls on identical radial scales, using completed pilot records.

No composite score, min-max scaling, selected classes or confirmation outcomes.
The CSV contains one row per policy/class and the JSON records source hashes.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/paper01_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ('forward', 'tool', 'qwen_assisted', 'granite_assisted')


def main():
    sources = ['classical_Testing_1_20.jsonl', 'qwen_pilot.jsonl', 'granite_pilot.jsonl']
    values = collections.defaultdict(dict)
    for name in sources:
        for line in (ROOT/'results'/name).read_text().splitlines():
            r = json.loads(line)
            policy = r['model']+'_'+r['policy'] if name.endswith('_pilot.jsonl') else r['policy']
            if policy not in POLICIES or r['budget'] != 3:
                continue
            key = (r['run'], r['label'], r['seed'])
            assert key not in values[policy], (policy, key)
            assert r['correct'] == int(r['prediction'] == r['label'])
            values[policy][key] = r['correct']
    recalls = {}
    for policy in POLICIES:
        seeds = (17, 29, 43) if policy.endswith('_assisted') else (0,)
        expected = {(r, c, s) for r in range(1, 21) for c in range(21) for s in seeds}
        assert set(values[policy]) == expected, policy
        # Average seeds within each identity before averaging the 20 run families.
        recalls[policy] = [100*float(np.mean([
            np.mean([values[policy][(r, c, s)] for s in seeds])
            for r in range(1, 21)])) for c in range(21)]
    analysis = json.loads((ROOT/'results/analysis.json').read_text())
    for policy in POLICIES:
        ba = next(r['balanced_accuracy'] for r in analysis['summaries']
                  if r['policy'] == policy and r['budget'] == 3)
        assert np.isclose(np.mean(recalls[policy]), 100*ba, atol=1e-10)
    heatmap = json.loads((ROOT/'results/diagnostic_figures.json').read_text())
    for policy in POLICIES:
        idx = heatmap['class_policy_order'].index(policy)
        assert np.allclose(recalls[policy], heatmap['class_recall_percent'][idx])
    with (ROOT/'results/class_recall_spider.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['policy', 'class_id', 'recall_percent', 'run_families', 'decoding_seeds'])
        for policy in POLICIES:
            for c, recall in enumerate(recalls[policy]):
                writer.writerow([policy, c, recall, 20, 3 if policy.endswith('_assisted') else 1])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.75), subplot_kw={'projection': 'polar'})
    theta = np.linspace(0, 2*np.pi, 21, endpoint=False)
    closed = np.r_[theta, theta[0]]
    style = {'forward': ('#0072B2', '--', 's', 'Forward selection'),
             'tool': ('#009E73', ':', '^', 'Entropy tool')}
    for ax, model, letter in zip(axes, ('qwen', 'granite'), ('A', 'B')):
        ax.set_theta_offset(np.pi/2)
        ax.set_theta_direction(-1)
        ax.set_xticks(theta, ['N' if c == 0 else str(c) for c in range(21)])
        ax.tick_params(axis='x', pad=2, labelsize=9)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100], ['25', '50', '75', '100'])
        ax.tick_params(axis='y', labelsize=7, colors='#555555')
        ax.set_rlabel_position(262)
        ax.grid(color='#d8dce0', linewidth=.6)
        ax.spines['polar'].set_color('#a7afb7')
        for policy in ('forward', 'tool', model+'_assisted'):
            color, line, marker, label = style.get(policy, ('#D55E00', '-', 'o', 'Assisted model'))
            ax.plot(closed, np.r_[recalls[policy], recalls[policy][0]], color=color,
                    linestyle=line, marker=marker, markersize=3, linewidth=1.4, label=label)
        ax.set_title(f'{letter}  {model.title()} assisted', fontsize=11, weight='bold', pad=21)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .075), ncol=3,
               frameon=False, fontsize=10, handlelength=2.6)
    fig.text(.5, .028, 'Class recall (%)  |  N = normal, 1–20 = fault IDs  |  All policies use B = 3',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.055, right=.945, top=.84, bottom=.22, wspace=.30)
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(ROOT/'paper'/f'class_recall_spider.{ext}', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    report = {'scope': 'Post-pilot descriptive display of all 21 classes. No new trials or inferential tests.',
              'budget': 3, 'radial_limits_percent': [0, 100], 'recall_percent': recalls,
              'zero_recall_class_counts': {p: sum(v == 0 for v in recalls[p]) for p in POLICIES},
              'source_sha256': {n: hashlib.sha256((ROOT/'results'/n).read_bytes()).hexdigest() for n in sources},
              'verified': ['complete unique identities', 'stored correctness', 'balanced accuracy agreement', 'heatmap agreement'],
              'interpretation': 'Unfilled lines aid class-wise comparison. Polygon area is not a metric. Class adjacency has no statistical meaning. No classwise confidence intervals.'}
    (ROOT/'results/class_recall_spider.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'policies': len(POLICIES), 'classes': 21, 'verified': report['verified']}))


if __name__ == '__main__':
    main()

"""Build evidence-linked manuscript sections only after every planned study passes."""
import argparse
import datetime
import json
import shutil
import tempfile
from pathlib import Path
from submission_gate import scientific_inputs, digest

ROOT=Path(__file__).resolve().parents[1]
LABELS={'forward':'Forward selection','rounded_tool':'Rounded entropy tool','tool':'Entropy tool',
        'full':'All 41 sensors','fixed':'Ascending sensor ID','mi':'Mutual information','importance':'Tree importance',
        'qwen_assisted':'Qwen assisted','granite_assisted':'Granite assisted','qwen_raw':'Qwen raw','granite_raw':'Granite raw',
        'qwen_fixed_plan':'Qwen fixed plan','granite_fixed_plan':'Granite fixed plan'}


def pct(x):return f'{100*x:.2f}'
def interval(x):return f'[{pct(x[0])}, {pct(x[1])}]'
def table(header,rows):return '\n'.join(['| '+' | '.join(header)+' |','| '+' | '.join(['---']*len(header))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])+'\n'


def build(root):
    issues=scientific_inputs(root)
    if issues:raise RuntimeError('Refusing to build from incomplete studies:\n'+'\n'.join(issues))
    def read(rel):return json.loads((root/rel).read_text())
    c=read('results/confirmation/analysis.json');d=read('results/revision/predictor_diagnostics.json')
    m=read('results/revision/controls_analysis.json');l=read('results/revision/closed_loop_analysis.json')
    # Require complete analysis schemas. Never silently drop unreported comparisons.
    assert len(c['primary_contrasts'])==4 and len(l['contrasts'])==4
    assert len(m['cells'])==12 and all(x['n']==100 for x in m['cells'])
    assert len(l['summaries'])==7
    by={(x['policy'],x['budget']):x for x in c['summaries']}
    for policy in ('qwen_raw','qwen_assisted','granite_raw','granite_assisted','forward','rounded_tool'):
        assert all((policy,b) in by for b in (1,3,5))
    for x in c['summaries']:
        assert x['independent_clusters']==50 and x['snapshot_identities']==1050
        assert 0 <= x['balanced_accuracy'] <= 1 and 0 <= x['ci95'][0] <= x['ci95'][1] <= 1
    out=root/'paper/postreview';out.parent.mkdir(exist_ok=True)
    if out.exists():raise RuntimeError('Post-review output already exists. Preserve it and explicitly select a new revision before rebuilding.')
    temporary=Path(tempfile.mkdtemp(prefix='.postreview-build-',dir=out.parent))
    try:
        sections=['# Results from the completed post-review studies\n',
          'This evidence-linked draft reports all planned follow-up studies. It is separate from the original pilot and requires integration into the abstract, results, discussion and conclusion, followed by scientific and visual review. All studies were specified after pilot inspection. Their plans were locally frozen, not publicly preregistered.\n',
          '## Confirmation on untouched run families\n',
          'The confirmation retains the pilot predictor, model artifacts, interfaces, rankings and one-time plans. It uses testing run families 21–70, with 1,050 snapshot identities across 21 classes, budgets of one, three and five sensors, and three decoding seeds. The complete record contains 37,800 model episodes and 35,700 reference records. Each budget starts independently from an empty observed state. The frozen primary comparison remains budget three. Family resampling keeps all classes together and averages the sampled decoding seeds within identities. The four primary contrasts use 98.75% marginal bootstrap intervals under the prespecified Bonferroni allocation. Secondary intervals are descriptive.\n',
          table(['Policy','Budget','Balanced accuracy (%)','95% interval (%)','Median seconds','95th percentile seconds'],[
          [LABELS.get(x['policy'],x['policy']),x['budget'],pct(x['balanced_accuracy']),interval(x['ci95']),f"{x['median_seconds']:.4f}",f"{x['p95_seconds']:.4f}"] for x in sorted(c['summaries'],key=lambda x:(x['budget'],x['policy']))]),
          '### All primary confirmation contrasts\n',
          table(['Model','Reference','Difference (pp)','98.75% interval (pp)','Upper bound below +3 pp'],[
          [x['model'],LABELS.get(x['reference'],x['reference']),pct(x['difference']),interval(x['simultaneous_interval']),str(x['excludes_three_point_benefit'])] for x in c['primary_contrasts']]),
          'Differences are assisted model minus reference. An interval excluding zero is not an equivalence test. A point estimate above three percentage points with a lower bound above zero does not establish a population gain of three points. A population gain above that target requires the lower bound to exceed three points. These estimates condition on the original fitted predictor and sampled decoding seeds.\n',
          '### Probability quality\n',
          table(['Policy at B=3 (full B=41)','Log loss','Brier score','Top-label ECE'],[
          [LABELS.get(x['policy'],x['policy']),f"{x['log_loss']:.4f}",f"{x['brier']:.4f}",f"{x['top_label_ece_10_bins']:.4f}"] for x in c['summaries'] if x['budget'] in (3,41)]),
          'Probability diagnostics use uncalibrated predictions. Log loss and multiclass Brier score are proper scoring rules. ECE uses ten fixed equal-width confidence bins and is a descriptive, bin-dependent summary. No calibration was fitted to testing outcomes.\n',
          '## Validation-selected predictor and diagnostic foundation\n',
          'Four masked predictors share training runs 1–350, 64 trees, the original scaling and training masks. Selection maximizes mean balanced accuracy across twelve fixed three-sensor masks on validation runs 351–400. Candidate order resolves ties. A separately trained full-observation classifier diagnoses the effect of a different training objective, but is not an acquisition comparator. No testing outcomes were used for selection.\n',
          table(['Candidate','Mean masked validation accuracy (%)','Full validation accuracy (%)','Full log loss','Full Brier','Full ECE'],[
          [x['name'],pct(x['validation_mask_mean_accuracy']),pct(x['full']['balanced_accuracy']),f"{x['full']['log_loss']:.4f}",f"{x['full']['brier']:.4f}",f"{x['full']['ece10']:.4f}"] for x in d['candidates']]),
          f"The selected candidate is {d['selected']}. The dedicated full-observation diagnostic reaches {pct(d['dedicated_full_observation']['metrics']['balanced_accuracy'])}% validation balanced accuracy. Validation scores used for selection are not unbiased estimates of subsequent testing performance. The four candidates do not exhaust predictor families or calibration methods.\n",
          '## Matched interface interventions\n',
          'Six intervention arms evaluate ordering, consistent ID remapping, anonymized descriptions, score removal and score shuffling against the original interface. Each model uses the same twenty development identities and five decoding seeds. The 1,200 episodes are development controls, not an independent diagnostic test set. All arms compute scores before serialization to keep that work comparable. Accuracy is ordinary accuracy because the development sample is not class-balanced. Trajectory changes do not by themselves establish diagnostic improvement or a reasoning mechanism.\n',
          table(['Model','Arm','Episodes','Paths changed versus original (%)','Development accuracy (%)'],[
          [x['model'],x['arm'],x['n'],pct(x['fraction_paths_changed']),pct(x['accuracy'])] for x in m['cells']]),
          '## Closed-loop evaluation with the selected predictor\n',
          'Testing runs 71–90 supply 420 new snapshot identities. Each of four model policies is evaluated across five decoding seeds at budget three. Forward selection is reselected using validation data only. The adaptive reference uses the same five-decimal scores exposed to the assisted models. The full-sensor reference is included separately. This evaluation reruns acquisition with the selected predictor and therefore differs from rescoring old trajectories. It contains 9,660 records. Crossed resampling uses twenty run families and five decoding seeds, with 20,000 replicates. The limited numbers of families and seeds constrain coverage.\n',
          table(['Policy','Balanced accuracy (%)','95% crossed interval (%)','Log loss','Brier','ECE','Median seconds'],[
          [LABELS.get(x['policy'],x['policy']),pct(x['metrics']['balanced_accuracy']),interval(x['ci95']),f"{x['metrics']['log_loss']:.4f}",f"{x['metrics']['brier']:.4f}",f"{x['metrics']['ece10']:.4f}",f"{x['median_seconds']:.4f}"] for x in l['summaries']]),
          table(['Model','Reference','Difference (pp)','98.75% crossed interval (pp)'],[
          [x['model'],LABELS.get(x['reference'],x['reference']),pct(x['difference']),interval(x['ci9875'])] for x in l['contrasts']]),
          '## Interpretation boundaries\n',
          'The original unfavorable pilot decision remains part of the record. The confirmation and selected-predictor studies use different testing families and answer different questions. Their absolute accuracies must not be treated as a paired causal estimate of changing the predictor. Generalization to other industrial datasets, learned or non-myopic acquisition methods, temporal fault detection, physical sensor costs and deployed safety remains untested. Measured runtimes are observational and include session and system effects. No energy or carbon measurements were made.\n']
        (temporary/'manuscript_sections.md').write_text('\n'.join(sections))
        # Vector figures are generated from audited aggregate statistics only.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        for ax,model in zip(axes,('qwen','granite')):
            for policy,color,marker in [(model+'_raw','#0072B2','o'),(model+'_assisted','#D55E00','s'),('forward','#009E73','^'),('rounded_tool','#CC79A7','D')]:
                rows=[by[policy,b] for b in (1,3,5)];y=np.array([100*x['balanced_accuracy'] for x in rows]);lo=np.array([100*x['ci95'][0] for x in rows]);hi=np.array([100*x['ci95'][1] for x in rows])
                ax.errorbar((1,3,5),y,yerr=[y-lo,hi-y],marker=marker,color=color,label=LABELS[policy],capsize=3)
            ax.set(title=model.capitalize(),xlabel='Acquired sensors',xticks=[1,3,5],ylim=(0,100));ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
        axes[0].set_ylabel('Balanced accuracy (%)');fig.tight_layout()
        for ext in ('svg','pdf','png'):fig.savefig(temporary/f'confirmation_budget.{ext}',dpi=300)
        plt.close(fig)
        (temporary/'figure_captions.md').write_text('Confirmation budget comparison. Each point uses 1,050 snapshot identities from 50 simulation-run families. Model points average three decoding seeds within each identity. Error bars are descriptive pointwise 95% family-bootstrap intervals from 20,000 resamples. Both panels share a 0–100% scale. Colors and marker shapes identify policies. Connecting lines link measured budgets only. These secondary intervals are not simultaneous. Visualization code was prepared with ChatGPT/Codex from recorded audited results.\n')
        inputs=['results/confirmation/analysis.json','results/confirmation/audit.json','results/confirmation/pipeline_complete.json',
                'results/revision/predictor_diagnostics.json','results/revision/controls_analysis.json','results/revision/closed_loop_analysis.json',
                'results/confirmation/qwen.jsonl','results/confirmation/granite.jsonl','results/confirmation/references.jsonl',
                'results/revision/qwen_controls.jsonl','results/revision/granite_controls.jsonl','results/revision/closed_loop.jsonl']
        manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete_audited_studies':True,
                  'main_manuscript_integrated':False,'scientific_review_complete':False,'visual_review_complete':False,'submission_ready':False,
                  'input_sha256':{p:digest(root/p) for p in inputs},'outputs':{p.name:digest(p) for p in temporary.iterdir()}}
        (temporary/'build_manifest.json').write_text(json.dumps(manifest,indent=2))
        temporary.rename(out)
        return out
    except BaseException:
        shutil.rmtree(temporary)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    out=build(args.root)
    from build_reproducibility_archive import build_archive
    archive=build_archive(args.root,out/'reproducibility_complete.zip',scope='complete')
    (out/'reproducibility_archive.json').write_text(json.dumps(archive,indent=2))
    print(out)

"""Integrate all audited studies into a reviewable anonymous manuscript.

This authoring step never changes experimental inputs or certifies human approval.
"""
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path
from submission_gate import scientific_inputs
from build_postreview_results import LABELS, pct, interval, table

ROOT = Path(__file__).resolve().parents[1]


def main():
    issues = scientific_inputs(ROOT)
    if issues:
        raise RuntimeError('All completed scientific audits are required: ' + str(issues))
    read = lambda name: json.loads((ROOT / name).read_text())
    c = read('results/confirmation/analysis.json')
    d = read('results/revision/predictor_diagnostics.json')
    m = read('results/revision/controls_analysis.json')
    l = read('results/revision/closed_loop_analysis.json')
    original = ROOT / 'paper/pilot_manuscript.md'
    text = original.read_text()
    cby = {(x['policy'], x['budget']): x for x in c['summaries']}
    lby = {x['policy']: x for x in l['summaries']}
    # Leave the original pilot source intact. The complete manuscript is separate.
    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise RuntimeError('Source paragraph changed or ambiguous: ' + old[:80])
        text = text.replace(old, new)

    abstract = (
        'Repeated language-model consultation adds computation to measurement selection, but valid requests alone do not establish diagnostic value. '
        'We contribute a paired empirical evaluation of small language models for budget-constrained industrial fault diagnosis, separating acquisition from prediction with a shared masked-input classifier. '
        'Tennessee Eastman Process simulations supply a 420-snapshot pilot, confirmation on 1,050 untouched snapshots, and a 420-snapshot closed-loop test with a validation-selected predictor. '
        'The studies record 52,440 model episodes, including 1,200 development-control episodes. '
        f'At three sensors in confirmation, assisted Qwen and Granite reach {pct(cby["qwen_assisted",3]["balanced_accuracy"])}% and {pct(cby["granite_assisted",3]["balanced_accuracy"])}% balanced accuracy, below {pct(cby["forward",3]["balanced_accuracy"])}% for forward selection. '
        f'With the selected predictor, they reach {pct(lby["qwen_assisted"]["metrics"]["balanced_accuracy"])}% and {pct(lby["granite_assisted"]["metrics"]["balanced_accuracy"])}%, versus {pct(lby["rounded_tool"]["metrics"]["balanced_accuracy"])}% for the matched entropy tool. '
        'Both assisted-minus-tool 98.75% bootstrap intervals lie below zero. Qwen versus forward selection remains inconclusive at this coverage. '
        'Primary comparisons account for four contrasts, with family resampling and additional decoding-seed resampling in the final test. '
        'Listing-order interventions change both models’ control trajectories without establishing a reasoning mechanism. '
        'Recorded model runtimes are higher, but observational timing limits precise speedup claims. '
        'The original unfavorable pilot criterion is retained. Repeated consultation provides no demonstrated incremental diagnostic benefit over the matched tool at the primary budget in the tested configuration. '
        'Different test partitions preclude a causal predictor comparison, and one simulator with untested learned acquisition baselines limits generalization.'
    )
    if len(abstract.split()) > 250:
        raise RuntimeError('Abstract exceeds journal word limit')
    text = re.sub(r'(?<=## Abstract\n\n).*?(?=\n\nKeywords:)', abstract, text, count=1, flags=re.S)
    replace('This study contributes a fully logged paired evaluation on common simulation snapshots, an analysis of acquisition concentration and tool-score agreement, and a predictor replay check that exposes dependence of policy rankings on the scoring classifier. These are empirical contributions within the tested setting. The study does not introduce a new acquisition algorithm or claim to be the first general benchmark of active feature acquisition.',
        'This study contributes a fully logged evaluation separating valid execution, acquisition behavior and diagnostic utility under a common predictor. An initial paired pilot is followed by independent confirmation on untouched simulation-run families, matched interventions on the development interface, and closed-loop testing with a predictor selected using validation data alone. The contribution is empirical evidence about the incremental value of the tested language-model selectors and the conditions of that evidence. No new acquisition algorithm, general reasoning mechanism or state-of-the-art acquisition performance is claimed.')
    replace('The repeated language-model policies are evaluated only at the primary budget of three sensors. Classical and fixed-plan policies additionally supply one- and five-sensor results. Consequently, their budget curves do not represent independently executed repeated-model trials at one or five sensors.',
        'The pilot evaluates repeated language-model policies at three sensors, while its classical and fixed-plan policies also supply one- and five-sensor results. Confirmation executes each repeated model independently at budgets one, three and five. The selected-predictor closed-loop study uses budget three. Curves and contrasts identify their study and budget rather than pooling these distinct executions.')
    replace('Testing runs 21–70 are allocated to a separately frozen confirmation, and runs 71–90 to a subsequent predictor study. Neither supplies results reported in this pilot manuscript. Runs 91–500 remain reserved. Snapshot conversion alone does not constitute outcome evaluation.',
        'Testing runs 21–70 provide 1,050 snapshots for confirmation under the original predictor, and runs 71–90 provide 420 new snapshots for closed-loop testing under a validation-selected predictor. These are separate test partitions. Runs 91–500 remain reserved. Snapshot conversion alone does not constitute outcome evaluation.')
    replace("Figure 2 distinguishes model fitting, policy selection, interface development and outcome evaluation. Run-family grouping reduces overlap between these roles. It does not remove the publisher partitions' distribution differences. The separate confirmation and predictor-extension partitions are shown to clarify their exclusion from the results reported here.",
        "Figure 2 distinguishes model fitting, policy selection, interface development and the three testing partitions. Run-family grouping reduces overlap between these roles. It does not remove the publisher partitions' distribution differences. Neither confirmation nor selected-predictor test outcomes are used for model or policy selection.")
    replace('Confirmation and predictor-extension allocations are separate studies and contribute no results to this manuscript.',
        'Confirmation uses 1,050 snapshots from runs 21–70. The selected-predictor closed-loop study uses 420 snapshots from runs 71–90. Their results are reported separately from the pilot.')
    replace('Tool-assisted prompts serialize scores to five decimal places. The tool-only baseline uses numerical scores before text serialization. This precision difference is an explicit implementation limitation.',
        'Tool-assisted prompts serialize scores to five decimal places. The original tool-only baseline uses scores before serialization, an explicit pilot limitation. Confirmation additionally includes a rounded tool reference. The selected-predictor evaluation compares assisted models with that five-decimal tool, including the same lowest-ID tie rule, to match their available numerical information.')
    replace('We have not performed a dedicated calibration experiment, so the present results cannot attribute tool behavior to calibrated or miscalibrated probabilities.',
        'The follow-up studies report log loss, multiclass Brier score and ten-bin top-label expected calibration error. These diagnose uncalibrated predictions without fitting calibration to testing outcomes. They do not establish that calibration causes any observed acquisition pattern.')
    methods = '''### 7.1 Confirmation and extension design

For a policy at fixed budget and predictor, let a(r,c,k) indicate a correct prediction for run family r, class c and decoding seed k. Its family score is the sum of these indicators divided by 21K, where K is the number of decoding seeds. Balanced accuracy is the mean family score because each family contains exactly one snapshot from each of the 21 classes. A paired contrast averages the differences of policy and reference family scores on matching identities. Deterministic policies use K = 1 and their scores are broadcast when compared with stochastic policies. Thus additional decoding seeds do not create additional independent simulation families.

Probability diagnostics are computed on individual recorded predictions and averaged over the evaluated episodes. Log loss is the mean negative natural logarithm of the true-class probability clipped below at 1e-12. Multiclass Brier score is the mean sum of squared probability errors over all 21 classes, without division by the class count. Top-label expected calibration error weights the absolute confidence–accuracy difference in each of ten equal-width confidence bins by its fraction of episodes. Bins include their lower boundary, exclude their upper boundary, and include confidence one in the last bin. These summaries do not add independent simulation families.

The confirmation plan was frozen locally on 26 September 2026 after pilot inspection. It retains the original predictor, model artifacts, interface, rankings and one-time plans, and evaluates testing runs 21–70 at budgets one, three and five using decoding seeds 17, 29 and 43. Every budget starts independently from an empty observed set. There are 37,800 model episodes and 35,700 reference records. The four primary budget-three comparisons remain assisted model minus forward selection and minus its same-model fixed plan. Each 98.75% marginal family-bootstrap interval uses 20,000 replicates. The allocation controls four comparisons approximately through the Bonferroni construction. Other budget and policy intervals are descriptive. The protocol is locally timestamped rather than publicly preregistered.

The separate extension plan was frozen on 27 September before its outcomes. It compares four masked predictors with 64 trees and depth/leaf combinations (14,3), (24,3), (unlimited,3) and (unlimited,1). All share the original training snapshots, scaling and 17 mask assignments. Selection maximizes mean balanced accuracy on twelve shared three-sensor masks generated with seed 20260926 on validation runs 351–400. Candidate order resolves ties. A dedicated full-observation, unlimited-depth, leaf-three classifier provides a validation diagnostic only. It has a different training objective and is not an acquisition comparator. No testing outcomes enter this selection, and no probability calibration is fitted.

Matched interface interventions use the original predictor, twenty previously selected development identities and seeds 17, 29, 43, 61 and 79. The six arms preserve the original interface, permute listing order, consistently remap sensor IDs, anonymize descriptions, remove scores or shuffle the score-to-sensor mapping. The same seed-specific permutation is applied consistently within a trajectory, and remapped requests are translated back before measurement revelation. Each model-arm cell contains 100 episodes, giving 1,200 episodes. Scores are computed in every arm before serialization to keep that work comparable. Outcomes are paired with the original arm by identity and seed. The sample is not class-balanced, so its reported accuracy is ordinary development accuracy. No population mechanism test or latency effect is inferred from these small controls.

Closed-loop testing uses runs 71–90 and the selected predictor. Forward selection is reselected using validation data only. Four model policies each contribute 2,100 episodes from 420 identities and five decoding seeds, and three deterministic references contribute 420 records each, for 9,660 records. The adaptive reference uses the same rounded scores available to assisted models. Acquisition is rerun with the selected predictor, rather than rescoring old paths. Crossed bootstrap resampling independently draws twenty run families and five seeds with replacement and averages their Cartesian product, retaining all classes within each family. Deterministic references are replicated across seed positions for paired contrasts. There are 20,000 replicates and four primary assisted-minus-forward/tool contrasts, each with a 98.75% marginal interval. Limited family and seed counts constrain coverage. A gain above three percentage points requires the lower bound to exceed three points, and nonsignificance is not equivalence.

Every planned record is audited against snapshot values, legal actions, disclosed probability states, tool scores, parsed model requests, fallbacks and final predictions. Scientific input hashes and exact identity sets are verified before manuscript integration. The closed-loop worker was restarted after two host interruptions on 30 September, at 16:04:16 and 19:42:30 UTC. Saved records were retained, their byte prefixes checked, and the frozen protocol left unchanged. Original initialization metadata is archived for each session. Timing summaries combine sessions and are observational, excluding controlled speedup, energy and carbon claims.

'''
    text = text.replace('## 8 Results\n', methods + '## 8 Pilot results\n', 1)
    # All original pilot evidence, including unfavorable findings, remains unchanged.
    before, tail = text.split('## 9 Discussion and threats to validity', 1)
    availability = tail.split('## Data and code availability', 1)[1]
    result = ['## 9 Confirmation and follow-up results\n',
        'All planned studies have completed their record audits. They were specified after inspection of the pilot and are reported in their own partitions. Tables 8–13 report the complete planned follow-up evidence. Supplemental numerical sections retain all budgets, policies and probability diagnostics.\n',
        '### 9.1 Independent confirmation\n',
        'Table 8. Confirmation accuracy and recorded episode runtime at three sensors, with the full-sensor reference shown separately. Each policy uses 1,050 identities from 50 run families. Models average three decoding seeds within identities. Intervals are descriptive pointwise 95% family-bootstrap intervals. Median and 95th-percentile seconds exclude loading and are observational.\n',
        table(['Policy','Balanced accuracy (%)','95% interval (%)','Median s','95th percentile s'],[
            [LABELS.get(x['policy'],x['policy']),pct(x['balanced_accuracy']),interval(x['ci95']),f"{x['median_seconds']:.4f}",f"{x['p95_seconds']:.4f}"]
            for x in c['summaries'] if x['budget'] in (3,41)]),
        'Table 9. All four frozen primary confirmation contrasts at three sensors. Differences are assisted model minus reference in percentage points. Intervals use 98.75% marginal coverage under the four-comparison Bonferroni allocation.\n',
        table(['Model','Reference','Difference (pp)','98.75% interval (pp)'],[
            [x['model'].capitalize(),LABELS.get(x['reference'],x['reference']),pct(x['difference']),interval(x['simultaneous_interval'])] for x in c['primary_contrasts']]),
        'The confirmation reproduces the unfavorable budget-three comparison with forward selection. Qwen’s difference is −6.00 percentage points with a 98.75% interval [−8.19, −3.90], and Granite’s is −11.46 [−13.75, −9.24]. Qwen’s difference from its fixed plan is exactly zero on these recorded identities, while Granite is lower by 5.46 points [−6.89, −4.06]. Identical recorded outcomes are not population equivalence. Qwen again follows one trajectory, IDs 1, 2 and 3, whereas Granite has 35 assisted trajectories. None of these budget-three model episodes uses fallback.\n',
        'Budget changes qualify the comparison. At one sensor, assisted Qwen and the rounded tool both score 14.48%, versus 10.29% for forward selection. At five sensors, the rounded tool scores 31.05%, versus 21.05% for Qwen and 15.75% for Granite. These secondary point estimates describe the recorded budgets and are not adjusted superiority tests. The frozen primary reference is not retrospectively changed to the best testing policy.\n',
        'Figure 9 shows independently executed model budget comparisons on the confirmation partition. Connecting measured budgets is descriptive and does not establish a monotone policy response. The full numerical budget results and probability diagnostics accompany the reproducibility package.\n',
        '![Confirmation budget comparison](postreview/confirmation_budget.png)\n',
        'Figure 9. Confirmation at one, three and five sensors. Each point uses 1,050 identities from 50 run families. Model correctness averages three decoding seeds within each identity. Error bars are descriptive pointwise 95% family-bootstrap intervals from 20,000 resamples, not simultaneous intervals. Panels share a 0–100% scale. Visualization code was prepared with ChatGPT/Codex from recorded audited results.\n',
        '### 9.2 Validation diagnostic foundation\n',
        'Table 10. Predictor candidates assessed on validation data only. Mean masked accuracy averages twelve shared three-sensor masks. Full-observation columns evaluate the same masked classifier with all channels. These selection scores are not unbiased estimates of testing performance. ECE is ten-bin top-label expected calibration error.\n',
        table(['Candidate','Masked accuracy (%)','Full accuracy (%)','Log loss','Brier','ECE'],[
            [x['name'],pct(x['validation_mask_mean_accuracy']),pct(x['full']['balanced_accuracy']),f"{x['full']['log_loss']:.4f}",f"{x['full']['brier']:.4f}",f"{x['full']['ece10']:.4f}"] for x in d['candidates']]),
        f"The selected candidate is {d['selected']}. The separately trained full-observation diagnostic reaches {pct(d['dedicated_full_observation']['metrics']['balanced_accuracy'])}% validation balanced accuracy. This diagnostic changes the learning objective and cannot be used to attribute a causal acquisition-policy effect. Proper scoring rules and bin-dependent ECE summaries describe probability quality without a fitted calibration intervention.\n",
        '### 9.3 Matched interface interventions\n',
        'Table 11. Development interface controls. Each row contains 100 episodes on the same twenty identities across five seeds. Changed paths are paired with the original arm after inverse ID remapping. Accuracy is ordinary development accuracy. The original row supplies the comparator.\n',
        table(['Model','Arm','Changed paths (%)','Development accuracy (%)'],[
            [x['model'].capitalize(),x['arm'],pct(x['fraction_paths_changed']),pct(x['accuracy'])] for x in m['cells']]),
        'Changing listing order alters all 100 control paths for each model. For Qwen, consistent ID remapping changes 13% of paths, anonymized descriptions change 19%, and removing or shuffling scores changes none. Granite’s paths change under every intervention, including 79% after score removal and 78% after score shuffling. These paired observations support presentation sensitivity within this sample. They do not prove universal score neglect or a general reasoning mechanism. The ordinary development accuracies in Table 11 are not evidence of held-out diagnostic gains.\n',
        'These interventions measure the response of the complete prompt-and-selector configuration to specific interface changes. A changed path can follow presentation changes without improving diagnosis, and an unchanged path does not prove that the model ignores every observation. Small, reused development samples constrain generalization. No classwise or population mechanism significance is claimed.\n',
        '### 9.4 Closed-loop evaluation with the selected predictor\n',
        'Table 12. Independent testing on runs 71–90 at budget three, with the full-sensor reference at 41 channels. Model rows average five seeds across 420 identities. Intervals are descriptive 95% crossed run-family/seed bootstrap intervals. ECE uses ten confidence bins. Median runtime is recorded episode seconds.\n',
        table(['Policy','Accuracy (%)','95% interval (%)','Log loss','Brier','ECE','Median s'],[
            [LABELS.get(x['policy'],x['policy']),pct(x['metrics']['balanced_accuracy']),interval(x['ci95']),f"{x['metrics']['log_loss']:.4f}",f"{x['metrics']['brier']:.4f}",f"{x['metrics']['ece10']:.4f}",f"{x['median_seconds']:.4f}"] for x in l['summaries']]),
        'Table 13. All four assisted-minus-reference contrasts in the selected-predictor evaluation. Differences are percentage points, with 98.75% marginal crossed-bootstrap intervals from 20,000 replicates. The rounded tool matches assisted prompt precision.\n',
        table(['Model','Reference','Difference (pp)','98.75% interval (pp)'],[
            [x['model'].capitalize(),LABELS.get(x['reference'],x['reference']),pct(x['difference']),interval(x['ci9875'])] for x in l['contrasts']]),
        'Figure 10 visualizes the four comparisons in Table 13 on a common difference scale. Supplemental Figures S5–S6 show class recall as a heatmap and spider comparison for this final study, with exact class values in the accompanying CSV.\n',
        '![Selected-predictor primary contrasts](postreview/closed_loop_contrasts.png)\n',
        'Figure 10. Assisted model minus reference in the selected-predictor study on testing runs 71–90. Points average five decoding seeds on 420 identities. Error bars are 98.75% marginal crossed run-family/seed bootstrap intervals under the four-comparison Bonferroni allocation. Solid and dashed vertical lines mark zero and the three-percentage-point target. A target-sized population gain requires the lower bound to exceed the target. Visualization code was prepared with ChatGPT/Codex from recorded audited results.\n',
        'Assisted Qwen is lower than the rounded tool by 8.10 percentage points [−12.38, −3.57], and assisted Granite by 14.81 [−19.71, −9.71]. Granite is also lower than forward selection by 10.52 points [−14.71, −5.95]. Qwen’s forward contrast is −3.81 [−7.14, 0.24]. Its interval includes zero, so this comparison does not establish a directional population difference or equivalence. Its upper bound remains below the three-point benefit target. All intervals quoted here use 98.75% marginal crossed-bootstrap coverage.\n',
        'Recorded median episode times are 3.56 seconds for assisted Qwen, 6.30 seconds for assisted Granite and 27.8 milliseconds for the rounded tool. These observational summaries exclude initial loading, combine execution sessions and do not establish precise speedup ratios. No consistent incremental diagnostic benefit over the matched tool is demonstrated at this budget.\n',
        'The selected-predictor study reruns acquisition on new snapshots. Its absolute accuracy differences from the confirmation are not a paired estimate of changing the predictor because the testing families also change. All planned contrasts are retained regardless of direction. The archive supplies per-seed accuracies and individual probability vectors.\n']
    discussion = '''## 10 Discussion and threats to validity

The study asks whether a language-model selector adds diagnostic value after the predictor and numerical acquisition tools are already available. The pilot's positive-effect criterion was not met, and that unfavorable decision remains part of the scientific record. Confirmation, validation diagnostics and closed-loop testing extend the evidence without relabeling post-pilot analyses as original hypotheses. Comparisons are conditional on their fitted predictor, interface, measurement budget and test partition. Confirmation places both assisted models below the frozen forward reference, while the selected-predictor study places both below the matched entropy tool. The Qwen–forward interval in that final study crosses zero and must remain inconclusive. Thus the evidence supports a configuration-specific absence of demonstrated incremental benefit over the tool, rather than a universal claim that language models cannot help acquire information.

### 10.1 Interpreting diagnostic and behavioral evidence

The four primary contrasts in each testing study quantify differences against explicit references and report uncertainty. Positive point estimates alone do not establish a minimum useful population gain, and zero or nonsignificant estimates do not establish equivalence. The same-model fixed plan controls the cost of repeated invocation imperfectly because assisted acquisition also adds observed values and tool scores. The separate rounded tool comparison more directly assesses the selector's incremental contribution given that numerical information.

The original Qwen trajectories remain concentrated on IDs 1, 2 and 3, whereas Granite varies its acquisitions. The matched interventions add evidence about sensitivity to listing order, remapped IDs, descriptions and scores within a development sample. They do not verify a reasoning mechanism. The class heatmap, spider chart and confusion matrices describe pilot failure patterns and must not be mistaken for classwise confirmation tests. Their common scales retain low or zero observed recalls instead of hiding them in an aggregate score.

### 10.2 Predictor quality and baseline scope

The original full-sensor accuracy of 40.48% is a material diagnostic limitation. Replay, validation-only selection and subsequent closed-loop testing address different aspects of predictor dependence. Selecting a predictor on twelve static masks does not guarantee performance on every adaptively selected subset. No probability calibration is fitted, and the nearest-donor tool uses only eight training donors. Its heuristic scores can be negative and need not equal conditional mutual information. Higher confidence need not mean a more correct diagnosis.

Implemented references include static rules, validation-selected forward acquisition, a nearest-donor entropy policy and one-time model plans. Competitive learned, amortized and non-myopic acquisition methods are not reproduced. Accordingly, these results cannot establish state-of-the-art performance or generalize the behavior of one tool to all classical acquisition algorithms. Predictor configurations have common training rows and tree counts, but this does not equalize effective capacity or search effort across all policy families. The models are off-the-shelf quantized systems without task fine-tuning, and alternative prompts, fault taxonomies and trained selectors remain untested.

### 10.3 Sampling and computational limits

One simulator, one snapshot time, one quantization level and one machine constrain transfer. Training and testing fault durations differ, and actual generator seeds are unavailable. Run identifiers are grouped across all classes as seed families based on publisher partition information. Confirmation adds 50 families, while the selected-predictor study has twenty families and five decoding seeds. Bootstrap coverage is approximate with finite clusters and seeds. Independent fits of the selected predictor are not included.

Recorded median and 95th-percentile runtime characterize the available executions. Warm prefix reuse, policy execution order, system activity and host restarts affect them. Process resident memory does not equal complete unified-memory demand. Pretraining is excluded, local fitting and setup are retained separately, and no hardware energy or carbon measurements were made. The paper therefore reports computational observations rather than precise speedup, monetary cost or environmental savings.

### 10.4 Evidence coverage

Table 14 states the coverage and unresolved questions. Uniform sensor-count budgets do not represent heterogeneous measurement price, delay or energy. The replay setting freezes values while the policy selects them and excludes physical sensor control.

Table 14. Evidence coverage and remaining limits. Unmeasured properties are not inferred from legal execution or accuracy.

| Dimension | Covered | Not established |
| --- | --- | --- |
| Measurement access | Sequential revelation of recorded channels | Physical sensing or streaming sensor control |
| Diagnosis | Three separate test partitions, 21 classes, sample 320 | External plants, early detection or robustness to shift |
| Acquisition policies | Selected rules, matched tool and two local models | Superiority to learned or non-myopic methods |
| Behavior | Logged paths and six development interface arms | General reasoning or population mechanism effects |
| Predictor | Validation selection and new closed-loop testing | Paired causal predictor effect or independent fit variability |
| Resources | Sensor count, recorded runtime and process memory | Energy, financial savings or operational safety |

### 10.5 Future research

The next comparison should reproduce learned, amortized and non-myopic acquisition baselines under common splits, information access and validation budgets. Predictor fitting and probability calibration should be separated from acquisition-policy selection, with independent predictor fits and a larger seed sample evaluated on untouched outcomes. Randomized interface interventions should be extended beyond the twenty development identities and analyzed with an explicit mechanism estimand.

Temporal evaluation should test early fault onset, noisy or unavailable channels, stale telemetry and changing process regimes. Unequal acquisition costs require a cost budget and an explicit stopping rule, together with accuracy–cost and accuracy–delay curves. Transfer should be evaluated on an unrelated industrial dataset before considering a non-intervening plant trial. Claims about energy or sensing savings require measured hardware power or sensor-level data. Advisory measurement selection must remain distinct from safety-critical plant control.

## 11 Conclusion

This study records 52,440 model episodes across an initial pilot, independent confirmation, matched development controls and selected-predictor closed-loop testing. It evaluates language-model consultation as an acquisition policy under a shared predictor, preserving the original unfavorable pilot criterion and all planned follow-up contrasts. Confirmation retains the unfavorable comparison with forward selection. In the selected-predictor test, assisted Qwen and Granite score 22.86% and 16.14%, versus 30.95% for the matched entropy tool, with both adjusted tool contrasts below zero. Qwen versus forward remains inconclusive. Listing-order controls demonstrate presentation sensitivity within the development sample. The complete evidence distinguishes legal requests, changes in acquisition paths and diagnostic utility. Interpretation depends on the predictor, budget, interface and testing distribution. The results support evidence-based assessment of incremental model value in retrospective industrial diagnosis, with learned acquisition baselines, temporal evaluation and external deployment remaining open.

'''
    text = before + '\n'.join(result) + '\n' + discussion + '## Data and code availability' + availability
    text = text.replace('The local research package contains code, source and model hashes, the analysis plan and amendments, split rules, predictions, action traces, timings and analysis outputs.',
        'The complete local reproducibility package contains source snapshots, code, fitted predictors, source and model hashes, frozen analysis plans and amendments, split rules, all planned predictions and acquisition traces, timings, statistical outputs and restart provenance. Supplemental numerical sections report all confirmation budgets, validation diagnostics and interface-control cells.')
    out = ROOT / 'paper/manuscript.md'
    text=text.replace('](study_design.png)','](study_design_full.png)')
    out.write_text(text)
    assert set(map(int,re.findall(r'^\[(\d+)\]',text,re.M))) == set(range(1,23))
    record = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_pilot_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
        manuscript_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        complete_audited_studies=True, main_manuscript_integrated=True,
        scientific_review_complete=False, all_pages_visually_reviewed=False,
        submission_ready=False, abstract_words=len(abstract.split()))
    (ROOT/'paper/postreview/integration_record.json').write_text(json.dumps(record,indent=2))
    manifest_path=ROOT/'paper/postreview/build_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    manifest.update(main_manuscript_integrated=True,integrated_markdown_sha256=record['manuscript_sha256'],scientific_review_complete=False,visual_review_complete=False,submission_ready=False)
    manifest_path.write_text(json.dumps(manifest,indent=2))
    print(out)


if __name__ == '__main__':
    main()

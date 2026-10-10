"""Refresh submission navigation and drafts after complete manuscript generation."""
import datetime
import json
from pathlib import Path
from submission_gate import scientific_inputs

ROOT=Path(__file__).resolve().parents[1]


def main():
    issues=scientific_inputs(ROOT)
    if issues:raise RuntimeError(str(issues))
    for rel in ('paper/manuscript.md','paper/manuscript.docx','paper/manuscript.pdf','paper/latex/manuscript.tex','paper/latex/manuscript.pdf','paper/postreview/reproducibility_complete.zip'):
        if not (ROOT/rel).is_file():raise FileNotFoundError(rel)
    meta=json.loads((ROOT/'submission/author_metadata.json').read_text())
    archive=json.loads((ROOT/'paper/postreview/reproducibility_archive.json').read_text())
    title=(ROOT/'paper/manuscript.md').read_text().splitlines()[0].removeprefix('# ')
    highlights=[
        '52,440 model episodes separate acquisition behavior from diagnostic utility.',
        'Confirmation uses 50 untouched simulation families and three sensor budgets.',
        'A validation-selected predictor is tested with new closed-loop acquisition.',
        'Matched controls probe sensor order, identifiers, descriptions and tool scores.',
        'All primary comparisons report multiplicity-adjusted bootstrap intervals.'
    ]
    assert 3<=len(highlights)<=5 and all(len(x)<=85 for x in highlights)
    (ROOT/'submission/highlights.txt').write_text('\n'.join('- '+x for x in highlights)+'\n')
    corr=meta['corresponding_author']
    letter=f'''Dear Editor,

Please consider our manuscript, “{title}”, for Engineering Applications of Artificial Intelligence.

The engineering application is budget-constrained access to recorded sensor measurements for retrospective fault diagnosis on public Tennessee Eastman Process simulations. The contribution is an empirical evaluation of the incremental diagnostic value of repeated small-language-model consultation under a shared predictor. The manuscript separates valid execution, acquisition behavior and diagnostic utility, rather than presenting interface reliability as evidence of improved diagnosis.

The complete evaluation contains 52,440 model episodes across a pilot, confirmation on untouched simulation families, matched development controls and closed-loop testing with a predictor selected using validation data alone. All planned primary contrasts and unfavorable results are retained. Source snapshots, fitted predictors, action records, statistical outputs, frozen plans and restart provenance are prepared in a reproducibility archive. The paper states its limits, including one simulator, retrospective snapshots and untested learned or non-myopic acquisition baselines. It makes no state-of-the-art or deployment claim.

The anonymous manuscript, separate author page, editable sources, figure assets, highlights and supplemental evidence are prepared. Use of ChatGPT/Codex in research preparation is disclosed separately from the Qwen and Granite systems evaluated experimentally. Funding and competing-interest statements reflect the information supplied by the authors.

Yours sincerely,
{corr['name']}
Corresponding author
{corr['institution']}
{corr['email']}
'''
    (ROOT/'submission/cover_letter_draft.md').write_text(letter)
    record={
        'prepared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Complete audited planned studies and integrated manuscript',
        'relative_path':'paper/postreview/reproducibility_complete.zip',
        'bytes':archive['bytes'],'sha256':archive['sha256'],
        'archive_crc_and_all_source_hashes_verified':True,
        'author_identity_scan_passed':False,
        'publicly_deposited':False,'verified_public_url':None,
        'author_approved_nonsharing_explanation':None,
        'complete_study_archive_pending':False,
        'note':'Local complete archive prepared. Final anonymity review, repository deposit decision and exact-version author approval remain required.'}
    (ROOT/'submission/archive_record.json').write_text(json.dumps(record,indent=2))
    (ROOT/'README.md').write_text(f'''# {title}

## Open the complete manuscript

- `paper/manuscript.docx` is the complete editable anonymous manuscript.
- `paper/manuscript.pdf` is its Word-rendered reading copy.
- `paper/manuscript.md` is the evidence-linked complete text source.
- `paper/latex/manuscript.tex` and `references.tex` are the editable LaTeX sources, with required assets and a compiled PDF.
- `submission/title_page.docx` contains the separate author details and supplied declarations.
- `submission/highlights.txt` and `cover_letter_draft.md` are editable submission drafts.
- `paper/postreview/reproducibility_complete.zip` packages all completed scientific evidence and sources. This is a local archive, not a public deposit.

The original `paper/pilot_manuscript.md` and `.docx` remain as the pilot record. Use `manuscript.docx` for the full paper.

## Evidence and remaining submission steps

All planned scientific studies have completed their record audits: 5,040 pilot model episodes, 37,800 confirmation model episodes, 1,200 development-control episodes and 8,400 selected-predictor model episodes. The final study also contains 1,260 deterministic reference records. Confirmation contains 35,700 reference records. The studies use separate testing run families and are reported separately.

The complete manuscript has been generated. Scientific interpretation review, inspection of every final page, exact-version author approval, originality/exclusive-submission confirmation and a repository link or author-approved non-sharing explanation remain required. These are tracked by `src/submission_gate.py`. No acceptance or submission-readiness guarantee is asserted.

## Scope and reproducibility

Sensor acquisition means sequential access to recorded measurements under a fixed sensor-count budget. It does not mean buying physical sensors or controlling a plant. The study uses Tennessee Eastman simulation snapshots, not measurements from a deployed plant.

`src/` contains experiment, audit, analysis, visualization and document builders. `results/` contains all episode records, audited analyses and fitted predictors. `data/` contains the four prepared snapshot arrays. `provenance/` contains frozen plans, source/model hashes and per-session restart metadata. Source data are available at https://doi.org/10.7910/DVN/6C3JR1. Model revisions and checksums identify the official downloadable weights.

Install the pinned dependencies in an isolated Python environment. Follow the full archive README and frozen plans for replay or re-execution. The recorded environment uses Metal on an Apple M2 with 8 GiB memory. Hardware changes invalidate direct timing comparisons. Do not launch duplicate live workers. The source and model binaries in the local delivery are distinct from the anonymous reproducibility ZIP, which excludes author contact files and large model binaries.
''')
    (ROOT/'submission/remaining_confirmations.md').write_text('''# Final author and archive decisions

Names, author order, affiliation, emails, corresponding author, no specific funding, no competing interests and the reported shared research perspectives/manuscript review are already recorded. They do not need to be supplied again.

After final scientific and visual review, confirm approval of the exact final manuscript and AI disclosure, confirm originality and no simultaneous submission, and approve an accurate CRediT mapping of the reported contributions. Supply or authorize the chosen repository deposit, or provide a genuine non-sharing explanation if applicable. Do not invent a DOI, contribution percentages or confidentiality restriction.
''')
    (ROOT/'submission/SUBMISSION_STATUS.md').write_text('''# Submission preparation status

All planned experiments and record audits are complete. The full manuscript is in paper/manuscript.docx, manuscript.md and manuscript.pdf, with compiled LaTeX sources in paper/latex. Complete evidence is packaged in paper/postreview/reproducibility_complete.zip. The original pilot source remains separately identified.

Submission is not yet certified. Final scientific interpretation and every-page visual review, exact-version author approval and AI-disclosure approval, originality/exclusive-submission confirmation, accurate CRediT roles and a repository link or author-approved non-sharing explanation remain to be finalized. Author names, affiliation, contact details, corresponding author, no specific funding and no competing interests are already recorded.

Run python src/submission_gate.py for the current evidence-bound status. A successful experiment audit alone does not certify the publication declarations or document layout.
''')
    (ROOT/'paper/author_information_needed.md').write_text('''# Author information status

The author names, order, common affiliation, emails, corresponding author, funding and competing-interest information are recorded in submission/author_metadata.json and the separate title_page.docx. They do not need to be supplied again. Only exact-final-version approval, AI disclosure approval, originality/exclusive-submission confirmation and an accurate CRediT mapping remain for the final author step. Keep all author details separate from the anonymous manuscript.
''')
    (ROOT/'submission/data_availability_draft.md').write_text('''# Data and code availability draft

The source Tennessee Eastman simulation data are available through Harvard Dataverse at https://doi.org/10.7910/DVN/6C3JR1, version 1.0. The complete local reproducibility archive includes prepared snapshots, fitted predictors, code, frozen plans, all planned recorded predictions and actions, statistical outputs and restart provenance. The archive is paper/postreview/reproducibility_complete.zip and its current hash is recorded in submission/archive_record.json.

A public archive link has not yet been assigned. The journal requires repository deposit and citation, or a genuine explanation if sharing is not possible. Do not insert a fabricated DOI or unsupported restriction. The final availability statement must be updated after the author-approved deposit decision.
''')
    (ROOT/'paper/review_response_status.md').write_text('''# Completed experimental revisions and review boundaries

The pilot remains part of the final paper, including its unfavorable primary decision. Confirmation and the separate extension were locally frozen after pilot inspection, and this timing is disclosed.

| Review concern | Completed revision | Interpretation boundary |
| --- | --- | --- |
| Independent testing | Confirmation on 50 new run families at three budgets | One simulator and one snapshot time |
| Weak diagnostic predictor | Four validation candidates and new closed-loop testing | No paired causal predictor comparison across test partitions |
| Probability quality | Log loss, multiclass Brier and ten-bin calibration summaries | Uncalibrated predictions, no calibration intervention |
| Unequal tool precision | Five-decimal entropy-tool comparator | Learned and non-myopic baselines remain untested |
| Acquisition mechanisms | Six paired development interface arms across five seeds | Small reused development sample, no general reasoning claim |
| Decoding variability | Five-seed crossed bootstrap in the selected-predictor test | Finite family and seed coverage remains approximate |
| Resource transparency | Recorded timings, process memory and session provenance | No energy, carbon or precise speedup estimates |
| Reproducibility | Full source, snapshots, fitted predictors and audited records | Public deposit and final author approval remain pending |

The complete manuscript requires scientific interpretation review and every-page visual inspection. This file records completed experimental work, not an acceptance recommendation or readiness certificate.
''')
    (ROOT/'REPRODUCIBILITY.md').write_text('''# Reproducing the complete study

The main source is paper/manuscript.md. The preserved pilot source is paper/pilot_manuscript.md. Confirmation and the selected-predictor extension have separate frozen plans and testing partitions. Never pool their records or shorten a study on the basis of outcomes.

## Dependencies and inputs

Use Python 3.11 or later in an isolated environment with the pinned requirements.txt packages. Document rendering also uses requirements-render.txt, a compatible LibreOffice/document renderer and Tectonic or XeLaTeX for the alternate PDF. The measured inference environment is llama-cpp-python 0.3.19 with Metal on an Apple M2 and 8 GiB memory. Different hardware and builds do not reproduce the timing conditions.

The data folder contains four prepared sample-320 snapshot arrays. Source files can be retrieved from https://doi.org/10.7910/DVN/6C3JR1, version 1.0, and verified against the retained publisher checksums before extraction. Model URLs, repository revisions and local hashes identify the official Q8_0 weights. The anonymous complete ZIP excludes those large downloadable binaries and author contact files. Fitted predictors and snapshots are included.

## Audit and analysis

Run python src/audit_results.py for the pilot and python src/submission_gate.py --science-only for exact completed-study identity sets, markers and frozen hashes. Confirmation replay/analysis is implemented in src/analyze_confirmation.py. The frozen revision_closed_loop.py skips existing identity keys and reproduces the final audit and crossed bootstrap after all 9,660 records are present. Run re-analysis in a separate extracted copy because scripts write outputs and model initialization metadata. Retain existing provenance before re-execution.

For independent full model execution, first inspect provenance/confirmation_plan_20260926.json and review_revision_plan.json. Run src/run_confirmation_pipeline.py followed by src/run_review_revision.py sequentially. Do not launch duplicate live workers on the preparation machine. Resume only missing records with unchanged settings. Restart provenance records preserved byte-prefix hashes and per-session initialization metadata.

## Figures and manuscript

Pilot figures use make_diagnostic_figures.py, make_spider_chart.py and make_submission_diagnostics.py. Full-study figures use make_full_study_figures.py after every planned audit passes. build_postreview_results.py creates the evidence sections and refuses to overwrite an existing numerical build. integrate_full_study.py preserves the pilot source and writes the complete manuscript from all audited analyses.

Build the full Word document with python src/build_manuscript_docx.py, then render it with src/render_paper.py and inspect every page. Export the full LaTeX source with python src/export_latex.py. Compile manuscript.tex from paper/latex with tectonic --untrusted --keep-logs manuscript.tex or a compatible XeLaTeX installation. The legacy write_paper.py is for the preserved pilot and refuses to overwrite an existing full manuscript.

The authorized watch_completion.py performs the audited integration, artifact builds and synchronization. It does not infer human author declarations, publish a repository or submit the paper. Its state is results/completion/status.json.

## Final review and archive

Rebuild the complete archive with python src/build_reproducibility_archive.py --scope complete --output paper/postreview/reproducibility_complete.zip after changes to manuscript sources or figures. Each included file is bound to its size and SHA-256, and ZIP integrity and all archived hashes are checked. A local ZIP is not a public deposit.

Run python src/submission_gate.py for submission checks. Exact-final-document scientific and visual review, author approvals, AI disclosure approval, originality/exclusive-submission confirmation and a repository URL or author-approved genuine non-sharing explanation remain necessary. Review flags must record work actually performed and bind to the final document hashes. The software gate does not judge novelty or guarantee acceptance.
''')
    from pypdf import PdfReader
    from submission_gate import digest
    latex=ROOT/'paper/latex/manuscript.pdf'
    (ROOT/'provenance/latex_build_verification.json').write_text(json.dumps({
        'built_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'compiler':'Tectonic','compiled':True,'pages':len(PdfReader(latex).pages),
        'pdf_sha256':digest(latex),'tex_sha256':digest(ROOT/'paper/latex/manuscript.tex'),
        'visually_reviewed':False,'note':'Compilation succeeded. Final layout must be inspected and recorded against this exact PDF.'},indent=2))
    print('Complete manuscript navigation and submission drafts refreshed')


if __name__=='__main__':main()

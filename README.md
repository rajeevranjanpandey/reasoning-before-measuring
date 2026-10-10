# Reasoning Before Measuring: Evaluating Small Language Models for Industrial Sensor Acquisition

Research project for **Budgeted Industrial Sensor Selection with Small Language Models: A Paired Evaluation with Thinking Disabled**.

This is an author-owned private working repository. It is not an anonymous reviewer link, a public archival deposit or a claim of journal acceptance. Anonymous reviewer access and a persistent archival identifier remain pending.

## Complete reproducibility package

Download all three `Review_Reproducibility_Part_*.zip` assets from the [v1.0 review-package release](https://github.com/rajeevranjanpandey/reasoning-before-measuring/releases/tag/v1.0-review-package). Extract all three into the same empty directory. These are independent ZIPs, not split volumes. Together they contain 155 files: research code, sample-320 snapshots, fitted predictors, frozen plans/amendments, model-download provenance, every analysed episode record, archived development checks, statistical outputs and supplemental verification text. PDF and LaTeX files are deliberately excluded, including inside archives.

`MANIFEST_SHA256.json` binds the extracted 155 files. The release also contains `RELEASE_SHA256.json`, which binds the downloaded ZIPs. Verify hashes before analysis. With Python and NumPy, run `python reviewer_record_check.py` from the extracted directory. It writes `review/episode_evidence_audit.json`; running the checker on the assembled delivery reproduced the audit byte for byte. No new model experiment was run during correction.

Original third-party full RData and GGUF downloads are not redistributed; obtain them from the cited Dataverse/model records using the pinned versions and recorded hashes. Dependencies in `requirements.txt` describe the recorded experiment; timing and inference results can depend on hardware and runtime. Use a fresh extracted copy because original research scripts can write analysis outputs.

## Files in this repository

- `src/`: original research scripts for readable inspection.
- `reviewer_record_check.py`: portable record audit and descriptive sign-flip re-analysis.
- `Code_And_Analysis.zip`: 144 smaller project files, preserving their original directories. Eleven larger logs/predictor files are available in the complete release package; the code archive is not a replacement for that package.
- `manuscript_blocks.drawio`: editable equation/algorithm/architecture blocks.
- `Supplementary_CSV_Data.zip`: four audited CSV tables.
- `run_with_models.py`: downloads and verifies missing pinned weights before launching an unchanged research script.
- `test_model_restore.py`: a small runnable check for download/caching/checksum behaviour.
- `MODEL_DOWNLOADS.md`: exact model revisions, hashes and launcher instructions.

## Verified scope and remaining gates

The paper reports 52,440 analysed model/control episodes, excluding 80 final development checks and 140 earlier parser/runtime-check records. Qwen raw/assisted paths and predictions match across all 12,810 paired identities. The four final-study exact sign-flip checks are post hoc and conditional on sign-exchangeability; they do not change frozen contrasts, bootstrap intervals or decisions.

Final all-author approval, accurate historical preparation-tool model/version details, project licensing/archival terms and journal submission remain outstanding. No manuscript PDF, LaTeX source, author title page, competing-interest form or internal correspondence is included here. Source scripts are retained as original research code; any historical document-generation scripts are not the current manuscript deliverable.

# Reasoning Before Measuring: Evaluating Small Language Models for Industrial Sensor Acquisition

Research materials for the manuscript. Reviewer-facing access: https://anonymous.4open.science/r/reasoning-before-measuring-C66F/

## Complete research package

Download the whole anonymous repository as a ZIP and extract it locally. The `review_evidence/` directory contains 21 binary parts of the three original research archives. GitHub release attachments are not needed.

Run this from the extracted repository directory:

```text
python assemble_review_archives.py
```

The script reconstructs the three ZIPs in `review_archives/` and checks their SHA-256 hashes against `RELEASE_SHA256.json`. It preserves invalid existing archives and rejects corrupt or missing parts. Extract all three independent ZIPs into the same empty directory. Their combined contents are the 155 research files listed in `MANIFEST_SHA256.json`: research code, sample-320 snapshots, fitted predictors, frozen plans/amendments, model provenance, analysed episode records, archived development checks, statistical outputs and supplementary verification text.

With Python and NumPy installed, run `python reviewer_record_check.py` inside that assembled research directory. It produces `review/episode_evidence_audit.json`. The locally assembled delivery reproduced the recorded audit; no fresh model experiment was run. `EVIDENCE_PARTS_SHA256.json` records each repository part's size and hash.

## Model weights

Full third-party RData and GGUF model downloads are not redistributed. The approved delivery launcher `run_with_models.py` restores missing weights from the pinned official revisions, checks byte sizes and SHA-256, then runs an unchanged research script. See `MODEL_DOWNLOADS.md`. Copy the launcher into the extracted research directory to use it with those scripts. Model downloads are outside the recorded episode timings.

## Other files

- `src/`: readable source snapshots of the recorded research scripts.
- `Code_And_Analysis.zip`: the smaller 144-file project bundle plus its delivery README. It omits eleven larger evidence files and does not replace the complete package.
- `Supplementary_CSV_Data.zip`: four audited CSV tables.
- `manuscript_blocks.drawio`: editable equation, algorithm and architecture blocks.
- `test_assemble_review_archives.py` and `test_model_restore.py`: small-file delivery checks.
- `UPLOAD_SHA256.json`: hashes of the delivery files; scientific-file hashes remain in `MANIFEST_SHA256.json`.

## Scope

The paper reports 52,440 analysed model/control episodes, excluding 80 final development checks and 140 earlier parser/runtime checks. Qwen raw/assisted paths and predictions match across all 12,810 paired identities. Exact sign-flip checks are post hoc and conditional on sign-exchangeability; frozen contrasts, bootstrap intervals and decisions are unchanged.

Historical preparation-tool model/application versions were not recorded and are confirmed unrecoverable. This research deposit does not establish all-author approval, project licensing or journal submission. Manuscript PDFs, LaTeX sources, author documents, correspondence and model weights are excluded, including inside the reconstructed archives.

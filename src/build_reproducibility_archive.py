"""Package audited evidence without partial studies or local author contact files."""
import argparse,datetime,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def build_archive(root,output,scope='pilot'):
    root=Path(root);output=Path(output)
    audit=json.loads((root/'results/pilot_audit.json').read_text())
    if audit.get('complete') is not True:raise RuntimeError('Completed pilot audit required')
    for model in ('qwen','granite'):
        if sha(root/f'results/{model}_pilot.jsonl')!=audit['models'][model]['sha256']:
            raise RuntimeError('Pilot records changed since audit')
    if scope=='complete':
        from submission_gate import scientific_inputs
        issues=scientific_inputs(root)
        if issues:raise RuntimeError('Incomplete studies cannot enter the full archive: '+str(issues))
    files=[]
    def add(rel):
        p=root/rel
        if not p.is_file():raise FileNotFoundError(rel)
        if p.resolve()==output.resolve():raise ValueError('Archive cannot include itself')
        files.append(p)
    for name in ('requirements.txt','requirements-render.txt','REPRODUCIBILITY.md'):add(name)
    for p in sorted((root/'src').glob('*.py')):
        if p.name not in ('build_title_page.py','watch_completion.py','package_submission.py'):
            add(str(p.relative_to(root)))
    for p in sorted((root/'data').glob('*_snapshot.npz')):add(str(p.relative_to(root)))
    for name in ('analysis.json','summary.csv','pilot_audit.json','training.joblib','training_summary.json',
                 'classical_Testing_1_20.jsonl','qwen_pilot.jsonl','granite_pilot.jsonl',
                 'qwen_development.jsonl','granite_development.jsonl','qwen_fixed_plan.json','granite_fixed_plan.json',
                 'qwen_fixed_Testing_1_20.jsonl','granite_fixed_Testing_1_20.jsonl',
                 'interface_control.json','predictor_sensitivity.json','predictor_sensitivity_records.jsonl',
                 'sensitivity_predictor.joblib','runtime_chart.json','class_recall_spider.csv','class_recall_spider.json',
                 'confusion_matrices.csv','submission_diagnostics.json','execution_gate.json'):
        add('results/'+name)
    for name in ('analysis_plan.json','effective_analysis_plan.json','data_checksums.json','model_checksums.json',
                 'qwen_model.json','granite_model.json','teprob.f','qwen_chat_template.jinja','granite_chat_template.jinja',
                 'frozen_code_hashes.json','legal_action_amendment.md','prefix_cache_amendment.md',
                 'runtime_amendment.md','exploratory_reframing_20260921.md',
                 'confirmation_plan_20260926.json','review_revision_plan.json'):
        add('provenance/'+name)
    for name in ('references.json','reference_numbering.json','pilot_manuscript.md'):add('paper/'+name)
    for p in (root/'paper').glob('*'):
        if p.suffix in ('.png','.svg','.pdf') and p.name!='manuscript.pdf':add(str(p.relative_to(root)))
    for p in (root/'paper/supplementary').glob('*'):
        if p.is_file():add(str(p.relative_to(root)))
    if scope=='complete':
        for name in ('manuscript.md','manuscript.docx','manuscript.pdf'):
            if (root/'paper'/name).exists():add('paper/'+name)
        for p in (root/'paper/latex').rglob('*'):
            if p.is_file() and p.suffix in ('.tex','.png','.svg','.pdf'):add(str(p.relative_to(root)))
        for folder in sorted((root/'provenance').glob('restart_*')):
            for p in sorted(folder.glob('*.json')):add(str(p.relative_to(root)))
        for folder in ('results/confirmation','results/revision'):
            for p in (root/folder).rglob('*'):
                if p.is_file() and p.suffix in ('.json','.jsonl','.joblib'):add(str(p.relative_to(root)))
        for p in (root/'paper/postreview').glob('*'):
            if p.is_file() and p.name!='reproducibility_archive.json' and p.suffix in ('.md','.json','.csv','.png','.svg','.pdf'):add(str(p.relative_to(root)))
    files=sorted(set(files))
    manifest={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    models=[]
    for model,filename in [('qwen','Qwen3-1.7B-Q8_0.gguf'),('granite','granite-3.3-2b-instruct-Q8_0.gguf')]:
        meta=json.loads((root/f'provenance/{model}_model.json').read_text())
        assert filename in [x['rfilename'] for x in meta['siblings']]
        details=json.loads((root/'provenance/model_checksums.json').read_text())[filename]
        models.append(dict(filename=filename,revision=meta['sha'],download_url=f"https://huggingface.co/{meta['id']}/resolve/{meta['sha']}/{filename}",**details))
    readme=f'''# Reproducibility archive

Scope: {scope}. This package is a local preparation artifact, not a claim of public deposit or journal submission.

The pilot contains 5,040 paired model episodes on 420 snapshots from 20 run families. {'All separately planned studies have also completed their audits.' if scope=='complete' else 'Confirmation and post-review results are excluded because they are unfinished. The included future-study scripts and frozen plans are not evidence of completion.'}

## Contents and verification

MANIFEST_SHA256.json lists the size and SHA-256 of every included source artifact. The JSON records retain model calls, actions, seeds, predictions and timings. The original classifier, replay classifier, source snapshots and plotting data are supplied. No author email files, account credentials or model binaries are included.

Install the pinned packages in requirements.txt in an isolated Python environment. Run `python src/audit_results.py` to reproduce recorded pilot predictions and check identities, actions and inputs. Run `python src/analyze.py`, `python src/make_diagnostic_figures.py`, `python src/make_spider_chart.py` and `python src/make_submission_diagnostics.py` to regenerate pilot statistics and figures. Work in a separate extracted directory because these commands write regenerated artifacts. Full model re-execution is different from prediction replay and table regeneration.

For model re-execution, MODEL_DOWNLOADS.json identifies exact revision URLs, filenames, sizes and SHA-256 hashes. Download into a models directory and verify hashes. Follow the original publisher licenses. The model binaries are excluded to avoid duplicating approximately 4.5 GB of publicly available pretrained weights. The recorded inference environment used Metal on an Apple M2 with 8 GiB memory. Hardware and build changes invalidate direct timing comparisons, and cross-platform bitwise determinism is not claimed.

Source dataset: https://doi.org/10.7910/DVN/6C3JR1, version 1.0. The four extracted sample-320 snapshots are included. Source RData files can be obtained from that public record and verified against provenance/data_checksums.json before using src/prepare_data.py. Source-data attribution and terms remain those of the original dataset. No new software license is assigned by this packaging operation.

The paper directory contains the preserved anonymous pilot source, figure assets and any integrated complete manuscript that has been generated. The postreview directory identifies audited follow-up evidence. Author approval and visual review status are recorded separately. Local preparation scripts do not certify all-author approval, final scientific interpretation, layout review or submission readiness.
'''
    output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.building.zip')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p in files:z.write(p,str(p.relative_to(root)))
        z.writestr('README.md',readme)
        z.writestr('MODEL_DOWNLOADS.json',json.dumps(models,indent=2))
        z.writestr('MANIFEST_SHA256.json',json.dumps(manifest,indent=2))
        z.writestr('ARCHIVE_STATUS.json',json.dumps({'scope':scope,'complete_planned_studies':scope=='complete','publicly_deposited':False,'submission_ready':False,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP integrity failure')
        for rel,info in manifest.items():
            h=hashlib.sha256()
            with z.open(rel) as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            if h.hexdigest()!=info['sha256']:raise RuntimeError('Archive hash mismatch: '+rel)
    temp.replace(output)
    return {'path':str(output),'bytes':output.stat().st_size,'sha256':sha(output),'files':len(manifest),'scope':scope,'publicly_deposited':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',choices=['pilot','complete'],default='pilot');a=p.parse_args()
    print(json.dumps(build_archive(a.root,a.output,a.scope),indent=2))

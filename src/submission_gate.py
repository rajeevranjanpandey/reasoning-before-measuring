"""Fail-closed submission checks. Never compute interim confirmation effects."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(root, name):
    path = root / name
    return json.loads(path.read_text()) if path.exists() else None


def record_keys(path, fields):
    seen = set()
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = tuple(row[k] for k in fields)
            if identity in seen:
                raise ValueError(f'Duplicate identity in {path.name}, line {number}')
            seen.add(identity)
    return seen


def scientific_inputs(root):
    """Return blockers without opening partial outcomes or using interim estimates."""
    root = Path(root)
    blockers = []
    required = {
        'results/confirmation/pipeline_complete.json': ('experiment_and_audit_complete', True),
        'results/confirmation/audit.json': ('passed', True),
        'results/confirmation/analysis.json': ('confirmation_complete', True),
        'results/revision/status.json': ('phase', 'experiments_complete'),
        'results/revision/controls_analysis.json': ('audit_passed', True),
        'results/revision/closed_loop_analysis.json': ('audit_passed', True),
    }
    for path, (key, value) in required.items():
        obj = read_json(root, path)
        if obj is None or obj.get(key) != value:
            blockers.append(f'{path}: requires {key}={value!r}')
    diagnostics = read_json(root, 'results/revision/predictor_diagnostics.json')
    if not diagnostics or diagnostics.get('test_outcomes_read') is not False or not diagnostics.get('completed_utc'):
        blockers.append('Validation-only predictor diagnostics incomplete or invalid')
    if blockers:
        return blockers
    # Markers are necessary but insufficient. Verify exact record identities and frozen inputs.
    try:
        for plan_name, key in [('confirmation_plan_20260926.json', 'files'), ('review_revision_plan.json', 'source_sha256')]:
            plan = read_json(root, 'provenance/' + plan_name)
            for name, expected in plan[key].items():
                if digest(root / name) != expected:
                    blockers.append('Frozen input changed: ' + name)
        expected = {(v, b, s, r, c) for v in ('raw', 'assisted') for b in (1, 3, 5)
                    for s in (17, 29, 43) for r in range(21, 71) for c in range(21)}
        for model in ('qwen', 'granite'):
            keys = record_keys(root / f'results/confirmation/{model}.jsonl', ('policy', 'budget', 'seed', 'run', 'label'))
            if keys != expected:
                blockers.append(f'{model}: confirmation identity set is not exactly 18,900 records')
        audit = read_json(root, 'results/confirmation/audit.json')
        if audit.get('checked_records') != 73500 or audit.get('expected_reference_records') != 35700:
            blockers.append('Confirmation audit counts differ from the frozen design')
        references = ('fixed','mi','importance','forward','random','tool','rounded_tool','full','qwen_fixed_plan','granite_fixed_plan')
        expected_ref = {(p,b,s,r,c) for p in references for b in ((41,) if p=='full' else (1,3,5))
                        for s in ((17,29,43) if p=='random' else (0,)) for r in range(21,71) for c in range(21)}
        if record_keys(root/'results/confirmation/references.jsonl',('policy','budget','seed','run','label')) != expected_ref:
            blockers.append('Reference identity set mismatch')
        arms = ('original','order','remap','anonymous','no_scores','shuffled_scores')
        expected_controls = {(a,s,i) for a in arms for s in (17,29,43,61,79) for i in range(20)}
        for model in ('qwen','granite'):
            if record_keys(root/f'results/revision/{model}_controls.jsonl',('arm','seed','index')) != expected_controls:
                blockers.append(model + ': mechanism control identity set mismatch')
        policies = ('forward','rounded_tool','full','qwen_raw','qwen_assisted','granite_raw','granite_assisted')
        expected_closed = {(p,s,r,c) for p in policies for s in ((17,29,43,61,79) if p.startswith(('qwen','granite')) else (0,))
                           for r in range(71,91) for c in range(21)}
        if record_keys(root/'results/revision/closed_loop.jsonl',('policy','seed','run','label')) != expected_closed:
            blockers.append('Closed-loop identity set mismatch')
        if len(diagnostics.get('candidates', [])) != 4 or diagnostics.get('selected') not in {x['name'] for x in diagnostics['candidates']}:
            blockers.append('Predictor selection report does not match the four-candidate design')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        blockers.append('Scientific artifact validation failed: ' + str(exc))
    return blockers


def author_blockers(root):
    record = read_json(root, 'submission/author_metadata.json')
    if record is None:
        return ['Author metadata and declarations have not been supplied']
    issues=[]
    def present(value):
        return isinstance(value,str) and bool(value.strip()) and not any(t in value.lower() for t in ('to be provided','todo','placeholder','[insert','[author'))
    authors=record.get('authors',[])
    if not authors:
        issues.append('Author list is empty')
    for i,a in enumerate(authors,1):
        for field in ('name','department','institution','postal_address','country'):
            if not present(a.get(field)):
                issues.append(f'Author {i}: missing {field}')
        if not a.get('contributions') or not all(present(x) for x in a['contributions']):
            issues.append(f'Author {i}: actual contributions missing')
    corr=record.get('corresponding_author',{})
    if corr.get('name') not in [a.get('name') for a in authors] or '@' not in corr.get('email',''):
        issues.append('Corresponding author and email must be supplied')
    for field in ('funding_statement','competing_interests_statement'):
        if not present(record.get(field)):
            issues.append(field + ' missing')
    for field in ('all_authors_approved','ai_disclosure_approved','originality_and_exclusive_submission_confirmed','credit_roles_approved'):
        if record.get(field) is not True:
            issues.append(field + ' not confirmed by authors')
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--science-only',action='store_true')
    args=parser.parse_args();root=args.root
    blockers=scientific_inputs(root)
    science_complete=not blockers
    if not args.science_only:
        blockers+=author_blockers(root)
        archive=read_json(root,'submission/archive_record.json') or {}
        if not (archive.get('verified_public_url') or archive.get('author_approved_nonsharing_explanation')):
            blockers.append('Verified public archive or author-approved non-sharing explanation missing')
        review=read_json(root,'submission/final_review.json') or {}
        for field in ('scientific_review_complete','all_pages_visually_reviewed','current_journal_requirements_checked'):
            if review.get(field) is not True:
                blockers.append(field + ' remains outstanding')
        markdown='paper/manuscript.md' if (root/'paper/manuscript.md').exists() else 'paper/pilot_manuscript.md'
        for field,rel in [('manuscript_sha256','paper/manuscript.pdf'),('markdown_sha256',markdown)]:
            if not (root/rel).exists() or review.get(field)!=digest(root/rel):
                blockers.append('Final review is not bound to the current '+rel)
        integration=read_json(root,'paper/postreview/build_manifest.json') or {}
        for rel,expected in integration.get('input_sha256',{}).items():
            if not (root/rel).exists() or digest(root/rel)!=expected:
                blockers.append('Post-review evidence changed after build: '+rel)
        if not integration.get('complete_audited_studies') or review.get('postreview_manifest_sha256') != (digest(root/'paper/postreview/build_manifest.json') if integration else None):
            blockers.append('Complete post-review evidence has not been integrated and reviewed')
    print(json.dumps({'scientific_inputs_complete':science_complete,'submission_ready':not blockers and not args.science_only,'blockers':blockers},indent=2))
    return 0 if not blockers else 2


if __name__=='__main__':
    raise SystemExit(main())

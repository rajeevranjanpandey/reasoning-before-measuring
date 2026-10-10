"""Resume the frozen closed-loop study after host restart, retaining saved episodes."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,re,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision'
def main():
 lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 def status(phase,**kw):
  temp=OUT/'resume_status.tmp';temp.write_text(json.dumps(dict(phase=phase,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));temp.replace(OUT/'status.json')
 try:
  for plan,key in [('confirmation_plan_20260926.json','files'),('review_revision_plan.json','source_sha256')]:
   for rel,expected in json.loads((ROOT/'provenance'/plan).read_text())[key].items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,'Frozen input changed: '+rel
  assert json.loads((ROOT/'results/confirmation/pipeline_complete.json').read_text())['experiment_and_audit_complete']
  assert json.loads((OUT/'controls_analysis.json').read_text())['audit_passed']
  assert json.loads((OUT/'predictor_diagnostics.json').read_text())['test_outcomes_read'] is False
  path=OUT/'closed_loop.jsonl';prefix=path.read_bytes();keys=set()
  for line in prefix.splitlines():
   r=json.loads(line);key=tuple(r[k] for k in ('policy','seed','run','label'))
   assert key not in keys,'Duplicate saved identity';keys.add(key)
  assert prefix.endswith(b'\n'),'Incomplete trailing record needs explicit recovery'
  for prior in (ROOT/'provenance').glob('restart_*/resume_record.json'):
   saved=json.loads(prior.read_text())
   assert hashlib.sha256(prefix[:saved['saved_prefix_bytes']]).hexdigest()==saved['saved_prefix_sha256'],'Previously saved prefix modified'
  boot=subprocess.check_output(['/usr/sbin/sysctl','-n','kern.boottime'],text=True)
  boot_utc=datetime.datetime.fromtimestamp(int(re.search(r'sec = (\d+)',boot).group(1)),datetime.timezone.utc).isoformat()
  archive=ROOT/'provenance'/('restart_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));archive.mkdir(exist_ok=False)
  for p in list(OUT.glob('*_runtime.json'))+[OUT/'closed_loop_setup.json']:
   if p.exists():shutil.copy2(p,archive/p.name)
  report={'restart_utc':now,'reason':'Resumed after worker interruption. Current host boot UTC: '+boot_utc,'saved_records':len(keys),'saved_prefix_bytes':len(prefix),'saved_prefix_sha256':hashlib.sha256(prefix).hexdigest(),'frozen_sources_verified':True,'settings_changed':False,'timing_note':'Episode runtimes span execution sessions. Model initialization metadata from before restart is preserved in this directory.'}
  (archive/'resume_record.json').write_text(json.dumps(report,indent=2))
  status('running',stage='revision_closed_loop.py',resumed_after_host_restart=True,saved_records=len(keys))
  with (OUT/'revision_closed_loop.py.log').open('a') as log:
   subprocess.run([sys.executable,str(ROOT/'src/revision_closed_loop.py')],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
  with path.open('rb') as f:assert hashlib.sha256(f.read(len(prefix))).hexdigest()==report['saved_prefix_sha256'],'Saved records modified'
  status('experiments_complete',manuscript_integration_complete=False,note='Completed evidence requires scientific integration and final review')
 except Exception as exc:
  status('failed',error=str(exc));raise
if __name__=='__main__':main()

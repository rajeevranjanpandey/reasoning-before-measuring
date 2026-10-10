"""Run the frozen post-review extension after confirmation releases the machine."""
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/revision'

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((ROOT/'provenance/review_revision_plan.json').read_text())
    def status(phase,**kw):
        (OUT/'status.json').write_text(json.dumps(dict(phase=phase,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
    try:
        status('waiting_for_confirmation',note='No additional compute experiments overlap the original confirmation')
        while not (ROOT/'results/confirmation/pipeline_complete.json').exists():time.sleep(30)
        complete=json.loads((ROOT/'results/confirmation/pipeline_complete.json').read_text())
        if not complete.get('experiment_and_audit_complete'):raise RuntimeError('Confirmation did not pass')
        for name,digest in plan['source_sha256'].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Frozen revision source changed: '+name)
        for script in ('revision_diagnostics.py','revision_controls.py','revision_closed_loop.py'):
            status('running',stage=script)
            with (OUT/(script+'.log')).open('a') as log:
                subprocess.run([sys.executable,str(ROOT/'src'/script)],stdout=log,stderr=subprocess.STDOUT,check=True,env=os.environ.copy())
        status('experiments_complete',manuscript_integration_complete=False,note='Results require scientific interpretation and manuscript integration before submission')
    except Exception as exc:
        status('failed',error=str(exc));raise

if __name__=='__main__':main()

"""Execute the frozen confirmation and full audit sequentially, with resource logs."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/confirmation'


def main():
    OUT.mkdir(exist_ok=True)
    for script, args in [('confirmation.py', ['run']), ('analyze_confirmation.py', [])]:
        with (OUT / (script + '.log')).open('a') as log:
            proc = subprocess.Popen([sys.executable, str(ROOT / 'src' / script), *args],
                                    stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
            with (OUT / 'resource_monitor.jsonl').open('a') as monitor:
                while proc.poll() is None:
                    swap = subprocess.run(['/usr/sbin/sysctl', 'vm.swapusage'], capture_output=True, text=True)
                    monitor.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        stage=script, pid=proc.pid, swap=swap.stdout.strip(), error=swap.stderr.strip())) + '\n')
                    monitor.flush(); time.sleep(10)
            if proc.returncode:
                raise RuntimeError(f'{script} failed with status {proc.returncode}; inspect retained logs')
    (OUT / 'pipeline_complete.json').write_text(json.dumps(dict(
        experiment_and_audit_complete=True, manuscript_integration_complete=False,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
    print('Confirmation and audit complete. Manuscript integration and visual review are still required.', flush=True)


if __name__ == '__main__': main()

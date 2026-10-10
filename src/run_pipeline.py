"""Sequential, resumable experiment driver. Never loads the two SLMs concurrently."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

def status(phase,**extra):
    data=dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),phase=phase,**extra)
    (ROOT/'results/status.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)

def command(arguments,name):
    status('running',step=name)
    with (ROOT/'results'/f'{name}.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,*arguments],stdout=log,stderr=subprocess.STDOUT,
                              cwd=ROOT.parent,env=os.environ.copy())
        with (ROOT/'results/resource_monitor.jsonl').open('a') as monitor:
            while proc.poll() is None:
                swap=subprocess.run(['/usr/sbin/sysctl','vm.swapusage'],capture_output=True,text=True)
                monitor.write(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                  step=name,swap=swap.stdout.strip(),swap_error=swap.stderr.strip()))+'\n')
                monitor.flush();time.sleep(5)
        if proc.returncode:raise RuntimeError(f'{name} failed with exit code {proc.returncode}')

def main():
    (ROOT/'provenance/execution_environment.json').write_text(json.dumps(dict(
        python=sys.version,platform=platform.platform(),machine=platform.machine(),
        memory_bytes=8589934592,chip='Apple M2',backend='Metal',threads=4),indent=2))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT/'src').glob('*.py')}
    (ROOT/'provenance/execution_code_hashes.json').write_text(json.dumps(manifest,indent=2))
    gate={}
    for model in ('qwen','granite'):
        path=ROOT/'results'/f'{model}_development.jsonl'
        if len(read_rows(path))!=40:
            command([str(ROOT/'src/agents.py'),model,'development'],f'{model}_development')
        rows=read_rows(path)
        for variant in ('raw','assisted'):
            cell=[r for r in rows if r['policy']==variant]
            legal=sum(r['fallback_count']==0 and len({a['sensor'] for a in r['actions']})==3 for r in cell)
            gate[f'{model}_{variant}']=dict(episodes=len(cell),legal_without_fallback=legal,
                                          passed=len(cell)==20 and legal>=19)
    (ROOT/'results/execution_gate.json').write_text(json.dumps(gate,indent=2))
    if not all(v['passed'] for v in gate.values()):
        status('execution_gate_failed',gate=gate)
        command([str(ROOT/'src/analyze.py')],'analysis')
        command([str(ROOT/'src/write_paper.py')],'paper')
        status('development_complete_pilot_blocked',reason='Execution gate failed; no scientific SLM pilot claim')
        return
    for model in ('qwen','granite'):
        if not (ROOT/'results'/f'{model}_fixed_Testing_1_20.jsonl').exists():
            command([str(ROOT/'src/agents.py'),model,'plan'],f'{model}_plan')
    for model in ('qwen','granite'):
        path=ROOT/'results'/f'{model}_pilot.jsonl'
        if len(read_rows(path))!=2520:
            command([str(ROOT/'src/agents.py'),model,'pilot'],f'{model}_pilot')
    command([str(ROOT/'src/analyze.py')],'analysis')
    command([str(ROOT/'src/write_paper.py')],'paper')
    status('pilot_complete',note='Confirmation, broader baselines and publication readiness require review of pilot decision')

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status('failed',error=str(exc));raise

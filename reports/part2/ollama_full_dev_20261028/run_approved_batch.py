"""Run the two human-approved conditions sequentially; no automatic retries."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
PYTHON=ROOT/'.venv/bin/python'
CONDITIONS=('A_original','D_both')


def stamp():return datetime.now(timezone.utc).isoformat()

def save(state):
    state['updated_utc']=stamp()
    temp=OUT/'batch_status.json.tmp'
    temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(OUT/'batch_status.json')


def main():
    with (OUT/'batch.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('An approved batch is already running')
        state=dict(status='running',started_utc=stamp(),approval='User explicitly approved both Llama settings, approximately 26 hours.',
            python=str(PYTHON),conditions={c:dict(status='queued',completed=0,total=19998) for c in CONDITIONS})
        save(state)
        for config in CONDITIONS:
            summary=OUT/(config+'_summary.json')
            if summary.exists():
                value=json.loads(summary.read_text())
                if value.get('status')!='complete' or value.get('n')!=19998:raise RuntimeError('Incomplete summary requires inspection')
                state['conditions'][config].update(status='complete',completed=19998);save(state);continue
            command=[str(PYTHON),str(ROOT/'src/experiment_ollama_full_dev.py'),'run','--output',str(OUT),'--config',config,'--resume']
            logpath=OUT/(config+'_run.log')
            with logpath.open('a') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['active_config']=config;state['conditions'][config].update(status='running',pid=child.pid,started_utc=stamp(),log=str(logpath));save(state)
                try:
                    while child.poll() is None:
                        progress_path=OUT/'progress.json'
                        if progress_path.exists():
                            progress=json.loads(progress_path.read_text())
                            if progress.get('config')==config:
                                for key in ['completed','invalid','estimated_remaining_seconds']:
                                    if key in progress:state['conditions'][config][key]=progress[key]
                        save(state)
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:pass
                except BaseException:
                    child.terminate();child.wait(timeout=30)
                    state['status']='stopped';save(state);raise
                code=child.returncode
            state['conditions'][config].update(exit_code=code,finished_utc=stamp())
            if code!=0 or not summary.exists():
                state['status']='stopped';state['conditions'][config]['status']='stopped';save(state)
                print('Stopped; inspect',logpath,flush=True);return 1
            state['conditions'][config].update(status='complete',completed=19998);save(state)
        state.update(status='complete',finished_utc=stamp(),active_config=None);save(state)
        print('Both full-development conditions complete.',flush=True)
        return 0


if __name__=='__main__':sys.exit(main())

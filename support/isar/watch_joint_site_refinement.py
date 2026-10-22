"""Refresh review plots as a specific offline fit completes its cases."""
import argparse,json,subprocess,sys,time
from pathlib import Path


def identity(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True)
    for name in ['experiment','reference','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();original=identity(a.pid)
    if original is None or 'fit_joint_continuous_sites.py' not in original:raise ValueError('Expected joint-fit process is not live')
    previous=None;a.output.mkdir(parents=True,exist_ok=True)
    while True:
        live=identity(a.pid)==original
        manifest=json.loads((a.experiment/'experiment.json').read_text())
        files=[a.experiment/c['case']/'analysis.json' for c in manifest['cases']]
        state=[(str(p),p.stat().st_mtime_ns) for p in files]
        names=['control-00002',*['joint-'+s for s in manifest['identity']['assignments']]]
        progress=[(name,(a.experiment/name/'progress.json').stat().st_mtime_ns) for name in names if (a.experiment/name/'progress.json').exists()]
        key=(state,progress,manifest['complete'])
        if key!=previous or not live:
            command=[sys.executable,str(Path(__file__).with_name('report_joint_continuous_sites.py')),'--experiment',str(a.experiment),'--reference',str(a.reference),'--output',str(a.output)]
            subprocess.run(command,check=True);previous=key
        status=dict(runner_live=live,completed_cases=len(files),complete=manifest['complete'],report=str(a.output/'index.html'))
        (a.output/'status.json').write_text(json.dumps(status,indent=2)+'\n')
        if not live:
            print(json.dumps(status),flush=True);break
        time.sleep(30)


if __name__=='__main__':main()

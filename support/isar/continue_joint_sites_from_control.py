"""Finish the live control, then seed each site refit from that same control."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path


def identity(pid):
    r=subprocess.run(['ps','-p',str(pid),'-o','lstart=,command='],capture_output=True,text=True)
    return r.stdout.strip() if r.returncode==0 else None


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pid',type=int,required=True)
    for name in ['experiment','motion','field','priors','review']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--iterations',type=int,default=100);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    original=identity(a.pid)
    if original is None or 'fit_joint_continuous_sites.py' not in original or '--output '+str(a.experiment)+' ' not in original:raise ValueError('Expected original fit is not live')
    control=a.experiment/'control-00002';required=[control/'analysis.json',control/'motion.json',*[control/f'brightness-{c}.npy' for c in range(5)]]
    while not all(p.exists() for p in required):
        if identity(a.pid)!=original:raise RuntimeError('Original fit ended before completing its control')
        time.sleep(5)
    result=json.loads((control/'analysis.json').read_text())
    if not result['complete'] or not result['fixed_antenna_locations'] or result['maximum_kinematic_or_box_violation']>1e-7:raise ValueError('Control is incomplete or infeasible')
    if identity(a.pid)==original:os.kill(a.pid,signal.SIGINT)
    while identity(a.pid)==original:time.sleep(1)
    # Only this job's unfinished first joint case can have started during the
    # handoff. Preserve its small trace; never remove completed fit results.
    partial=a.experiment/'joint-00002'
    if partial.exists() and not (partial/'analysis.json').exists():
        target=a.experiment/'joint-00002-before-control-initializer'
        if partial.is_symlink() or target.exists():raise ValueError('Unexpected handoff trace')
        partial.rename(target)
    command=[sys.executable,str(Path(__file__).with_name('fit_joint_continuous_sites.py')),'--motion',str(a.motion),'--field',str(a.field),
        '--priors',str(a.priors),'--output',str(a.experiment),'--iterations',str(a.iterations),'--workers',str(a.workers)]
    if not result['numerical_convergence']:
        # Continue an unfinished control with the profiled-brightness
        # preconditioner before using it as the matched comparison.
        retained=a.experiment/'control-00002-before-brightness-projection'
        if retained.exists():raise ValueError('Control continuation already recorded')
        control.rename(retained);command.extend(['--initial-control',str(retained/'analysis.json')])
        path=a.experiment/'experiment.json';manifest=json.loads(path.read_text())
        manifest['cases']=[c for c in manifest['cases'] if c['case']!='control-00002'];manifest['complete']=False
        path.write_text(json.dumps(manifest,indent=2)+'\n')
    log=Path('docs/data/isar_20261018/native-spatial-validation/joint-jerk-sites.log').open('a')
    job=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
    state=dict(stage='site-refits-seeded-from-completed-control',pid=job.pid,control=str(control),command=command)
    (a.experiment/'runner.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    review_log=Path('docs/data/isar_20261018/native-spatial-validation/joint-jerk-site-review-seeded.log').open('w')
    watcher=subprocess.Popen([sys.executable,str(Path(__file__).with_name('watch_joint_site_refinement.py')),'--pid',str(job.pid),'--experiment',str(a.experiment),
        '--reference',str(a.motion),'--output',str(a.review)],stdout=review_log,stderr=subprocess.STDOUT)
    code=job.wait();watcher.wait();log.close();review_log.close()
    state.update(stage='runner-finished',exit_code=code);(a.experiment/'runner.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    if code:raise RuntimeError('Site sweep ended before all cases completed; see its retained log and review')


if __name__=='__main__':main()

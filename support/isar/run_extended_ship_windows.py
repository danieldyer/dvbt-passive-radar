"""Run exact extended-scene/motion checks over every supported aperture.

This records successful and failed experiments without filling gaps or
independently rotating/rephasing video frames. The local results are inputs
to a later continuous-motion and static-geometry fit, not that final fit.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--centres',nargs='+',type=float);p.add_argument('--iterations',type=int,default=120);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);reports=[];started=time.monotonic()
    for source in sorted(a.windows.glob('window-*/analysis.json')):
        mid=json.loads(source.read_text())['mid_s']
        if a.centres and min(abs(mid-c) for c in a.centres)>.1:continue
        out=a.output/f'window-{mid:07.2f}';out.mkdir(exist_ok=True)
        stages=[('scene',[
            'support/isar/fit_ship_extended_scene.py','--motion',str(a.motion),'--windows',str(a.windows),
            '--centre',str(mid),'--candidates','80','--points','12','--output',str(out/'scene')],out/'scene/analysis.json'),
            ('motion',['support/isar/refine_ship_extended_motion.py','--scene',str(out/'scene/analysis.json'),
            '--windows',str(a.windows),'--count','9','--iterations',str(a.iterations),'--output',str(out/'motion')],out/'motion/analysis.json'),
            ('phase',['support/isar/calibrate_local_ship_scene.py','--motion',str(out/'motion/analysis.json'),
            '--windows',str(a.windows),'--output',str(out/'motion/local-calibrated-model.json')],out/'motion/local-calibrated-model.json'),
            ('closure',['support/isar/validate_local_ship_motion.py','--motion',str(out/'motion/analysis.json'),
            '--windows',str(a.windows),'--output',str(out/'motion/native-phase-closure.json')],out/'motion/native-phase-closure.json')]
        error=None
        for name,args,product in stages:
            if product.exists():
                saved=json.loads(product.read_text())
                if saved.get('complete',True) and (name!='closure' or saved.get('passed',False)):continue
            # An interrupted stage is retained for inspection rather than
            # silently overwritten or treated as a completed experiment.
            if product.parent.exists() and name in ['scene','motion'] and any(product.parent.iterdir()):
                error=f'Incomplete {name} stage requires explicit restart';break
            with (out/(name+'.log')).open('w') as log:
                r=subprocess.run([sys.executable,*args],stdout=log,stderr=subprocess.STDOUT)
            if r.returncode:error=f'{name} exited {r.returncode}';break
        record=dict(mid_s=mid,directory=str(out),error=error)
        if error is None:
            fit=json.loads((out/'motion/analysis.json').read_text());record.update(numerically_converged=fit['numerical_convergence'],parameters=fit['parameters'])
            for label,key in [('before','unused_before'),('after','unused_after')]:
                values=fit[key];record['unused_explained_fraction_'+label]=1-sum(v['unused_rss'] for v in values)/sum(v['unused_data_energy'] for v in values)
        reports.append(record);(a.output/'analysis.json').write_text(json.dumps(dict(complete=False,scope=__doc__,motion=str(a.motion),windows=reports,seconds=time.monotonic()-started),indent=2)+'\n')
        print(json.dumps(record),flush=True)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),windows=reports,seconds=time.monotonic()-started),indent=2)+'\n')
if __name__=='__main__':main()

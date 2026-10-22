"""Conditional exact-waveform variable-projection motion refinement.

The scene's physical locations come from training-only extended-scene pursuit.
Five common rigid-motion corrections change position, phase, envelope delay,
velocity and orientation together. Every point/band coefficient is constant
within this aperture. Unused range/slow-time data decide whether it transfers.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from audit_ship_exact_aperture import state
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch,PatchBand
from ship_window import Dataset
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True);p.add_argument('--count',type=int,default=9)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--iterations',type=int,default=100);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);scene=json.loads(a.scene.read_text());motion=JointMotion(Path(scene['motion']));r=motion.result
    mid=scene['mid_s'];source=a.windows/f'window-{mid:07.2f}';win=Window(source,r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds'])
    selection=next(q for q in scene['counts'] if q['count']==a.count);points=np.asarray(scene['candidate_positions'])[selection['support']]
    dataset=Dataset();window=dataset.window_from_description(win.result);bands=[]
    for c,old in enumerate(window['fast']):
        s=state(win,np.asarray(r['parameters']),c)
        kin=RigidPatch(win.time[c],mid,s['position'],s['velocity'],window['local_to_ecef'],motion.tx[3],motion.rx,s['frame'],s['frame_rate'])
        bands.append(PatchBand(old.field,kin,points,old.training,motion.tx[c],motion.rx))
    scale=np.array([.05,.005,.0005,.05,.005]);bounds=[(-.8,.8),(-.08,.08),(-.008,.008),(-.2,.2),(-.02,.02)]
    energy=sum(b.energy for b in bands);cache=[None,None];started=time.monotonic();trace=[]
    def evaluate(q):
        if cache[0] is None or not np.array_equal(q,cache[0]):
            values=[b.evaluate(q*scale) for b in bands]
            cache[:]=[q.copy(),dict(value=float(sum(v['value'] for v in values)/energy),
                gradient=sum(v['gradient'] for v in values)*scale/energy,condition=max(v['condition'] for v in values))]
        return cache[1]
    def objective(q):
        d=evaluate(q);return d['value'],d['gradient']
    initial_motion=np.asarray(scene.get('initial_motion_parameters',np.zeros(5)));initial=initial_motion/scale
    before=evaluate(initial);unused0=[b.validation(initial_motion) for b in bands]
    result=dict(complete=False,scope=__doc__,source_scene=str(a.scene),motion=str(motion.path),mid_s=mid,points=points,
        parameter_names=['path_rate_mps','path_acceleration_mps2','path_jerk_mps3','yaw_rate_deg_ps','yaw_acceleration_deg_ps2'],
        bounds=bounds,initial_motion_parameters=initial_motion,training_before=before,unused_before=unused0,trace=trace)
    def save():(a.output/'analysis.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    def callback(q):
        d=evaluate(q);trace.append(dict(iteration=len(trace)+1,parameters=q*scale,objective=d['value'],maximum_gradient=float(max(abs(d['gradient']))),seconds=time.monotonic()-started))
        result['parameters']=q*scale;save();print(json.dumps(serial(trace[-1])),flush=True)
    fit=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(lo/s,hi/s) for (lo,hi),s in zip(bounds,scale)],callback=callback,
        options=dict(maxiter=a.iterations,gtol=1e-7,ftol=1e-12,maxcor=20,maxls=20))
    final=fit.x*scale;result.update(complete=True,parameters=final,training_after=evaluate(fit.x),message=str(fit.message),
        unused_after=[b.validation(final) for b in bands],seconds=time.monotonic()-started,
        numerical_convergence=bool(fit.success and max(abs(fit.jac))<1e-5),accepted_as_global_motion=False)
    save()
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    axes[0].plot([t['iteration'] for t in trace],[t['objective'] for t in trace]);axes[0].set(xlabel='Iteration',ylabel='Training residual / data energy')
    for label,key in [('Before','unused_before'),('After','unused_after')]:
        axes[1].plot(range(5),[100*(1-v['unused_rss']/v['unused_data_energy']) for v in result[key]],'.-',label=label)
    axes[1].set(xticks=range(5),xticklabels=['Seven','SBS','Nine','Ten','ABC'],ylabel='Unused energy predicted (%)');axes[1].legend()
    fig.suptitle(f'{mid-15:.2f}–{mid+15:.2f} s · exact extended-scene motion likelihood\n{len(points)} fixed training-selected locations; one physical motion; constant complex point/band coefficients')
    fig.savefig(a.output/'motion-transfer.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()

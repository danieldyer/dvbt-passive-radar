"""Fit shared physical motion while profiling positive scattering envelopes."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from audit_ship_exact_aperture import state
from envelope_patch import EnvelopePatch
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch,PatchBand
from ship_window import Dataset
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--envelope-seed',type=Path,required=True);p.add_argument('--spacing-s',type=float,default=2.);p.add_argument('--iterations',type=int,default=150)
    p.add_argument('--refine-positions-m',type=float,default=0.)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);old=json.loads(a.motion.read_text());motion=JointMotion(Path(old['motion']));r=motion.result;mid=old['mid_s']
    win=Window(a.windows/f'window-{mid:07.2f}',r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds']);dataset=Dataset()
    exact=dataset.window_from_description(win.result);points=np.asarray(old['points']);bands=[]
    for c,b in enumerate(exact['fast']):
        s=state(win,np.asarray(r['parameters']),c);kin=RigidPatch(win.time[c],mid,s['position'],s['velocity'],exact['local_to_ecef'],motion.tx[3],motion.rx,s['frame'],s['frame_rate'])
        bands.append(PatchBand(b.field,kin,points,b.training,motion.tx[c],motion.rx))
    model=EnvelopePatch(bands,win.result['interval_s'],a.spacing_s,refine_positions=a.refine_positions_m>0)
    seed=next(q for q in json.loads(a.envelope_seed.read_text())['models'] if q['spacing_s']==a.spacing_s)
    initial=np.r_[old['parameters'],seed['parameters'],np.zeros(model.size-model.phase_end)];scale=np.r_[.05,.005,.0005,.05,.005,np.ones(model.phase_end-5),np.full(model.size-model.phase_end,.5)]
    bounds=[(-.8,.8),(-.08,.08),(-.008,.008),(-.2,.2),(-.02,.02)]+[(None,None)]*(model.phase_end-5)+[(-a.refine_positions_m,a.refine_positions_m)]*(model.size-model.phase_end);cache=[None,None];trace=[];started=time.monotonic()
    def evaluate(q):
        if cache[0] is None or not np.array_equal(q,cache[0]):cache[:]=[q.copy(),model.evaluate(q*scale)]
        return cache[1]
    def objective(q):
        d=evaluate(q);return d['value'],d['gradient']*scale
    before=evaluate(initial/scale);unused0=model.validation(initial,before)
    result=dict(old);result.update(complete=False,source_motion_fit=str(a.motion),envelope_spacing_s=a.spacing_s,trace=trace,
        envelope_training_before=before['value'],envelope_unused_before=unused0,
        scope='Exact physical motion with training-only nonnegative brightness envelopes; phase per point/band stays constant. No independent band motion or temporal phase screen.')
    def save():(a.output/'analysis.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    def callback(q):
        d=evaluate(q);record=dict(iteration=len(trace)+1,objective=d['value'],maximum_scaled_gradient=float(max(abs(d['gradient']*scale))),seconds=time.monotonic()-started)
        trace.append(record);result.update(parameters=(q*scale)[:5],phase_parameters=np.r_[(q*scale)[5:5+len(model.free)],d['point_phase_radians']],points=d['points'],optimizer_parameters=q*scale);save();print(json.dumps(record),flush=True)
    fit=minimize(objective,initial/scale,jac=True,method='L-BFGS-B',bounds=[(None if lo is None else lo/s,None if hi is None else hi/s) for (lo,hi),s in zip(bounds,scale)],
        callback=callback,options=dict(maxiter=a.iterations,gtol=1e-7,ftol=1e-12,maxcor=30,maxls=25))
    p=fit.x*scale;after=evaluate(fit.x);gradient=fit.jac.copy()
    for j,(lo,hi) in enumerate(bounds):
        if lo is not None and p[j]<lo+1e-7:gradient[j]=min(gradient[j],0.)
        if hi is not None and p[j]>hi-1e-7:gradient[j]=max(gradient[j],0.)
    beta=np.zeros(5);beta[model.free]=p[5:5+len(model.free)];g=abs(motion.gains)*np.exp(1j*beta)
    arrays={}
    for c,s in enumerate(after['solutions']):arrays[f'coefficient_{c}']=s['coefficient'].reshape(model.npoints,len(model.knots))
    np.savez_compressed(a.output/'envelope-coefficients.npz',knots_s=model.knots,**arrays)
    result.update(complete=True,parameters=p[:5],phase_parameters=np.r_[p[5:5+len(model.free)],after['point_phase_radians']],points=after['points'],physical_gains_real_imag=np.c_[g.real,g.imag],
        optimizer_parameters=p,optimizer_initial_points=points,position_phase_gauge_radians_per_m=model.position_phase_gauge,
        envelope_training_after=after['value'],envelope_unused_after=model.validation(p,after),
        numerical_convergence=bool(fit.success and max(abs(gradient))<1e-5),maximum_projected_scaled_gradient=float(max(abs(gradient))),message=str(fit.message),
        calibration_scope='Shared constant band/scatterer phases with positive piecewise-linear magnitudes; this aperture only.',seconds=time.monotonic()-started)
    save();print(json.dumps(serial(dict(complete=True,parameters=p[:5],numerical_convergence=result['numerical_convergence'],unused=result['envelope_unused_after']))),flush=True)
if __name__=='__main__':main()

"""Continuous physical-pixel peak refinement with one frozen calibration.

All bands move to the SAME subpixel position. Only coordinates of training
peaks are refined; no band phase, motion curve or amplitude is refitted.
Unused epochs then test the retained cross-band phase and power gain.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from dense_patch_image import DenseImages
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from report_measured_scatterer_phases import serial


def pixel_geometry(window,parameters,worker,motion):
    geo,_,_=window.geometry(parameters,worker,motion.gauge_point,motion.result['transmitter_groups'],motion.receiver_local,antennas=bool(motion.result['parameter_counts']['antenna']))
    geo[:,:,10:]=0.
    geo[:,:,10,:3]=geo[:,:,2,:3];geo[:,:,11,:3]=geo[:,:,6,:3]
    geo[:,:,18,:3]=geo[:,:,3,:3];geo[:,:,19,:3]=geo[:,:,7,:3]
    return geo


def statistics(images,gains,information):
    noise=np.sum(abs(gains)**2*information);z=np.einsum('c,cp->p',gains.conj(),images);power=abs(z)**2/noise
    independent=abs(images)**2/information[:,None];phase_upper=np.sum(abs(gains[:,None]*images),axis=0)**2/noise
    return dict(individual_power=independent,combined_power=power,gain_over_best_band_db=10*np.log10(power/independent.max(axis=0)),
        phase_efficiency=power/phase_upper,amplitude_efficiency=phase_upper/independent.sum(axis=0),independent_efficiency=power/independent.sum(axis=0),
        phase_residual_deg=np.degrees(np.angle(images*gains.conj()[:,None]*z.conj()[None,:])))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False);motion=JointMotion(a.motion);result=motion.result
    gains=np.asarray([complex(*z) for z in result['calibration_gains_real_imag']]);count=result['parameter_counts'];at=count['path']+count.get('cross',0)+count['antenna']
    if count['phase']:gains[[0,1,2,4]]*=np.exp(1j*np.asarray(result['parameters'][at:]))
    paths=sorted(a.windows.glob('window-*/analysis.json'));paths=paths[:a.limit] if a.limit else paths;worker=None;records=[]
    try:
        for k,path in enumerate(paths):
            window=Window(path.parent,result['interval_s'],result['path_spacing_seconds'],result.get('cross_spacing_seconds',15.));source=path.parent/'training/model.bin'
            if worker is None:worker=DenseImages.reopen(source,derivatives=8)
            else:worker.replace_archive(source)
            worker.training=True;geo=pixel_geometry(window,np.asarray(result['parameters']),worker,motion)
            images=worker.evaluate_geometry(geo,derivatives=False)[:,0]
            archive=np.load(path.parent/'training-images.npz');pixels=archive['pixels'];u=archive['forward_m'];v=archive['port_m'];nu=len(u)
            association=np.c_[(pixels[1,:2]-pixels[0,:2])/(u[1]-u[0]),(pixels[nu,:2]-pixels[0,:2])/(v[1]-v[0])]
            power=abs(np.einsum('c,cp->p',gains.conj(),images))**2;allowed=np.ones(len(pixels),bool);selected=[]
            for j in range(6):
                index=int(np.argmax(np.where(allowed,power,-np.inf)));selected.append(index);allowed&=np.sum((pixels[:,:2]-pixels[index,:2])**2,axis=1)>=35**2
            initial=pixels[selected].copy();scale=power[selected];cache=[None,None];calls=0
            def evaluate(q):
                nonlocal calls
                if cache[0] is None or not np.array_equal(q,cache[0]):
                    positions=initial.copy();positions[:,:2]+=q.reshape(-1,2)@association.T;worker.set_pixels(positions)
                    values=worker.evaluate_cached();z=np.einsum('c,cp->p',gains.conj(),values[:,0]);derivative=np.einsum('c,cjp->pj',gains.conj(),values[:,1:3])@association
                    value=-np.mean(abs(z)**2/scale);gradient=(-2*np.real(z.conj()[:,None]*derivative)/scale[:,None]/len(initial)).ravel()
                    cache[:]=[q.copy(),dict(value=value,gradient=gradient,positions=positions,images=values[:,0])];calls+=1
                return cache[1]
            def objective(q):
                r=evaluate(q);return r['value'],r['gradient']
            before=evaluate(np.zeros(12));fit=minimize(objective,np.zeros(12),jac=True,method='L-BFGS-B',bounds=[(-1.,1.)]*12,
                options=dict(maxiter=50,maxls=15,ftol=1e-8,gtol=5e-4));after=evaluate(fit.x)
            worker.replace_archive(path.parent/'unused/model.bin');worker.training=False;test_geo=pixel_geometry(window,np.asarray(result['parameters']),worker,motion)
            worker.set_pixels(initial);test0=worker.evaluate_geometry(test_geo,derivatives=False)[:,0]
            worker.set_pixels(after['positions']);test1=worker.evaluate_cached(derivatives=False)[:,0]
            information=np.asarray([8192*np.sum(window.context[f'information_{c}'][~window.training[c]]) for c in range(5)])
            record=dict(mid_s=window.result['mid_s'],initial_positions=initial,refined_positions=after['positions'],subpixel_forward_port_m=fit.x.reshape(-1,2),
                training_objective_before=before['value'],training_objective_after=after['value'],unused_before=statistics(test0,gains,information),unused_after=statistics(test1,gains,information),
                maximum_gradient=float(max(abs(fit.jac))),message=str(fit.message),gpu_evaluations=calls)
            records.append(record);(out/'analysis.json').write_text(json.dumps(serial(dict(complete=False,motion=str(a.motion),windows=records)),indent=2)+'\n')
            print(json.dumps(dict(window=k+1,windows=len(paths),mid_s=window.result['mid_s'],phase_efficiency=float(record['unused_after']['phase_efficiency'][0]),gain_db=float(record['unused_after']['gain_over_best_band_db'][0]))),flush=True)
    finally:
        if worker is not None:worker.close()
    data=dict(complete=True,motion=str(a.motion),scope=__doc__,windows=records);(out/'analysis.json').write_text(json.dumps(serial(data),indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained');time=[r['mid_s'] for r in records]
    for key,label in [('unused_before','Coarse 2 m pixel'),('unused_after','Continuous physical position')]:
        axes[0].plot(time,[r[key]['gain_over_best_band_db'][0] for r in records],'.-',label=label)
        axes[1].plot(time,[r[key]['phase_efficiency'][0] for r in records],'.-',label=label)
    axes[0].set_ylabel('Unused gain over best band (dB)');axes[0].legend();axes[0].axhline(0,color='grey');axes[1].set(ylabel='Phase-only coherent efficiency',ylim=(0,1));axes[1].legend()
    for c in range(5):axes[2].plot(time,[r['unused_after']['phase_residual_deg'][c,0] for r in records],'.',label=['Seven','SBS','Nine','Ten','ABC'][c])
    axes[2].set(ylabel='Unused phase residual (deg)',ylim=(-180,180),xlabel='Capture time (s)');axes[2].legend(ncol=5)
    fig.suptitle('Physical subpixel refinement: one shared location across all five bands\nTraining-only pixel selection/refinement; frozen whole-capture motion and channel calibration; unused-epoch checks')
    fig.savefig(out/'subpixel-coherence.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()

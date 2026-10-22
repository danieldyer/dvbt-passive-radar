"""Fit continuous path corrections within declared, full ship apertures.

The objective is mean single-band image Renyi-2 entropy. One physical path
spline drives every band, envelope delay, RF phase and symbol Doppler. No
per-band phase trajectory, image warp or pixel normalization is fitted.
Unused half-second blocks are scored after the fit, with no refitting.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

from dense_patch_image import DenseImages,entropy
from ship_window import Dataset,local_geometry
from spline_motion import basis
from imaging import attitude_basis
from report_measured_scatterer_phases import serial


def prepare(window,pixels,out,training=True):
    return DenseImages(window['fast'],pixels,out,training=training)


def packet(model,parameters,bases,rates,heading=None):
    result=[]
    for c,b in enumerate(model.bands):
        path=bases[c]@parameters;rate=rates[c]@parameters
        zero=np.zeros(len(path))
        result.append(local_geometry(b,path,rate,zero,zero,model.selected[c],model.stride))
    return np.asarray(result)


def focus_window(dataset,first,last,out,*,spacing=2.,iterations=80,knot_seconds=3.,fit=True,source_model=None,forward_limits=(-330.,330.),ais_centred=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);window=dataset.window(first,last,model=source_model)
    # Candidate association only; every pixel of this declared image patch
    # is imaged. No fitted hull mask or external target response is applied.
    a=dataset.track.sample(np.array([dataset.epoch+window['mid']]),max_gap_s=400.)
    assoc=(window['local_to_ecef'].T@attitude_basis(dataset.track,a,'reported-course-prior')[0])[:2,:2]
    u=np.arange(forward_limits[0],forward_limits[1]+.01,spacing);v=np.arange(-100.,100.01,spacing);uu,vv=np.meshgrid(u,v)
    origin=(a['position_ecef_m'][0]-dataset.state(np.array([window['mid']]),source_model)['position_ecef_m'][0])@window['local_to_ecef'] if ais_centred else np.zeros(3)
    pixels=np.c_[np.c_[uu.ravel(),vv.ravel()]@assoc.T+origin[:2],np.full(uu.size,20.)]
    bases=[basis(b.field.time,[first,last],knot_seconds,fixed_midpoint=True)[:,:-1] for b in window['fast']]
    rates=[basis(b.field.time,[first,last],knot_seconds,derivative=1)[:,:-1] for b in window['fast']]
    model=prepare(window,pixels,out/'training',True)
    context={}
    for c,b in enumerate(window['fast']):
        f=b.field;kin=b.kinematics
        for name,value in dict(time=f.time,information=f.data['information'],position=kin.position,velocity=kin.velocity,
                frame=kin.frame,frame_rate=kin.frame_rate,direction=kin.direction,direction_rate=kin.direction_rate,
                observer=f.data['centre_path_m'],carrier=f.data.get('carrier_reference_path_m',f.data['centre_path_m']),
                tx=b.tx,rx=b.rx,rf_hz=f.metadata['rf_hz'],training=b.training).items():context[f'{name}_{c}']=value
    np.savez(out/'geometry-context.npz',**context)
    n=bases[0].shape[1];initial=np.zeros(n);trace=[];cache=[None,None]
    def evaluate(p):
        if cache[0] is None or not np.array_equal(p,cache[0]):
            image=model.evaluate_geometry(packet(model,p,bases,rates))[:,0]
            power=abs(image)**2;total=power.sum(axis=1);squared=(power**2).sum(axis=1)
            value=np.mean(2*np.log(total)-np.log(squared))
            cotangent=2*image*(1/total[:,None]-power/squared[:,None])/len(image)
            local=model.adjoint(cotangent)
            gradient=sum(b.T@g[:len(b),0]+db.T@g[:len(b),1] for b,db,g in zip(bases,rates,local))
            cache[:]=[p.copy(),dict(value=float(value),gradient=gradient,image=image)]
        return cache[1]
    before=evaluate(initial)
    def objective(p):
        result=evaluate(p);return result['value'],result['gradient']
    def callback(p):
        r=evaluate(p);trace.append(dict(entropy=r['value'],gradient=float(max(abs(r['gradient'])))))
        print(json.dumps(dict(mid=window['mid'],iteration=len(trace),**trace[-1])),flush=True)
    try:
        fitted=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-6.,6.)]*n,callback=callback,
            options=dict(maxiter=iterations,gtol=2e-5,ftol=2e-8,maxls=15,maxcor=15)) if fit else None
        p=fitted.x if fit else initial;after=evaluate(p)
        result=dict(interval_s=[first,last],mid_s=window['mid'],training_epoch_s=3.32,source_model=window['model'],path_knot_seconds=knot_seconds,
            path_parameters_m=np.r_[p,0.],association_origin_east_north_m=origin[:2],forward_limits_m=forward_limits,training_entropy_before=before['value'],training_entropy_after=after['value'],
            message=str(fitted.message) if fit else 'Baseline only',iterations=len(trace),trace=trace,
            maximum_gradient=float(max(abs(after['gradient']))),gpu_calls=model.calls,gpu_seconds=model.seconds,
            waveform_erasures=window['erasures'],assignment=dataset.assignment,spacing_m=spacing,complete=False)
        np.savez_compressed(out/'training-images.npz',pixels=pixels,forward_m=u,port_m=v,before=before['image'],after=after['image'])
    finally:model.close()
    check=prepare(window,pixels,out/'unused',False)
    try:
        unused0=check.evaluate_geometry(packet(check,initial,bases,rates))[:,0]
        unused1=check.evaluate_geometry(packet(check,p,bases,rates))[:,0]
    finally:check.close()
    result.update(unused_entropy_before=[entropy(z) for z in unused0],unused_entropy_after=[entropy(z) for z in unused1],
        training_individual_before=[entropy(z) for z in before['image']],training_individual_after=[entropy(z) for z in after['image']],complete=True)
    np.savez_compressed(out/'unused-images.npz',pixels=pixels,forward_m=u,port_m=v,before=unused0,after=unused1)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(18,8),layout='constrained')
    for row,z in enumerate([unused0,unused1]):
        for c in range(5):
            # Fixed common matched-power scale; this diagnostic has half the
            # temporal support of the final 30-second original-symbol image.
            norm=np.sum(window['fast'][c].field.root_information[~window['fast'][c].training]**2)
            power=10*np.log10(np.maximum(abs(z[c])**2/norm,1e-12)).reshape(v.size,u.size)
            axes[row,c].imshow(power,extent=[u[0],u[-1],v[0],v[-1]],origin='lower',aspect='equal',cmap='magma',vmin=0,vmax=50)
            axes[row,c].plot(0,0,'+',color='#00e5ff');axes[row,c].set_title(['Seven','SBS','Nine','Ten','ABC'][c])
    fig.suptitle(f'{first:.2f}–{last:.2f} s · unused epochs · before / shared-path autofocus\nForward/port diagnostic axes; raw complex images retained for north-up rendering')
    fig.savefig(out/'unused-comparison.png',dpi=130);plt.close(fig)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--centres',type=float,nargs='+',required=True)
    p.add_argument('--seconds',type=float,default=30.);p.add_argument('--iterations',type=int,default=80);p.add_argument('--spacing-m',type=float,default=2.)
    p.add_argument('--reported-course',action='store_true');p.add_argument('--forward-limits',nargs=2,type=float,default=[-330.,330.]);p.add_argument('--ais-centred',action='store_true')
    p.add_argument('--assignment',default='00002');p.add_argument('--baseline-only',action='store_true');p.add_argument('--local-initializers',action='store_true')
    args=p.parse_args();dataset=Dataset(assignment=args.assignment)
    results=[]
    for mid in args.centres:
        directory=args.output/f'window-{mid:07.2f}';existing=directory/'analysis.json'
        if existing.exists():results.append(json.loads(existing.read_text()));continue
        model=None
        if args.local_initializers:
            fits=[json.loads(f.read_text()) for f in Path('build/isar-20261015/full-response-radar-motion').glob('window-???.json')]
            first,last=mid-args.seconds/2,mid+args.seconds/2
            usable=[m for m in fits if m['interval_s'][0]<=first and m['interval_s'][1]>=last]
            if usable:
                model={**dataset.model,**min(usable,key=lambda m:abs(np.mean(m['interval_s'])-mid))}
                if 'path_reference_time_s' not in min(usable,key=lambda m:abs(np.mean(m['interval_s'])-mid)):model.pop('path_reference_time_s',None)
        if args.reported_course:
            from spline_motion import SplineMotion
            model=dict(dataset.model if model is None else model);curve=SplineMotion(model);params=curve.parameters.copy();params[curve.sections[1]]=0.
            model.update(attitude_prior='reported-course-prior',parameters=params.tolist())
        results.append(focus_window(dataset,mid-args.seconds/2,mid+args.seconds/2,directory,spacing=args.spacing_m,iterations=args.iterations,fit=not args.baseline_only,source_model=model,forward_limits=args.forward_limits,ais_centred=args.ais_centred))
        args.output.mkdir(parents=True,exist_ok=True);(args.output/'analysis.json').write_text(json.dumps([json.loads(p.read_text()) for p in sorted(args.output.glob('window-*/analysis.json'))],indent=2)+'\n')

if __name__=='__main__':main()

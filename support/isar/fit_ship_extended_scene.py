"""Fit a shared-location extended scene before attributing defocus to motion.

Orthogonal least-squares pursuit uses only training complex range/slow-time
observations and exact waveform templates. Independent per-band coefficients
and a shared complex scene are compared on unused data. This is a sparse
forward-model diagnostic, not a replacement image or proof of a unique hull.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import maximum_filter
from audit_ship_exact_aperture import state
from dense_patch_image import DenseImages,trial_geometry
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from joint_support import select
from rigid_patch import profile_static,RigidPatch,PatchBand
from ship_window import Dataset
from shared_scene import solve,phase_coherent
from coherent_amplitude_glrt import statistic
from report_measured_scatterer_phases import serial


def report_fit(a,y,at,selected,*,gains=None):
    training=[];testing=[];coefficient=[];gram=[];rhs=[];energy=[]
    for design,data,mask in zip(a,y,selected):
        x=design[mask][:,at];d=data[mask];coef=np.linalg.lstsq(x,d,rcond=1e-10)[0]
        coefficient.append(coef);gram.append(x.conj().T@x);rhs.append(x.conj().T@d);energy.append(np.vdot(d,d).real)
        training.append(float(np.linalg.norm(d-x@coef)**2))
        xt=design[~mask][:,at];dt=data[~mask];testing.append(float(np.linalg.norm(dt-xt@coef)**2))
    shared=solve(np.asarray(gram),np.asarray(rhs),np.asarray(energy));phase=phase_coherent(np.asarray(gram),np.asarray(rhs),np.asarray(energy))
    validation={}
    for label,coef in [('independent',coefficient),('shared',shared['gain'][:,None]*shared['scene'][None]),('shared-phase',phase['coefficient'])]:
        cost=0.;cross=0.;model_power=0.;total=0.
        for design,data,mask,c in zip(a,y,selected,coef):
            prediction=design[~mask][:,at]@c;d=data[~mask]
            cost+=np.linalg.norm(d-prediction)**2;cross+=np.vdot(prediction,d).real;model_power+=np.vdot(prediction,prediction).real;total+=np.vdot(d,d).real
        validation[label]=dict(unused_rss=float(cost),unused_energy=float(total),predicted_energy=float(model_power),cross_energy=float(cross),
            explained_fraction=float(1-cost/total))
    for result in [shared,phase]:
        for key in list(result):
            if np.iscomplexobj(result[key]):
                z=np.asarray(result.pop(key));result[key+'_real_imag']=np.stack([z.real,z.imag],axis=-1)
    return dict(support=at,training_rss=float(sum(training)),unused=validation,
        coefficient_real_imag=np.stack([np.asarray(coefficient).real,np.asarray(coefficient).imag],axis=-1),shared=shared,phase_coherent=phase)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--initial-motion',type=Path,help='Reimage and reselect the scene after this local physical motion fit')
    p.add_argument('--centre',type=float,required=True);p.add_argument('--candidates',type=int,default=80);p.add_argument('--points',type=int,default=24);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.monotonic();motion=JointMotion(args.motion);r=motion.result
    source=args.windows/f'window-{args.centre:07.2f}';win=Window(source,r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds'])
    saved=np.load(source/'training-images.npz');pixels=saved['pixels'];shape=(len(saved['port_m']),len(saved['forward_m']))
    initial_pose=np.asarray(json.loads(args.initial_motion.read_text())['parameters']) if args.initial_motion else np.zeros(5)
    dataset=Dataset();window=dataset.window_from_description(win.result);bands=[]
    for c,old in enumerate(window['fast']):
        s=state(win,np.asarray(r['parameters']),c);kin=RigidPatch(win.time[c],args.centre,s['position'],s['velocity'],window['local_to_ecef'],motion.tx[3],motion.rx,s['frame'],s['frame_rate'])
        bands.append(PatchBand(old.field,kin,np.array([[0.,0.,20.]]),old.training,motion.tx[c],motion.rx))
    worker=DenseImages.reopen(source/'training/model.bin',derivatives=8);worker.training=True
    geometry=np.zeros((5,worker.stride,74,4),'<f4')
    for c,b in enumerate(bands):geometry[c,:,:50]=trial_geometry(b,initial_pose,b.training,worker.stride)
    try:images=worker.evaluate_geometry(geometry,derivatives=False)[:,0]
    finally:worker.close()
    # Candidate acquisition uses the declared conditional noise metric,
    # without renormalising a band's image by its observed total energy.
    # Exact dictionary norms enter the later orthogonal-LS selection.
    normal=np.asarray([8192*np.sum(b.field.data['information'][b.training]) for b in bands])
    independent=np.sum(abs(images)**2/normal[:,None],axis=0);score=independent
    acquisition='Sum of conditionally noise-normalised independent matched powers'
    if args.initial_motion:
        fitted=json.loads(args.initial_motion.read_text())
        if 'physical_gains_real_imag' in fitted:
            gains=np.asarray([complex(*z) for z in fitted['physical_gains_real_imag']])
            score,_=statistic(images/np.sqrt(normal[:,None])*np.exp(-1j*np.angle(gains))[:,None])
            acquisition='Coherent positive-amplitude Gaussian GLRT with frozen initial band phases'
    maxima=[]
    for z in list(images)+[score.astype(complex)**.5]:
        power=abs(z).reshape(shape)**2;candidate=np.flatnonzero(power.ravel()==maximum_filter(power,size=7).ravel())
        maxima.extend(candidate[np.argsort(power.ravel()[candidate])[::-1][:args.candidates//3]].tolist())
    maxima=np.unique(maxima);order=maxima[np.argsort(score[maxima])[::-1]];chosen=[]
    for index in order:
        if all(np.linalg.norm(pixels[index,:2]-pixels[j,:2])>=8. for j in chosen):chosen.append(index)
        if len(chosen)==args.candidates:break
    points=pixels[chosen]
    designs=[];data=[];selected=[]
    for c,band in enumerate(bands):
        f=band.field;h,_=band.design(initial_pose,points=points);y=f.y;train=band.training
        # Project the nuisance separately in each fold, on both the model and
        # data. Per-row image phase is not treated as an unbiased observation.
        a=np.zeros_like(h);d=np.zeros_like(y)
        for mask in [train,~train]:
            a[mask]=profile_static(h,f.root_information,mask)[mask];d[mask]=profile_static(y,f.root_information,mask)[mask]
        designs.append(a.reshape(-1,len(points)));data.append(d.ravel());selected.append(np.repeat(train,len(f.offsets)))
        print(json.dumps(dict(channel=c,candidates=len(points),seconds=time.monotonic()-started)),flush=True)
    training_a=[a[s] for a,s in zip(designs,selected)];training_y=[y[s] for y,s in zip(data,selected)];records=[]
    for support,step in select(training_a,training_y,args.points,tolerance=1e-8):
        fit=report_fit(designs,data,support,selected);fit.update(selection=step,count=len(support));records.append(fit)
        report=dict(complete=False,scope=__doc__,motion=str(args.motion),mid_s=args.centre,candidate_positions=points,counts=records,seconds=time.monotonic()-started,
            initial_motion_parameters=initial_pose,initial_motion_source=str(args.initial_motion) if args.initial_motion else None,candidate_acquisition=acquisition)
        (out/'analysis.json').write_text(json.dumps(serial(report),indent=2)+'\n')
        print(json.dumps(dict(count=len(support),unused=fit['unused'],seconds=time.monotonic()-started)),flush=True)
    report['complete']=True;(out/'analysis.json').write_text(json.dumps(serial(report),indent=2)+'\n')
    plot(out,report,pixels,shape,images)


def plot(out,report,pixels,shape,images):
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    points=np.asarray(report['candidate_positions']);records=report['counts'];fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for label in ['independent','shared','shared-phase']:
        axes[0,0].plot([r['count'] for r in records],[100*r['unused'][label]['explained_fraction'] for r in records],'.-',label=label)
    axes[0,0].set(xlabel='Number of shared locations',ylabel='Unused data energy predicted (%)');axes[0,0].legend()
    axes[0,1].plot([r['count'] for r in records],[r['training_rss'] for r in records]);axes[0,1].set(xlabel='Number of shared locations',ylabel='Training residual energy')
    for ax,record in zip(axes[1],records[::max(1,len(records)-1)]):
        power=np.sum(abs(images)**2,axis=0);ax.scatter(pixels[:,0],pixels[:,1],c=10*np.log10(power/power.max()),cmap='magma',vmin=-30,vmax=0,s=3)
        chosen=points[record['support']];ax.scatter(chosen[:,0],chosen[:,1],facecolors='none',edgecolors='#00ccff',s=70)
        ax.set(title=f'{record["count"]} shared locations',xlabel='East-coordinate at midpoint (m)',ylabel='North-coordinate at midpoint (m)',aspect='equal')
    fig.suptitle(f'{report["mid_s"]-15:.2f}–{report["mid_s"]+15:.2f} s · exact extended-scene prediction\nTraining-only shared-location pursuit; same motion for all bands; unused complex range data evaluate scene transfer')
    fig.savefig(out/'scene-transfer.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()

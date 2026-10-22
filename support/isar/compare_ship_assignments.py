"""Full-recording primary/backup comparisons with frozen common motion.

Two physical ray hypotheses per band are backprojected once per window.
All 32 licensed combinations then fit only four constant interband phases
on training images. Unused epochs retain the selected phases. This is a
conditional site comparison; it does not refit a different ship per band.
"""
import argparse,json,itertools
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from dense_patch_image import DenseImages
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from motion import C,excess_path
from ship_site_signature import licensed_options
from report_measured_scatterer_phases import serial

CODES={'12013':'0','12015':'1','12014':'2'}


def phase_fit(image,gains,initial):
    free=[0,1,2,4]
    def evaluate(p):
        g=gains.copy();g[free]*=np.exp(1j*p);z=np.einsum('c,wcp->wp',g.conj(),image)
        power=abs(z)**2;total=power.sum(axis=1);square=(power**2).sum(axis=1)
        h=np.mean(2*np.log(total)-np.log(square));cot=2*z*(1/total[:,None]-power/square[:,None])
        gradient=np.asarray([2*np.real(np.sum(cot.conj()*(-1j*g[c].conjugate()*image[:,c])))/len(image) for c in free])
        return float(h),gradient
    best=min([minimize(evaluate,p,jac=True,method='BFGS',options=dict(maxiter=100,gtol=1e-7)) for p in [initial,np.zeros(4)]],key=lambda r:r.fun)
    return dict(phases=best.x,training_entropy=float(best.fun),maximum_gradient=float(max(abs(best.jac))),message=str(best.message))


def entropy_score(image,gains,phase):
    g=gains.copy();g[[0,1,2,4]]*=np.exp(1j*phase);z=np.einsum('c,wcp->wp',g.conj(),image);p=abs(z)**2
    return float(np.mean(2*np.log(p.sum(axis=1))-np.log((p**2).sum(axis=1))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=False);motion=JointMotion(a.motion);result=motion.result;root=Path('docs/data/calibration_20261016_sites')
    options=licensed_options(root);choices=[sorted([(CODES[o['site_id']],np.asarray(o['transmitter_ecef_m'])) for o in options if o['channel']==c],key=lambda x:x[0]) for c in range(5)]
    if any(len(o)!=2 for o in choices):raise ValueError('Exactly the declared primary and backup sites are required')
    current='00002';unique=list(dict.fromkeys(current));counts=result['parameter_counts'];n=counts['path']+counts.get('cross',0)
    offsets=result['parameters'][n:n+counts['antenna']];offsets=np.asarray(offsets).reshape(-1,3)@motion.receiver_local.T if offsets else np.zeros((len(unique),3))
    offset_by_site=dict(zip(unique,offsets));windows=[Window(p.parent,result['interval_s'],result['path_spacing_seconds'],result.get('cross_spacing_seconds',15.)) for p in sorted(a.windows.glob('window-*/analysis.json'))]
    training=[];unused=[];normal=[];worker=None
    try:
        for index,w in enumerate(windows):
            channels=[]
            for train,target in [(True,training),(False,unused)]:
                source=w.directory/('training' if train else 'unused')/'model.bin'
                if worker is None:worker=DenseImages.reopen(source,derivatives=8)
                else:worker.replace_archive(source)
                worker.training=train;images=[];original=[w.context[f'tx_{c}'].copy() for c in range(5)]
                for option in range(2):
                    references=[]
                    for c in range(5):
                        code,position=choices[c][option];tx=position+offset_by_site.get(code,np.zeros(3));w.context[f'tx_{c}']=tx
                        ref=(excess_path(motion.gauge_point[None],tx,motion.rx)[0]-excess_path(motion.gauge_point[None],original[c],motion.rx)[0])[0]
                        references.append(float(ref))
                    geo,_,_=w.geometry(np.asarray(result['parameters']),worker,motion.gauge_point,result['transmitter_groups'],motion.receiver_local,antennas=False)
                    for c,reference in enumerate(references):
                        z=(geo[c,:,9,1]+1j*geo[c,:,9,2])*np.exp(-2j*np.pi*motion.rf[c]/C*reference)
                        geo[c,:,9,1]=z.real;geo[c,:,9,2]=z.imag
                    images.append(worker.evaluate_geometry(geo,derivatives=False)[:,0])
                for c in range(5):w.context[f'tx_{c}']=original[c]
                target.append(images)
            normal.append([8192*np.sum(w.context[f'information_{c}'][~w.training[c]]) for c in range(5)])
            print(json.dumps(dict(formed_window=index+1,windows=len(windows))),flush=True)
    finally:
        if worker is not None:worker.close()
    training=np.asarray(training,dtype='<c8');unused=np.asarray(unused,dtype='<c8');normal=np.asarray(normal)
    np.savez_compressed(out/'image-bank.npz',training=training,unused=unused,normal=normal,centres=[w.result['mid_s'] for w in windows])
    gains=np.asarray([complex(*z) for z in result['calibration_gains_real_imag']]);initial=np.asarray(result['parameters'][n+counts['antenna']:]) if counts['phase'] else np.zeros(4)
    comparisons=[]
    for selected in itertools.product([0,1],repeat=5):
        code=''.join(choices[c][option][0] for c,option in enumerate(selected));index=np.arange(5)
        train=training[:,selected,index];test=unused[:,selected,index];fit=phase_fit(train,gains,initial)
        fit.update(assignment=code,options=selected,unused_entropy=entropy_score(test,gains,fit['phases']))
        comparisons.append(fit);print(json.dumps(serial(fit)),flush=True)
        (out/'analysis.json').write_text(json.dumps(serial(dict(complete=False,motion=str(a.motion),comparisons=comparisons)),indent=2)+'\n')
    comparisons.sort(key=lambda r:r['training_entropy']);best=comparisons[0];selection=best['options'];image=unused[:,selection,np.arange(5)]
    train=training[:,selection,np.arange(5)];g=gains.copy();g[[0,1,2,4]]*=np.exp(1j*best['phases'])
    reports=[]
    for w,i,j,info in zip(windows,image,train,normal):
        scientific=np.load(w.directory/'training-images.npz');pixels=scientific['pixels'];score=np.sum(abs(j)**2/info[:,None],axis=0)
        allowed=np.ones(len(pixels),bool);points=[]
        for k in range(6):
            at=int(np.argmax(np.where(allowed,score,-np.inf)));allowed&=np.sum((pixels[:,:2]-pixels[at,:2])**2,axis=1)>=35**2
            powers=abs(i[:,at])**2/info;combined=abs(np.vdot(g,i[:,at]))**2/np.sum(abs(g)**2*info)
            phase=np.degrees(np.angle(i[:,at]*i[3,at].conjugate()*g.conj()*g[3]))
            points.append(dict(pixel_index=at,point=pixels[at],individual_power=powers,combined_power=combined,
                gain_over_best_band_db=10*np.log10(combined/max(powers)),coherent_fraction_of_independent_power=combined/powers.sum(),phase_residual_degrees=phase))
        reports.append(dict(mid_s=w.result['mid_s'],points=points))
    final=dict(complete=True,motion=str(a.motion),selection='Training only; unused scores do not select the winner',comparisons=comparisons,phase_diagnostics=reports)
    (out/'analysis.json').write_text(json.dumps(serial(final),indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(3,1,figsize=(14,10),layout='constrained');times=[w.result['mid_s'] for w in windows]
    for c in [0,1,2,4]:ax[0].scatter(times,[r['points'][0]['phase_residual_degrees'][c] for r in reports],s=16,label=['Seven','SBS','Nine','Ten','ABC'][c])
    ax[0].set(ylabel='Unused phase residual (deg)',ylim=(-180,180));ax[0].legend(ncol=4)
    ax[1].plot(times,[r['points'][0]['gain_over_best_band_db'] for r in reports],'.-');ax[1].axhline(0,color='grey');ax[1].set_ylabel('Unused matched-power gain (dB)')
    rank=sorted(comparisons,key=lambda r:r['training_entropy']);ax[2].plot([r['assignment'] for r in rank[:12]],[r['training_entropy'] for r in rank[:12]],'o',label='Training');ax[2].plot([r['assignment'] for r in rank[:12]],[r['unused_entropy'] for r in rank[:12]],'x',label='Unused');ax[2].legend();ax[2].set_ylabel('Image entropy')
    fig.suptitle(f'Frozen common-motion site comparison; best training assignment {best["assignment"]}\nOne constant interband calibration over the full capture; brightest training-selected vessel cell per window')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
if __name__=='__main__':main()

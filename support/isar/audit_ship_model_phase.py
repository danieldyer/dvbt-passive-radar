"""Residual phase relative to a fitted extended target, not dirty-image peaks.

Each temporal statistic is the exact matched projection of the range data
onto the complete predicted scattering scene. Predicted signal energy gives
its conditional information. These diagnostics include noise/model error;
they are not isolated oscillator phase measurements or phase corrections.
"""
import argparse,json
from pathlib import Path
import numpy as np
from audit_ship_exact_aperture import state
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from rigid_patch import RigidPatch,PatchBand,profile_static
from ship_window import Dataset
from gpga import Isolation
from matched_phase import history
from audit_ship_scattering_envelopes import linear_basis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.motion.read_text());motion=JointMotion(Path(d['motion']));r=motion.result;mid=d['mid_s']
    win=Window(a.windows/f'window-{mid:07.2f}',r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds']);dataset=Dataset()
    exact=dataset.window_from_description(win.result);points=np.asarray(d['points']);parameters=np.asarray(d['parameters'])
    envelopes=None
    if 'envelope_spacing_s' in d:envelopes=np.load(a.motion.parent/'envelope-coefficients.npz')
    else:
        if 'coherent_scene_refinement' in d:coef=np.asarray(d['coherent_scene_refinement']['coefficient_real_imag'])
        else:coef=np.asarray(d['scene_response_comparison']['phase_coherent']['coefficient_real_imag'])
        coef=coef[...,0]+1j*coef[...,1]
    arrays={};reports=[]
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(14,11),layout='constrained',sharex=True)
    for c,old in enumerate(exact['fast']):
        s=state(win,np.asarray(r['parameters']),c);kin=RigidPatch(win.time[c],mid,s['position'],s['velocity'],exact['local_to_ecef'],motion.tx[3],motion.rx,s['frame'],s['frame_rate'])
        band=PatchBand(old.field,kin,points,old.training,motion.tx[c],motion.rx);h,_=band.design(parameters)
        if envelopes is not None:
            b,nodes=linear_basis(band.field.time,win.result['interval_s'],d['envelope_spacing_s']);np.testing.assert_array_equal(nodes,envelopes['knots_s'])
            coefficients=b@envelopes[f'coefficient_{c}'].T;raw=np.einsum('trp,tp->tr',h,coefficients)
        else:raw=np.einsum('trp,p->tr',h,coef[c])
        model=np.zeros_like(raw);data=np.zeros_like(raw)
        for mask in [band.training,~band.training]:
            model[mask]=profile_static(raw,band.field.root_information,mask)[mask]
            data[mask]=profile_static(band.field.y,band.field.root_information,mask)[mask]
        z,info=history(data,model);t=band.field.time;train=band.training
        kernel=Isolation(t,info,train,.6);smoothed,_=kernel.apply(z);phase=np.unwrap(np.angle(smoothed[:,0]));phase-=np.average(phase,weights=info*train)
        axes[0].plot(t,np.degrees(phase),label=['Seven','SBS','Nine','Ten','ABC'][c]);axes[1].plot(t,abs(smoothed[:,0]));axes[2].plot(t,10*np.log10(np.maximum(info,1e-30)))
        selected=~train;weight=info*selected;power=abs(np.sum(weight*z[:,0]))**2/weight.sum();energy=np.sum(weight*abs(z[:,0])**2)
        report=dict(channel=c,unused_coherent_fraction=float(power/energy),unused_matched_model_power=float(power),
            unused_information=float(weight.sum()),display_smoothed_phase_span_degrees=float(np.ptp(np.degrees(phase))))
        reports.append(report)
        for key,value in dict(time_s=t,value=z[:,0],information=info,training=train,prediction=model,observation=data).items():arrays[f'{key}_{c}']=value
    axes[0].set(ylabel='Residual phase (deg)',title='Training-only display smoothing; one constant phase origin removed');axes[0].legend(ncol=5)
    axes[1].set(ylabel='Matched residual magnitude',title='Unity is the prediction of the fitted scene');axes[1].axhline(1,color='grey')
    axes[2].set(ylabel='Predicted scene energy (dB)',xlabel='Capture time (s)')
    fig.suptitle(f'{mid-15:.2f}–{mid+15:.2f} s · measured phase relative to the full fitted scattering scene\nOne physical motion and shared scatterer phases; model and data receive the same nuisance projection; no correction applied')
    fig.savefig(a.output/'model-phase.png',dpi=140);plt.close(fig)
    np.savez_compressed(a.output/'histories.npz',**arrays)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),bands=reports),indent=2)+'\n');print(json.dumps(reports),flush=True)
if __name__=='__main__':main()

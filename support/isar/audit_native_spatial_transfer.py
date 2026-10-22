"""Training-selected neighbours: exact native-row deblending and band transfer.

Both temporal folds share fixed physical support, motion and calibration.
Independent complex point coefficients are measured jointly, not by reading
phase from individual bright pixels. Unknown stationary coefficients are fit
on training only and frozen for prediction. The output is a model diagnostic,
not a claim that the selected peaks are separate material ship scatterers.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from recorded_injection_fields import load_fields,read_profiles,common_projection
from fixed_range_field import stationary_profile_design
from native_folds import training_mask
from render_injected_point import cartesian_state
from stationary import Stationary
from recovery_experiment import NAMES
from report_measured_scatterer_phases import serial


def candidates(image,count=5):
    east,north=image['east_m'],image['north_m'];xx,yy=np.meshgrid(east,north)
    power=image['power'][:5].sum(axis=0)
    maxima=(power==maximum_filter(power,size=9))&(abs(xx)<220)&(abs(yy)<260)
    order=np.flatnonzero(maxima);order=order[np.argsort(-power.ravel()[order])]
    points=[np.zeros(3)]
    for i in order:
        q=np.array([xx.ravel()[i],yy.ravel()[i],0.])
        if any(np.linalg.norm(q[:2]-p[:2])<70 for p in points):continue
        points.append(q)
        if len(points)==count:break
    return np.asarray(points)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=Path('docs/data/isar_20261018/native-spatial-validation'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=5)
    p.add_argument('--expected-profiles',type=Path,help='Project an explicit synthetic control through the identical frozen observation map')
    p.add_argument('--points-from',type=Path,help='Replay fixed candidate coordinates instead of selecting again')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path('docs/data/isar_20261018/native-shared-radial-point-6s')
    model=json.loads((root/'analysis.json').read_text());render=json.loads((root/'render-fit.json').read_text())
    first,last=model['interval_s'];mid=model['midpoint_s'];state=np.asarray(model['physical_state_enu'])
    with np.load(args.directory/'image-training/images.npz') as z:image=dict(z)
    points=np.load(args.points_from) if args.points_from else candidates(image,args.count)
    np.save(out/'points.npy',points)
    fields,rotation,rx,tx,_,_=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'),
        offsets=np.arange(0,25),fixed_epoch_s=mid)
    expected=None
    if args.expected_profiles:
        expected_header,expected=read_profiles(args.expected_profiles)
    g=np.asarray(render['fits']['constant_velocity_gls']['coefficients_real_imag'])[:,0];gains=g[:,0]+1j*g[:,1]
    training=[];testing=[];info=[];records=[]
    for c,original in enumerate(fields):
        f=original.windowed(first,last);keep=training_mask(f.time,first);selected=np.repeat(keep,len(f.offsets))
        if expected is not None:
            np.testing.assert_array_equal(expected[c]['rows'][:,1],f.data['row_time'])
            f.y=common_projection(f,expected[c],expected_header['first_range_sample']).reshape(f.y.shape)
        motion=cartesian_state(f.data['row_time'],state,mid,rotation,rx,-.22678571428571473)
        h=[]
        for q in points:
            pos=motion['position_ecef_m']+motion['body_to_ecef']@q
            vel=motion['velocity_ecef_mps']+motion['frame_velocity']@q
            h.append(f.trajectory(pos,vel,position_offset=pos-f.data['centre_ecef_m']).ravel())
        h=np.column_stack(h);stationary=stationary_profile_design(f);nuisance=Stationary(stationary,selected)
        a=nuisance.remove(h[selected]);y=nuisance.remove(f.y.ravel()[selected])
        coefficient=np.linalg.lstsq(a,y,rcond=1e-10)[0]
        beta=nuisance.solve(f.y.ravel()[selected]-h[selected]@coefficient)
        clean=f.y.ravel()[~selected]-stationary[~selected]@beta;test=h[~selected]
        measured=np.linalg.lstsq(test,clean,rcond=1e-10)[0]
        reference=nuisance.solve(f.y.ravel()[selected]);baseline=f.y.ravel()[~selected]-stationary[~selected]@reference
        residual=clean-test@coefficient
        training.append(coefficient);testing.append(measured);info.append(np.linalg.pinv(test.conj().T@test,rcond=1e-10).diagonal().real)
        records.append(dict(channel=c,name=NAMES[c],training_condition=float(np.linalg.cond(a)),testing_condition=float(np.linalg.cond(test)),
            excluded_prediction_improvement=float(1-np.vdot(residual,residual).real/np.vdot(baseline,baseline).real),
            point_information_retention=np.sum(abs(a)**2,axis=0)/np.sum(abs(h[selected])**2,axis=0)))
        print(NAMES[c],serial(records[-1]),flush=True)
    training,testing,variance=np.asarray(training),np.asarray(testing),np.asarray(info)
    phases=[np.degrees(np.angle(a/gains[:,None])) for a in (training,testing)]
    snr=10*np.log10(np.maximum(abs(testing)**2/variance,1e-30))
    result=dict(scope=__doc__,complete=True,point_selection='Training-only independent-band image; anchor fixed by training fit',
        points_east_north_up_m=points,channels=records,training_coefficient_real_imag=np.stack([training.real,training.imag],axis=-1),
        excluded_coefficient_real_imag=np.stack([testing.real,testing.imag],axis=-1),
        phase_relative_to_frozen_calibration_deg=phases,
        excluded_conditional_thermal_point_power_db=snr,
        noise_scope='Known-waveform conditional thermal metric only; residual interference and uncertainty of training stationary coefficients are not included in quoted point powers',
        geographic_localisation_certified=False,motion_refitted=False,band_calibration_refitted=False,
        expected_profiles=None if args.expected_profiles is None else str(args.expected_profiles))
    fig,axes=plt.subplots(1,3,figsize=(16,6),layout='constrained')
    power=image['power'][:5].sum(axis=0);db=10*np.log10(np.maximum(power,1e-30))
    axes[0].imshow(db,origin='lower',extent=[image['east_m'][0],image['east_m'][-1],image['north_m'][0],image['north_m'][-1]],
        cmap='magma',vmin=db.max()-40,vmax=db.max(),aspect='equal')
    for j,q in enumerate(points):axes[0].plot(*q[:2],'+',color='cyan');axes[0].annotate(str(j),q[:2],color='cyan',xytext=(5,4),textcoords='offset points')
    axes[0].set(title='Training-only candidate support',xlabel='East from fitted point (m)',ylabel='North (m)')
    for ax,phase,label in zip(axes[1:],phases,['Training','Excluded']):
        im=ax.imshow(phase,cmap='twilight',vmin=-180,vmax=180,aspect='auto')
        ax.set(title=label+' jointly deblended phase',xlabel='Candidate',yticks=range(5),yticklabels=NAMES)
        for c in range(5):
            for j in range(len(points)):ax.text(j,c,f'{phase[c,j]:.0f}°',ha='center',va='center',fontsize=9,color='white')
    fig.colorbar(im,ax=axes[1:].tolist(),label='Phase relative to frozen anchor calibration (degrees)',shrink=.8)
    fig.suptitle('Exact original-row candidate histories; no motion or phase refit on excluded epochs\nCandidates may include PSF lobes, stationary clutter and unresolved echo mixtures')
    fig.savefig(out/'spatial-phase-transfer.png',dpi=160);plt.close(fig)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

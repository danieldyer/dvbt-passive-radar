"""Independent rigid-CV echoes through actual native recording coordinates.

The waveform pulses, information, epochs, covariance, bistatic sites and C0/C1
operator come from the recording. Direct distances and numerical OFDM symbol
quadrature generate the synthetic data independently of the fit's analytic
geometry. This control fixes source support and excludes the recorded scene;
it does not replace blind point injection into the measured interference.
"""
import argparse,json
from pathlib import Path
import numpy as np
from cartesian_brightness_scene import CartesianBrightnessBand,CartesianBrightnessFit
from cv_rigid_patch import CvRigidPatch
from fixed_range_field import stationary_profile_design
from native_folds import training_mask
from recorded_injection_fields import load_fields
from report_measured_scatterer_phases import serial
from test_cartesian_brightness_scene import independent_echo


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiles',type=Path,default=Path('build/isar-20261018/current-native-full/profiles.bin'))
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    root=Path('docs/data/isar_20261018/native-spatial-validation')
    reference=json.loads((root/'ais-near-shared-point-4s/analysis.json').read_text())
    first,last=reference['interval_s'];mid=reference['midpoint_s'];state=np.asarray(reference['physical_state_enu'])
    points=np.load(root/'ais-near-three-point-refined-support.npy')
    fields,rotation,rx,tx,_,_=load_fields(args.profiles,offsets=np.arange(-12,13),fixed_epoch_s=mid,interval_s=(first,last))
    bands=[]
    for c,f in enumerate(fields):
        kin=CvRigidPatch(f.data['row_time'],mid,state[0],state[1],rotation,rx)
        knots=np.linspace(f.data['row_time'][0],f.data['row_time'][-1],5)
        bands.append(CartesianBrightnessBand(f,training_mask(f.time,first),knots,kin,points,.3*c,stationary_profile_design(f),channel=c))
    generator=CartesianBrightnessFit(bands)
    truth=np.r_[10.,-15.,.03,-.02,.24,40.,-80.,.2,-.5,.8,.1,-.2,.2,-.1]
    strengths=[]
    for b in bands:
        coefficient=.02*(1+.2*np.sin(np.arange(15)+b.channel)).reshape(3,5);strengths.append(coefficient)
        b.field.y[:]=independent_echo(generator,truth,b.channel,coefficient)
        b.__init__(b.field,b.training,b.knots,b.kinematics,b.points,b.band_phase,b.design,channel=b.channel)
    model=CartesianBrightnessFit(bands);seed=np.zeros(model.size);seed[4]=.15
    truth_value,truth_gradient=model.objective(truth)
    if truth_value>1e-12:raise ValueError('Independent synthesis and native analytic response disagree')
    result=dict(scope=__doc__,complete=False,profiles=str(args.profiles),interval_s=[first,last],midpoint_s=mid,
        reference_state_enu=state,points_east_north_up_m=points,truth=truth,seed=seed,
        independent_synthesis_native_objective=truth_value,independent_synthesis_native_gradient=truth_gradient,
        source_support_known=True,recorded_scene_added=False,geographic_localisation_certified=False)
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    fit=model.fit(seed,max_iterations=200);result['initial_fit']=fit;save();print('Coarse-seed fit',serial(fit),flush=True)
    fit=model.polish(fit['parameters'],max_iterations=160);result['fit']=fit;save();print('Native polish',serial(fit),flush=True)
    error=np.asarray(fit['parameters'])-truth
    result.update(position_error_m=float(np.linalg.norm(error[:2])),velocity_error_mps=float(np.linalg.norm(error[2:4])),
        spin_error_degps=float(abs(error[4])),rotation_centre_offset_error_m=float(np.linalg.norm(error[5:7])),
        unused_prediction=[b.prediction(fit['parameters']) for b in bands],complete=True)
    save();print('Errors',serial({k:v for k,v in result.items() if k.endswith('_error_m') or k.endswith('_error_mps')}),flush=True)


if __name__=='__main__':main()

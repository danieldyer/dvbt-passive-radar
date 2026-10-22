"""Certify fit/native physical phase closure and the local velocity derivative."""
import argparse,json
from pathlib import Path
import numpy as np
from audit_ship_exact_aperture import state
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from local_ship_motion import LocalShipMotion
from motion import enu_basis,excess_path
from rigid_patch import RigidPatch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    model=LocalShipMotion(a.motion);base=JointMotion(Path(model.refinement['motion']));r=base.result
    window=Window(a.windows/f'window-{model.mid:07.2f}',r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds'])
    sample=base.track.sample(np.array([base.epoch+model.mid]),max_gap_s=400.)
    local=enu_basis(sample['latitude'][0],sample['longitude'][0]);gauge=base.base(np.array([model.mid]))['body_to_ecef'][0].T@local
    phase=[];velocity=[]
    for c in range(5):
        t=window.time[c];s=state(window,np.asarray(r['parameters']),c)
        kin=RigidPatch(t,model.mid,s['position'],s['velocity'],model.refinement_local,model.tx[3],model.rx,s['frame'],s['frame_rate'])
        actual=model.state(t)
        for point in np.asarray(model.refinement['points']):
            expected,speed=kin.point_state(model.local_parameters,point)
            physical=actual['position_ecef_m']+np.einsum('tij,j->ti',actual['body_to_ecef']@gauge,point)
            vel=actual['velocity_ecef_mps']+np.einsum('tij,j->ti',actual['frame_velocity']@gauge,point)
            phase.append(float(np.max(abs(excess_path(physical,model.tx[c],model.rx)[0]-excess_path(expected,model.tx[c],model.rx)[0]))*360*model.rf[c]/299792458.))
            velocity.append(float(np.max(np.linalg.norm(vel-speed,axis=1))))
    # Absolute Unix epochs are binary64: millisecond finite differences can
    # magnify their ~0.2 us quantisation into an apparent velocity error.
    t=np.linspace(*np.array(model.interval)+np.array([.03,-.03]),1001);h=.02
    finite=(model.state(t+h)['position_ecef_m']-model.state(t-h)['position_ecef_m'])/(2*h)
    error=float(np.max(np.linalg.norm(finite-model.state(t)['velocity_ecef_mps'],axis=1)))
    # Changing a display/view interval must not redefine any spline knots.
    t0=np.array([model.mid]);before=base.state(t0);base.interval=model.interval
    after=base.state(t0)
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    result=dict(motion=str(a.motion),maximum_fit_native_phase_error_degrees=max(phase),maximum_velocity_closure_mps=max(velocity),
        maximum_time_derivative_velocity_error_mps=error,view_interval_preserves_curve=True,passed=max(phase)<.001 and max(velocity)<1e-5 and error<1e-4)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    if not result['passed']:raise ValueError('Physical motion closure failed')
if __name__=='__main__':main()

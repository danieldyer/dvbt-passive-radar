"""Original-field phase/adjoint checks before residual-driven reconstruction."""
import argparse,json,tempfile,copy
from pathlib import Path
import numpy as np
from block_pulse import BlockPulse
from waveform_kernels import KernelField
from test_full_path_point import problem
from physical_scene_adjoint import PhysicalSceneAdjoint
from full_path_point import carrier_bin
from bistatic_path_lift import lift
from rigid_patch import profile_static


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--motion',type=Path,help='Instead compare the complete saved local recorded scene')
    p.add_argument('--freeze-body-attitude',action='store_true')
    a=p.parse_args()
    rng=np.random.default_rng(848);bands=[];pixels=np.array([[0.,0.,0.],[27.,-19.,10.],[-35.,22.,20.],[45.,12.,10.]])
    frequencies=[177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]
    for c,rf in enumerate(frequencies):
        b=problem();f=copy.copy(b.field);f.k=2*np.pi*rf/299792458.;f.metadata={**f.metadata,'rf_hz':rf}
        # As in the saved recording, retain a fixed carrier origin. The
        # fixture's original moving deramp makes its origin point stationary
        # and exactly removed by the nuisance, leaving undefined test phase.
        f.data={**f.data,'carrier_reference_path_m':np.full(len(b.t),f.data['centre_path_m'][0])}
        kernel=BlockPulse(np.repeat(f.pulse.samples[None,:],len(b.t),axis=0));b.field=KernelField(f,kernel,np.arange(len(b.t)))
        b.groups=(b.t>=8).astype(int);b.field.y=rng.normal(size=f.y.shape)+1j*rng.normal(size=f.y.shape)
        b.field.y+=f.root_information[:,None]*(.4+.2j)*(1+b.groups[:,None]);bands.append(b)
    if a.motion:
        from full_scene_context import open_scene
        fit,_,scene,_=open_scene(a.motion)
        if scene[0].points.ndim!=2:raise ValueError('Use a single complete local scene')
        bands=[v.point for v in scene];pixels=scene[0].points
    reference_error=None
    if a.freeze_body_attitude:
        if not a.motion:raise ValueError('Frozen-body validation requires a recorded physical scene')
        from fit_distributed_physical_scene import freeze_body_attitude
        before=[b.waveform(np.zeros(b.b.shape[1]),c,False,np.zeros(3)) for c,b in enumerate(bands)]
        freeze_body_attitude(bands,fit)
        reference_error=0.
        for c,(b,old) in enumerate(zip(bands,before)):
            after=b.waveform(np.zeros(b.b.shape[1]),c,False,np.zeros(3))
            np.testing.assert_array_equal(after,old)
    reports=[]
    with tempfile.TemporaryDirectory(prefix='physical-scene-adjoint-',dir='build') as directory:
        operator=PhysicalSceneAdjoint(bands,pixels,Path(directory))
        try:
            for trial in range(2):
                values=[b.field.y if trial==0 else (.6+.3j)*b.field.y for b in bands]
                actual=operator.evaluate(values);expected=np.zeros_like(actual)
                for c,(b,y) in enumerate(zip(bands,values)):
                    state=lift(b.anchor,b.velocity,b.direction,b.direction_rate,b.base_path,b.base_rate,b.motion.tx[3],b.motion.rx)
                    projected=profile_static(y,b.field.root_information,b.training,b.groups)*b.training[:,None]
                    for j,point in enumerate(pixels):
                        position=state['position_ecef_m']+b.frame@(point-b.reference_point)
                        velocity=state['velocity_ecef_mps']+b.frame_rate@(point-b.reference_point)
                        _,_,g=b.field.path_kinematics(position,velocity,transmitter=b.motion.tx[c],receiver=b.motion.rx,position_offset=position-b.field.data['centre_ecef_m'])
                        factor,_=carrier_bin(b.t,np.sum(g*velocity,axis=1),b.coordinate['native_time_s'],b.coordinate['native_information'],b.field.k)
                        exact=b.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point)
                        # The proposal is explicitly the representative-epoch
                        # operator, not the final original-bin likelihood.
                        h=exact/factor[:,None]
                        expected[c,j]=np.vdot(h,projected)
                error=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected));phase=float(np.degrees(max(abs(np.angle(actual/expected)).ravel())))
                if error>8e-4 or phase>.08:raise ValueError(f'Physical residual adjoint changed phase/operator: {error}, {phase}')
                reports.append(dict(trial=trial,relative_complex_error=error,maximum_phase_error_degrees=phase))
        finally:operator.close()
    a.output.write_text(json.dumps(dict(passed=True,scope=__doc__,motion=str(a.motion) if a.motion else None,phase_alignment_fitted=False,checks=reports,
        frozen_body_attitude=a.freeze_body_attitude,reference_waveform_relative_error=reference_error),indent=2)+'\n')
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()

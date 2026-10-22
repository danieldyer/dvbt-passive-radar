"""Check recorded NNLS normals against the original projected design actions."""
import argparse,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_path_scene import SceneBand
from sparse_positive_envelope import Normal
from motion import enu_basis
from rigid_patch import profile_static
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--channel',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();start=time.monotonic();fit,motion,old,saved=open_scene(a.motion);scene=json.loads(a.scene.read_text());c=a.channel;b=old[c].point
    at=np.array([scene['mid_s']]);state=motion.state(at);sample=motion.track.sample(motion.epoch+at,max_gap_s=400.)
    gauge=state['body_to_ecef'][0].T@enu_basis(sample['latitude'][0],sample['longitude'][0]);points=np.asarray(scene['points'])@gauge.T
    band=SceneBand(b,points,scene['scene_window_edges_s'],fit['amplitude_spacing_s'],scene.get('active_points'))
    h=band.design(np.zeros(b.b.shape[1]),c,False);normal=Normal(h,b.field.y,b.field.root_information,b.training,b.amplitude,b.groups)
    rng=np.random.default_rng(409);checks=[]
    def forward(coefficient):
        value=b.amplitude@coefficient.reshape(normal.knots,normal.points)
        return profile_static(np.sum(h*value[:,None,:],axis=2),b.field.root_information,b.training,b.groups)*b.training[:,None]
    for j in range(3):
        coefficient=(rng.normal(size=len(normal.rhs))+1j*rng.normal(size=len(normal.rhs)))/np.sqrt(np.maximum(normal.diagonal,1.))
        model=forward(coefficient);actual=float(np.vdot(model,model).real);expected=float(np.vdot(coefficient,normal.multiply(coefficient)).real)
        rhs=np.vdot(model,b.data[True]);check=np.vdot(coefficient,normal.rhs)
        checks.append(dict(normal_relative_error=abs(actual-expected)/max(actual,1e-300),rhs_relative_error=abs(rhs-check)/max(abs(rhs),1e-300)))
    phi=np.asarray(scene['phase_parameters'])[4:];phase=phi-phi[0]+scene['band_phase_initial_radians'][c]
    result=normal.solve(phase,phase_groups=band.phase_groups);rot=np.exp(1j*phase[band.phase_groups]);coef=rot*result['amplitude']
    residual=forward(coef)-b.data[True]
    gradient=np.asarray(b.amplitude.T@np.einsum('trp,tr->tp',h.conj(),residual)).ravel()
    actual_gradient=normal.multiply(coef)-normal.rhs
    scale=np.sqrt(np.maximum(normal.diagonal,16*np.finfo(float).eps*max(normal.diagonal.max(),1e-30)));power=max(float(max(abs((rot.conj()*normal.rhs).real/scale))),1e-30)
    projected=2*(rot.conj()*gradient).real/scale/power;x=result['amplitude']*scale/power
    kkt=np.where(x>1e-10,projected,np.minimum(projected,0))
    report=dict(complete=True,scope=__doc__,channel=c,scene=str(a.scene),normal_actions=checks,
        numerical_convergence=result['numerical_convergence'],normal_kkt=result['maximum_scaled_kkt'],original_design_kkt=float(max(abs(kkt))),
        gradient_difference_scaled=float(max(abs(2*(rot.conj()*(gradient-actual_gradient)).real/scale/power))),
        maximum_amplitude=float(max(result['amplitude'])),iterations=result['iterations'],polish_iterations=result['normal_polish_iterations'],
        direct_polish=result['direct_cholesky_nnls_polish'],temporal_blocks=len(normal.temporal_blocks),
        block_sizes=[len(x) for x in normal.temporal_blocks],seconds=time.monotonic()-start)
    a.output.write_text(json.dumps(serial(report),indent=2)+'\n');print(json.dumps(serial(report)),flush=True)


if __name__=='__main__':main()

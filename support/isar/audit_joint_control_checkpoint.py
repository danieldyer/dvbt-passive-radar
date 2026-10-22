"""Independent first-order audit of a running fixed-location control."""
import argparse,json
from pathlib import Path
import numpy as np
from antenna_motion import Antennas
from continuous_ship_motion import ContinuousShipMotion
from continuous_ship_fit import ContinuousScene
from full_path_scene import SceneBand
from full_ship_field import load
from joint_continuous_ship import JointCoordinates,JointPointBand,JointPrior
from kinematic_prior import KinematicPrior
from fit_joint_continuous_sites import feasible_improvement


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['motion','field','checkpoint','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile-brightness-preconditioner',action='store_true')
    a=p.parse_args();raw=a.checkpoint.read_bytes();checkpoint=json.loads(raw)
    if checkpoint['assignment']!='00002':raise ValueError('This audit is for the unchanged-geometry control')
    m=ContinuousShipMotion(a.motion);fit=m.result['radar_fit'];antennas=Antennas(m.rotation,list('00002'));model=JointCoordinates(m,antennas)
    fields,coordinates,_=load(a.field);points=np.asarray(m.result['reference_points_east_north_up_m']);edges=np.asarray(fit['aspect_edges_s']);bands=[]
    for c,(field,coord) in enumerate(zip(fields,coordinates)):
        b=JointPointBand(field,coord,model,c,fit['amplitude_spacing_s'])
        bands.append(SceneBand(b,np.broadcast_to(points,(len(edges)-1,*points.shape)),edges,fit['amplitude_spacing_s'],adjoint_gradient=True))
    settings=fit['motion_prior'];prior=JointPrior(model,KinematicPrior(model.base,acceleration_change_mps2=settings['acceleration_change_mps2'],angular_rate_change_degps=settings['angular_rate_change_degps'],timescale_s=settings['timescale_s']))
    scene=ContinuousScene(bands,model,np.angle(m.gains),workers=2,prior=prior,fit_band_phases=True);q=np.asarray(checkpoint['parameters'])
    value,g,values=scene.objective(q,True);matrix,lo,hi=model.constraints();matrix=np.pad(matrix,((0,0),(0,len(q)-model.n)))
    base=model.base;bound=np.r_[np.full(base.np,150.),np.full(base.np,50.),np.full(base.na-1,np.radians(20)/.01),3.,np.zeros(antennas.count),np.full(scene.p+len(scene.free_bands),np.inf)]
    lower,upper=-bound,bound;gap=feasible_improvement(q,g,matrix,lo,hi,lower,upper);violation=float(max(np.max(lo-matrix@q),np.max(matrix@q-hi),np.max(lower-q),np.max(q-upper),0.))
    import hashlib
    report=dict(complete=True,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(raw).hexdigest(),iteration=checkpoint['history'][-1]['iteration'],objective=value,
        radar_rss_fraction=sum(v['cost'] for v in values)/scene.energy,maximum_linearized_feasible_improvement=gap,violation=violation,
        maximum_free_gradient=float(max(abs(g[lower<upper]))),maximum_fixed_coordinate_gradient=float(max(abs(g[lower==upper]))),
        first_order_gate_satisfied=bool(gap is not None and gap<1e-7 and violation<1e-7),scope=__doc__)
    if a.profile_brightness_preconditioner:
        scene.profile_brightness=True;_,information=scene.preconditioner(q)
        report['profiled_brightness_preconditioning']=information
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

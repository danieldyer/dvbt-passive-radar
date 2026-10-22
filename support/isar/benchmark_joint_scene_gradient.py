"""Compare exact materialised and adjoint gradients on a full recorded band."""
import argparse,cProfile,io,json,pstats,time
from pathlib import Path
import numpy as np
from antenna_motion import Antennas
from continuous_ship_motion import ContinuousShipMotion
from full_path_scene import SceneBand
from full_ship_field import load
from joint_continuous_ship import JointCoordinates,JointPointBand


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['motion','field','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();m=ContinuousShipMotion(a.motion);fields,coordinates,_=load(a.field)
    model=JointCoordinates(m,Antennas(m.rotation,['o','o','o','o','r']));b=JointPointBand(fields[0],coordinates[0],model,0)
    fit=m.result['radar_fit'];edges=fit['aspect_edges_s'];points=np.asarray(m.result['reference_points_east_north_up_m'])
    scene=SceneBand(b,np.broadcast_to(points,(len(edges)-1,*points.shape)),edges,fit['amplitude_spacing_s'])
    q=np.zeros(model.n);phase=np.asarray(fit['parameters'])[-scene.phase_count:]+np.angle(m.gains[0]);outcomes=[];records=[]
    a.output.mkdir(parents=True,exist_ok=True)
    for adjoint in [False,True]:
        scene.adjoint_gradient=adjoint;scene.warm=None;profiler=cProfile.Profile();started=time.monotonic()
        result=profiler.runcall(scene.evaluate,q,phase,0);seconds=time.monotonic()-started
        output=io.StringIO();pstats.Stats(profiler,stream=output).sort_stats('cumtime').print_stats(30)
        (a.output/f'profile-{adjoint}.txt').write_text(output.getvalue())
        outcomes.append(result);records.append(dict(adjoint=adjoint,seconds=seconds,cost=result['cost'],maximum_amplitude_kkt=result['maximum_amplitude_kkt']))
        print(json.dumps(records[-1]),flush=True)
    reference,actual=outcomes
    error=float(np.linalg.norm(actual['path_gradient']-reference['path_gradient'])/np.linalg.norm(reference['path_gradient']))
    np.testing.assert_allclose(actual['phase_gradient'],reference['phase_gradient'],rtol=1e-10,atol=1e-9)
    if error>1e-9 or abs(actual['cost']-reference['cost'])>1e-9*reference['energy']:raise AssertionError('Adjoint changes the joint likelihood')
    report=dict(complete=True,channel='Seven',native_16_symbol_bins=len(b.t),runs=records,relative_path_and_antenna_gradient_error=error,
        scope='Full original-record band and shared scene; same NNLS/statistical model, materialised versus exact factored adjoint')
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

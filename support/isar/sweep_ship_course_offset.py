"""Test a global delay of the reported-course orientation hypothesis.

Translation is frozen. Every source's physical midpoint position is retained
under the alternative orientation, so this tests its angular evolution rather
than changing its geographic association. Source and band phases are constant;
only training epochs enter phase/amplitude profiling. No hypothesis is applied
to the original data or adopted automatically.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from declared_ship_orientation import state
from full_path_point import PointBand
from full_path_scene import SceneBand, ScenePath, warm_phases
from full_ship_field import load
from joint_ship_motion import JointMotion
from local_ship_motion import load_motion
from motion import excess_path, enu_basis


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--offsets-s',type=float,nargs='+',default=[-45.,-30.,-15.,0.,15.,30.,45.,60.])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.motion.read_text());scene=json.loads(a.scene.read_text())
    native=load_motion(a.motion);old=JointMotion(Path(d['motion']))
    fields,coordinates,_=load(Path(d['field']));edges=np.asarray(scene['scene_window_edges_s'])
    mid=(edges[:-1]+edges[1:])/2
    at=np.array([scene['mid_s']]);sample=old.track.sample(old.epoch+at,max_gap_s=400.)
    gauge=old.state(at)['body_to_ecef'][0].T@enu_basis(sample['latitude'][0],sample['longitude'][0])
    points=np.asarray(scene['points'])@gauge.T
    original_frame=native.state(mid)['body_to_ecef']
    beta=np.asarray(scene['band_phase_initial_radians'])
    phi=np.asarray(scene['phase_parameters'])[4:];reports=[]
    for offset in a.offsets_s:
        trial_frame=state(native,mid,'reported-course',offset)['body_to_ecef']
        trial_points=np.einsum('wij,wjk,wpk->wpi',trial_frame.transpose(0,2,1),original_frame,points)
        bands=[];maximum_identity_error=0.
        for c,(f,coordinate) in enumerate(zip(fields,coordinates)):
            b=PointBand(f,coordinate,native,[0.,0.,0.],np.asarray(d['path_knots_s']),
                        np.asarray(d['path_coefficients_m']),162.,0.,'reported-course',offset)
            current=native.state(b.t)
            path,gradient=excess_path(current['position_ecef_m'],native.tx[3],native.rx)
            b.base_path=path;b.base_rate=np.sum(gradient*current['velocity_ecef_mps'],axis=1)
            band=SceneBand(b,trial_points,edges,0.);bands.append(band)
            if offset==0:
                reference=PointBand(f,coordinate,old,d['point_body_m'],np.asarray(d['path_knots_s']),
                    np.asarray(d['path_coefficients_m']),162.,0.,d['orientation'])
                original=SceneBand(reference,points,edges,0.)
                h=band.design(np.zeros(b.b.shape[1]),c,False)
                expected=original.design(np.zeros(reference.b.shape[1]),c,False)
                error=float(np.linalg.norm(h-expected)/np.linalg.norm(expected))
                maximum_identity_error=max(maximum_identity_error,error)
                if error>1e-7:raise AssertionError('Changing the virtual reference point changed the physical forward operator')
        likelihood=ScenePath(bands)
        initial=np.r_[np.zeros(likelihood.n),beta,phi[1:]-phi[0]]
        parameters,warm=warm_phases(likelihood,initial,200)
        value,gradient,models=likelihood.objective(parameters,True)
        source_phase=np.r_[0.,parameters[likelihood.n+5:]]
        unused=[b.evaluate(parameters[:likelihood.n],source_phase+parameters[likelihood.n+c],c,
                  training=False,derivatives=False,frozen_amplitude=m['amplitude']) for c,(b,m) in enumerate(zip(bands,models))]
        report=dict(offset_s=offset,training_objective=value,
            unused_explained_fraction=1-sum(m['cost'] for m in unused)/sum(m['energy'] for m in unused),
            band_unused_explained_fraction=[m['explained_fraction'] for m in unused],
            phase_block=warm,maximum_zero_offset_operator_error=maximum_identity_error if offset==0 else None,
            calibration_scope='One constant band phase over all aspect supports; constant source phase and brightness per support')
        reports.append(report);print(json.dumps(report),flush=True)
        (a.output/'analysis.json').write_text(json.dumps(dict(complete=False,scope=__doc__,motion=str(a.motion),scene=str(a.scene),trials=reports),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    offset=[r['offset_s'] for r in reports]
    axes[0].plot(offset,[1-r['training_objective'] for r in reports],'.-',label='Training')
    axes[0].plot(offset,[r['unused_explained_fraction'] for r in reports],'.-',label='Excluded');axes[0].legend()
    for c,name in enumerate(['Seven','SBS','Nine','Ten','ABC']):axes[1].plot(offset,[r['band_unused_explained_fraction'][c] for r in reports],'.-',label=name)
    axes[1].legend()
    for ax in axes:ax.set(xlabel='Reported-course time offset (s)',ylabel='Full-crop complex energy predicted')
    fig.suptitle('Orientation-hypothesis diagnostic; frozen physical centre translation\nSame source midpoint positions, known waveform/epochs, constant phases; no motion solution adopted')
    fig.savefig(a.output/'orientation-offset-test.png',dpi=145);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),scene=str(a.scene),trials=reports),indent=2)+'\n')


if __name__=='__main__':main()

"""Diagnose cancellation in a joint reflector-position/brightness fit.

A decreasing prediction residual is not sufficient evidence of physical
reflector recovery. Nearby opposing components can approximate a derivative
of the point response while their separate powers diverge. This report uses
unchanged original data and the exact known-waveform operator.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from full_ship_field import load,select_range
from full_path_point import PointBand
from full_path_scene import SceneBand
from joint_ship_motion import JointMotion
from rigid_patch import profile_static
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motions',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in a.motions:
        fit=json.loads(path.read_text());motion=JointMotion(Path(fit['motion']));motion.interval=fit['interval_s']
        fields,coordinates,_=load(fit['field']);saved=np.load(path.parent/'trajectory.npz')
        points=np.asarray(fit['scene_points_body_m']);phi=np.asarray(fit['scene_point_phase_radians'])
        beta=np.angle([complex(*z) for z in fit['physical_gains_real_imag']])
        distance=np.linalg.norm(points[:,None]-points[None,:],axis=-1);distance+=np.eye(len(points))*1e30
        pair=np.unravel_index(np.argmin(distance),distance.shape)
        record=dict(motion=str(path),numerical_convergence=fit['numerical_convergence'],minimum_source_separation_m=distance[pair],closest_pair=pair,
            closest_pair_phase_difference_degrees=float(np.degrees(np.angle(np.exp(1j*(phi[pair[0]]-phi[pair[1]]))))),
            unused_full_crop_explained_fraction=fit['unused_explained_fraction'],points_body_m=points,channels=[])
        for c,(old,coordinate) in enumerate(zip(fields,coordinates)):
            field=old.windowed(*fit['interval_s']);index=np.searchsorted(old.time,field.time)
            coord={k:coordinate[k].reshape(-1,16)[index].ravel() for k in ['native_time_s','native_information']}
            coord['training']=coordinate['training'][index]
            if 'stationary_groups' in coordinate:coord['stationary_groups']=coordinate['stationary_groups'][index]
            def band_at(f):
                b=PointBand(f,coord,motion,fit['point_body_m'],fit['path_knots_s'],fit['path_coefficients_m'],fit['anchor_path_time_s'],fit['amplitude_spacing_s'],fit['orientation'])
                return SceneBand(b,points)
            band=band_at(field);b=band.point;h=band.design(np.zeros(b.b.shape[1]),c,False)
            mask=~b.training;amp=saved[f'amplitude_{c}'];parts=h*(amp*np.exp(1j*(phi+beta[c])))[:,None,:]
            parts=profile_static(parts,field.root_information,mask,b.groups)[mask]
            component_energy=np.sum(abs(parts)**2,axis=(0,1));sum_energy=float(np.sum(abs(parts.sum(axis=2))**2))
            small=band_at(select_range(field,-16,-4));hs=small.design(np.zeros(b.b.shape[1]),c,False)
            original=profile_static(h,field.root_information,mask,b.groups)[mask]
            restricted=profile_static(hs,field.root_information,mask,b.groups)[mask]
            retained=np.sum(abs(restricted)**2,axis=(0,1))/np.sum(abs(original)**2,axis=(0,1))
            record['channels'].append(dict(channel=c,component_power_to_summed_power=float(component_energy.sum()/sum_energy),
                total_component_power=float(component_energy.sum()),summed_model_power=sum_energy,maximum_fitted_brightness=float(amp.max()),
                declared_range_gate_offsets=[-16,-4],known_point_template_information_retained=retained,
                range_gate_scope='Known-point norm only, before brightness/position nuisance. Not a joint motion CRLB or a universal information-loss bound.'))
        records.append(record)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for i,record in enumerate(records):
        label='Fixed support' if i==0 else 'Unrestricted position/brightness trial'
        points=np.asarray(record['points_body_m']);axes[0].plot(points[:,0],points[:,1],'.',label=label,ms=7)
        axes[1].plot(range(5),[10*np.log10(c['component_power_to_summed_power']) for c in record['channels']],'.-',label=label)
        axes[2].plot(range(5),[c['maximum_fitted_brightness'] for c in record['channels']],'.-',label=label)
    axes[0].set(xlabel='Body x (m)',ylabel='Body y (m)',title='Same source associations; common to all bands',aspect='equal');axes[0].legend(fontsize=8)
    axes[1].set(xticks=range(5),xticklabels=['Seven','SBS','Nine','Ten','ABC'],ylabel='Sum of component powers / summed power (dB)',title='Destructive cancellation grows without better focus');axes[1].legend(fontsize=8)
    axes[2].set(xticks=range(5),xticklabels=['Seven','SBS','Nine','Ten','ABC'],yscale='log',ylabel='Maximum brightness (same original units)',title='Diverging coefficients are not recovered reflectors')
    fig.suptitle('Original 153.27–183.27 s waveform field; excluded epochs; no image normalization\nPoint refinement exposes an unstable inverse decomposition. It is rejected as a physical improvement.')
    fig.savefig(a.output/'source-cancellation.png',dpi=150);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,records=records,
        accepted_position_refinement=False)),indent=2)+'\n')
    rows=[]
    for record in records:
        rows.append(f'<tr><td>{Path(record["motion"]).parent.name}</td><td>{record["minimum_source_separation_m"]:.6g} m</td><td>{100*record["unused_full_crop_explained_fraction"]:.3f}%</td><td>{record["numerical_convergence"]}</td></tr>')
    (a.output/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Joint-source instability</title><style>body{font:17px/1.5 system-ui;max-width:1200px;margin:40px auto}img{width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>Why the free-source fit is rejected</h1>
<p>The stable 18-source fit retains fixed training-selected support. Allowing each source to move makes pairs nearly coincident while opposing brightness coefficients grow by orders of magnitude. Excluded prediction improves only slightly and the original-data native image remains broadly similar. This is an unstable inverse decomposition, not evidence of better recovered vessel motion.</p>
<table><tr><th>Fit</th><th>Minimum source distance</th><th>Excluded full-crop energy predicted</th><th>Numerically converged</th></tr>'''+''.join(rows)+'''</table><img src="source-cancellation.png">
<p>The limiting mechanism is explicit: <code>[h(x+δ) − h(x−δ)]/(2δ) → ∂h/∂x</code>. The predicted field can remain finite while each component amplitude grows as <code>1/δ</code>. Nonnegative magnitudes do not prevent this when the two fixed complex phases oppose. A low residual or an excellent fitted-mixture phase score cannot certify physical scatterers.</p>
<p>The new unknown-position trials are retained but rejected. Their checkpoints and interruption reasons are preserved. The fixed-support likelihood is retained for motion work; no raw profile, production processing or accepted geometry is changed. The range-gate information figures in the JSON describe conditional single-point template norms, not the full joint estimation problem.</p>''')


if __name__=='__main__':main()

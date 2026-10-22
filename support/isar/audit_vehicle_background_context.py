"""Verify a previous vehicle fit against surrounding original profile cells.

Freeze its physical point, complex amplitude and stationary coefficients.
Check original observation identity and the same prediction in the fitted
crop, then evaluate the unchanged coefficients in a wider recorded context.
No new calibration or interpretation of a road association is made here.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coherent_arc import CoherentArc
from extract_full_aperture import load as load_aperture
from field import Field
from fit_road_arc_raw import load_problem
from recorded_field_context import expand
from report_background_extrapolation import original_values
from report_measured_scatterer_phases import serial
from road_arc_raw import RawRoadGeometry


def main():
    root=Path('docs/data/isar_20261017');source=root/'road-sites-32/fits.json'
    old=json.loads(source.read_text());code=[0,2,0,0,2]
    trial=next(t for t in old['trials'] if t['assignment']==code)
    summary=root/'road-arc-geometry/fits.json'
    model,associations,sources=load_problem(summary,code,['A'],fixed_calibration=trial)
    old_association=next(a for a in trial['associations'] if a['label']=='A')
    if associations[0]['route']!=old_association['route']:raise ValueError('Road identity changed')
    index=old['labels'].index('A');p=np.asarray(trial['road_states'][index])/model.geometry.scale
    out=root/'vehicle-background-context-A';out.mkdir(exist_ok=True)
    if (out/'analysis.json').exists():raise ValueError('Keep previous audits')
    fig,axes=plt.subplots(5,1,figsize=(13,13),layout='constrained');rows=[]
    for c,(b,ax) in enumerate(zip(model.likelihoods[0],axes)):
        f=b.field;aperture=load_aperture(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'))
        offsets=np.arange(f.offsets[0]-2,f.offsets[-1]+11)
        wide=expand(f,aperture,offsets)
        # Road patches used an additional known constant carrier phase. It is
        # free in the likelihood, but must be restored for an identity check.
        offset=f.metadata['phase_reference_offset_m']
        values=wide.data['value']*np.exp(-1j*f.k*offset)
        wide=Field({**wide.data,'value':values},f.pulse.samples)
        inside=np.isin(offsets,f.offsets);data_error=float(np.max(abs(values[:,inside]-f.data['value'])))
        if data_error>1e-11:raise ValueError('Archived vehicle observations are not reproduced')
        expanded=CoherentArc(wide,b.training,stationary_paths=b.delays)
        # Reuse the exact road/antenna template with only this band's field replaced.
        likelihoods=[list(model.likelihoods[0])];likelihoods[0][c]=expanded
        full=RawRoadGeometry(model.geometry,likelihoods)
        h,_,_=model.band_template(p,0,c);hp=b.nuisance.remove(h[b.selected])
        alpha=np.vdot(hp,b.clean)/np.vdot(hp,hp).real
        beta=b.nuisance.solve(f.y.ravel()[b.selected]-alpha*h[b.selected])
        old_prediction=original_values(f,(alpha*h+b.design@beta).reshape(f.y.shape))
        hw,_,_=full.band_template(p,0,c)
        prediction=original_values(wide,(alpha*hw+expanded.design@beta).reshape(wide.y.shape))
        error=float(np.max(abs(prediction[:,inside]-old_prediction)))
        operation=float(abs(alpha)*max(abs(f.pulse.samples))+sum(abs(beta))*max(abs(f.pulse.samples)))
        bound=256*np.finfo(float).eps*operation*np.linalg.cond(wide.whiten)*np.linalg.cond(f.whiten)
        if error>max(1e-11,bound):raise ValueError('Frozen prediction changed inside the original crop')
        def power(x):return np.average(abs(x)**2,axis=0,weights=wide.data['information'])
        observed,predicted=power(values),power(prediction)
        outside=~inside
        ratio=float(10*np.log10(predicted[outside].sum()/observed[outside].sum()))
        singular=b.nuisance.singular_values
        row=dict(channel=c,original_offsets=f.offsets,expanded_offsets=offsets,
            original_observations_maximum_error=data_error,inside_frozen_prediction_error=error,
            inside_roundoff_bound=bound,stationary_columns=len(b.delays),stationary_rank=b.nuisance.rank,
            stationary_condition=float(singular[0]/singular[b.nuisance.rank-1]),
            maximum_coefficient_magnitude=float(max(abs(beta))),outside_prediction_to_observed_power_db=ratio,
            observed_mean_power=observed,predicted_mean_power=predicted)
        rows.append(row)
        ax.semilogy(offsets,observed,'o-',label='Original captured profiles')
        ax.semilogy(offsets,predicted,'o-',label='Frozen point + stationary fit')
        ax.axvspan(f.offsets[0]-.5,f.offsets[-1]+.5,color='#ddd',alpha=.4)
        ax.set(title=f'Channel {c}: outside predicted/observed {ratio:+.1f} dB',ylabel='Mean original profile power')
        print(c,'data error',data_error,'condition',row['stationary_condition'],'outside dB',ratio,flush=True)
    axes[0].legend();axes[-1].set_xlabel('Range offset from the empirical reference (samples)')
    fig.suptitle('Vehicle A: old fitted coefficients tested on surrounding recorded range cells\nShading: original fitted crop. No coefficient or physical parameter is refitted.',fontsize=12)
    fig.savefig(out/'context-prediction.png',dpi=150);plt.close(fig)
    result=dict(scope=__doc__,complete=True,accepted_calibration=False,production_changed=False,
        source=str(source),assignment=code,association=associations[0],channels=rows,
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [source,summary,Path(__file__),Path('support/isar/recorded_field_context.py'),Path('support/isar/road_arc_raw.py')]})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))


if __name__=='__main__':main()

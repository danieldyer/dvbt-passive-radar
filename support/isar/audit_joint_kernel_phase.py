"""Test known-pulse changes in a real joint four-return subtraction.

The isolated self-pulse phase comparison does not bound cross-talk from bright
neighbours. This control keeps all nonlinear paths fixed and re-estimates only
the same brightness/static nuisance under mean versus known original-weight
waveform kernels. Both models use identical observations, metric and splits.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from component_phase import component_measurements
from fit_fixed_grid_phases import prepare_fixed
from fit_shared_echo_cluster import subset_prediction
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import ReferencedAmplitudeArc,SharedEchoCluster
from waveform_kernels import bind_subset,read_kernels


def known_band(band,grid,records,samples):
    field=bind_subset(band.field,grid['time'],grid['information'],samples,
        np.array([r['valid'] for r in records]))
    return ReferencedAmplitudeArc(field,band.training,band.knots,
        band.epoch-band.scale,band.epoch+band.scale,band.known_path,band.known_rate,
        stationary_design=band.design,curve=band.base.curve)


def main():
    root=Path('docs/data/isar_20261017');out=root/'joint-kernel-phase';out.mkdir(exist_ok=True)
    source=root/'fixed-grid-multiple-phases/fits.json';saved=json.loads(source.read_text())
    geometry=Path('docs/data/calibration_20261016_sites')
    fields=load_fields(Path('build/isar-20261015/full-fields'),geometry)
    bands,parts,_=prepare_fixed(fields,geometry,*saved['interval_s'],stationary=True)
    kernel=Path('build/isar-20261016/waveform-kernels-30s');_,records,samples=read_kernels(kernel/'response.kernels')
    grids=np.load(kernel/'rows.npz');fig,axes=plt.subplots(5,2,figsize=(14,13),layout='constrained')
    result=dict(scope=__doc__,complete=True,production_changed=False,accepted_calibration=False,channels=[])
    for c,(band,entry) in enumerate(zip(bands,saved['independent'])):
        known=known_band(band,grids[f'channel_{c}'],records[c],samples[c])
        original=SharedEchoCluster([band],saved['selected_origins_samples'])
        corrected=SharedEchoCluster([known],saved['selected_origins_samples'])
        p=np.asarray(entry['trials'][entry['best_index']]['parameters'])
        r,_=original.evaluate(p);cost=r@r*original.energy
        if abs(cost-entry['trials'][entry['best_index']]['cost'])>1e-7:raise ValueError('Original objective changed')
        a=component_measurements(original.clusters[0],p[original.indices[0]])
        b=component_measurements(corrected.clusters[0],p[corrected.indices[0]])
        scores={name:subset_prediction(model,p,[parts[c]]) for name,model in [('mean',original),('known',corrected)]}
        row=dict(name=entry['name'],prediction=scores,components=[])
        for s in (1,3):
            x,y=a[s],b[s];use=parts[c]==3
            phase=np.angle(y['complex_coefficient']*x['complex_coefficient'].conj())
            weight=np.sum(abs(x['frozen_component_prediction'])**2,axis=1)
            good=use&(weight>0)
            rms=np.degrees(np.sqrt(np.average(phase[good]**2,weights=weight[good])))
            row['components'].append(dict(label=saved['labels'][s],time_s=x['time_s'],
                phase_change_deg=np.degrees(phase),testing=use,weighted_unused_phase_change_rms_deg=rms))
            axes[c,0 if s==1 else 1].plot(x['time_s'],np.degrees(phase),'.',ms=2)
        result['channels'].append(row)
        for ax,label in zip(axes[c],('B','D')):
            ax.set(title=entry['name']+' · '+label,xlabel='Capture time (s)',ylabel='Known minus mean phase (°)',xlim=saved['interval_s'])
        print(entry['name'],'unused mean/known',[scores[k]['testing'][0]['heldout_error'] for k in ('mean','known')],
            'phase changes',[v['weighted_unused_phase_change_rms_deg'] for v in row['components']],flush=True)
    fig.suptitle('Known range-pulse variation: actual neighbouring-return subtraction\n'
        'Same fixed nonlinear paths, data, covariance and stationary design. Brightness/static coefficients re-estimated on training only.',fontsize=12)
    fig.savefig(out/'phase-change.png',dpi=150);plt.close(fig)
    inputs=[source,kernel/'response.kernels',kernel/'rows.npz',Path(__file__),
        Path('support/isar/waveform_kernels.py'),Path('support/isar/component_phase.py')]
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))


if __name__=='__main__':main()

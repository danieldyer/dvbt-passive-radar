"""Review joint range-resolved echoes and their cross-band phase trajectories."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compare_ais_phase import NAMES,COLOURS,path_geometry
from fit_shared_echo_cluster import prepare
from report_background_extrapolation import original_values
from report_measured_scatterer_phases import load_fields,serial
from shared_echo_cluster import SharedEchoCluster
from motion import C


def parts(cluster,parameters):
    result=cluster.evaluate(parameters);k=len(cluster.base.knots)
    return np.stack([result['design'][:,i*k:(i+1)*k]@result['amplitude'][i*k:(i+1)*k] for i in range(cluster.components)])


def component_phase(cluster,parameters,testing):
    b=cluster.base;p=np.asarray(parameters).reshape(cluster.components,4);signals=parts(cluster,parameters)
    total=signals.sum(axis=0);out=[]
    for s,q in enumerate(p):
        raw,_=b.response(q);unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
        residual=b.field.y-(total-signals[s]).reshape(b.field.y.shape)
        product=np.sum(unit.conj()*residual,axis=1);phase=np.angle(product)
        predicted=signals[s].reshape(b.field.y.shape);weight=np.sum(abs(predicted)**2,axis=1)
        valid=testing&(weight>0)
        rms=float(np.degrees(np.sqrt(np.average(phase[valid]**2,weights=weight[valid])))) if np.any(valid) else None
        out.append(dict(time_s=b.field.time,phase_residual_deg=np.degrees(phase),test_phase_rms_deg=rms,
            testing=testing,predicted_information=weight))
    return out


def conditional_observables(model,parameters):
    """Keep the complete cross-component conditional covariance; never pinv."""
    b=model.bands[0];cluster=model.clusters[0];index=model.indices[0];value=cluster.evaluate(parameters[index])
    information=np.zeros((model.count,model.count));information[np.ix_(index,index)]=value['information']
    scales=np.sqrt(np.diag(information));supported=scales>0
    if not supported.all():return dict(identifiable=False,reason='Zero conditional information column')
    normalized=information/scales[:,None]/scales[None,:];eigen=np.linalg.eigvalsh(normalized)
    if eigen[0]<=eigen[-1]*1e-10:return dict(identifiable=False,reason='Rank-deficient conditional information',eigenvalues=eigen)
    lower=np.linalg.cholesky(normalized);inverse=np.linalg.solve(lower.T,np.linalg.solve(lower,np.eye(model.count)))
    covariance=inverse/scales[:,None]/scales[None,:]
    n=model.components;transform=np.zeros((3*n,model.count))
    for s in range(n):
        transform[3*s,2*s]=C/8e6;transform[3*s+1,2*n+2*s]=1/b.scale;transform[3*s+2,2*n+2*s+1]=2/b.scale**2
    local=parameters[index].reshape(n,4)
    # These are increments from the saved/nominal path, not geographic fixes.
    measured=np.column_stack([local[:,0]*C/8e6,local[:,2]/b.scale,2*local[:,3]/b.scale**2])
    return dict(identifiable=True,range_origin_m_rate_residual_mps_curvature_residual_mps2=measured,
        covariance=transform@covariance@transform.T,normalized_information_condition=float(eigen[-1]/eigen[0]),
        scope='Local proper-complex thermal Fisher information after all real brightness and constant phases, retaining covariance between components. No residual-noise rescaling, priors or active-bound information. Model mismatch and source association are not included; this is not a calibrated physical confidence interval.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/shared-echo-cluster/fits.json'))
    args=parser.parse_args();out=args.source.parent;data=json.loads(args.source.read_text())
    if not data['complete']:raise ValueError('Wait for completed fits')
    fields=load_fields(Path('build/isar-20261015/full-fields'),Path('docs/data/calibration_20261016_sites'))
    first,last=data['interval_s'];extent=data.get('range_extent_samples',8);origins=data['selected_origins_samples'];labels=data['labels'];n=len(labels)
    bands,partition=prepare(fields,Path('docs/data/calibration_20261016_sites'),first,last,range_extent=extent)
    radius=data['origin_radius_samples'];independent=[];reproduction=[]
    for ch,record in enumerate(data['independent']):
        model=SharedEchoCluster([bands[ch]],origins,origin_radius=radius)
        for fit in record['trials']:
            r,_=model.evaluate(np.asarray(fit['parameters']));reproduction.append(abs(r@r*model.energy-fit['cost'])/max(1.,fit['cost']))
        best=record['trials'][record['best_index']];p=np.array(best['parameters'])
        independent.append(dict(model=model,parameters=p,fit=best,
            phases=component_phase(model.clusters[0],p[model.indices[0]],partition[ch]==3),
            conditional_observables=conditional_observables(model,p)))
    common=SharedEchoCluster(bands,origins,origin_radius=radius)
    for fit in data['common']:
        r,_=common.evaluate(np.asarray(fit['parameters']));reproduction.append(abs(r@r*common.energy-fit['cost'])/max(1.,fit['cost']))
    best=data['common'][data['best_common_index']];p=np.asarray(best['parameters'])
    if max(reproduction)>1e-9:raise ValueError('Saved likelihood reproduction failed')
    common_phases=[component_phase(c,p[index],part==3) for c,index,part in zip(common.clusters,common.indices,partition)]
    # Frozen complex prediction in the original measured range coordinates.
    originals=[parts(c,p[i]) for c,i in zip(common.clusters,common.indices)]
    raw_predictions=[original_values(b.field,v.sum(axis=0).reshape(b.field.y.shape)) for b,v in zip(bands,originals)]
    peak=max(np.max(abs(b.field.data['value'])) for b in bands)
    fig,axes=plt.subplots(5,3,figsize=(16,14),layout='constrained',sharex=True,sharey=True)
    for ch,(b,pred) in enumerate(zip(bands,raw_predictions)):
        observed=b.field.data['value'];offsets=b.field.offsets*C/8e6
        for col,value in enumerate((observed,pred,observed-pred)):
            image=axes[ch,col].imshow(20*np.log10(np.maximum(abs(value.T)/peak,1e-6)),origin='lower',aspect='auto',
                extent=[first,last,offsets[0]-C/16e6,offsets[-1]+C/16e6],cmap='magma',vmin=-50,vmax=0)
            for s,label in enumerate(labels):
                origin=p[common.indices[ch]].reshape(n,4)[s,0]*C/8e6
                axes[ch,col].axhline(origin,color='#00e5ff',lw=.6,ls=':')
                if col==0:axes[ch,col].text(first+.02,origin+7,label,color='#00e5ff',fontsize=8)
        axes[ch,0].set_ylabel(NAMES[ch]+'\nOffset from saved AIS range (m)')
    for ax,title in zip(axes[0],('Original measured field','Frozen shared-trajectory prediction','Residual')):ax.set_title(title)
    for ax in axes[-1]:ax.set_xlabel('Capture time (s)')
    fig.colorbar(image,ax=axes,label='Magnitude relative to one measured maximum across all panels (dB)',shrink=.65)
    fig.suptitle(f'{n} selected returns fitted jointly across {2*extent+1} measured range cells\n'
        'Same colour scale throughout. No new stationary projection, multipath model or local normalization.',fontsize=13)
    fig.savefig(out/'01-range-prediction.png',dpi=165);plt.close(fig)
    fig,axes=plt.subplots(n,3,figsize=(17,3.5*n),layout='constrained',squeeze=False)
    grid=np.linspace(first,last,160);u=(grid-(first+last)/2)/bands[0].scale;shared=p[-2*n:].reshape(n,2)
    for s,label in enumerate(labels):
        for ch,(entry,colour) in enumerate(zip(independent,COLOURS)):
            local=entry['parameters'][entry['model'].indices[0]].reshape(n,4);curve=local[s,2]*u+local[s,3]*u*u
            axes[s,0].plot(grid,100*curve,color=colour,lw=1.3,label=NAMES[ch])
            measured=common_phases[ch][s];t=np.asarray(measured['time_s']);tu=(t-(first+last)/2)/bands[ch].scale
            path=shared[s,0]*tu+shared[s,1]*tu*tu-np.radians(measured['phase_residual_deg'])/bands[ch].field.k
            use=np.asarray(measured['testing']);axes[s,1].plot(t[use],100*path[use],'.',ms=2.5,color=colour,alpha=.65)
        shared_curve=shared[s,0]*u+shared[s,1]*u*u
        for ax in axes[s,:2]:ax.plot(grid,100*shared_curve,'k--',lw=1.5,label='Shared fit');ax.set_xlabel('Capture time (s)');ax.set_ylabel('Residual path (cm)')
        axes[s,0].set_title(label+' · independent fits versus shared trajectory')
        axes[s,1].set_title(label+' · unused measurements after subtracting other fitted returns')
        a=[e['phases'][s]['test_phase_rms_deg'] for e in independent];b=[r[s]['test_phase_rms_deg'] for r in common_phases]
        x=np.arange(5);axes[s,2].bar(x-.17,a,.34,label='Independent');axes[s,2].bar(x+.17,b,.34,label='Shared')
        axes[s,2].set_xticks(x,NAMES,rotation=25);axes[s,2].set_ylabel('Conditional unused phase RMS (degrees)');axes[s,2].set_title(label+' · phase prediction')
    axes[0,0].legend(ncol=3,fontsize=8);axes[0,2].legend(fontsize=8)
    fig.suptitle('Jointly resolved components: independent phase changes and shared-motion control\n'
        'Other returns use frozen training amplitudes. Measured dots are circular innovations around the shared branch, not a global phase unwrap.',fontsize=13)
    fig.savefig(out/'02-component-trajectories.png',dpi=170);plt.close(fig)
    ind_error=sum(e['fit']['prediction']['testing'][0]['heldout_error'] for e in independent)
    common_error=sum(r['heldout_error'] for r in best['prediction']['testing']);ratio=common_error/ind_error
    origins_matrix=p[:-2*n].reshape(5,n,2)[:,:,0]*C/8e6
    split_diagnostic=None;split_note=''
    if data.get('split_delayed_source'):
        a,b=[labels.index(name) for name in ('D','E')]
        split_diagnostic=dict(abc_component_range_separation_m=float(abs(origins_matrix[4,a]-origins_matrix[4,b])),
            abc_component_to_sum_energy_ratio=best['component_to_sum_energy_ratio'][4],
            abc_brightness_condition=best['active_brightness_condition'][4],
            accepted_as_physical_association=False)
        split_note=f'''<p class="note"><b>The ABC split is not accepted as an identified physical association.</b> In the shared fit, D and E at ABC are separated by only {split_diagnostic['abc_component_range_separation_m']:.4f} m, while the component/sum energy ratio is {split_diagnostic['abc_component_to_sum_energy_ratio']:.2f} and brightness condition is {split_diagnostic['abc_brightness_condition']:.1f}. This is an unresolved opposing-component explanation. The improved prediction alone does not identify two physical points or provide reliable ABC phase-centre information.</p>'''
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    for s,label in enumerate(labels):ax.plot(np.arange(5),origins_matrix[:,s],'.-',label=label)
    ax.set_xticks(range(5),NAMES);ax.set(ylabel='Measured envelope origin from saved AIS reference (m)',title='Envelope origins remain independently measured in each band')
    ax.legend();fig.savefig(out/'03-envelope-origins.png',dpi=170);plt.close(fig)
    rows=''.join(f'<tr><td>{NAMES[ch]}</td><td>{e["fit"]["prediction"]["testing"][0]["heldout_error"]:.2f}</td><td>{best["prediction"]["testing"][ch]["heldout_error"]:.2f}</td><td>{best["active_brightness_condition"][ch]:.2f}</td><td>{best["component_to_sum_energy_ratio"][ch]:.3f}</td></tr>' for ch,e in enumerate(independent))
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Joint selected-echo phase trajectories</title>
<style>body{{font:16px/1.5 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ddd}}.note{{background:#eef4f8;padding:15px}}</style>
<h1>Jointly resolve selected returns before comparing their phases</h1>
<p class="note">At {first:g}–{last:g} seconds, {n} selected returns are fitted simultaneously in {2*extent+1} original measured range cells and all five channels. A shared set of phase trajectories changes final unused-quarter squared error by {100*(ratio-1):+.3f}% relative to independent band trajectories. These are conditional prediction results for the declared component model, not an antenna-location certificate.</p>
{split_note}
<p>Each component has an independent constant phase and envelope origin in every band. Quadratic paths drive both delay migration and RF phase. Positive real brightness at 0.25 s knots is fitted using training blocks only. Original C0/C1 symbol response, sixteen-row averaging and range covariance enter before fitting. No new stationary nuisance or multipath model is present. Original frozen capture-time background subtraction is already part of the data.</p>
<p>Components start at the selected magnitude-map features {dict(zip(labels,origins))} samples relative to the saved AIS reference, with envelope searches ±{radius:g} samples around those associations. Expanded data, when used, reproduce all seventeen existing cells bit for bit. Constant band phases absorb unknown scattering phase; no fitted time-varying phase screen is allowed.</p>
<h2>Independent and shared phase trajectories</h2><a href="02-component-trajectories.png"><img src="02-component-trajectories.png" alt="Resolved component phase trajectories and unused measured innovations"></a>
<p>The centre column subtracts the other components using coefficients frozen from training, then measures the remaining component’s phase with its range template. The dots retain a wavelength ambiguity and are conditional on the other fitted returns. Model subtraction uncertainty is not included in the phase bars. The <a href="../measured-scatterer-phases/index.html">unfitted cell histories</a> remain the direct measurement control.</p>
<h2>Prediction of the full measured field</h2><a href="01-range-prediction.png"><img src="01-range-prediction.png" alt="Original complex-range magnitudes, frozen prediction and residual"></a>
<table><tr><th>Band</th><th>Independent unused error</th><th>Shared unused error</th><th>Brightness condition</th><th>Component/sum energy ratio</th></tr>{rows}</table>
<p>The shared fit has range-bound indices {best['bound_indices']} and maximum projected gradient {best['maximum_projected_gradient']:.3g}. Bounds and unresolved residual structure are retained as limitations. Brightness conditioning and component/sum energy ratios diagnose cancelling components; these fits do not use the unstable short-crop stationary model found previously.</p>
<h2>Range measurements for subsequent geometry checks</h2><a href="03-envelope-origins.png"><img src="03-envelope-origins.png" alt="Independently measured envelope origins across bands"></a>
<p>These range origins and the independent phase coefficients are retained with their complete cross-component conditional Fisher covariance, where invertible. The covariance is thermal and conditional on the chosen point model; it is not a physical confidence certificate. Bounds and coherent residuals must be addressed before accepting an antenna calibration. No positions have been adopted.</p>
<p>All {len(reproduction)} saved objectives reproduce with maximum relative error {max(reproduction):.3g}. <a href="tests.log">Numerical tests</a> include known multi-return recovery, unused prediction, exact profiled derivatives and original-row phase/envelope/symbol-rate bookkeeping.</p>
<p><a href="fits.json">All starts and fitted parameters</a> · <a href="analysis.json">Measured components, covariance and reproduction results</a> · <a href="../piecewise-measured-phases/index.html">38 local phase tests</a> · <a href="../index.html">Analysis index</a></p></html>''')
    summary=dict(scope=__doc__,complete=True,accepted_calibration=False,interval_s=[first,last],labels=labels,
        independent_error=ind_error,shared_error=common_error,shared_to_independent_error_ratio=ratio,
        independent_conditional_observables=[e['conditional_observables'] for e in independent],
        independent_component_phases=[e['phases'] for e in independent],shared_component_phases=common_phases,
        shared_envelope_origin_m=origins_matrix,reproduced_objectives=len(reproduction),maximum_objective_error=max(reproduction),
        split_association_diagnostic=split_diagnostic,
        source_sha256={str(args.source):hashlib.sha256(args.source.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out/'analysis.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False))
    print(out/'index.html','unused error ratio',ratio,flush=True)


if __name__=='__main__':main()

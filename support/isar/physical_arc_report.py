"""Plot physical site fits separately from independent-band phase continuity."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv
from cluster_phase import crop
from compare_ais_phase import NAMES
from flash_geometry import Flash,assignments
from flash_siting_report import EPOCH
from imaging import sites
from physical_amplitude_arc import PhysicalAmplitudeArc
from ship_site_signature import licensed_options


def information(record,source,interval):
    """Mean-response Fisher information; all fitted nuisance is eliminated."""
    directory=Path('docs/data/calibration_20261016_sites')
    _,receiver,_=sites(directory)
    rotation=np.array(json.loads((directory/'geometry-prior.json').read_text())['ecef_to_enu'])
    code=lambda a:''.join({'12013':'0','12015':'1','12014':'2'}[v['site_id']] for v in a)
    assignment=next(a for a in assignments(licensed_options(directory)) if code(a)==record['assignment'])
    track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    midpoint=(interval['first_s']+interval['last_s'])/2
    geometry=Flash(np.zeros((5,3)),np.tile(np.eye(3),(5,1,1)),track,EPOCH+midpoint,rotation,receiver,assignment,
        receiver_sigma=(0.,0.,7.5),receiver_up_bounds_m=(-10.,10.))
    rows=[]
    for ch,saved in enumerate(interval['channels']):
        fit=saved['alternatives'][saved['chosen_index']]
        with np.load(f'build/isar-20261015/full-fields/channel-{ch}.npz') as data:
            field=crop(data,interval['first_s'],interval['last_s'],np.arange(-2,3))
        model=PhysicalAmplitudeArc(field,np.array(fit['test']['training']),fit['test']['amplitude_knots_s'],geometry,ch,midpoint)
        _,_,_,amplitude,design,derivative=model.evaluate(np.array(record['parameters']),True)
        a=np.r_[design.real,design.imag][:,amplitude>0]
        j=np.r_[derivative.real,derivative.imag]
        j-=a@np.linalg.lstsq(a,j,rcond=1e-10)[0]
        rows.append(j)
    jacobian=np.concatenate(rows)*np.sqrt(2)
    antennas=jacobian[:,np.r_[np.arange(15),17]]
    nuisance=jacobian[:,18:]
    u,s,_=np.linalg.svd(nuisance,full_matrices=False)
    q=u[:,s>s[0]*1e-9]
    projected=antennas-q@(q.T@antennas)
    singular=np.linalg.svd(projected,compute_uv=False)
    return {'data_only_antenna_information_eigenvalues':(singular**2).tolist(),
        'nuisance_rank':q.shape[1], 'antenna_coordinates':16,
        'scope':'Conditional thermal mean-response information after stationary, active positive amplitudes, band phases and physical target motion. No location priors included; receiver horizontal coordinates fixed.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixed',type=Path,default=Path('docs/data/isar_20261017/ship-physical-arc-sites-32/fits.json'))
    p.add_argument('--refined',type=Path,default=Path('docs/data/isar_20261017/ship-physical-arc-refined/fits.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    fixed,refined=[json.loads(path.read_text()) for path in (args.fixed,args.refined)]
    if not fixed['complete'] or not refined['complete'] or len(fixed['trials'])!=32:
        raise ValueError('Wait for the complete declared site comparison')
    source=json.loads(Path(fixed['independent_arc_source']).read_text())
    interval=next(v for v in source['intervals'] if [v['first_s'],v['last_s']]==fixed['interval_s'])
    ordered=sorted(fixed['trials'],key=lambda r:r['cost']);best=ordered[0]
    refined_best=min(refined['trials'],key=lambda r:r['cost'])
    info=information(refined_best,source,interval)
    fig,axes=plt.subplots(2,1,figsize=(13,8),sharex=True,layout='constrained')
    for ax,key,title in zip(axes,(lambda r:r['cost'],lambda r:sum(v['test_error'] for v in r['prediction'])),
            ('Training: shared physical point, fixed nominal antennas','Unused quarter: complex prediction error')):
        values=np.array([key(r) for r in ordered]);values-=values.min()
        ax.bar(np.arange(32),values,color=['#0072b2' if r['assignment'][-1]=='2' else '#d55e00' for r in ordered])
        for i,r in enumerate(ordered):
            if r['bound_indices']:ax.scatter(i,values[i]+.02*values.max(),marker='x',s=25,color='black')
        ax.set_ylabel('Difference from minimum');ax.set_title(title)
    axes[1].set_xticks(np.arange(32),[r['assignment'] for r in ordered],rotation=65)
    axes[1].set_xlabel('Seven / SBS / Nine / Ten / ABC · 0 Ornata, 1 Eyre, 2 Ridge')
    fig.suptitle('163–166 s physical-point comparison · blue ABC Ridge, orange ABC Ornata\n× denotes a bound contact; model mismatch and local alternatives limit interpretation')
    fig.savefig(args.output/'sites.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    observed=np.array(fixed['independent_observables'])
    for col,label in enumerate(('Excess path (m)','Excess-path rate (m/s)','Excess-path acceleration (m/s²)')):
        reference=observed[:,col].mean() if col==0 else 0.
        axes[col].plot(NAMES,observed[:,col]-reference,'o-',color='black',label='Independent band quadratic')
        for row,name in [(best,'Best fixed assignment '+best['assignment']),
                         (refined_best,'Refined antennas '+refined_best['assignment'])]:
            axes[col].plot(NAMES,np.array(row['predicted_observables'])[:,col]-reference,'o--',label=name)
        axes[col].set_ylabel(label+(f' minus {reference:.2f} m' if col==0 else ''))
    axes[0].legend(fontsize=8)
    fig.suptitle('Independent phase summaries versus a single physical trajectory\nFits use original complex observations; summaries are initialization and diagnostic only')
    fig.savefig(args.output/'observables.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    independent=[100*r['alternatives'][r['chosen_index']]['test']['test_complex_prediction_fraction'] for r in interval['channels']]
    axes[0].plot(NAMES,independent,'o-',color='black',label='Independent band quadratic')
    for row,name in [(best,'Best fixed antennas'),(refined_best,'Refined antennas')]:
        axes[0].plot(NAMES,[100*v['test_complex_prediction_fraction'] for v in row['prediction']],'o--',label=name)
    axes[0].set_ylabel('Unused-quarter energy explained (%)');axes[0].legend(fontsize=8)
    eigen=np.array(info['data_only_antenna_information_eigenvalues'])
    axes[1].semilogy(np.arange(1,17),np.maximum(eigen,1e-14),'.-')
    axes[1].axhline(1,color='black',ls=':',label='Unit prior-coordinate information')
    axes[1].set(xlabel='Physical antenna mode, strongest first',ylabel='Data-only information',title='Motion, strengths, static clutter and phases profiled')
    axes[1].legend(fontsize=8)
    fig.savefig(args.output/'prediction-information.png',dpi=160);plt.close(fig)
    offsets=np.array(refined_best['antenna_fit']['transmitter_offsets_enu_m'])
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for j,label in enumerate(('East','North','Up')):axes[0].plot(NAMES,offsets[:,j],'.-',label=label)
    axes[0].axhline(0,color='black',lw=.5);axes[0].set_ylabel('Fitted transmitter displacement (m)');axes[0].legend()
    scales=np.array([150.,150.,20.,.5,.5,.1,.1,.1,.05,1.])
    for row,label in [(best,'Fixed antennas'),(refined_best,'Refined antennas')]:
        q=np.array(row['reflector_nuisance_standardised'])*scales
        axes[1].plot(q[0],q[1],'o',label=f'{label}: E {q[0]:.0f}, N {q[1]:.0f} m')
    axes[1].plot(0,0,'*',color='black',ms=12,label='AIS antenna prior')
    axes[1].set(xlabel='Reflector offset east (m)',ylabel='Reflector offset north (m)',title='Offset from AIS plus fitted time shift')
    axes[1].set_aspect('equal');axes[1].legend(fontsize=8)
    fig.suptitle('Unadopted nuisance solutions · parameter bounds are retained, not relaxed to obtain agreement')
    fig.savefig(args.output/'offsets.png',dpi=160);plt.close(fig)
    summary={'accepted_calibration':False,'new_focused_videos':False,'fixed_trials':32,
        'best_fixed_assignment':best['assignment'],'best_refined_assignment':refined_best['assignment'],
        'best_fixed_cost':best['cost'],'best_refined_cost':refined_best['cost'],
        'best_fixed_bound_indices':best['bound_indices'],'best_refined_bound_indices':refined_best['bound_indices'],
        'information':info,'sources':[str(args.fixed),str(args.refined),fixed['independent_arc_source']]}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{r["assignment"]}</td><td>{r["cost"]:.2f}</td><td>{r["maximum_projected_gradient"]:.2g}</td><td>{r["bound_indices"]}</td></tr>' for r in ordered)
    (args.output/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Physical ship arc and site comparison</title>
<style>body{{font:16px/1.55 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}}img{{max-width:100%}}td,th{{padding:5px 12px;text-align:left}}.note{{background:#fff1d3;padding:16px}}</style>
<h1>Physical ship arc: motion continuity and transmitter hypotheses</h1>
<p class="note">No new phase centres are accepted. Better independent-band phase prediction has not yet produced a physically credible, well-conditioned common-point calibration. The best fixed-site trial is {best['assignment']}; the two refined-site controls also retain parameter-bound contacts. This is progress in model testing, not a completed geographic calibration or a new focused video.</p>
<p>The 163–166 s interval uses one common Cartesian position, velocity and acceleration; the AIS interpolation supplies their declared priors and a bounded timing offset. Receiver horizontal coordinates are fixed at the user’s supplied position. The refined trials allow receiver height and transmitter coordinates to vary under the existing correlated site/antenna priors. There is no yaw, roll or pitch fit. Five constant band phases and nonnegative brightness envelopes are nuisance variables. The physical path drives range, carrier phase and OFDM response together at original row epochs, before recorded averaging and range whitening.</p>
<p>Unlike fitting only extracted phase summaries, this optimizer evaluates the original complex data at every iteration. Those summaries supply additional starts. The amplitude knots were selected using a separate validation quarter in the <a href="../ship-phase-continuity/index.html">continuity experiment</a>; final prediction uses its untouched quarter. Stationary coefficients, brightness and phases are frozen during prediction. Real-data directional gradient relative errors are below 10⁻⁴; the independent numerical suite also verifies the physical phase gauge and antenna/direct-path derivatives.</p>
<img src="sites.png" alt="All permitted site assignments on the raw physical arc likelihood">
<p>All 32 permitted primary/backup combinations are present. The receiver/site priors and target priors are held identical. Several trials are poor local fits, and all initialization costs and gradient checks remain in the JSON. A strong conditional preference between sites does not establish the sites when the common-point hypothesis itself has structured residuals or needs implausible nuisance values.</p>
<img src="observables.png" alt="Independent and physically linked range, rate and curvature">
<img src="prediction-information.png" alt="Predictive error and antenna information after nuisance elimination">
<p>The information plot uses the derivative of the predicted mean, eliminates stationary coefficients, active amplitude coefficients, physical motion and constant band phases, and excludes location priors. Sixteen antenna coordinates remain after fixing horizontal receiver location. Values below the plot floor are displayed at 10⁻¹⁴. These are conditional thermal information directions, not model-error-aware location confidence intervals. Unknown constant scattering phase is not equated between different RF bands.</p>
<img src="offsets.png" alt="Unaccepted transmitter and reflector offsets">
<p><a href="summary.json">Summary JSON</a> · <a href="../ship-physical-arc-sites-32/fits.json">All fixed-site fits</a> · <a href="../ship-physical-arc-refined/fits.json">Refined-antenna controls</a> · <a href="../road-refinement/index.html">Vehicle control with road associations held fixed</a></p>
<details><summary>Training scores and optimizer checks</summary><table><tr><th>Sites</th><th>Cost</th><th>Projected gradient</th><th>Bound contacts</th></tr>{rows}</table></details></html>''')


if __name__=='__main__':main()

"""Replay common physical RF-phase controls and plot measured discrepancies."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import numpy as np

from ais import read_csv
from common_physical_pair import CommonPhysicalPair
from compare_ais_phase import NAMES,COLOURS
from component_phase import physical_primary_measurements
from fit_physical_primary_association import assignment_code,geometry_for,make_model,prepare_window,measured
from fit_physical_primary_coherence import independent_free,common_free
from flash_geometry import assignments
from joint_path_information import expected_jacobian,functional_bounds
from nearby_echo_control import load,ROOT
from report_fixed_grid_guided_return import phase_comparison
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def antenna_bounds(independent,model,p,free,*,common):
    if common:mapped,transform=model.mapping(p);transform=transform[:,free]
    else:mapped,transform=p,np.eye(independent.count)[:,free]
    matrix,norm=expected_jacobian(independent,mapped,transform)
    antenna=np.r_[np.arange(15),17];interest=np.flatnonzero(np.isin(free,antenna))
    if len(interest)!=len(antenna):raise ValueError('All nonzero antenna coordinates must be estimated')
    functionals=independent.geometry.antennas.factor[:,antenna]
    result=functional_bounds(matrix,norm,interest,functionals)
    result.update(physical_offset_standard_deviation_enu_m=result.pop('standard_deviation').reshape(6,3),
        physical_offset_estimable=result.pop('estimable').reshape(6,3),
        policy='All estimated motion, primary phase(s), neighbour trajectories/origins/phases, active real brightness and static profile marginalized. No geographic/jerk prior, active guard or frozen nuisance supplies information. Two inherited Ten C trajectory coordinates and receiver horizontal position remain fixed as in every fit.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'physical-primary-coherence/fits.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=json.loads(args.source.read_text())
    if not source['complete']:raise ValueError('Wait for completed phase controls and release checks')
    if args.output.exists() and (not (args.output/'analysis.json').exists() or
            json.loads((args.output/'analysis.json').read_text()).get('source')!=str(args.source)):
        raise ValueError('Only an existing report of the same source may be refreshed')
    old=json.loads(Path(source['source']).read_text());empirical=json.loads(Path(old['source']).read_text())
    control=load();track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
    options=assignments(licensed_options(control['geometry']));out=args.output;out.mkdir(parents=True,exist_ok=True)
    errors=[];phase_errors=[];reviews=[];records=[]
    for number,group in enumerate(source['groups']):
        window=empirical['windows'][group['window_index']];first,last=group['interval_s'];midpoint=(first+last)/2
        bands,parts,models,parameters,_=prepare_window(control,window)
        assignment=next(a for a in options if assignment_code(a)==group['assignment'])
        geometry=geometry_for(control,assignment,midpoint,track)
        independent=make_model(bands,models,control,geometry,midpoint);common=CommonPhysicalPair(independent)
        cases=[('Independent phases',False,group['independent']),
            ('Common physical phase',True,group['common_trials'][group['best_common_index']]),
            ('Released independent phases',False,group['relaxed'])]
        for label,is_common,record in [*cases,*[('Alternative common',True,r) for i,r in enumerate(group['common_trials']) if i!=group['best_common_index']]]:
            model=common if is_common else independent;p=np.asarray(record['parameters']);r,_=model.evaluate(p)
            errors.append(abs(r@r*model.energy-record['cost']))
            mapped=common.mapping(p)[0] if is_common else p
            fresh=measured(independent,mapped)
            for a,b in zip(fresh,record['measured_primary']):
                for key in ('time_s','physical_path_change_m','complex_coefficient_real_imag','conditional_information'):
                    np.testing.assert_allclose(a[key],b[key],rtol=1e-10,atol=1e-13)
                    phase_errors.append(float(np.max(abs(np.asarray(a[key])-np.asarray(b[key])))))
            records.append(dict(antenna_mode=group['antenna_mode'],label=label,cost=record['cost'],
                gradient=record['audit']['projected_gradient'],bounds=record['audit']['bound_indices'],
                total_unused_error=record['total_unused_error'],prior_cost=record['prior_cost'],
                maximum_component_to_sum_ratio=max(v['component_to_sum_energy_ratio'] for v in record['diagnostics'])))
        variants=[];fig,axes=plt.subplots(3,3,figsize=(16,10),layout='constrained',sharex=True)
        for col,(label,is_common,record) in enumerate(cases):
            model=common if is_common else independent;p=np.asarray(record['parameters'])
            mapped=common.mapping(p)[0] if is_common else p
            native=[physical_primary_measurements(c,mapped[index]) for c,index in zip(independent.clusters,independent.indices)]
            m=phase_comparison(native,bands,first,last);reference=m['channels'][2]['measured_path_change_m']
            rms=[]
            for c,(ch,colour,rec,cluster,index) in enumerate(zip(m['channels'],COLOURS,native,independent.clusters,independent.indices)):
                axes[0,col].plot(m['time_s'],ch['measured_path_change_m'],'.-',color=colour,ms=2,lw=.8,label=NAMES[c])
                if c!=2:axes[1,col].plot(m['time_s'],100*(ch['measured_path_change_m']-reference),'.-',color=colour,ms=2,lw=.8)
                b=cluster.base;raw,_=cluster.response_blocks(mapped[index])[0]
                unit=b.reduced(raw[:,:,None])[:,0].reshape(b.field.y.shape)
                norm=np.sum(abs(unit)**2,axis=1)
                prediction=rec['frozen_component_prediction']
                expected=np.divide(np.sum(unit.conj()*prediction,axis=1),norm,out=np.zeros(len(norm),complex),where=norm>0)
                observed=rec['complex_coefficient']*np.exp(1j*b.field.k*rec['physical_path_change_m'])
                residual=np.angle(observed*expected.conj());testing=parts[c]==3
                weight=np.sum(abs(prediction)**2,axis=1);use=testing&(weight>0)
                rms.append(float(np.degrees(np.sqrt(np.average(residual[use]**2,weights=weight[use])))) if use.any() else None)
                axes[2,col].plot(rec['time_s'][testing],np.degrees(residual[testing]),'.',ms=2,color=colour)
            axes[0,col].set(title=label,ylabel='Measured path change (m)')
            axes[1,col].set(ylabel='Measured channel minus Nine (cm)');axes[1,col].axhline(0,color='#aaa',lw=.5)
            axes[2,col].set(ylabel='Unused phase innovation (°)',xlabel='Capture time (s)')
            for ax in axes[:,col]:
                ax.set_xlim(first,last);ax.xaxis.set_major_locator(MultipleLocator(.5));ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            info=antenna_bounds(independent,model,p,record['audit']['free_indices'],common=is_common) if group['antenna_mode']=='refined' else None
            variants.append(dict(label=label,measurement=m,unused_conditional_phase_rms_deg=rms,antenna_information=info,
                antenna_fit=record['antenna_fit'],cost=record['cost'],total_unused_error=record['total_unused_error'],audit=record['audit']))
        axes[0,0].legend(ncol=3,fontsize=8)
        fig.suptitle(f'{first:g}–{last:g} s · {group["antenna_mode"]} antennas · common physical RF phase versus independent phases\n'
            'Measured coefficients after training-frozen neighbour subtraction; constant origins only. No slope, heading or phase-screen correction.\n'
            'Unused innovations compare observations with the complete predicted component, including within-bin brightness.',fontsize=11)
        plot=f'{number+1:02d}-{group["antenna_mode"]}-measured-phases.png';fig.savefig(out/plot,dpi=145);plt.close(fig)
        reviews.append(dict(interval_s=group['interval_s'],antenna_mode=group['antenna_mode'],plot=plot,variants=variants,
            common_unused_change_percent=100*(cases[1][2]['total_unused_error']/group['independent']['total_unused_error']-1)))
    if max(errors)>1e-7:raise ValueError('Original-data objectives changed')
    refined=[r for r in reviews if r['antenna_mode']=='refined']
    if refined:
        review=refined[0];fig,axes=plt.subplots(1,4,figsize=(20,4.8),layout='constrained')
        for variant,style in zip(review['variants'],('o-','s--','^:')):
            offsets=np.vstack((variant['antenna_fit']['transmitter_offsets_enu_m'],variant['antenna_fit']['receiver_offset_enu_m']))
            for axis in range(3):axes[axis].plot([*NAMES,'RX'],offsets[:,axis],style,label=variant['label'])
            information=variant['antenna_information'];singular=np.asarray(information['normalized_singular_values'])
            axes[3].semilogy(np.arange(1,len(singular)+1),np.maximum(singular,1e-16),style,label=f'{variant["label"]} · rank {information["target_rank"]}/{information["target_dimension"]}')
        for ax,name in zip(axes[:3],('East','North','Up')):ax.set(title=name+' offsets',ylabel='Prior displacement (m)')
        axes[3].set(title='Antenna data-only information after nuisance',xlabel='Normalized singular direction',ylabel='Singular value')
        axes[3].axhline(1e-8,color='#999',lw=.7,ls=':');axes[3].legend(fontsize=7)
        axes[0].legend(fontsize=8)
        fig.suptitle('A local antenna adjustment is not a conclusive geographic calibration\n'
            'Shared site / per-channel priors are unchanged. Weak and null directions remain unmeasured; no prior information enters the singular values.',fontsize=11)
        fig.savefig(out/'03-antenna-adjustment-and-information.png',dpi=145);plt.close(fig)
    files=[args.source,Path(source['source']),Path(old['source']),Path(__file__),*[Path('support/isar')/(n+'.py') for n in
        ('fit_physical_primary_coherence','fit_physical_primary_association','physical_echo_cluster','physical_amplitude_arc',
         'common_physical_pair','common_physical_arc','referenced_common_arc','component_phase','joint_path_information')]]
    result=dict(complete=True,accepted_calibration=False,production_changed=False,source=str(args.source),reviews=reviews,
        fits=records,reproduced_objectives=len(errors),maximum_absolute_cost_error=max(errors),
        maximum_absolute_reproduced_measurement_error=max(phase_errors),
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    rows=''.join(f'<tr><td>{r["antenna_mode"]}</td><td>{r["label"]}</td><td>{r["cost"]:.3f}</td><td>{r["total_unused_error"]:.3f}</td><td>{r["gradient"]:.3g}</td><td>{r["bounds"]}</td><td>{r["maximum_component_to_sum_ratio"]:.3g}</td></tr>' for r in records)
    images=''.join(f'<h2>{r["antenna_mode"].title()} antennas</h2><p>Common-phase unused error change from the independent control: {r["common_unused_change_percent"]:+.4f}%.</p><img src="{r["plot"]}" alt="Measured common and independently phased primary trajectories">' for r in reviews)
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Common physical primary phase</title>
<style>body{{font:16px/1.6 system-ui;max-width:1500px;margin:30px auto;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ddd;padding:6px}}</style>
<h1>Common physical RF phase with independently moving neighbours</h1>
<p>One physical point drives delay, phase and symbol Doppler. This experiment additionally tests one physical RF scattering phase across the five bands, correctly undoing their conditioning-phase references. Independent constant band phases provide the larger-model control. No phase correction is applied to data. The same original samples, kernels, range covariance, observed-cell stationary profile, brightness knots, independent neighbouring curves and priors remain in both models.</p>
<p>Each antenna mode screens all 14 carrier-cycle proposals by original training likelihood, refines the best two and releases the winning common solution back to independent phases. Those local cycle initializers do not exhaust the full geographic priors. Minimum training cost selects a start; unused samples select neither starts nor models. These previously inspected data supply exploratory consistency checks, not external validation. Receiver horizontal location and redundant AIS lag remain fixed; the same two inherited Ten C path coefficients are fixed throughout.</p>
<table><tr><th>Antennas</th><th>Model</th><th>Penalized training cost</th><th>Unused squared error</th><th>Projected gradient</th><th>Active bounds</th><th>Maximum component / sum energy</th></tr>{rows}</table>
{images}{'<img src="03-antenna-adjustment-and-information.png" alt="Antenna offsets and data-only identifiability">' if refined else ''}
<p>The singular-value curves are evaluated at each model's own fitted parameters and use their own original-derivative column norms. Numerical ranks use the explicit 1e−8 cutoff. They diagnose unresolved directions; they are not a same-scene information-gain comparison between the independent and restricted model classes. A scalar prediction-error percentage also does not bound location uncertainty.</p>
<p>A common scattering phase is an optimistic physical hypothesis: unknown frequency-dependent scattering or instrumental phase can invalidate it. A close prediction fit on one two-second episode does not identify transmitter sites or uniquely locate antennas. Data-only information marginalizes every estimated scene nuisance and excludes geographic/jerk priors and active guards. Refinement is conditional on the nominal licensed assignment 00002, not a comparison of all 32 assignments. No fitted coordinates are adopted.</p>
<p>{len(errors)} original likelihoods reproduce with maximum absolute discrepancy {max(errors):.3g}; saved native complex phase diagnostics reproduce within {max(phase_errors):.3g}. <a href="analysis.json">Measured arrays and full diagnostics</a> · <a href="../physical-primary-phase-review/index.html">Original physical-primary association</a> · <a href="../multiscatterer-phase-review-complete/index.html">All selected scatterer phase plots</a>.</p></html>''')
    snapshot=out/'sources';snapshot.mkdir(exist_ok=True)
    for p in files:
        if p.suffix=='.py':shutil.copyfile(p,snapshot/p.name)
    print('report',out/'index.html','objectives',len(errors),'max error',max(errors),flush=True)


if __name__=='__main__':main()

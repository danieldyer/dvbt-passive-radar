"""Reproducible synthetic recovery, exact backprojection, and recorded-data controls."""
import argparse
from copy import copy
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from recovery_experiment import Experiment, SCALE, NAMES, load_fields
from report_measured_scatterer_phases import serial


def coefficient(a):
    a=np.asarray(a);return a[:,0]+1j*a[:,1]


def phase_diagnostics(ex,p,data):
    _,_,alpha,pred=ex.profile(p,data);records=[]
    for c,(f,train) in enumerate(zip(ex.fields,ex.training)):
        h=ex.templates(p,c)[:,0].reshape(f.y.shape);use=np.repeat(train,len(f.offsets))
        static=np.linalg.lstsq(ex.designs[c][use],data[c][use]-h.ravel()[use]*alpha[c][0],rcond=1e-10)[0]
        cleaned=(data[c]-ex.designs[c]@static).reshape(f.y.shape)
        norm=np.sum(abs(h)**2,axis=1);z=np.sum(h.conj()*cleaned,axis=1)/norm
        phase=np.angle(z*alpha[c][0].conj());power=abs(alpha[c][0])**2*norm
        valid=(~train)&(power>4)
        rms=np.degrees(np.sqrt(np.mean(phase[valid]**2))) if valid.any() else None
        records.append(dict(name=NAMES[c],time_s=f.time,amplitude_relative=abs(z)/max(abs(alpha[c][0]),1e-100),
            wrapped_residual_degrees=np.degrees(phase),predicted_thermal_phase_sigma_degrees=np.degrees(1/np.sqrt(2*power)),
            testing=~train,valid_test=valid,conditional_unused_phase_rms_degrees=rms,
            test_rss_per_complex_cell=np.mean(abs(data[c].reshape(f.y.shape)[~train]-pred[c].reshape(f.y.shape)[~train])**2)))
    return records


def antenna_information(ex,p,data,*,step_m=.1):
    """Profile unknown reflectivity, stationary coefficients, and all six motion coordinates."""
    _,motion,_,_=ex.profile(p,data)
    site=np.array([0,0,0,0,1]);columns=[];labels=[]
    for s in (0,1):
        for axis in range(3):
            delta=ex.rotation[axis]*step_m
            plus,minus=copy(ex),copy(ex)
            plus.transmitters=ex.transmitters.copy();minus.transmitters=ex.transmitters.copy()
            plus.transmitters[site==s]+=delta;minus.transmitters[site==s]-=delta
            columns.append((plus.profile(p,data,derivative=False)[0]-minus.profile(p,data,derivative=False)[0])/(2*step_m))
            labels.append(('Ornata' if s==0 else 'Ridge')+' '+['E','N','U'][axis])
    plus,minus=copy(ex),copy(ex);plus.receiver=ex.receiver+ex.rotation[2]*step_m;minus.receiver=ex.receiver-ex.rotation[2]*step_m
    columns.append((plus.profile(p,data,derivative=False)[0]-minus.profile(p,data,derivative=False)[0])/(2*step_m));labels.append('RX U')
    antenna=np.column_stack(columns);u,s,_=np.linalg.svd(motion,full_matrices=False)
    q=u[:,s>s[0]*1e-11];projected=antenna-q@(q.T@antenna)
    _,sv,vh=np.linalg.svd(projected,full_matrices=False)
    # Full inverse only when numerically full rank; a pseudoinverse must not
    # manufacture finite variances by deleting poorly measured coordinates.
    sd=np.sqrt(np.sum((vh.T/sv)**2,axis=1)/2)
    return dict(labels=labels,singular_values_per_metre=sv,right_singular_vectors=vh,finite_difference_step_m=step_m,
        best_measured_unit_antenna_combination_local_sigma_m=1/(np.sqrt(2)*sv[0]),
        local_standard_deviation_m=sd,projection_rank=int(q.shape[1]),
        note='Local conditional bound for this synthetic point, nominal 00002 assignment, known height and fixed RX horizontal coordinates. Seven coordinates, no site priors. Small singular values are reported, not deleted.')


def render_images(ex,truth,motion,gains,out,side):
    """Exact original-symbol templates; no interpolation of a fitted phase image."""
    axis=np.linspace(-200,200,side);east,north=np.meshgrid(axis,axis)
    offsets=np.c_[east.ravel(),north.ravel(),np.zeros(east.size)]
    points=np.array([[0.,0.,0.],[-55.,110.,0.],[55.,-110.,0.]])
    reflectivity=np.array([1.+0j,.65*np.exp(.7j),.5*np.exp(-.9j)])
    scene=Experiment(ex.fields,ex.rotation,ex.receiver,ex.transmitters,ex.state,ex.midpoint,offsets=points)
    data=[scene.independent_signal(truth,c,gains[c]*reflectivity) for c in range(5)]
    synthetic_fit,unused=scene.fit(data,motion,120)
    print('three-point known-support fit',synthetic_fit['physical_correction'],'RSS',synthetic_fit['training_rss'],flush=True)
    scores={};peaks={};losses={}
    for label,p in [('oracle',truth),('recovered',synthetic_fit['parameters']),('AIS',np.zeros(6))]:
        matrices=[]
        for c,(f,train,q) in enumerate(zip(ex.fields,ex.training,ex.q)):
            use=np.repeat(train,len(f.offsets));dictionary=[]
            for offset in offsets:
                h=ex.templates(p,c,offsets=[offset])[:,0][use]
                dictionary.append(h-q@(q.conj().T@h))
            matrices.append(np.column_stack(dictionary))
        cross=np.zeros(len(offsets),complex);norm=np.zeros(len(offsets));channel_maps=[]
        point_cross=np.zeros(1,complex);point_norm=0.;reference=0.
        for c,(h,f,train,q) in enumerate(zip(matrices,ex.fields,ex.training,ex.q)):
            use=np.repeat(train,len(f.offsets));clean=data[c][use]-q@(q.conj().T@data[c][use])
            product=h.conj().T@clean;energy=np.sum(abs(h)**2,axis=0)
            channel_maps.append(abs(product)**2/np.maximum(energy,1e-100))
            cross+=gains[c].conjugate()*product;norm+=abs(gains[c])**2*energy
            # Coherence loss uses an isolated point, avoiding brightness from neighbours.
            h0=ex.templates(p,c,offsets=[[0.,0.,0.]])[:,0][use];h0-=q@(q.conj().T@h0)
            z=ex.independent_signal(truth,c,[gains[c]])[use];z-=q@(q.conj().T@z)
            point_cross+=gains[c].conjugate()*np.vdot(h0,z);point_norm+=abs(gains[c])**2*np.vdot(h0,h0).real
            reference+=np.vdot(z,z).real
        score=abs(cross)**2/np.maximum(norm,1e-100);scores[label]=score.reshape(east.shape)
        peaks[label]=offsets[np.argmax(score)];losses[label]=float(-10*np.log10(abs(point_cross[0])**2/point_norm/reference))
        if label=='oracle':scores['per_channel']=np.asarray(channel_maps).reshape(5,*east.shape)
        print('image',label,'isolated-point loss dB',losses[label],flush=True)
    upper=max(v.max() for k,v in scores.items() if k!='per_channel')
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,label in zip(axes,('AIS','oracle','recovered')):
        image=ax.imshow(10*np.log10(np.maximum(scores[label]/upper,1e-8)),origin='lower',extent=[-200,200,-200,200],
            cmap='magma',vmin=-35,vmax=0,interpolation='nearest')
        ax.scatter(points[:,0],points[:,1],marker='+',c='cyan',s=65)
        ax.set(title=f'{label}: {losses[label]:.3f} dB point coherence loss',xlabel='East offset (m)',ylabel='North offset (m)',aspect='equal')
    fig.colorbar(image,ax=axes,label='Matched power relative to common maximum (dB)')
    fig.suptitle('Independent three-point synthetic scene · exact 30 s five-band backprojection\nCyan marks truth; band gains known. Recovered motion uses known point support, not blind scene recovery.')
    fig.savefig(out/'03-images.png',dpi=150);plt.close(fig)
    np.savez(out/'images.npz',axis_m=axis,points_enu_m=points,**scores)
    return dict(known_support_fit=synthetic_fit,isolated_point_loss_db=losses,image_peaks_enu_m=peaks,
        scope='Conditional imaging control with known static scatterer offsets and known band gains; not blind multiscatterer/phase-centre recovery.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--side',type=int,default=31)
    args=parser.parse_args();out=args.input;result=json.loads((out/'analysis.json').read_text())
    baseline=json.loads((args.baseline/'analysis.json').read_text())
    first,last=result['interval_s'];fields,rot,rx,tx,state,mid=load_fields(first,last,result['stride'])
    ex=Experiment(fields,rot,rx,tx,state,mid);truth=np.asarray(result['truth_parameters'])
    with np.load(out/'observations.npz') as values:
        clean=[values[f'clean_{c}'] for c in range(5)];static=[values[f'static_{c}'] for c in range(5)];noise=[values[f'noise_{c}'] for c in range(5)]
    best=lambda case:min((f for f in result['fits'] if f['case']==case),key=lambda f:f['training_rss'])
    data=[z+s+n for z,s,n in zip(clean,static,noise)]
    diagnostics={name:phase_diagnostics(ex,np.asarray(best(name)['parameters']), y) for name,y in
        [('noisy',data),('recorded',[f.y.ravel() for f in fields])]}
    fig,axes=plt.subplots(5,2,figsize=(15,13),layout='constrained',sharex=True,sharey=True)
    for col,name in enumerate(('noisy','recorded')):
        for c,r in enumerate(diagnostics[name]):
            ax=axes[c,col];use=np.asarray(r['testing']);t=np.asarray(r['time_s']);sigma=np.asarray(r['predicted_thermal_phase_sigma_degrees'])
            ax.fill_between(t,-2*sigma,2*sigma,color='grey',alpha=.25)
            ax.plot(t[use],np.asarray(r['wrapped_residual_degrees'])[use],'.',ms=3)
            ax.set(title=NAMES[c]+' · '+name,ylim=(-180,180),ylabel='Wrapped phase residual (°)',xlabel='Capture time (s)')
    fig.suptitle('Frozen prediction on unused epochs · synthetic and original recording\nGrey: ±2σ conditional thermal prediction. No phase unwrapping or fitted time-varying phase correction.')
    fig.savefig(out/'02-phase-residuals.png',dpi=150);plt.close(fig)
    info=antenna_information(ex,truth,[z+s for z,s in zip(clean,static)])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for source,label in ((baseline,'Local refinement only'),(result,'Delay/rate/curvature acquisition')):
        fits=[f for f in source['fits'] if f['case']=='noiseless']
        axes[0].plot([f['start'] for f in fits],[max(f['training_rss'],1e-18) for f in fits],'o-',label=label)
    axes[0].set(yscale='log',xlabel='Independent displaced start',ylabel='Noiseless training residual power');axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(6),result['truth_information']['local_standard_deviation_physical'])
    axes[1].set(yscale='log',xticks=np.arange(6),xticklabels=['E (m)','N (m)','vE (m/s)','vN (m/s)','aE (m/s²)','aN (m/s²)'],ylabel='Local conditional 1σ bound')
    axes[1].tick_params(axis='x',labelrotation=35)
    axes[2].plot(np.arange(1,8),info['singular_values_per_metre'],'o-')
    axes[2].set(yscale='log',ylabel='Profiled antenna singular value (1/m)',xlabel='Mode, strongest first')
    fig.suptitle('Acquisition failure and geometric information are separate questions\nAntenna bound profiles all six motion coordinates and independent constant band phases; no site priors.')
    fig.savefig(out/'01-recovery-and-information.png',dpi=150);plt.close(fig)
    t=np.linspace(-1,1,301);theta=.4*np.sin(3*t);phi=np.pi/2-.001
    a=(np.cos(theta)/np.cos(phi)+np.sin(theta)/np.sin(phi))/2
    b=(np.cos(theta)/np.cos(phi)-np.sin(theta)/np.sin(phi))/2
    z=a*np.exp(1j*phi)+b*np.exp(-1j*phi)
    ratio=np.mean(a*a+b*b)/np.mean(abs(z)**2)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(t,np.degrees(theta),label='Changing phase to reproduce');axes[0].plot(t,np.degrees(np.angle(z)),'--',label='Two fixed-phase positive components')
    axes[0].set(xlabel='Time (arbitrary units)',ylabel='Phase (°)');axes[0].legend(fontsize=8)
    axes[1].plot(t,a,label='Positive amplitude A');axes[1].plot(t,b,label='Positive amplitude B');axes[1].set(xlabel='Time',ylabel='Individual amplitude (sum has unit amplitude)');axes[1].legend(fontsize=8)
    fig.suptitle(f'Why overlapping, flexible components can fake motion · component/sum energy ratio {ratio:,.0f}')
    fig.savefig(out/'04-degeneracy.png',dpi=150);plt.close(fig)
    gains=np.array([np.linalg.lstsq(ex.templates(truth,c),clean[c],rcond=None)[0][0] for c in range(5)])
    imaging=render_images(ex,truth,np.asarray(best('noiseless')['parameters']),gains,out,args.side)
    summary=dict(phase_diagnostics=diagnostics,antenna_information=info,imaging=imaging,
        overlapping_component_counterexample=dict(maximum_complex_error=np.max(abs(z-np.exp(1j*theta))),minimum_amplitude=min(a.min(),b.min()),component_to_sum_energy_ratio=ratio))
    (out/'diagnostics.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False)+'\n')
    report=f'''<!doctype html><meta charset="utf-8"><title>Synthetic recovery and failure audit</title>
<style>body{{font:16px/1.55 system-ui;max-width:1350px;margin:30px auto;padding:0 20px}}img{{width:100%}}.note{{background:#fff1d3;padding:15px}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #ccc}}</style>
<h1>Synthetic recovery and failure audit</h1>
<p>The previous tests did not establish recovery from realistic motion errors. This experiment isolates acquisition, forward-model correctness, physical information, and image formation.</p>
<div class="note">No new antenna calibration or real-data focused image is claimed. A successful synthetic point fit establishes a tested subproblem; it does not identify material scatterers in the recording.</div>
<h2>Experiment</h2><p>Actual {last-first:g} s aperture ({first:g}–{last:g} s), five frequencies, approved site hypothesis 00002, actual original symbol epochs/information, recorded range covariance, and known reconstructed DVB-T pulse responses. Keep every {result['stride']}th complete 16-symbol bin, retaining every symbol within it ({result['retained_epochs'][0]} retained bins per band). Truth is planar quadratic motion, known height, constant complex reflectivity per band, plus declared stationary echoes and proper-complex Gaussian noise. Mean per-bin matched information is 10 dB. Carrier phase, range migration and C0/C1 factors use exact two-leg geometry.</p>
<p>Independent synthesis evaluates Cartesian norms and 24-node numerical symbol quadrature. Maximum relative forward discrepancy: {max(result['independent_forward_relative_error']):.3g}. Estimation uses the existing analytic Field operator. Every fitted coefficient is frozen for the unused one-in-four epochs. No phase unwrapping, amplitude splines or phase-screen corrections.</p>
<h2>1. The optimiser could fail even when the model was exactly correct</h2>
<p>Original starts displaced the target by 40–60 m per axis and velocity by approximately 0.1 m/s. Direct local refinement failed both noiseless starts. Delay/rate/curvature acquisition followed by the same exact likelihood recovers both to the numerical floor. This is an acquisition failure, not evidence that point motion is intractable. The coarse tone model is used only to initialize the exact geometric estimator.</p><img src="01-recovery-and-information.png">
<p>Under the stated synthetic SNR and unknown independent band phases, conditional motion-position bounds are {result['truth_information']['local_standard_deviation_physical'][0]:.1f} m east and {result['truth_information']['local_standard_deviation_physical'][1]:.1f} m north. These are local bounds, not confidence intervals for the real distributed ship. Antenna bounds additionally profile the six motion coordinates; weak singular values are retained. A small residual alone cannot demonstrate accurate site coordinates.</p>
<h2>2. The recording does not behave like this one persistent point</h2><img src="02-phase-residuals.png">
<p>The original recording is passed through exactly the same estimator and acquisition. Residual plots show wrapped measured phase on unused epochs after training-frozen stationary subtraction. Grey bands use only the existing conditional thermal covariance: excess scatter is evidence against that combined one-point/noise hypothesis, not a measurement of a uniquely identified physical scatterer. Other moving echoes and temporal brightness changes are deliberately not claimed to be thermal noise. Real fits reaching search bounds remain rejected.</p>
<h2>3. Exact image-formation control</h2><img src="03-images.png">
<p>The independent three-point scene has a fixed north/east layout, no fitted yaw, and common material reflectivities across bands after known band gains. Motion recovery in this control knows the three relative offsets; image formation scans every pixel through the original-symbol operator. This is a conditional control, not blind joint scene/motion recovery. Cyan crosses show truth. Isolated-point coherent loss: oracle {imaging['isolated_point_loss_db']['oracle']:.6g} dB, recovered {imaging['isolated_point_loss_db']['recovered']:.6g} dB, uncorrected AIS {imaging['isolated_point_loss_db']['AIS']:.3f} dB. Sidelobes and broad responses visible even with truth are real properties of this aperture, not autofocus errors.</p>
<h2>4. Why the earlier overlapping-scatterer model was unsafe</h2><img src="04-degeneracy.png">
<p>Two almost opposite constant phases with independently changing positive amplitudes can synthesize a changing resultant phase. This example reproduces a ±23° trajectory with positive amplitudes and complex error below {np.max(abs(z-np.exp(1j*theta))):.2g}, despite neither component moving. The component/sum energy ratio is {ratio:,.0f}. Consequently, nonnegative brightness does not protect inferred motion when components overlap. This explains how a very small likelihood residual could coexist with implausible or unstable fitted motion. Such solutions must be rejected or avoided by an identifiable scene model.</p>
<h2>What is and is not established</h2><p>Established: independent operator agreement; full six-coordinate noiseless recovery from displaced starts after acquisition; an explicit conditional imaging control; measured geometric information; a constructive counterexample to the earlier brightness-model assumption. Outstanding: blind support recovery, an interference/brightness model that predicts unused real data, conclusive site assignment/coordinates, and full-capture autofocus. The failed initial and rate-only acquisition runs are retained beside this report.</p>
<p>Method: <a href="https://www.cs.umd.edu/~oleary/software/varpro.pdf">O’Leary and Rust, Variable Projection for Nonlinear Least Squares Problems</a>. Linear complex reflectivities are eliminated analytically with the residual-dependent Jacobian term retained. Exact bistatic backprojection follows the forward-model approach described by <a href="https://arxiv.org/abs/1302.5340">Wang and Yazıcı, Bistatic SAR Imaging of Moving Targets</a>; this experiment makes its own geometry and noise assumptions explicit.</p>
<p><a href="analysis.json">Fit metrics</a> · <a href="diagnostics.json">Diagnostics and arrays</a> · <a href="images.npz">Image arrays</a> · <a href="../recovery-audit-v1/analysis.json">Failed local-only control</a></p>'''
    (out/'index.html').write_text(report)
    print('report',out/'index.html',flush=True)


if __name__=='__main__':main()

"""Validate and exercise the windowed Radon-FRFT motion kernel on recorded data."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.legendre import leggauss

from field import FS,average
from motion import C
from gpga import fit as fit_gpga
from quadratic_motion_kernel import QuadraticMotionKernel,wavenumber_focus
from recovery_experiment import Experiment,NAMES,load_fields
from report_measured_scatterer_phases import serial


def independent_echo(kernel,p,alpha):
    f=kernel.field;d=f.data;t=kernel.dt
    path=kernel.reference+p[0]+p[1]*t+.5*p[2]*t*t
    omega=-f.k*.001024*(p[1]+p[2]*t)
    nodes,weight=leggauss(24);nodes,weight=nodes/2,weight/2
    tone=np.exp(1j*omega[:,None]*nodes)
    moments=tone@weight-1j*d['centre_omega']*(tone@(nodes*weight))
    lag=f.offsets[None,:]-(path-d['centre_path_m'])[:,None]*FS/C
    raw=alpha*f.pulse(lag)*moments[:,None]*np.exp(-1j*f.k*(path-d.get('carrier_reference_path_m',d['centre_path_m']))[:,None])
    return f.apply(average(raw,d['row_information'],f.metadata['block_rows'])[0]).ravel()


def execute(kernel,data,*,acceleration_halfwidth=.08):
    before=time.monotonic();origin=np.arange(-150.,150.1,C/FS/2)
    acquired=kernel.search(data,origin,rate_halfwidth=.6,acceleration_halfwidth=acceleration_halfwidth,keep=4)
    bounds=([-185.,kernel.nominal[1]-.65,kernel.nominal[2]-acceleration_halfwidth-.01],
            [185.,kernel.nominal[1]+.65,kernel.nominal[2]+acceleration_halfwidth+.01])
    fits=[]
    for seed in acquired:
        p=np.clip(seed['parameters'],np.asarray(bounds[0])+1e-9,np.asarray(bounds[1])-1e-9)
        fits.append(kernel.refine(data,p,bounds=bounds))
    best=min(fits,key=lambda f:f['training_rss']);best['seconds']=time.monotonic()-before
    best['coarse_seeds']=acquired
    return best


def slim(fit):return {k:v for k,v in fit.items() if k not in ('prediction','residual','unit_template')}


def plot_recorded_controls(maps,out):
    fig,axes=plt.subplots(len(maps),3,figsize=(15,3*len(maps)),layout='constrained',squeeze=False)
    for row,(name,t,z,alpha,raw,focused,delta) in enumerate(maps):
        upper=10*np.log10(max(np.max(abs(raw)**2),np.max(abs(focused)**2),1e-20))
        for col,v in enumerate((raw,focused)):
            power=10*np.log10(np.maximum(abs(v)**2,1e-20));ax=axes[row,col]
            ax.imshow(power.T,origin='lower',aspect='auto',extent=[t[0],t[-1],-8*C/FS,8*C/FS],
                cmap='magma',vmin=upper-35,vmax=upper)
            ax.set(title=name+(' · before' if col==0 else ' · wavenumber retiming'),xlabel='Time (s)',ylabel='Excess-path offset (m)')
        axes[row,2].plot(t,np.degrees(np.angle(z*alpha.conjugate())),'.',ms=3)
        axes[row,2].set(title=name+' · measured residual phase',xlabel='Time (s)',ylabel='Wrapped phase (°)',ylim=(-180,180))
    fig.suptitle('Original recorded data · independent quadratic-path fits\nIdentical colour scale within each before/after pair; residual phase is still measured.')
    fig.savefig(out/'03-recorded-controls.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--windows',type=float,nargs='+',default=[2.,4.,8.,16.,30.])
    parser.add_argument('--stride',type=int,default=8)
    parser.add_argument('--channels',type=int,nargs='+',default=[0,1,2,3,4])
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    fields,rot,rx,tx,state,mid=load_fields(149.,179.,args.stride)
    result=dict(scope=__doc__,complete=False,production_changed=False,accepted_calibration=False,
        policy='Independent band path acquisition; exact quadratic excess path links range migration, RF phase and C0/C1. Constant complex reflectivity and declared stationary basis. No heading or antenna fit. Training-only search and refinement; unused rows never enter optimisation.',
        scope_limits='CPU reference/acquisition kernel. Not a GPU implementation, blind multi-scatterer solver, full omega-k/Stolt imager or proven calibrated multiband autofocus.',synthetic={},recorded=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    save();rng=np.random.default_rng(12017)
    # Independent, low per-bin SNR synthetic echo at the actual 30 s sampling.
    ex=Experiment(fields,rot,rx,tx,state,mid);c=2;k=QuadraticMotionKernel(fields[c],ex.training[c],ex.designs[c])
    truth=k.nominal+np.array([13.7,.23,.018]);h=independent_echo(k,truth,1.)
    alpha=np.sqrt(len(fields[c].time)/np.vdot(h,h).real)*np.exp(.7j)
    signal=independent_echo(k,truth,alpha)
    noise=(rng.normal(size=h.size)+1j*rng.normal(size=h.size))/np.sqrt(2)
    data=signal+noise;fit=execute(k,data)
    hfit=fit['unit_template'][~k.selected];z=signal[~k.selected]
    loss=-10*np.log10(abs(np.vdot(hfit,z))**2/np.vdot(hfit,hfit).real/np.vdot(z,z).real)
    result['synthetic']=dict(truth=truth,fit=slim(fit),fitted_parameter_error=fit['parameters']-truth,
        independent_forward_relative_error=np.linalg.norm(signal-k.response(truth)*alpha)/np.linalg.norm(signal),
        unused_true_signal_coherence_loss_db=loss,per_retained_epoch_matched_information_db=0.)
    # A generously optimized constant-rate comparator, with truth range known.
    # This avoids attributing ordinary frequency-bin mismatch to chirp recovery.
    constant_rate=[]
    for rate in np.linspace(truth[1]-.7,truth[1]+.7,141):
        h0=k.response([truth[0],rate,0.])[~k.selected]
        constant_rate.append(abs(np.vdot(h0,z))**2/np.vdot(h0,h0).real/np.vdot(z,z).real)
    result['synthetic']['best_constant_rate_true_signal_coherence_loss_db']=-10*np.log10(max(constant_rate))
    print('synthetic truth',truth,'fit',fit['parameters'],'loss',loss,flush=True);save()
    # Higher-order PGA is a separate truth-controlled test, with frozen unused epochs.
    t=fields[c].time;u=(t-mid)/15;selected=np.arange(len(t))%4!=3
    error=1.5*(np.polynomial.legendre.legval(u,[0,0,0,1])+.4*np.polynomial.legendre.legval(u,[0,0,0,0,1]))
    x=np.array([1.,.7*np.exp(.6j),.5*np.exp(-.3j)])[None,:]*np.exp(1j*error[:,None])
    x+=(rng.normal(size=x.shape)+1j*rng.normal(size=x.shape))*.08
    pga=fit_gpga(t,x,np.ones(len(t)),selected,bandwidths_hz=(1.,.5,.25),iterations_per_bandwidth=12)
    recovered=pga['phase_radians'];basis=np.column_stack([np.ones(len(t)),u,u*u])
    # Motion and phase gauges are explicit. Higher-order-only PGA must not
    # silently change the fitted quadratic motion or geographic origin.
    project=lambda p:p-basis@np.linalg.lstsq(basis[selected],p[selected],rcond=None)[0]
    projected,reality=project(recovered),project(error)
    result['synthetic']['pga']=dict(time_s=t,training=selected,true_higher_phase_rad=reality,
        recovered_higher_phase_rad=projected,unused_phase_error_rms_deg=np.degrees(np.sqrt(np.mean((projected[~selected]-reality[~selected])**2))),
        heldout_gain_db=pga['heldout_coherent_power_gain_db'],algorithm='Generalized PGA on fixed independently synthesized histories; no data-derived image-peak selection.',converged=pga['converged'])
    save()
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    axes[0].plot(t,np.degrees(reality),label='Injected higher-order phase');axes[0].plot(t,np.degrees(projected),'--',label='Training-only PGA')
    axes[0].set(xlabel='Capture time (s)',ylabel='Higher-order phase (°)');axes[0].legend(fontsize=8)
    axes[1].plot(t[~selected],np.degrees((projected-reality)[~selected]),'.');axes[1].set(xlabel='Unused epoch (s)',ylabel='Recovery error (°)')
    fig.suptitle('Separate higher-order autofocus control · constant, linear and quadratic gauges removed')
    fig.savefig(out/'02-pga-control.png',dpi=150);plt.close(fig)
    maps=[];previous={}
    for duration in args.windows:
        window=[f.windowed(mid-duration/2,mid+duration/2) for f in fields]
        sub=Experiment(window,rot,rx,tx,state,mid)
        for c in args.channels:
            kernel=QuadraticMotionKernel(window[c],sub.training[c],sub.designs[c])
            # Broader curvature support on brief flash intervals; long-window
            # boundaries are reported rather than called a physical solution.
            halfwidth=.5 if duration<=4 else .15 if duration<=8 else .08
            fitted=execute(kernel,window[c].y.ravel(),acceleration_halfwidth=halfwidth)
            continuation=None
            if c in previous:
                old,pold=previous[c];delta_time=kernel.epoch-old.epoch
                seed=np.array([old.reference+pold[0]+pold[1]*delta_time+.5*pold[2]*delta_time**2-kernel.reference,
                    pold[1]+pold[2]*delta_time,pold[2]])
                bounds=([-185.,kernel.nominal[1]-.65,kernel.nominal[2]-halfwidth-.01],
                        [185.,kernel.nominal[1]+.65,kernel.nominal[2]+halfwidth+.01])
                continuation=kernel.refine(window[c].y.ravel(),np.clip(seed,np.asarray(bounds[0])+1e-8,np.asarray(bounds[1])-1e-8),bounds=bounds)
                previous[c]=(kernel,continuation['parameters'])
            else:previous[c]=(kernel,fitted['parameters'])
            entry=dict(name=NAMES[c],channel=c,duration_s=duration,epochs=len(window[c].time),
                nominal=kernel.nominal,acceleration_search_halfwidth=halfwidth,
                continuation=slim(continuation) if continuation is not None else None,**slim(fitted))
            result['recorded'].append(entry);save()
            print(NAMES[c],duration,'p',fitted['parameters'],'train',fitted['training_rss'],'test',fitted['test_rss'],'sec',fitted['seconds'],flush=True)
            f=window[c];h=fitted['unit_template'].reshape(f.y.shape)
            alpha=complex(*fitted['alpha_real_imag'])
            static=fitted['prediction']-fitted['unit_template']*alpha
            clean=(f.y.ravel()-static).reshape(f.y.shape)
            z=np.sum(h.conj()*clean,axis=1)/np.sum(abs(h)**2,axis=1)
            p=fitted['parameters'];dt=f.data['row_time']-kernel.epoch
            path=kernel.reference+p[0]+p[1]*dt+.5*p[2]*dt*dt
            delta=average(path-f.data['centre_path_m'],f.data['row_information'],f.metadata['block_rows'])[0]
            # Undo only the fixed conditional whitening to show physical range profiles.
            raw=np.linalg.solve(f.whiten,(clean/f.root_information[:,None]).T).T
            focused=wavenumber_focus(raw,C/FS,delta,f.metadata['rf_hz'])
            if duration==max(args.windows):maps.append((NAMES[c],f.time,z,alpha,raw,focused,delta))
            np.savez(out/f'channel-{c}-{duration:g}s.npz',time_s=f.time,training=kernel.training,
                matched_coefficient=z,unit_template=h,raw_clean=raw,wavenumber_focused=focused,path_shift_m=delta)
    plot_recorded_controls(maps,out)
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for c in args.channels:
        entries=[e for e in result['recorded'] if e['channel']==c]
        axes[0].plot([e['duration_s'] for e in entries],[10*np.log10(max(e['score'],1e-100)) for e in entries],'o-',label=NAMES[c])
        axes[1].plot([e['duration_s'] for e in entries],[e['parameters'][1] for e in entries],'o-')
        axes[2].plot([e['duration_s'] for e in entries],[e['parameters'][2] for e in entries],'o-')
    for ax in axes:ax.set_xlabel('Coherent window (s)')
    axes[0].set_ylabel('Training matched statistic (dB, conditional metric)');axes[0].legend(fontsize=8)
    axes[1].set_ylabel('Fitted excess-path rate (m/s)');axes[2].set_ylabel('Fitted excess-path acceleration (m/s²)')
    fig.suptitle('Variable-window Radon-FRFT search · search maxima are not calibrated detection SNR or false-alarm probabilities')
    fig.savefig(out/'01-window-search.png',dpi=150);plt.close(fig)
    result['complete']=True
    result['sources_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in [Path(__file__),Path('support/isar/quadratic_motion_kernel.py')]}
    save()
    text=f'''<!doctype html><meta charset="utf-8"><title>Quadratic motion acquisition audit</title>
<style>body{{font:16px/1.55 system-ui;max-width:1450px;margin:30px auto;padding:0 20px}}img{{width:100%}}.note{{padding:14px;background:#fff1d3}}</style>
<h1>Windowed Radon–fractional Fourier motion kernel</h1>
<p>Implemented: dual-scale coherent path acquisition; exact original-symbol quadratic refinement; range-frequency migration correction with carrier-phase bookkeeping; and a separate higher-order phase-gradient autofocus recovery control.</p>
<p class="note">This is an offline CPU reference and validation package. It does not establish transmitter locations or a calibrated all-band ship image. Range-frequency retiming is the migration-correction operator, not a complete omega-k/Stolt geographic imager. Real candidate phases remain contaminated by unresolved echoes.</p>
<h2>Synthetic sensitivity</h2><p>An independently synthesized migrating quadratic echo uses actual 30 s native epochs, actual reconstructed waveform pulses, the existing noise metric, and 0 dB mean matched information per retained epoch. Training-only acquisition/refinement leaves <b>{loss:.4f} dB</b> true-signal coherent loss on unused epochs. Parameter error [range m, rate m/s, acceleration m/s²]: {serial(fit['parameters']-truth)}. Forward discrepancy: {result['synthetic']['independent_forward_relative_error']:.3g}. This is one controlled noise realization, not a probability-of-detection or global optimality claim.</p>
<h2>Window/rate/curvature acquisition on all requested channels</h2><img src="01-window-search.png">
<p>Each band is independently acquired, without forcing a common AIS phase history. The coarse Radon step follows envelope motion; the finer nonuniform FRFT chirp bank searches carrier rate/curvature. Top candidates are refined using the known range response and exact original-row C0/C1 operator. All three physical path parameters remain linked. Unknown band phases are constant coefficients, never time-varying corrections. Search support is ±150 m and ±0.6 m/s about the coarse reference, with declared curvature support per window. Missed support and boundary fits remain possible.</p>
<h2>Wavenumber correction and measured residuals</h2><img src="03-recorded-controls.png">
<p>Range-frequency multiplication exp(+i 2π f ΔL/c), followed by the RF factor exp(+i 2π f_c ΔL/c), removes the fitted path history without fitting yaw or introducing monostatic geometry. Fourfold zero padding supplies guard space. Finite-crop edge support still limits fidelity. The physical range grid is already AIS-referenced in these archived fields; the stored shift explicitly accounts for that reference. The residual-phase plots retain every measured epoch and show why a quadratic bright-cell fit alone is insufficient evidence for persistent material-scatterer identity.</p>
<h2>Higher-order autofocus control</h2><img src="02-pga-control.png">
<p>Three noisy synthetic histories receive known cubic/quartic phase error. Training-only generalized PGA recovers higher-order phase with unused RMS error {result['synthetic']['pga']['unused_phase_error_rms_deg']:.3f}°. Constant, linear and quadratic components are explicitly removed so this control cannot silently change image origin or quadratic motion. Reported PGA gain before that additional quadratic gauge removal: {result['synthetic']['pga']['heldout_gain_db']:.3f} dB. Independent band autofocus is not yet adopted for coherent combination: its phase gauge must remain tied to geometry.</p>
<h2>Literature and next gate</h2><p><a href="https://arxiv.org/abs/2403.06788">Li et al., dual-scale GRFT</a>; <a href="https://arxiv.org/abs/1511.03122">Tian et al., nonuniform Radon–FRFT</a>; <a href="https://arxiv.org/abs/1904.00169">Li et al., windowed Radon–FRFT</a>. The immediate remaining gate is blind multi-scatterer association and a jointly predictive geographic motion model, followed by a validated wavenumber image mapping or exact backprojection. A sharper phase-fitted cell is not treated as proof of that result.</p>
<p><a href="analysis.json">Complete metrics and search seeds</a> · <a href="../recovery-audit-delay-acquired/index.html">Synthetic geometry/recovery failure audit</a>. Per-band .npz files contain unchanged-reference data, fitted parameters, complex matched histories and focused profiles.</p>'''
    (out/'index.html').write_text(text);print('report',out/'index.html',flush=True)


if __name__=='__main__':main()

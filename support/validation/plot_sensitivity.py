#!/usr/bin/env python3
"""Create standalone scientific figures from the reproducible audit outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sensitivity import BINS, C, FS, NFFT, RF, window


def save(fig, directory, name):
    fig.savefig(directory/f'{name}.png', dpi=180)
    fig.savefig(directory/f'{name}.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--doppler-hz',type=float,default=200)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.18})
    theory=json.loads((a.audit/'theory.json').read_text())

    matched=json.loads((a.audit/'matched-filter.json').read_text())
    cases=[r for r in matched['cases'] if r['delay_samples'] in (0,800) and r['doppler_hz'] in (0,100,200)]
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    x=np.arange(len(cases))
    template=[r['gain_from_exact_template_with_hann_db'] for r in cases]
    taper=[r['exact_hann_loss_db'] for r in cases]
    ax.bar(x,template,label='Template mismatch (recoverable while retaining Hann)')
    ax.bar(x,taper,bottom=template,label='Hann white-noise cost (ambiguity trade-off)')
    ax.set_xticks(x,[f"{r['delay_samples']*C/FS/1000:.2f} km\n{r['doppler_hz']:g} Hz" for r in cases])
    ax.set(ylabel='SNR loss relative to exact white-noise optimum (dB)',ylim=(0,3.3),
           title='Fixed 0.999872 s CPI: decoded symbols, frozen H, finite echo operator')
    ax.legend(loc='upper left',fontsize=9)
    save(fig,a.output,'fixed_cpi_loss')

    doppler=json.loads((a.audit/'doppler-bound.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for excerpt in doppler['excerpts']:
        stem=Path(excerpt['source']).stem
        profile=np.load(a.audit/f'{stem}-doppler-noise.npz')
        keep=abs(profile['frequency'])>5
        reference=np.median(profile['mean_psd'][keep])
        axes[0].plot(profile['frequency'],10*np.log10(profile['mean_psd']/reference),
                     label=stem.replace('excerpt-','Start ')+' s',lw=1)
    control=doppler['white_noise_profile_control']['noise_profile_db_quantiles']
    axes[0].axhspan(control[1],control[3],color='grey',alpha=.25,
                   label='White-noise control: central 90%')
    axes[0].set(xlabel='Doppler (Hz)',ylabel='Noise-like background relative to median (dB)',
                xlim=(-a.doppler_hz,a.doppler_hz),ylim=(-1.6,2.7),
                title='31 nonoverlapping 1 s windows per excerpt; 3.75–29.2 km')
    axes[0].legend(fontsize=8)
    for name in ('hann','hamming'):
        records=[r for r in doppler['white_noise_bounds'] if r['window']==name]
        axes[1].semilogx([r['cells'] for r in records],
                         [r['gain_vs_single_cell_db'] for r in records],'o-',label=name)
    axes[1].axhline(10*np.log10(1.5),ls=':',color='C0',label='Hann full single-cell loss')
    axes[1].axhline(10*np.log10(1.3628257888),ls=':',color='C1',label='Hamming full single-cell loss')
    axes[1].set(xlabel='Complex Doppler cells processed jointly',ylabel='Gain vs current window single cell (dB)',
                xticks=[1,3,5,9,17,33],title='Exact white-noise information bound, on-bin target')
    axes[1].set_xticklabels(['1','3','5','9','17','33']);axes[1].legend(fontsize=8)
    save(fig,a.output,'doppler_background_and_bound')

    records=[r for e in doppler['excerpts'] for r in e['local_bounds']]
    fig,ax=plt.subplots(figsize=(12,4.7),layout='constrained');x=np.arange(len(records))
    for offset,key,label in ((-.24,'heldout_trained_weights_gain_db','Sample covariance + tiny ridge'),
                             (0,'heldout_regularized_weights_gain_db','Validation-selected regularisation'),
                             (.24,'heldout_known_window_weights_gain_db','Fixed analytical Hann weights')):
        ax.bar(x+offset,[r[key] for r in records],width=.24,label=label)
    ax.scatter(x,[r['heldout_oracle_local_gain_db'] for r in records],marker='_',s=180,
               color='black',label='Same-test covariance conditional ceiling')
    ax.axhline(1.34698573897,color='grey',ls=':',label='White-noise nine-cell gain')
    labels=[f"{Path(e['source']).stem[-3:]} s\n{r['centre_hz']:g} Hz"
            for e in doppler['excerpts'] for r in e['local_bounds']]
    ax.set_xticks(x,labels);ax.tick_params(axis='x',labelsize=8)
    ax.set(ylabel='Modelled SNR gain vs single Doppler cell (dB)',ylim=(0,3.5),
           title='Nine-cell processing: train 10 s, validate 10 s, evaluate separate final 11 s')
    ax.legend(fontsize=8,ncol=2)
    save(fig,a.output,'doppler_heldout')

    fig, axes=plt.subplots(1,2,figsize=(11,4.1),layout='constrained')
    for record in theory['windows']:
        name=record['name'];w=window(name,8192)
        response=abs(np.fft.fft(w,8192*128))[:128*10]/w.sum()
        label=name.replace('_',' ')
        axes[0].plot(np.arange(len(response))/128,20*np.log10(np.maximum(response,1e-8)),label=label)
        axes[1].scatter(record['white_noise_loss_db'],-record['peak_sidelobe_db'])
        axes[1].annotate(label,(record['white_noise_loss_db'],-record['peak_sidelobe_db']),xytext=(5,5),textcoords='offset points')
    axes[0].set(xlabel='Fourier-bin offset',ylabel='Response (dB)',ylim=(-115,1),xlim=(0,10))
    axes[0].legend();axes[1].set(xlabel='White-noise SNR cost (dB)',ylabel='Peak sidelobe rejection (dB)',xlim=(-.1,3.8),ylim=(0,105))
    fig.suptitle('Taper selection: more sidelobe rejection costs sensitivity and resolution')
    save(fig,a.output,'window_tradeoff')

    data=np.genfromtxt(a.audit/'response-injection.csv',delimiter=',',names=True)
    delays=np.unique(data['delay_samples']);frequencies=np.unique(data['doppler_hz'])
    gain=np.array([[20*np.log10(max(data[(data['delay_samples']==d)&(data['doppler_hz']==f)]['refit_echo_voltage_gain'][0],1e-6)) for d in delays] for f in frequencies])
    fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    im=ax.imshow(gain,origin='lower',aspect='auto',vmin=-60,vmax=0,cmap='viridis')
    ax.set_xticks(np.arange(len(delays)),[f'{d*C/FS/1000:.3g}' for d in delays]);ax.set_yticks(np.arange(len(frequencies)),[f'{f:.4g}' for f in frequencies])
    ax.set(xlabel='Injected excess path (km)',ylabel='Injected Doppler (Hz)',title='Echo retention with contaminated response fit: diagonal OFDM model')
    ax.grid(False)
    for j in range(len(frequencies)):
        for i in range(len(delays)):ax.text(i,j,f'{gain[j,i]:.1f}',ha='center',va='center',color='white' if gain[j,i]<-30 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Matched echo-response gain (dB; 0 means retained)')
    save(fig,a.output,'echo_retention')

    n=8192;t=(np.arange(n)+.5)/n-.5;w=window('hann',n)
    durations=np.geomspace(.05,16,220)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for f in (0,25,50,100,200):
        walk=durations*(BINS*FS/NFFT)/RF*f
        g=(np.sinc(walk[:,None]*t)@w/w.sum())**2
        axes[0].semilogx(durations,10*np.log10(durations*g/1.5),label=f'{f} Hz')
    axes[0].set(xlabel='Coherent integration interval (s)',ylabel='SNR vs 1 s rectangular, stationary case (dB)',ylim=(-22,12),title='Centered fixed-range bin, ideal constant Doppler match')
    axes[0].legend(ncol=2,fontsize=9)
    cycles=np.geomspace(.02,6,250);time=np.arange(n)/n;s=np.exp(2j*np.pi*cycles[:,None]*time)
    constant=(s.conj()@w/w.sum())[:,None];g=w*(s.conj()-constant)
    snr=abs(np.sum(g*s,axis=1))**2/np.sum(abs(g)**2,axis=1)/n
    optimum=1-abs(s.sum(axis=1)/n)**2
    axes[1].semilogx(cycles,10*np.log10(snr),label='Current Hann + weighted DC removal')
    axes[1].semilogx(cycles,10*np.log10(optimum),label='Optimal unknown-DC, white-noise bound')
    axes[1].axhline(-10*np.log10(1.5),color='grey',ls=':',label='Hann without DC removal')
    axes[1].set(xlabel='Absolute Doppler × interval (cycles)',ylabel='SNR efficiency relative to known white-noise optimum (dB)',ylim=(-32,1),title='Slow targets: DC rejection is a sensitivity trade-off')
    axes[1].legend(fontsize=8)
    fig.suptitle('Integration-time models, not measured target SNR')
    save(fig,a.output,'integration_tradeoff')

    maps=np.load(a.audit/'excerpt-050-maps-maps.npz')
    raw_peak=abs(maps['raw']).max()**2
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained',sharex=True,sharey=True)
    extent=(0,800*C/FS/1000,maps['frequency'][0],maps['frequency'][-1])
    for ax,key,title in zip(axes,('dc','flat'),('Current residual cross-spectrum + DC removal','Pilot-balanced residual + DC removal')):
        power=10*np.log10(np.maximum(abs(maps[key])**2/raw_peak,1e-15))
        im=ax.imshow(power,origin='lower',aspect='auto',extent=extent,vmin=-105,vmax=-45,cmap='magma',interpolation='nearest')
        ax.set(title=title,xlabel='Excess path (km)',ylim=(-a.doppler_hz,a.doppler_hz));ax.grid(False)
    axes[0].set_ylabel('Doppler (Hz)')
    fig.colorbar(im,ax=axes,label='Power relative to uncancelled zero-delay peak (dB)')
    fig.suptitle('C31 recording: one-second CPI near 82 s, shared absolute display scale')
    save(fig,a.output,'pilot_balanced_map')


if __name__=='__main__':main()

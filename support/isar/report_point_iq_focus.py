"""Publish native-IQ point recovery and same-data oracle focusing controls."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates

from report_measured_scatterer_phases import serial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('docs/data/injection_20261018/constant-velocity-iq'))
    args=parser.parse_args();root=args.root
    plan=json.loads((root/'plan.json').read_text());comparison=json.loads((root/'operator-common-projection.json').read_text())
    fits={name:json.loads((root/f'{name}-fit/analysis.json').read_text()) for name in ['linear','iq']}
    renders={name:json.loads((root/name/'analysis.json').read_text()) for name in ['oracle-image','recovered-point-image','observed-image']}
    if any(not x['complete'] for x in fits.values()):raise ValueError('Finish both physical fits first')
    arrays={name:np.load(root/name/'images.npz') for name in renders}
    peak_loss=np.asarray(renders['oracle-image']['peak_power_db'])-renders['recovered-point-image']['peak_power_db']
    iq_score=fits['iq']['scores']['constant_velocity_gls']
    worst_relative=max(z['excluded']['relative_complex_error'] for z in iq_score['channels'])
    bound=-10*np.log10(1-worst_relative**2)
    x=np.asarray(plan['receiver_ecef_m'])+np.asarray(plan['state_enu'])[0]@np.asarray(plan['ecef_to_enu'])
    g=sum((x-np.asarray(z))/np.linalg.norm(x-np.asarray(z)) for z in [plan['transmitters_ecef_m'][2],plan['receiver_ecef_m']])@np.asarray(plan['ecef_to_enu']).T
    normal=g[:2]/np.linalg.norm(g[:2]);tangent=np.array([-normal[1],normal[0]])
    oracle=arrays['oracle-image'];s=np.linspace(-110,110,2201);cuts=[];widths=[]
    for direction in [normal,tangent]:
        position=s[:,None]*direction;coordinates=np.array([position[:,1]-oracle['north_m'][0],position[:,0]-oracle['east_m'][0]])
        values=[];fwhm=[]
        for power in oracle['power']:
            cut=map_coordinates(power,coordinates,order=1,mode='constant');cut/=cut[len(s)//2]
            right=np.flatnonzero(cut[len(s)//2:]<.5)
            fwhm.append(None if not len(right) else 2*s[len(s)//2+right[0]])
            values.append(cut)
        cuts.append(values);widths.append(fwhm)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,values,label in zip(axes,cuts,['Along excess-range gradient','Along equal-range contour']):
        for i in [0,2,4,5]:ax.plot(s,10*np.log10(np.maximum(values[i],1e-12)),label=['Seven','SBS','Nine','Ten','ABC','Coherent sum'][i],lw=1.)
        ax.set(xlabel='Physical displacement (m)',ylabel='Power relative to central point peak (dB)',title=label,ylim=(-45,1))
    axes[0].set_xlim(-40,40);axes[0].legend(fontsize=9);axes[1].legend(fontsize=9)
    fig.suptitle('Oracle point-spread response, 30 s aperture\nCuts interpolate power from the 1 m raster; sidelobes contain no extra scatterers')
    fig.savefig(root/'point-spread-cuts.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,8),sharex=True,sharey=True,layout='constrained')
    high=float(np.max(oracle['power']));vmax=10*np.log10(high)+.1;vmin=vmax-45
    for ax,name,title in zip(axes,arrays,['Oracle point increment','Recovered point increment','Recovered complete IQ scene']):
        a=arrays[name];im=ax.imshow(10*np.log10(np.maximum(a['power'][5],1e-30)),origin='lower',
            extent=[a['east_m'][0],a['east_m'][-1],a['north_m'][0],a['north_m'][-1]],cmap='magma',vmin=vmin,vmax=vmax,interpolation='nearest')
        ax.plot(0,0,'+',color='#00ffff');ax.set(title=title,xlabel='East from injected point (m)',ylabel='North (m)')
    fig.colorbar(im,ax=axes,label='Conditional matched power (dB)',shrink=.7)
    fig.suptitle('Five-band complex backprojection · fixed north/east axes · no image alignment')
    fig.savefig(root/'focus-comparison.png',dpi=160);plt.close(fig)
    original=json.loads((root.parent/'constant-velocity/analysis.json').read_text())
    result=dict(scope=__doc__,additional_point_power_db=20,
        profile_position_error_m=fits['linear']['scores']['constant_velocity_gls']['horizontal_error_at_epoch_m'],
        iq_position_error_m=iq_score['horizontal_error_at_epoch_m'],iq_velocity_error_mps=iq_score['velocity_error_at_epoch_mps'],
        maximum_operator_mismatch_db=max(v['coherence_loss_db'] for v in comparison),
        maximum_operator_mean_phase_error_deg=max(abs(v['phase_bias_deg']) for v in comparison),
        identical_data_peak_loss_db=peak_loss,excluded_five_band_coherence_loss_upper_bound_db=bound,
        approximate_raster_fwhm_range_direction_m=widths[0],approximate_raster_fwhm_contour_direction_m=widths[1],
        original_strength_position_error_m=original['scores']['constant_velocity_gls']['horizontal_error_at_epoch_m'],ship_recovery_complete=False,production_changed=False)
    (root/'summary.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    rows=''.join(f'<tr><td>{["Seven","SBS","Nine","Ten","ABC"][v["channel"]]}</td><td>{v["coherence_loss_db"]:.5f}</td><td>{v["phase_bias_deg"]:.5f}°</td><td>{v["changed_iq_native_epochs"]}</td><td>{v["native_epoch_difference_s"]*1e9:.1f} ns</td></tr>' for v in comparison)
    (root/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native-IQ point focusing validation</title>
<style>body{{font:16px/1.6 system-ui;max-width:1300px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}</style>
<h1>Point localisation, motion and native-IQ focusing</h1>
<p class="note">The full IQ receiver recovers this constant-velocity point to {result['iq_position_error_m']:.2f} m and {result['iq_velocity_error_mps']*1000:.2f} mm/s, compared with {result['profile_position_error_m']:.2f} m for the profile control. On identical data and weights, the blindly recovered motion loses {peak_loss[-1]:.5f} dB of combined backprojection peak power relative to the oracle. The goal of a focused ship over the capture duration remains unfinished.</p>
<p>This is the 500 m relocated point with its power increased by 20 dB. The weaker original-strength test still has a 68 m geographic displacement; its working-model conditional bound is about 41 m. Increasing strength is a sensitivity control, not a replacement for that result. The <a href="../colored-point-study/index.html">independent coloured-noise trials</a> test estimator behaviour under the declared covariance. All fitting starts at coarse data-acquired proposals; true position, velocity, phase and band coefficients enter only synthesis, scoring and the oracle control.</p>
<h2>Actual waveform and receiver path</h2>
<p>The echo is added to all five saved channelised IQ streams before OFDM synchronisation, equalisation, FEC, reconstruction, response/noise estimation and finite C0/C1 compression. The control instead adds its independently FFT-evaluated waveform after the frozen baseline finite compressor. Both process 142–172 s. Every FEC word passes and decoded packet checksums match the baseline in all five bands. Source path interpolation is accurate to 2 nm. Capture data are unchanged.</p>
<table><tr><th>Band</th><th>Profile/IQ mismatch (dB)</th><th>Mean phase difference</th><th>Changed FFT epochs</th><th>Largest epoch difference</th></tr>{rows}</table>
<p>The stronger echo makes the synchroniser move the FFT start by one sample on 30 Ten symbols. The strict identical-epoch comparator fails and its log is retained. The explicit common-projection check retains those measured epochs separately and applies one fixed range interpolation, phase convention, averaging and noise metric to both sets of native C0/C1 rows. No time interpolation, fitted alignment or production-data correction occurs. Symbol counts, valid identities and decoded packets agree; the maximum permitted FFT-window difference is one 8 MS/s sample. The numerical 0.05 dB mismatch and 0.1 dB gain gates are unchanged.</p>
<h2>Focused images</h2><img src="focus-comparison.png" alt="Unaligned oracle, recovered point and complete observed scene images">
<p>The first two panels use the identical finite-profile point increment and original weights; background subtraction is a validation control only, never input to motion recovery. The third uses the complete actual IQ replay, including all recorded background. Its conditional power is about 0.3 dB lower mainly because the stronger echo changes the pipeline's estimated noise weights; the native complex amplitude differs by only 0.007–0.026 dB per band. That scale change is distinct from motion coherence loss.</p>
<p><a href="observed-image/point-images.png">Five individual IQ images and coherent sum</a> · <a href="oracle-image/point-images.png">Six oracle panels</a> · <a href="recovered-point-image/point-images.png">Six recovered point-control panels</a>. Every native row is backprojected on fixed north/east axes; no image translation, rotation, sharpening or peak registration is applied. The final 10.5 s was excluded from motion fitting and band calibration. Its per-band full-complex prediction errors give a conservative concatenated five-band coherence-loss bound of {bound:.6f} dB under the declared waveform/thermal metric; the finite-profile/IQ mismatch above is a separate receiver-path check.</p>
<h2>A focused point is still an extended point-spread response</h2><img src="point-spread-cuts.png" alt="Physical range and cross-range point-spread cuts">
<p>Even one true point produces long blobs and strong stripes here. The 30 s raster gives roughly {widths[0][-1]:.0f}–{widths[0][-1]+1:.0f} m combined main-lobe width along the range gradient and {widths[1][-1]:.0f} m along the equal-range contour. Individual bands have roughly 20–26 m and 76–95 m widths. These are approximate power-FWHM cuts on a 1 m raster, with motion held fixed. They are not joint position/velocity uncertainty bounds. The sidelobes and multi-band fringes in this controlled point image contain no vessel structure; a similar ship-like shape alone cannot validate a resolved hull.</p>
<p><a href="summary.json">Summary</a> · <a href="operator-common-projection.json">Complete operator comparison</a> · <a href="linear-fit/analysis.json">Profile fit</a> · <a href="iq-fit/analysis.json">IQ fit</a> · <a href="plan.json">Synthesis plan</a>. Reproduction tools are under <code>support/isar</code> and <code>support/validation/recorded_injection_probe.cpp</code>.</p></html>''')


if __name__=='__main__':main()

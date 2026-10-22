"""Quantify the ship/point comparison using the frozen matched-statistic model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from coherent_amplitude_glrt import statistic
from finite_facet import C
from point_spread_cuts import central_width
from recovery_experiment import NAMES
from render_injected_point import fitted_state
from template_information import evaluate_cached
from report_measured_scatterer_phases import serial


def combine(score,info,gain):
    individual=abs(score)**2/info
    coherent=abs(np.einsum('c,cp->p',gain.conj(),score))**2/np.einsum('c,cp->p',abs(gain)**2,info)
    calibrated=score*np.exp(-1j*np.angle(gain))[:,None]/np.sqrt(info)
    positive,phase=statistic(calibrated)
    independent=individual.sum(axis=0)
    return np.r_[individual,coherent[None]],positive,independent,phase


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,default=Path('docs/data/isar_20261018/native-spatial-validation/ais-near-projected-likelihood-brightness-full-source'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reuse-cuts',action='store_true',help='Refresh this report using its existing immutable model queries')
    args=p.parse_args();image,out=args.image,args.output;out.mkdir(parents=True,exist_ok=args.reuse_cuts)
    meta=json.loads((image/'analysis.json').read_text())
    root=Path('docs/data/isar_20261018/native-spatial-validation');fit=json.loads((root/'ais-near-three-point-30s/render-fit.json').read_text())
    plan=json.loads((root/'baseline-plan.json').read_text());rx,tx,rotation=map(np.asarray,(plan['receiver_ecef_m'],plan['transmitters_ecef_m'],plan['ecef_to_enu']))
    state=np.asarray(fit['coarse_state_enu']);mid=fit['midpoint_s'];pose=fitted_state(np.array([mid]),state,mid,rotation,rx,fit)
    position=pose['position_ecef_m'][0];g=sum((position-a)/np.linalg.norm(position-a) for a in (tx[2],rx))@rotation.T
    normal=g[:2]/np.linalg.norm(g[:2]);tangent=np.array([normal[1],-normal[0]])
    axes=np.array([normal,tangent]);distance=np.arange(-200.,200.01,.25)
    pixels=np.r_[*[np.c_[distance[:,None]*axis,np.zeros(len(distance))] for axis in axes]]
    with np.load(image/'images.npz') as source:saved=dict(source)
    raw=np.fromfile(image/'information-worker/information.bin','<f4').reshape(2,5,-1,4)
    scale=saved['information'][:,0,3]/raw[0,:,0,3]
    np.testing.assert_allclose(saved['information'][:,:,3],raw[0,:,:,3]*scale[:,None],rtol=1e-12)
    if args.reuse_cuts:
        previous=json.loads((out/'analysis.json').read_text())
        if previous['image']!=str(image):raise ValueError('Cached cuts belong to a different image model')
        with (out/'cuts-worker/pixels.bin').open('rb') as stream:
            import struct
            count,=struct.unpack('<I',stream.read(4));original=np.fromfile(stream,'<f4').reshape(count,4)
        np.testing.assert_array_equal(original,np.c_[pixels,np.zeros(len(pixels))].astype('<f4'))
        cached=np.fromfile(out/'cuts-worker/information.bin','<f4').reshape(2,5,len(pixels),4).astype(float)
        info=cached[0]*scale[:,None,None];observed=cached[1,:,:,:2]@np.array([1.,1j]);point=cached[1,:,:,2:]@np.array([1.,1j])
    else:info,observed,point=evaluate_cached(image/'information-worker/model.bin',pixels,scale,out/'cuts-worker')
    gain=saved['gains'];o,op,oi,_=combine(observed,info[:,:,1],gain);s,sp,si,_=combine(point,info[:,:,1],gain)
    o,s=[a.reshape(6,2,-1) for a in [o,s]]
    widths=[[central_width(distance,s[c,j]) for j in range(2)] for c in range(6)]
    centre=len(distance)//2
    grid_origin=np.flatnonzero(np.all(saved['pixels_east_north_up_m']==0,axis=1))
    if len(grid_origin)!=1:raise ValueError('The physical anchor must occur exactly on the raster')
    np.testing.assert_allclose(point[:,centre],saved['point_score'][:,grid_origin[0]],rtol=2e-6)
    point_phase=np.degrees(np.angle(point[:,centre]/gain));observed_phase=np.degrees(np.angle(observed[:,centre]/gain))
    point_ratio=sp[centre]/si[centre];observed_ratio=op[centre]/oi[centre]
    range_maxima=np.flatnonzero((s[5,0,1:-1]>s[5,0,:-2])&(s[5,0,1:-1]>=s[5,0,2:]))+1
    range_maxima=range_maxima[abs(distance[range_maxima])>1.]
    adjacent=range_maxima[np.argmax(s[5,0,range_maxima])]
    ambiguity_db=float(10*np.log10(s[5,0,adjacent]/s[5,0,centre]))
    # Local physical spatial-frequency support, including the occupied RF band,
    # over the aperture and viewport corners. This is a raster sampling check,
    # not a geographic-information or target-resolution bound.
    times=np.linspace(*meta['interval_s'],121);pose=fitted_state(times,state,mid,rotation,rx,fit)
    corners=np.array([[e,n,0.] for e in [saved['east_m'][0],saved['east_m'][-1]] for n in [saved['north_m'][0],saved['north_m'][-1]]])
    minimum=np.full(2,np.inf);maximum=-minimum.copy()
    for q in corners:
        moved=pose['position_ecef_m']+np.einsum('nij,j->ni',pose['body_to_ecef'],q)
        for antenna,rf in zip(tx,plan['rf_hz']):
            grad=sum((moved-a)/np.linalg.norm(moved-a,axis=1)[:,None] for a in (antenna,rx))
            body=np.einsum('ni,nij->nj',grad,pose['body_to_ecef'])[:,:2]
            for frequency in [rf-6.65625e6/2,rf+6.65625e6/2]:
                wavenumber=2*np.pi*frequency/C*body
                minimum=np.minimum(minimum,wavenumber.min(axis=0));maximum=np.maximum(maximum,wavenumber.max(axis=0))
    sampling=np.pi/(maximum-minimum)
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    names=NAMES+['Coherent five-band template']
    for j,a in enumerate(ax):
        peak=s[5,j,centre]
        a.plot(distance,10*np.log10(np.maximum(o[5,j]/peak,1e-12)),label='Observed scene')
        a.plot(distance,10*np.log10(np.maximum(s[5,j]/peak,1e-12)),label='One point; same visibility')
        a.axvline(0.,color='#777',lw=.6);a.axhline(-3.0103,color='#999',lw=.6)
        a.set(title=['Bistatic range direction','Equal-range direction'][j],xlabel='Distance from fitted anchor (m)',ylabel='Power relative to point control at origin (dB)',ylim=(-40,8),xlim=([-20,20] if j==0 else [-200,200]))
    ax[0].legend(fontsize=9);fig.suptitle('Direct 0.25 m physical cuts, not interpolation of the raster\nFrozen known-waveform model, original carrier-bin moments, nuisance projection and conditional covariance')
    fig.savefig(out/'point-spread-cuts.png',dpi=160);plt.close(fig)
    east,north=saved['east_m'],saved['north_m'];shape=(len(north),len(east));extent=[east[0],east[-1],north[0],north[-1]]
    info_grid=saved['information'][:,:,1]
    recorded,positive,independent,phase=combine(saved['observed_score'],info_grid,gain)
    synthetic,_,_,_=combine(saved['point_score'],info_grid,gain)
    residual,_,_,_=combine(saved['observed_score']-saved['point_score'],info_grid,gain)
    upper=10*np.log10(max(recorded[5].max(),synthetic[5].max()))+.1;lower=upper-40
    camera=json.loads((root/'ais-near-three-point-30s/camera.json').read_text())
    fig,axs=plt.subplots(1,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    for a,value,title in zip(axs,[recorded[5],synthetic[5],residual[5]],['Observed scene','One-point waveform prediction','Observed minus frozen point prediction']):
        im=a.imshow(10*np.log10(np.maximum(value.reshape(shape),1e-30)),origin='lower',extent=extent,cmap='magma',vmin=lower,vmax=upper)
        a.plot(*camera['marker_east_north_m'],'+',color='cyan',ms=8)
        a.set(title=title,xlabel='East from fitted anchor (m)',ylabel='North (m)',aspect='equal')
    fig.colorbar(im,ax=axs.tolist(),label='Conditional nuisance-projected matched statistic (dB)',shrink=.6)
    fig.suptitle('One-metre north/east raster; identical physical coordinates and fixed colour scale\nThe point prediction is frozen from training; no image registration or residual rescaling')
    fig.savefig(out/'measured-point-residual.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(14,9),sharex=True,sharey=True,layout='constrained')
    for a,value,title in zip(axs,[recorded[5],positive,independent],['Known anchor spectrum','Unknown positive band amplitudes; shared phase','Independent-band phase/amplitude bound']):
        im=a.imshow(10*np.log10(np.maximum(value.reshape(shape),1e-30)),origin='lower',extent=extent,cmap='magma',vmin=lower,vmax=upper)
        a.plot(*camera['marker_east_north_m'],'+',color='cyan',ms=8)
        a.set(title=title,xlabel='East from fitted anchor (m)',ylabel='North (m)',aspect='equal')
    fig.colorbar(im,ax=axs.tolist(),label='Conditional profile-likelihood improvement (dB; different nuisance dimensions)',shrink=.6)
    fig.suptitle('Frozen instrument-band phases in the first two models\nThe last permits arbitrary band phases for diagnosis; these models do not have equal false-alarm thresholds')
    fig.savefig(out/'coherent-model-comparison.png',dpi=160);plt.close(fig)
    result=dict(complete=True,focused_ship_certified=False,image=str(image),point_fwhm_range_and_equal_range_m=widths,
        cut_axes_east_north=axes,cut_spacing_m=.25,local_intensity_nyquist_spacing_east_north_m=sampling,
        observed_calibrated_band_phase_at_anchor_deg=observed_phase,point_calibrated_band_phase_at_anchor_deg=point_phase,
        positive_amplitude_coherent_to_independent_at_anchor=dict(observed=float(observed_ratio),point=float(point_ratio)),
        source_waveform_checks=meta['source_checks'],native_score_checks=meta['score_checks'],information_checks=meta['information_checks'])
    result['strongest_noncentral_range_cut_peak']=dict(distance_m=float(distance[adjacent]),power_relative_to_anchor_db=ambiguity_db)
    result['native_anchor_transfer']=json.loads((out/'anchor-transfer.json').read_text()) if (out/'anchor-transfer.json').exists() else None
    np.savez(out/'cuts.npz',distance_m=distance,axes_east_north=axes,point_power=s,observed_power=o,information=info)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    rows=''.join(f'<tr><td>{name}</td><td>{w[0]:.2f}</td><td>{w[1]:.2f}</td></tr>' for name,w in zip(names,widths))
    max_gpu=max(v['gpu_error_relative_to_raw_information'] for v in meta['information_checks'])
    max_native=max(v['representative_native_error_relative_to_raw_information'] for v in meta['information_checks'])
    max_score=max(v['native_score_relative_error'] for v in meta['score_checks'])
    max_source_phase=max(abs(v['constant_phase_error_deg']) for v in meta['source_checks'])
    (out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Projected ship-image validation</title>
<style>body{{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:0 24px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:left}}.note{{padding:16px;background:#eef6fa}}</style>
<h1>Noise-corrected ship image versus an independently synthesised point</h1>
<p class="note">The remaining north-northwest feature must be judged against the point response below, not its ship-like outline. The reference includes the actual fitted visibility, original symbol epochs, full processing history and stationary-clutter projection. A coherent ship image across the capture remains unproved; these controls establish the image statistic and its actual resolution before further motion fitting.</p>
<h2>Measured scene, point expectation and residual</h2><img src="measured-point-residual.png">
<p>Cyan marks the AIS antenna prior. The synthetic waveform is generated from decoded DVB-T, propagated along the physical fitted trajectory and passed through the finite production compressor. No point is inserted into the measured image. Both images use the same original rows, weights, kernels, physical coordinates, band calibration and covariance. The residual subtracts the training-fitted point prediction in the complex domain, before taking power; it is not an image-level contrast trick.</p>
<h2>Resolution from direct physical cuts</h2><img src="point-spread-cuts.png">
<table><tr><th>Band</th><th>Point power FWHM, range direction (m)</th><th>Equal-range direction (m)</th></tr>{rows}</table>
<p>The point's RF bands are phase calibrated without registration. Its calibrated phases at the physical origin are {', '.join(f'{q:+.3f}°' for q in point_phase)}. The direct quarter-metre cuts query the immutable GPU model rather than interpolating image pixels. Width is measured around the known point, using the first half-power crossings; it is not a motion/localisation CRLB. The real scene may contain additional reflectors, changing scattering and residual clutter.</p>
<p><strong>The narrow central width does not imply an unambiguous three-metre range response.</strong> The sparse RF bands produce a noncentral range-cut peak at {distance[adjacent]:+.2f} m with power {ambiguity_db:.2f} dB relative to the anchor. These strong ambiguity lobes must remain in the forward model when separating reflectors. The equal-range cut uses the instantaneous geometry at the nominal midpoint; it is not a full two-dimensional resolution contour.</p>
<h2>Correct covariance and interference projection</h2>
<p>In whitened coordinates, model y = αh + Db + n with n a proper complex unit Gaussian. D is a fixed constant-profile nuisance basis; P = I−D(DᴴD)<sup>−1</sup>Dᴴ. Profiling α and b gives T = |hᴴPy|²/(hᴴPh). The denominator is the projected template information, including the uncertainty introduced by eliminating the stationary nuisance. It is not an estimated local floor, fitted colour normalization or Doppler notch. Under this declared known covariance and fixed template, T has a unit-mean exponential noise-only distribution. Unknown real-data interference covariance and uncertainty in the fitted motion/calibration remain outside that conditional claim.</p>
<p>This is the rank-one interference-rejecting matched-subspace construction described by <a href="https://mountainscholar.org/items/06a68de8-560b-47ac-93a8-c9839b17aab3">Scharf and Friedlander, Matched Subspace Detectors (1994)</a>. Its optimality applies to the stated model, not arbitrary extended ship scattering or a globally unknown motion trajectory.</p>
<p>The GPU computes data contractions and template information together, using the same phase, known Chebyshev pulse and covariance. Native weighted bin moments retain carrier averaging and put real visibility before averaging. The largest checked GPU/CPU information discrepancy is {max_gpu:.3g} relative to raw information; the largest approximation/native discrepancy is {max_native:.3g}. Native complex-score discrepancy is at most {max_score:.3g}. Source-waveform phase discrepancy is below {max_source_phase:.4f}°. No fitted constant phase alignment is removed from that check.</p>
<h2>Spatial sampling and band combination</h2><img src="coherent-model-comparison.png">
<p>The locally sampled physical wavenumber support, including the 6.65625 MHz occupied bandwidth and the aperture's physical rotation, gives intensity sampling limits of approximately {sampling[0]:.2f} m east and {sampling[1]:.2f} m north over this viewport. The one-metre raster resolves those fringes; the earlier 2–4 m diagnostic rasters can undersample them. This is a raster criterion, not extra physical resolution gained by adding pixels.</p>
<p>The positive-amplitude GLRT keeps the instrument-band phases fixed but permits different real nonnegative brightness in each band and one common reflector phase per pixel. The independent-band comparator additionally permits arbitrary phases. Their noise-only distributions differ, so higher raw likelihood is not automatically an SNR or detection gain. At the fitted physical anchor, the positive-amplitude/independent ratio is {observed_ratio:.2%} for the measured data and {point_ratio:.2%} for the point control.</p>
<p><a href="anchor-transfer.html">Native-symbol excluded-data phase transfer and visibility support</a> reports the separate training/excluded checks. The nominal 30-second interval must not be mistaken for a uniform 30-second signal-bearing aperture.</p>
<p><a href="{Path('../')/image.name/'projected-image.png'}">Five individual bands and coherent template image</a> · <a href="analysis.json">All numerical checks</a> · <a href="cuts.npz">Physical cuts</a> · <a href="../motion-controls-review/index.html">Previous stationary-ridge analysis</a>. Captured data and production processing are unchanged by this support validation.</p></html>''')
    print(json.dumps(serial(dict(widths_m=widths,sampling_limits_m=sampling,coherence_ratio=result['positive_amplitude_coherent_to_independent_at_anchor']))),flush=True)


if __name__=='__main__':main()

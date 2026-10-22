"""Render actual ship images and test a frozen interband registration.

All five bands use one physical north/east grid. Complex baseband envelopes
are interpolated before their different spatial carriers are restored to a
common carrier. A single complex gain per band is fitted at one training-only
reference pixel, then frozen over the whole image and all unused epochs.
"""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


def resample_envelope(value,x,y,xx,yy):
    coordinate=np.asarray([(yy-y[0])/(y[1]-y[0]),(xx-x[0])/(x[1]-x[0])])
    z=map_coordinates(value.real,coordinate,order=3,mode='nearest')
    z=z+1j*map_coordinates(value.imag,coordinate,order=3,mode='nearest')
    valid=(xx>=x[0])&(xx<=x[-1])&(yy>=y[0])&(yy<=y[-1])
    return np.where(valid,z,np.nan)


def common_carrier(envelope,carrier,reference,xy):
    """Change only the declared image-envelope carrier, in radians/metre."""
    return envelope*np.exp(1j*np.einsum('i,i...->...',np.asarray(carrier)-reference,xy))


def combine(numerator,information,gains):
    """Independent whitened bands; gains are fixed before seeing these data."""
    scale=np.sqrt(np.sum(abs(gains)**2*information))
    return np.einsum('c,c...->...',gains.conj(),numerator)/scale


def db(z):return 10*np.log10(np.maximum(abs(z)**2,1e-30))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--pixel-m',type=float,default=.5)
    args=parser.parse_args();out=args.directory
    source=json.loads((out/'analysis.json').read_text())
    prior=json.loads(Path('docs/data/calibration_20261016_sites/geometry-prior.json').read_text())
    rotation=np.asarray(prior['ecef_to_enu']);basis=np.asarray(source['basis_x_along_up_ecef'])
    local=rotation@basis
    east=np.arange(-250.,250.01,args.pixel_m);north=east.copy();ee,nn=np.meshgrid(east,north)
    xy=np.einsum('ij,j...->i...',local[:2,:2].T,np.asarray([ee,nn]))
    arrays=[np.load(out/f'channel-{c}.npz') for c in range(5)]
    carriers=np.asarray([a['carrier_rad_per_m'] for a in arrays]);reference=carriers[2]
    keys=['stationary_profiled','training','unused','quadratic_only','unit_point_response','reference']
    images={key:np.asarray([common_carrier(resample_envelope(a[key],a['x_m'],a['y_m'],*xy),
        carriers[c],reference,xy) for c,a in enumerate(arrays)]) for key in keys}
    info={key:np.asarray([float(a[key]) for a in arrays]) for key in
        ('origin_information','training_information','unused_information')}
    train_power=np.sum(abs(images['training'])**2/info['training_information'][:,None,None],axis=0)
    # AIS supplies only this broad association region; it supplies no fitted
    # phase or hull outline. Selection and all five gains use training only.
    roi=(ee**2+nn**2<=150.**2)&np.isfinite(train_power)
    anchor=np.unravel_index(np.argmax(np.where(roi,train_power,-np.inf)),ee.shape)
    gains=images['training'][:,anchor[0],anchor[1]]/info['training_information']
    gains/=np.max(abs(gains))
    combined={key:combine(images[key],info[ikey],gains) for key,ikey in
        [('stationary_profiled','origin_information'),('training','training_information'),
         ('unused','unused_information'),('quadratic_only','origin_information')]}
    unregistered=combine(images['unused'],info['unused_information'],np.ones(5))
    norms=info['origin_information']
    singles=images['stationary_profiled']/np.sqrt(norms[:,None,None])
    singles_test=images['unused']/np.sqrt(info['unused_information'][:,None,None])
    panels=np.concatenate([singles,combined['stationary_profiled'][None]])
    panels_test=np.concatenate([singles_test,combined['unused'][None]])
    finite=np.isfinite(db(panels));top=float(np.max(db(panels)[finite]));limits=(top-40,top)
    extent=[east[0],east[-1],north[0],north[-1]]
    names=[r['name'] for r in source['channels']]+['Coherent combination · frozen training gains']

    def decorate(ax):
        ax.plot(0,0,'+',color='#00ffff',ms=9,mew=1.4,label='AIS antenna prior')
        ax.plot(ee[anchor],nn[anchor],'o',mec='#ff35ff',mfc='none',ms=7,mew=1.2,label='Training registration point')
        ax.set(xlabel='East of AIS prior at 164 s (m)',ylabel='North (m)')

    def render_six(values,path,title,vlim=limits,relative=False):
        fig,axes=plt.subplots(2,3,figsize=(16,10),layout='constrained',sharex=True,sharey=True)
        for ax,z,name in zip(axes.ravel(),values,names):
            power=db(z)
            if relative:power-=np.nanmax(power)
            im=ax.imshow(power,extent=extent,origin='lower',cmap='inferno',vmin=vlim[0],vmax=vlim[1],interpolation='nearest')
            ax.set_title(name);decorate(ax)
        axes.flat[0].legend(loc='upper right',fontsize=7)
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Power relative to panel maximum (dB)' if relative else 'Conditional matched power (dB; origin information)',shrink=.8)
        fig.suptitle(title)
        fig.savefig(out/path,dpi=150);plt.close(fig)

    render_six(panels,'01-channels-and-combined.png','Actual recording · 149–179 s · north up\nQuadratic range/phase compensation + higher-order phase fit; static profile treated as a nuisance')
    render_six(panels_test,'02-unused-epochs.png','Unused recording epochs · frozen motion, phase correction and interband gains\nThe fitting epochs are excluded; same fixed colour scale as the full aperture')
    for c,z in enumerate(panels):
        fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
        im=ax.imshow(db(z),extent=extent,origin='lower',cmap='inferno',vmin=limits[0],vmax=limits[1],interpolation='nearest')
        decorate(ax);ax.legend(loc='upper right',fontsize=8);ax.set_title(names[c]+' · actual 30 s aperture · north up')
        fig.colorbar(im,ax=ax,label='Conditional matched power (dB; origin information)')
        fig.savefig(out/('coherent.png' if c==5 else f'channel-{c}.png'),dpi=150);plt.close(fig)
    # A known point goes through the complete archived waveform/row operator.
    # Give it the fitted band response, then apply exactly the frozen combiner.
    point=images['unit_point_response']
    point_combined=combine(point*gains[:,None,None],norms,gains)
    render_six(np.concatenate([point/np.sqrt(norms[:,None,None]),point_combined[None]]),
        '03-known-point-response.png','Known point through the same waveform, geometry and 30 s aperture\nEach panel scaled to its own peak; these are attainable ambiguity functions, not ship observations',(-40,0),True)
    before=images['quadratic_only']/np.sqrt(norms[:,None,None])
    fig,axes=plt.subplots(2,5,figsize=(19,8.5),layout='constrained',sharex=True,sharey=True)
    for row,values,label in [(0,before,'Quadratic only'),(1,singles,'+ Higher-order phase fit')]:
        for c,ax in enumerate(axes[row]):
            im=ax.imshow(db(values[c]),extent=extent,origin='lower',cmap='inferno',vmin=limits[0],vmax=limits[1],interpolation='nearest')
            ax.set_title(names[c]+' · '+label,fontsize=10);decorate(ax)
    fig.colorbar(im,ax=axes.ravel().tolist(),label='Conditional matched power (dB; same fixed scale)',shrink=.75)
    fig.suptitle('Does the higher-order phase correction actually focus the recording?\nIdentical samples, fixed geometry and no fitted yaw')
    fig.savefig(out/'04-motion-comparison.png',dpi=150);plt.close(fig)

    z_train=images['training'][:,anchor[0],anchor[1]]
    z_test=images['unused'][:,anchor[0],anchor[1]]
    response=z_test/info['unused_information']
    phase_error=np.angle(response*gains.conj());phase_error-=phase_error[2];phase_error=np.angle(np.exp(1j*phase_error))
    independent=float(np.sum(abs(z_test)**2/info['unused_information']))
    coherent=float(abs(combined['unused'][anchor])**2)
    best=float(np.max(abs(z_test)**2/info['unused_information']))
    efficiency=coherent/independent
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].bar(names[:5],np.degrees(phase_error));axes[0].set(ylabel='Unused relative phase error (degrees)',title='Registration predicted from training; Nine reference',ylim=(-180,180));axes[0].axhline(0,color='black',lw=.6)
    powers=[best,float(abs(unregistered[anchor])**2),coherent,independent]
    axes[1].bar(['Best band','No registration','Frozen gains','Energy bound'],10*np.log10(np.maximum(powers,1e-30)))
    axes[1].set(ylabel='Conditional matched power (dB)',title='Unused data at the frozen registration point');axes[1].tick_params(axis='x',labelsize=9)
    fig.savefig(out/'05-interband-validation.png',dpi=150);plt.close(fig)
    records=[]
    for c in range(5):
        records.append(dict(name=names[c],rf_hz=source['channels'][c]['rf_hz'],
            fitted_gain_real_imag=[gains[c].real,gains[c].imag],unused_relative_phase_error_degrees=float(np.degrees(phase_error[c])),
            unused_matched_power_db=float(db(z_test[c]/np.sqrt(info['unused_information'][c]))),
            before_phase_fit_anchor_power_db=float(db(before[c][anchor])),after_phase_fit_anchor_power_db=float(db(singles[c][anchor])),
            waveform_operator_snr_loss_db=[t['matched_snr_loss_db'] for t in source['channels'][c]['checks']]))
    result=dict(interval_s=source['interval_s'],names=names,colour_scale_db=limits,spatial_pixel_m=args.pixel_m,
        coordinate_frame='Fixed north/east axes relative to the AIS antenna prior at 164 s; 20 m above the AIS reference; no yaw fit',
        reference_carrier_rad_per_m=reference.tolist(),native_to_east_north=local[:2,:2].tolist(),
        registration_point_east_north_m=[float(ee[anchor]),float(nn[anchor])],association_region_m=150,
        training_epochs=[r['training_epochs'] for r in source['channels']],unused_epochs=[r['unused_epochs'] for r in source['channels']],
        unused_combined_vs_best_band_db=float(10*np.log10(coherent/best)),unused_combined_to_independent_energy=efficiency,
        unused_loss_to_known_response_combination_db=float(-10*np.log10(efficiency)),
        channels=records,phase_centres_calibrated=False,common_cartesian_motion_validated=False,
        limitations=['Per-band range/phase corrections are independent, not a common fitted Cartesian motion.',
            'Constant gains are trained at one common pixel; that does not prove a shared resolved scatterer or antenna calibration.',
            'The unused epochs are interleaved within this aperture, not a new time interval.',
            'Power uses the origin template information for a fixed per-band display scale; it is not an exact pixelwise SNR map.',
            'No adaptive image normalization, pixelwise phase alignment, fitted yaw or Doppler notch is used.'])
    (out/'image-results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(out/'north-up-complex-images.npz',east_m=east,north_m=north,
        individual_common_carrier_numerator=images['stationary_profiled'],combined=combined['stationary_profiled'],
        unused_combined=combined['unused'],gains=gains,reference_carrier_rad_per_m=reference,
        origin_information=norms,unused_information=info['unused_information'])
    rows=''.join(f'<tr><td>{r["name"]}</td><td>{r["unused_relative_phase_error_degrees"]:+.2f}°</td><td>{r["after_phase_fit_anchor_power_db"]-r["before_phase_fit_anchor_power_db"]:+.2f}</td><td>{min(r["waveform_operator_snr_loss_db"]):.3f}–{max(r["waveform_operator_snr_loss_db"]):.3f}</td></tr>' for r in records)
    links=' · '.join(f'<a href="channel-{c}.png">{names[c]}</a>' for c in range(5))+' · <a href="coherent.png">Coherent combination</a>'
    body=f'''<!doctype html><meta charset="utf-8"><title>Ship image formation — 17 October 2026</title>
<style>body{{max-width:1300px;margin:32px auto;font:17px/1.5 system-ui;padding:0 20px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{padding:8px;border:1px solid #bbb}}.note{{background:#fff1d0;padding:15px}}</style>
<h1>Actual ship image formation: independent channels and coherent sum</h1>
<p class="note">These are actual recorded-data images, not a rendered sparse scene. Broad ambiguity ridges remain: this is not yet a resolved ship hull or validated five-channel geographic calibration. The fixed interband registration reaches {100*efficiency:.1f}% of the independent-band energy bound on unused epochs at its reference pixel ({-10*np.log10(efficiency):.2f} dB loss), with {10*np.log10(coherent/best):+.2f} dB matched power relative to the best band there. That is a conditional point test, not proof of coherent hull imaging.</p>
<p>{links}</p><p>30 s aperture, capture time 149–179 s, north up, no fitted yaw. Cyan: AIS antenna prior at the midpoint. Magenta: reference pixel selected using fitting epochs only, within 150 m of that prior. All panels use one fixed colour scale. No hull outline is imposed.</p>
<a href="01-channels-and-combined.png"><img src="01-channels-and-combined.png"></a>
<h2>Unused-data check</h2><p>The exact epochs used by the earlier motion/phase fit are excluded. One complex gain per band is estimated at the training reference pixel and frozen across every pixel and unused epoch. Complex envelopes are interpolated before restoring the band-dependent spatial carriers to one common carrier. The sixth image sums complex values before taking power; it is not an incoherent magnitude overlay.</p>
<a href="02-unused-epochs.png"><img src="02-unused-epochs.png"></a><img src="05-interband-validation.png">
<h2>Motion compensation and achievable resolution</h2><p>Each band retains the recursively followed quadratic range/phase trajectory plus its previously fitted smooth higher-order phase correction. The comparison holds geometry, original observations and colour scale fixed. It does not silently turn these separate corrections into a common physical ship trajectory.</p>
<a href="04-motion-comparison.png"><img src="04-motion-comparison.png"></a>
<p>The known point is independently generated through every original native row, known waveform pulse and C0/C1 observation response, then imaged with the same operator. Its ambiguity ridges show the aperture’s localization limits for this straight, translating target reference. Each point-response panel uses a relative scale; the ship panels do not.</p>
<a href="03-known-point-response.png"><img src="03-known-point-response.png"></a>
<table><tr><th>Band</th><th>Unused interband phase error</th><th>Higher-order phase gain at reference (dB)</th><th>Operator loss at three test pixels (dB)</th></tr>{rows}</table>
<h2>Scope and reproducibility</h2><p>Antennas use the current Ornata/Ridge trial assignment. Transmitter phase centres remain uncalibrated. Independent per-band corrections are not yet a validated common Cartesian trajectory. The static return is an explicit constant complex range-profile nuisance, removed by an orthogonal projection in the unchanged whitened metric; no Doppler interval is masked. Separate fitting and unused epoch sets receive separate fixed nuisance projections. Template information at the origin sets each band’s display scale; no local image-driven power normalization is applied.</p>
<p>The spectral reference imager uses exact two-leg geometry. The Stolt image and full original-data sensitivity checks are saved alongside it. The displayed images use the numerical spectral reference, keeping Stolt approximation out of the imaging conclusion.</p>
<p><a href="image-results.json">Image measurements and limitations</a> · <a href="analysis.json">Observation and operator provenance</a> · <a href="north-up-complex-images.npz">Complex north-up images</a> · <a href="../quadratic-motion-kernel-recursive/index.html">Motion/phase estimation audit</a></p>'''
    (out/'index.html').write_text(body)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

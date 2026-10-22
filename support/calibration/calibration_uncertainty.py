"""Propagate joint site/phase covariance to conditional imaging coherence.

This is a local Gaussian design diagnostic, not a measured SNR loss or a global
posterior. It includes site/channel-phase correlations and compares the linear
phase calculation against nonlinear geometry under the same Gaussian draws.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from geometry import C, RF, ecef, unit
from terrain import RasterTerrain


def coherent_fraction(derivative, covariance):
    """Expected equal-band coherent power / perfectly calibrated power."""
    n = derivative.shape[-2]
    result = np.full(derivative.shape[:-2], float(n))
    for i in range(n):
        for j in range(i+1,n):
            contrast = derivative[...,i,:]-derivative[...,j,:]
            variance = np.einsum('...i,ij,...j->...',contrast,covariance,contrast)
            if np.any(variance < -1e-8):
                raise ValueError('Negative phase-contrast variance')
            result += 2*np.exp(-np.maximum(variance,0)/2)
    return result/n**2


def phase_derivative(x, sites, assignment, offsets):
    derivative = np.zeros((len(x),5,13))
    lengths = []
    for ch,site in enumerate(assignment):
        tx = sites[site]+offsets[ch]
        lengths.append(np.linalg.norm(x-tx,axis=1)+np.linalg.norm(x,axis=1)-np.linalg.norm(tx))
        derivative[:,ch,3*site:3*site+3] = -2*np.pi*RF[ch]/C*(-unit(x-tx)-unit(tx))*[30,30,75]
    for column,ch in enumerate([0,2,3,4]):
        derivative[:,ch,9+column] = 1.
    return derivative, np.array(lengths)


def nonlinear_check(x,sites,assignment,offsets,covariance,expected,draws=100000):
    rng = np.random.default_rng(2754)
    perturb = rng.normal(size=(draws,13))@np.linalg.cholesky(covariance).T
    trial = sites[None,:,:]+perturb[:,:9].reshape(draws,3,3)*[30,30,75]
    beta = np.zeros((draws,5));beta[:,[0,2,3,4]] = perturb[:,9:]
    result = []
    for point,power in zip(x,expected):
        error = np.zeros((draws,5))
        for ch,site in enumerate(assignment):
            tx = sites[site]+offsets[ch]; tt = trial[:,site]+offsets[ch]
            reference = np.linalg.norm(point-tx)-np.linalg.norm(tx)
            change = np.linalg.norm(point-tt,axis=1)-np.linalg.norm(tt,axis=1)-reference
            error[:,ch] = beta[:,ch]-2*np.pi*RF[ch]/C*change
        samples = abs(np.exp(1j*error).sum(axis=1))**2/25
        mean = float(samples.mean())
        result.append({'point_enu_m':point.tolist(),'linear_expected_power':float(power),
            'nonlinear_monte_carlo_power':mean,'monte_carlo_standard_error':float(samples.std(ddof=1)/np.sqrt(draws)),
            'loss_difference_db':float(10*np.log10(mean/power))})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['data','terrain','dem']:
        p.add_argument('--'+key,type=Path,required=True)
    args = p.parse_args()
    saved = json.loads((args.data/'calibration-predictive.json').read_text())
    prior = json.loads((args.data/'geometry-prior.json').read_text())
    terrain = RasterTerrain(args.terrain,prior,args.dem)
    longitude,latitude = np.linspace(144.755,145.595,256),np.linspace(-38.095,-37.535,224)
    lon,lat = np.meshgrid(longitude,latitude)
    height = terrain.elevation(lat,lon)
    x = ((ecef(lat,lon,height+terrain.undulation(lat,lon))-np.array(prior['receiver_ecef_m']))@
         np.array(prior['ecef_to_enu']).T).reshape(-1,3)
    sites = np.array(saved['site_positions_enu_m'])+np.array(saved['transmitter_corrections_enu_m'])
    offsets,covariance = np.array(saved['channel_offsets_enu_m']),np.array(saved['posterior_covariance'])
    derivative,lengths = phase_derivative(x,sites,saved['channel_sites'],offsets)
    valid = np.isfinite(x).all(axis=1)&np.all((lengths>=600)&(lengths<=30000),axis=0)
    power = coherent_fraction(derivative,covariance)
    loss = -10*np.log10(power)
    quantiles = [0,1,5,25,50,75,95,99,100]
    order = np.flatnonzero(valid)[np.argsort(loss[valid])]
    selected = order[np.round(np.linspace(0,len(order)-1,7)).astype(int)]
    check = nonlinear_check(x[selected],sites,saved['channel_sites'],offsets,covariance,power[selected])
    report = {'status':'conditional Gaussian calibration-risk diagnostic; not an accepted calibration or a measured loss',
        'assumptions':'Equal band gains/noise, common non-dispersive scatterer, fixed physical image point and receiver, local single-branch fitted covariance with 10 m site priors. Target and gain nuisance parameters eliminated. No receiver-position, terrain, scattering or alternate-branch uncertainty is added.',
        'equation':'E[|sum exp(j delta_theta_i)|^2]/N^2 = (N + 2 sum_(i<j) exp(-Var(delta_theta_i-delta_theta_j)/2))/N^2',
        'range_support_m':[600,30000],'quantiles_percent':quantiles,
        'calibration_loss_db_quantiles':np.percentile(loss[valid],quantiles).tolist(),
        'grid_fraction_below_1db_loss':float(np.mean(loss[valid]<1)),
        'nonlinear_geometry_checks':check,'monte_carlo_draws':100000}
    (args.data/'calibration-imaging-risk.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(10,7.7),layout='constrained')
    image=ax.imshow(np.where(valid.reshape(lat.shape),loss.reshape(lat.shape),np.nan),origin='lower',
        extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]],cmap='magma',vmin=0,vmax=10*np.log10(5),
        aspect=1/np.cos(np.deg2rad(-37.8)))
    ax.contour(longitude,latitude,height,levels=[50,100,300,500,700],colors='#a3a6aa',linewidths=.35,alpha=.5)
    import matplotlib.patheffects as effects
    style={'color':'#17212d','path_effects':[effects.withStroke(linewidth=2,foreground='white')]}
    rx=prior['receiver'];ax.plot(rx['longitude_deg'],rx['latitude_deg'],'+',color='#17212d',markersize=8)
    ax.annotate('Receiver',(rx['longitude_deg'],rx['latitude_deg']),xytext=(5,6),textcoords='offset points',**style)
    ax.plot(145.34661,-37.837476,'^',color='#17212d',markersize=5)
    ax.annotate('Mt Dandenong',(145.34661,-37.837476),xytext=(5,6),textcoords='offset points',**style)
    ax.set(xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)',
        title='Conditional calibration uncertainty in coherent imaging\nLocal Gaussian model · equal band weights · fixed image points')
    fig.colorbar(image,ax=ax,label='Expected coherent-power loss from calibration uncertainty (dB)')
    fig.savefig(args.data/'calibration-imaging-risk.png',dpi=160);plt.close(fig)
    print('Loss quantiles',report['calibration_loss_db_quantiles'])
    print('Maximum nonlinear check difference dB',max(abs(x['loss_difference_db']) for x in check))


if __name__ == '__main__':
    main()

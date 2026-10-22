"""Separate calibration-induced geographic displacement from coherent defocus.

A fixed-world-pixel phase loss is not a peak-focus loss. The local image point
has two ground coordinates, and a common phase does not affect image power.
This diagnostic projects those directions explicitly, reports the displacement
uncertainty, and keeps the physical antenna covariance visible. It is not a
replacement for withheld-target validation or an assertion of surveyed sites.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from calibration_uncertainty import phase_derivative,coherent_fraction
from coherent_fit import CoherentProblem
from fit_geometry import observations
from geographic_io import read_profiles
from geometry import C,RF,ecef,unit
from phase_fit import load_histories
from range_fit import load_fields
from road_geometry import RoadCurve
from road_model import RoadTrajectory,RoadProblem
from terrain import RasterTerrain
from range_fit import Pulse
from scipy.optimize import least_squares


def calibration_covariance(problem,state):
    expected=copy.copy(problem);expected.data=problem.fitted_observations(state)
    j=expected.residual(state,True,True,True)
    selected=np.r_[np.arange(9),np.arange(problem.beta_offset,problem.magnitude_offset)]
    nuisance=np.delete(j,selected,axis=1)
    u,s,_=np.linalg.svd(nuisance,full_matrices=False);u=u[:,s>s[0]*1e-10]
    calibrated=j[:,selected]-u@(u.T@j[:,selected])
    information=calibrated.T@calibrated
    covariance=np.linalg.inv(information)
    covariance=(covariance+covariance.T)/2
    np.linalg.cholesky(covariance)
    return covariance,np.linalg.eigvalsh(information)


def ground_phase_jacobian(x,sites,assignment,offsets,ground_gradient):
    jacobian=np.empty((len(x),5,2))
    slopes=-ground_gradient[:,:2]/ground_gradient[:,2,None]
    for ch,site in enumerate(assignment):
        gradient=-2*np.pi*RF[ch]/C*(unit(x-sites[site]-offsets[ch])+unit(x))
        jacobian[:,ch]=gradient[:,:2]+gradient[:,2,None]*slopes
    return jacobian


def image_point_projection(derivative,ground_jacobian,covariance):
    # Local least-squares phase registration with one irrelevant common phase.
    design=np.concatenate([np.ones((*ground_jacobian.shape[:-1],1)),ground_jacobian],axis=-1)
    inverse=np.linalg.pinv(design,rcond=1e-10)
    shift=-inverse@derivative
    remainder=derivative+design@shift
    displacement=shift[:,1:]
    location_covariance=displacement@covariance@np.swapaxes(displacement,-1,-2)
    singular=np.linalg.svd(design,compute_uv=False)
    return remainder,displacement,location_covariance,singular


def ground_point(xy,height,terrain):
    point=np.r_[xy,height][None,:]
    for _ in range(3):
        clearance,gradient=terrain.clearance(point,True)
        point[:,2]-=clearance/gradient[:,2]
    _,gradient=terrain.clearance(point,True)
    if not np.isfinite(point).all() or not np.isfinite(gradient).all():
        raise ValueError('Registered point outside valid terrain support')
    return point,gradient


def nonlinear_registration(points,sites,assignment,offsets,covariance,terrain,pulses,draws):
    """Follow the local geographic branch through finite calibration changes.

    Exact bistatic distances and terrain replace the tangent-plane prediction.
    The waveform envelope is evaluated after phase registration; no extra
    frequency-domain phase correction is fitted to each realization.
    """
    perturb=np.random.default_rng(87523).normal(size=(draws,13))@np.linalg.cholesky(covariance).T
    results=[];k=2*np.pi*RF/C
    for point in points:
        _,base=phase_derivative(point[None,:],sites,assignment,offsets);base=base[:,0]
        samples=[];failures=[]
        for delta in perturb:
            beta=np.zeros(5);beta[[0,2,3,4]]=delta[9:]
            position=point[:2].copy()
            try:
                for fraction in np.linspace(.125,1.,8):
                    trial=sites+fraction*delta[:9].reshape(3,3)*[30,30,75]
                    def residual(xy,jacobian=False):
                        x,gradient=ground_point(xy,point[2],terrain)
                        if jacobian:
                            h=ground_phase_jacobian(x,trial,assignment,offsets,gradient)[0]
                            return h-h.mean(axis=0)
                        _,length=phase_derivative(x,trial,assignment,offsets)
                        phase=fraction*beta-k*(length[:,0]-base)
                        return phase-phase.mean()
                    fit=least_squares(residual,position,jac=lambda q:residual(q,True),
                        x_scale=100.,max_nfev=80,ftol=1e-10,xtol=1e-8,gtol=1e-7)
                    if not fit.success:raise ValueError('Local geographic continuation did not converge')
                    position=fit.x
                x,_=ground_point(position,point[2],terrain)
                _,length=phase_derivative(x,trial,assignment,offsets);change=length[:,0]-base
                phase=beta-k*change
                envelope=np.array([pulse(change[ch]*8e6/C)/pulse(0) for ch,pulse in enumerate(pulses)])
                phase_power=float(abs(np.exp(-1j*phase).mean())**2)
                total_power=float(abs((envelope*np.exp(-1j*phase)).mean())**2)
                samples.append({'displacement_m':float(np.linalg.norm(position-point[:2])),
                    'phase_power_fraction':phase_power,'waveform_power_fraction':total_power,
                    'range_error_m':change.tolist()})
            except ValueError as error:
                failures.append(str(error))
        result={'point_enu_m':point.tolist(),'draws':draws,'successful_draws':len(samples),
            'failure_reasons':{reason:failures.count(reason) for reason in sorted(set(failures))},
            'samples':samples}
        if samples:
            for key in ('displacement_m','phase_power_fraction','waveform_power_fraction'):
                values=np.array([s[key] for s in samples])
                result[key+'_quantiles_5_50_95']=np.percentile(values,[5,50,95]).tolist()
                result[key+'_mean']=float(values.mean())
        results.append(result)
        print('Nonlinear registration',np.round(point[:2],1).tolist(),len(samples),'/',draws,
            'mean waveform power',result.get('waveform_power_fraction_mean'),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','prior','terrain','dem','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--draws',type=int,default=0,help='Nonlinear geometry/actual-waveform checks at six geographic points')
    args=parser.parse_args();saved=json.loads(args.fit.read_text())
    prior=json.loads(args.prior.read_text());terrain=RasterTerrain(args.terrain,prior,args.dem)
    sites=np.array(saved['site_positions_enu_m']);offsets=np.array(saved['channel_offsets_enu_m'])
    routes=json.loads(args.routes.read_text())['routes'];_,records=read_profiles(args.templates)
    problems=[]
    for label,coarse,selected in zip(saved['labels'],observations(args.histories,saved['labels']),saved['selected_roads']):
        channels,description=load_histories(args.histories,label,1.)
        model=RoadTrajectory(channels[1]['seed'],RoadCurve(routes[selected['route']]),selected['coarse_state'],saved['assignment'],offsets)
        problems.append(RoadProblem(channels,coarse,sites,load_fields(args.patches,label,description,records),model))
    problem=CoherentProblem(problems,site_prior_sigma=np.array(saved['site_prior_sigma_m']))
    state=np.array(saved['state']);covariance,eigenvalues=calibration_covariance(problem,state)
    sites=sites+np.array(saved['site_corrections_enu_m'])
    longitude=np.linspace(144.76,145.59,192);latitude=np.linspace(-38.09,-37.54,160)
    lon,lat=np.meshgrid(longitude,latitude);height=terrain.elevation(lat,lon)
    x=((ecef(lat,lon,np.where(np.isfinite(height),height,0)+terrain.undulation(lat,lon))-
        np.array(prior['receiver_ecef_m']))@np.array(prior['ecef_to_enu']).T).reshape(-1,3)
    derivative,lengths=phase_derivative(x,sites,saved['assignment'],offsets)
    valid=np.isfinite(height).ravel()&np.all((lengths>=600)&(lengths<=30000),axis=0)
    _,gradient=terrain.clearance(x,True)
    h=ground_phase_jacobian(x,sites,saved['assignment'],offsets,gradient)
    valid&=np.isfinite(h).all(axis=(1,2))
    remaining=np.full_like(derivative,np.nan)
    shift=np.full((len(x),2,derivative.shape[-1]),np.nan)
    location_covariance=np.full((len(x),2,2),np.nan)
    remaining[valid],shift[valid],location_covariance[valid],singular=image_point_projection(
        derivative[valid],h[valid],covariance)
    fixed=-10*np.log10(coherent_fraction(derivative,covariance))
    registered=-10*np.log10(coherent_fraction(remaining,covariance))
    radii=np.full(len(x),np.nan)
    radii[valid]=np.sqrt(5.991464547*np.maximum(np.linalg.eigvalsh(location_covariance[valid])[:,-1],0))
    quantiles=[0,5,25,50,75,95,100]
    result={'source_fit':str(args.fit),'assignment':saved['assignment'],'labels':saved['labels'],
        'status':'local Gaussian design diagnostic; not calibration acceptance',
        'assumptions':'Conditional on selected roads, point scatterers, site assignment, fixed receiver, terrain and empirical covariance. Includes stated site and road/motion priors. Equal channel weights. Image registration is linear and unconstrained; a large displacement can invalidate the local approximation. Finite range-envelope loss, clutter and alternative phase/road branches are not included here.',
        'parameter_order':'nine normalized antenna corrections (30,30,75 m per site), then four channel phases relative to SBS (radians)',
        'covariance':covariance.tolist(),'calibration_information_eigenvalues':eigenvalues.tolist(),
        'quantiles_percent':quantiles,'fixed_world_pixel_loss_db':np.percentile(fixed[valid],quantiles).tolist(),
        'registered_phase_loss_db':np.percentile(registered[valid],quantiles).tolist(),
        'geographic_95pct_radius_m':np.percentile(radii[valid],quantiles).tolist(),
        'fraction_with_95pct_radius_below_100m':float(np.mean(radii[valid]<100)),
        'correction_to_previous_interpretation':'Fixed-pixel coherent loss includes geographic misregistration and must not be reported as peak defocusing loss.'}
    if args.draws:
        ordered=np.flatnonzero(valid)[np.argsort(radii[valid])]
        selected=ordered[np.round(np.array([.05,.25,.5,.75,.9,.95])*(len(ordered)-1)).astype(int)]
        pulses=[]
        for channel in range(5):
            rows=records[(records['channel']==channel)&(records['information']>0)]
            pulses.append(Pulse(np.average(rows['psf'],axis=0,weights=rows['information'])))
        result['nonlinear_registration']=nonlinear_registration(x[selected],sites,saved['assignment'],offsets,
            covariance,terrain,pulses,args.draws)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(args.output.with_suffix('.npz'),points=x,valid=valid,derivative=derivative,
        ground_jacobian=h,displacement=shift,location_covariance=location_covariance,
        covariance=covariance,remaining=remaining,longitude=longitude,latitude=latitude)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,values,maximum,title,unit_label in zip(axes,(fixed,registered,np.minimum(radii,500)),
            (7,1,500),('At a fixed world pixel','After local ground-position registration','Geographic displacement uncertainty'),
            ('Expected coherent-power loss (dB)','Remaining phase-only loss (dB)','95% ellipse enclosing radius (m; clipped at 500)')):
        image=ax.imshow(np.where(valid,values,np.nan).reshape(lat.shape),origin='lower',
            extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]],cmap='magma',vmin=0,vmax=maximum,
            aspect=1/np.cos(np.deg2rad(-37.8)))
        ax.set(title=title,xlabel='Longitude (degrees east)',ylabel='Latitude (degrees north)')
        fig.colorbar(image,ax=ax,label=unit_label,orientation='horizontal',shrink=.95)
    fig.suptitle('Conditional calibration uncertainty: displacement and defocus are different\nLinear model; road and site priors included; not held-out validation')
    fig.savefig(args.output.with_suffix('.png'),dpi=150);plt.close(fig)
    for key in ('fixed_world_pixel_loss_db','registered_phase_loss_db','geographic_95pct_radius_m'):
        print(key,result[key],flush=True)


if __name__=='__main__':main()

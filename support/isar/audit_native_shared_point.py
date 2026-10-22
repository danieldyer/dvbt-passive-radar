"""Frozen interband calibration transfer at the training-fitted point.

This is an observed coherent/independent matched-power ratio, not a target
SNR bound. Background overlap and uncertainty of the training-estimated
stationary profile are not removed by renormalising the observations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from recorded_injection_fields import load_fields
from fixed_range_field import stationary_profile_design
from fading_point_motion import RadialGeometry,FadingPointArc,FadingPointFit
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--profiles',type=Path,default=Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    p.add_argument('--output',type=Path)
    p.add_argument('--windowed-read',action='store_true',help='Bound memory for a long archive; use its local known-waveform covariance')
    p.add_argument('--range-samples',type=int,nargs=2,help='Inclusive observed offsets; must match the fitted range crop')
    args=p.parse_args();root=args.fit;description=json.loads((root/'analysis.json').read_text())
    if not description['transverse_motion_fixed_to_ais']:raise ValueError('This audit requires the declared radial subspace')
    first,last=description['interval_s'];mid=description['midpoint_s'];parameters=np.asarray(description['fit']['parameters'])
    observed=args.range_samples or description.get('observed_range_samples',[0,24])
    fields,rotation,rx,tx,_,_=load_fields(args.profiles,
        offsets=np.arange(observed[0],observed[1]+1),fixed_epoch_s=mid,interval_s=(first,last) if args.windowed_read else None)
    geometry=RadialGeometry(rx,rotation,tx,np.asarray(description['prior_mean_state_enu']))
    render=json.loads((root/'render-fit.json').read_text());g=np.asarray(render['fits']['constant_velocity_gls']['coefficients_real_imag'])[:,0]
    gains=g[:,0]+1j*g[:,1];scores=[];information=[];selected=[]
    for c,original in enumerate(fields):
        f=original.windowed(first,last);keep=np.floor((f.time-first)/.25).astype(int)%4!=1
        knots=np.r_[f.data['row_time'][0],np.arange(first+1.,last,1.),f.data['row_time'][-1]]
        b=FadingPointArc(f,keep,knots,geometry,c,mid,stationary_profile_design(f))
        _,_,beta=b.original_prediction(parameters);clean=f.y.ravel()-b.design@beta
        raw,_=b.response(parameters)
        phase=parameters[geometry.count+c]+f.k*(geometry.prediction(parameters)[c,0]-b.reference_path)
        h=b.reduced(raw[:,:,None])[:,0]*np.exp(-1j*phase)
        band_scores=[];band_information=[]
        for mask in (keep,~keep):
            use=np.repeat(mask,len(f.offsets));band_scores.append(np.vdot(h[use],clean[use]));band_information.append(np.vdot(h[use],h[use]).real)
        scores.append(band_scores);information.append(band_information)
    scores=np.asarray(scores);information=np.asarray(information)
    for fold,label in enumerate(['training','excluded']):
        s,i=scores[:,fold],information[:,fold]
        independent=np.sum(abs(s)**2/i);coherent=abs(np.vdot(gains,s))**2/np.sum(abs(gains)**2*i)
        selected.append(dict(fold=label,coherent_to_independent_power=float(coherent/independent),
            difference_db=float(10*np.log10(coherent/independent)),
            band_phase_error_deg=np.degrees(np.angle(s/gains)),
            score_real_imag=np.c_[s.real,s.imag],information=i))
    result=dict(scope=__doc__,profiles=str(args.profiles),bounded_read=args.windowed_read,
        geographic_localisation_certified=False,point_selected_on_training=True,
        band_calibration_refit_on_excluded=False,stationary_refit_on_excluded=False,folds=selected)
    (args.output or root/'calibration-transfer.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(serial(result)),flush=True)


if __name__=='__main__':main()

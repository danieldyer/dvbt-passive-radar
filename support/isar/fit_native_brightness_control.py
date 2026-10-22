"""Separate changing real brightness from phase error in a declared range crop.

Constant and positive-spline brightness models use identical training epochs,
fixed observation coordinates, exact native-symbol operators and stationary
nuisance. All acquisition uses the training fold. Phase stays quadratic and
continuous; no complex per-time coefficient or phase screen is allowed.
"""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from amplitude_arc import AmplitudeArc
from coherent_arc import CoherentArc
from fixed_range_field import stationary_profile_design
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES
from scan_native_ship_chirps import acquire
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--first',type=float,default=158.);p.add_argument('--last',type=float,default=164.)
    p.add_argument('--knot-s',type=float,default=1.)
    p.add_argument('--profiles',type=Path,help='Use a bounded context from this native archive')
    p.add_argument('--range-samples',type=int,nargs=2,default=[0,24],help='Inclusive observed cell offsets from the fixed AIS reference')
    p.add_argument('--origin-bounds',type=float,nargs=2,help='Restrict both acquisition and refinement to these sample offsets')
    args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    first,last=args.first,args.last
    origin_bounds=(-1.,25.) if args.origin_bounds is None else tuple(args.origin_bounds)
    if not origin_bounds[0]<origin_bounds[1] or args.range_samples[0]>=args.range_samples[1]:raise ValueError('Ordered range bounds are required')
    fields,*_=load_fields(args.profiles or Path('build/injection-20261018/deterministic-baseline/profiles.bin'),
        offsets=np.arange(args.range_samples[0],args.range_samples[1]+1),fixed_epoch_s=(first+last)/2,
        interval_s=(first,last) if args.profiles else None)
    result=dict(scope=__doc__,interval_s=[first,last],knot_s=args.knot_s,
        training='Quarter-second blocks: ordinal modulo four differs from one',channels=[],complete=False,
        origin_bounds_samples=origin_bounds,observed_range_samples=args.range_samples,
        profiles=str(args.profiles or Path('build/injection-20261018/deterministic-baseline/profiles.bin')))
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(5,2,figsize=(13,14),layout='constrained')
    for c,original in enumerate(fields):
        f=original.windowed(first,last);keep=np.floor((f.time-first)/.25).astype(int)%4!=1
        design=stationary_profile_design(f);constant=CoherentArc(f,keep,stationary_design=design)
        proposals,_,_=acquire(constant,count=3,origin_bounds=args.origin_bounds)
        cf=constant.fit(origin_bounds=origin_bounds,rate_bound=3.,seeds=[q['seed'] for q in proposals])
        cd=constant.diagnostics(cf)
        knots=np.r_[f.data['row_time'][0],np.arange(first+args.knot_s,last,args.knot_s),f.data['row_time'][-1]]
        fading=AmplitudeArc(f,keep,knots,stationary_design=design)
        ff=fading.fit(origin_bounds=origin_bounds,rate_bound=3.,seeds=[cf.parameters])
        fd=fading.diagnostics(ff)
        record=dict(channel=c,name=NAMES[c],constant=dict(parameters=cf.parameters,objective=cf.objective,
            gradient=cf.gradient,prediction=cd['test_complex_prediction_fraction']),
            fading=dict(parameters=ff.parameters,objective=ff.objective,gradient=ff.gradient,
                prediction=fd['test_complex_prediction_fraction'],amplitudes=fd['amplitude_coefficients'],
                information_rank=ff.information_rank),knots_s=knots)
        result['channels'].append(record);save()
        for d,label,colour in [(cd,'Constant brightness','tab:blue'),(fd,'Positive brightness','tab:orange')]:
            phase=np.asarray(d['phase_residual_deg']);axes[c,0].plot(f.time,phase,'.',color=colour,ms=2,label=label,alpha=.65)
        axes[c,0].set(title=NAMES[c],ylabel='Residual phase (degrees)',ylim=(-180,180),xlabel='Capture time (s)')
        axes[c,1].plot(f.time,fd['amplitude']);axes[c,1].plot(knots,fd['amplitude_coefficients'],'o')
        axes[c,1].set(title=f"Excluded complex prediction: {cd['test_complex_prediction_fraction']:.1%} → {fd['test_complex_prediction_fraction']:.1%}",
            ylabel='Fitted positive strength',xlabel='Capture time (s)')
        for ax in axes[c]:
            for t in np.arange(first+.25,last,1.):ax.axvspan(t,t+.25,color='#999',alpha=.15)
        np.savez(out/f'channel-{c}-histories.npz',time_s=f.time,training=keep,constant_phase_deg=cd['phase_residual_deg'],
            fading_phase_deg=fd['phase_residual_deg'],amplitude=fd['amplitude'])
        fig.savefig(out/'brightness-and-phase.png',dpi=150)
        print(NAMES[c],record['constant']['prediction'],record['fading']['prediction'],'gradient',ff.gradient,flush=True)
    result['complete']=True;save();plt.close(fig)


if __name__=='__main__':main()

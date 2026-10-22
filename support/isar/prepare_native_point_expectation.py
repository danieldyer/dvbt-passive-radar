"""Finite-waveform point expectation with training-estimated real brightness.

The paired linear replay supplies the actual waveform increment. Only that
increment is scaled; recorded observations are never changed. Native phases,
C0/C1 moments, epochs, known pulses, weights and grouping are retained.
"""
import argparse,json
from pathlib import Path
import numpy as np
from native_folds import native_indices
from recorded_injection_fields import read_profiles,write_profiles


def paired_support(linear,baseline,first,last):
    """Return exact row pairs, using only the baseline's native group origin."""
    at=native_indices(baseline['rows'],baseline['information'],first,last,'all')
    use=np.searchsorted(linear['rows'][:,1],baseline['rows'][at,1])
    if not len(at) or use.max()>=len(linear['rows']):raise ValueError('Point control lacks complete original native groups')
    np.testing.assert_array_equal(linear['rows'][use],baseline['rows'][at])
    np.testing.assert_array_equal(linear['information'][use],baseline['information'][at])
    np.testing.assert_array_equal(linear['pulse'][use],baseline['pulse'][at])
    return use,at


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--fit',type=Path,default=Path('docs/data/isar_20261018/native-shared-radial-point-6s'))
    p.add_argument('--linear',type=Path,default=Path('build/injection-20261018/native-point-spread/profiles.bin'))
    p.add_argument('--baseline',type=Path,default=Path('build/injection-20261018/deterministic-baseline/profiles.bin'))
    p.add_argument('--component',type=int,default=0)
    p.add_argument('--interval',type=float,nargs=2,help='Explicit common native support inside the fitted aperture')
    args=p.parse_args();model=json.loads((args.fit/'analysis.json').read_text());first,last=model['interval_s']
    if args.interval:
        if not first<=args.interval[0]<args.interval[1]<=last:raise ValueError('Comparison interval must stay inside the fitted aperture')
        first,last=args.interval
    header,linear=read_profiles(args.linear,interval_s=(first,last))
    baseline_header,baseline=read_profiles(args.baseline,interval_s=(first,last))
    if (header['first_range_sample'],header['range_count'])!=(baseline_header['first_range_sample'],baseline_header['range_count']):
        raise ValueError('Paired point/control range support changed')
    output={}
    for c,ch in linear.items():
        base=baseline[c]
        # Grouping follows the original baseline archive, never the first
        # output row of the shorter derived export. A new file is not a new
        # native averaging origin.
        use,at=paired_support(ch,base,first,last)
        time=ch['rows'][use,1]
        brightness=model.get('phase_histories',{}).get('brightness_model')
        if brightness is not None:
            a=np.asarray(brightness['bands'][c]['coefficients'])[args.component]
            knots=np.asarray(brightness['bands'][c]['knots_s'])
        else:
            if args.component:raise ValueError('A one-point fit has no other component')
            a=np.asarray(model['prediction'][c]['amplitude_coefficients'])
            knots=np.r_[time[0],np.arange(first+1.,last,1.),time[-1]]
        if len(knots)!=len(a) or a.mean()<=0:raise ValueError('Fitted native brightness support disagrees')
        if time[0]<knots[0]-1e-9 or time[-1]>knots[-1]+1e-9:raise ValueError('Do not extrapolate a fitted brightness envelope')
        amplitude=np.interp(time,knots,a)/a.mean()
        values=(ch['values'][:,use]-base['values'][:,at])*amplitude[None,:,None]
        output[c]=dict(values=values,pulse=ch['pulse'][use],rows=ch['rows'][use],information=ch['information'][use])
    header.update(injection_level='training-estimated-point-expectation',background_included=False,
        synthesis_scope=__doc__,brightness_fit=str(args.fit),interval_s=[first,last])
    write_profiles(args.output,header,output)
    print('Wrote',args.output,'native rows',[len(v['rows']) for v in output.values()],flush=True)


if __name__=='__main__':main()

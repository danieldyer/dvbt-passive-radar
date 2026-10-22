"""Test unresolved-echo beating with training-acquired coherent chirp mixtures."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from coherent_arc import CoherentArc
from coherent_chirp_scene import ChirpScene
from recorded_injection_fields import load_fields
from recovery_experiment import NAMES
from scan_native_ship_chirps import acquire
from stationary import Stationary
from fixed_range_field import stationary_profile_design
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--components',type=int,default=3)
    args=p.parse_args();description=json.loads((args.scan/'analysis.json').read_text())
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    offsets=np.asarray(description['offset_samples']);first,last=description['interval_s']
    fixed_epoch=description.get('fixed_reference_epoch_s')
    fields,*_=load_fields(Path('build/injection-20261018/deterministic-baseline/profiles.bin'),offsets=offsets,fixed_epoch_s=fixed_epoch)
    result=dict(scope=__doc__,interval_s=[first,last],complete=False,channels=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    for c,original in enumerate(fields):
        f=original.windowed(first,last);keep=np.floor((f.time-first)/2).astype(int)%4!=1
        arc=CoherentArc(f,keep,stationary_design=stationary_profile_design(f)) if fixed_epoch is not None else CoherentArc(f,keep)
        bounds=(float(offsets.min()-1),float(offsets.max()+1))
        initial=np.asarray(description['channels'][c]['parameters'])[[0,2,3]]
        fits=[];record=dict(name=NAMES[c],channel=c,fits=fits);result['channels'].append(record)
        for count in range(1,args.components+1):
            model=ChirpScene(arc,count)
            if count==1:seeds=[initial]
            else:
                old=ChirpScene(arc,count-1);previous=np.asarray(fits[-1]['parameters'])
                h=old.evaluate(previous)['templates'];acquisition=copy.copy(arc)
                acquisition.nuisance=Stationary(np.column_stack([arc.design,h]),arc.selected)
                acquisition.clean=acquisition.nuisance.remove(f.y.ravel()[arc.selected])
                proposals,_,_=acquire(acquisition,count=2)
                seeds=[np.r_[previous,q['seed']] for q in proposals]
            trials=[model.fit(seed,bounds) for seed in seeds]
            best=min(trials,key=lambda q:q['training_rss']);fits.append(best);save()
            print(NAMES[c],count,'training',best['training_rss']/arc.energy,
                'excluded',best['prediction']['explained_fraction'],'bounds',best['active_bounds'],flush=True)
        result['complete']=False;save()
    result['complete']=True;save()


if __name__=='__main__':main()

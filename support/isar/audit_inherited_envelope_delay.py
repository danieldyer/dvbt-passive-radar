"""Test the older reference's envelope delays without adopting calibration.

Physical positions, motion and source support stay fixed. Only the same
constant phases and positive brightness coefficients are refitted. Zero and
both signs of the older reference's saved delay values are explicit controls.
An envelope offset does not change RF propagation phase or physical Doppler.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from full_path_scene import ScenePath,warm_phases
from report_measured_scatterer_phases import serial


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fit,motion,bands,saved=open_scene(a.motion,full_field=True)
    delays=np.asarray(motion.model['range_bias_m']);likelihood=ScenePath(bands)
    initial=np.r_[np.zeros(likelihood.n),np.angle([complex(*z) for z in fit['physical_gains_real_imag']]),np.asarray(fit['scene_point_phase_radians'])[1:]]
    records=[]
    for sign in [0.,1.,-1.]:
        for c,b in enumerate(bands):b.point.range_bias_m=float(sign*delays[c]);b.warm=None
        parameter,diagnostic=warm_phases(likelihood,initial,200)
        value,gradient,solutions=likelihood.objective(parameter,True)
        phase=np.r_[0.,parameter[likelihood.n+5:]];testing=[]
        for c,(band,solution) in enumerate(zip(bands,solutions)):
            test=band.evaluate(parameter[:likelihood.n],phase+parameter[likelihood.n+c],c,training=False,derivatives=False,frozen_amplitude=solution['amplitude'])
            testing.append({k:test[k] for k in ['cost','energy','explained_fraction']})
        record=dict(sign=sign,fixed_envelope_delay_m=sign*delays,training_objective=value,
            excluded_explained_fraction=1-sum(t['cost'] for t in testing)/sum(t['energy'] for t in testing),excluded=testing,
            phase_optimization=diagnostic,seconds=time.monotonic()-started)
        records.append(record);print(json.dumps(serial(record)),flush=True)
        (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,motion=str(a.motion),records=records)),indent=2)+'\n')
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,motion=str(a.motion),records=records,calibration_adopted=False)),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for record in records:
        ax.plot(range(5),[100*t['explained_fraction'] for t in record['excluded']],'.-',label=f'{record["sign"]:+g} × inherited delay')
    ax.set(xticks=range(5),xticklabels=['Seven','SBS','Nine','Ten','ABC'],ylabel='Excluded full-crop energy predicted (%)',
        title='Fixed physical motion and source support; phase/positive brightness refitted on training only')
    ax.legend();fig.savefig(a.output/'inherited-envelope-delay.png',dpi=145);plt.close(fig)


if __name__=='__main__':main()

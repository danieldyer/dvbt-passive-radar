"""Conditional sensitivity of shared nearby paths to 32 licensed assignments.

This is a frozen-path sensitivity screen, not a position fit or site posterior.
Change only model geometry, retain every original observation and reference,
and rebase the common path if its reference transmitter changes. Positive
brightness and observed-cell stationary coefficients are profiled on training
rows. No per-band phase/rate realignment is allowed in the screen.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from compare_ais_phase import path_geometry
from fit_shared_echo_cluster import subset_prediction
from nearby_echo_control import load,shared_model,project_reference_change
from report_measured_scatterer_phases import serial
from ship_site_signature import licensed_options


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();saved=json.loads(args.source.read_text())
    if not saved['complete']:raise ValueError('Wait for all declared common-path starts')
    control=load();old_code=saved.get('assignment','00002');base=shared_model(control,old_code)
    selected=saved['trials'][saved['best_index']];p=np.array(selected['parameters'])
    r,_=base.evaluate(p)
    if abs(r@r*base.energy-selected['cost'])>1e-7:raise ValueError('Original joint likelihood changed')
    choices=licensed_options(control['geometry']);code_ids={'0':'12013','1':'12015','2':'12014'}
    receiver=np.array(json.loads((control['geometry']/'geometry-prior.json').read_text())['receiver_ecef_m'])
    state=control['nominal'][2];time=control['bands'][2].field.data['row_time']
    refs={}
    for code in '01':
        transmitter=next(o['transmitter_ecef_m'] for o in choices if o['channel']==2 and o['site_id']==code_ids[code])
        refs[code]=path_geometry(state['position'],state['velocity'],np.array(transmitter),receiver)[0]
    trials=[]
    for code in map(''.join,itertools.product('01','02','01','01','02')):
        model=shared_model(control,code);q=p.copy()
        delta,error=project_reference_change(control['curve'],time,refs[code[2]]-refs[old_code[2]])
        if error>1e-6:raise ValueError('Inaccurate reference-path representation')
        for s in (1,4):q[model.local_path_indices[0][s]]+=delta
        r,_=model.evaluate(q);cost=float(r@r*model.energy)
        prediction=subset_prediction(model,q,control['parts'])
        trials.append(dict(assignment=code,cost=cost,data_cost=cost-model.prior_cost(q),
            reference_projection_max_error_m=error,prediction=prediction))
        print(code,'training',cost,'unused',sum(v['heldout_error'] for v in prediction['testing']),flush=True)
    best=int(np.argmin([f['cost'] for f in trials]))
    result=dict(scope=__doc__,complete=True,accepted_calibration=False,production_changed=False,
        source=str(args.source),source_start=saved['best_index'],reference_assignment=old_code,
        selection='Minimum training cost; no selection using unused observations.',
        best_conditional_index=best,trials=trials,
        limitation='No nonlinear common path, constant phase, range-envelope or physical coordinate refinement per hypothesis. '
            'Small conditional differences cannot certify site assignment; this screen identifies whether geometry needs a full original-row control before interpreting interband trajectory discrepancies.',
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.source,Path(__file__),Path('support/isar/nearby_echo_control.py'))})
    args.output.parent.mkdir(exist_ok=True,parents=True)
    args.output.write_text(json.dumps(serial(result),indent=2,allow_nan=False))
    print('conditional best',trials[best]['assignment'],flush=True)


if __name__=='__main__':main()

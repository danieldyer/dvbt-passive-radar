"""Hold flash-derived antennas fixed while predicting other ship intervals.

Candidate regions and independent-path initializers come from training-only
discovery/fits. The antenna calibration never sees these intervals. Independent
band phases and the more restrictive common physical phase are explicit
alternative hypotheses; neither permits independent band phase rates.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ais import read_csv
from cluster_phase import crop
from common_physical_arc import CommonPhysicalFit
from flash_geometry import Flash, assignments
from flash_siting_report import EPOCH
from imaging import sites
from physical_amplitude_arc import PhysicalAmplitudeArc, PhysicalAmplitudeFit
from piecewise_ais import LocalAisReference
from ship_site_signature import licensed_options


def frozen_coordinates(assignment, receiver, rotation, calibration=None):
    """Apply a saved ENU displacement once, retaining the physical baseline."""
    result = deepcopy(assignment)
    receiver = np.array(receiver, copy=True)
    if calibration is not None:
        tx = np.asarray(calibration['transmitter_offsets_enu_m'])
        rx = np.asarray(calibration['receiver_offset_enu_m'])
        if tx.shape != (5, 3) or rx.shape != (3,) or not np.isfinite(tx).all() or not np.isfinite(rx).all():
            raise ValueError('A frozen calibration needs five transmitter and one receiver offsets')
        for entry, offset in zip(result, tx):
            entry['transmitter_ecef_m'] = (np.asarray(entry['transmitter_ecef_m'])+offset@rotation).tolist()
        receiver += rx@rotation
    return result, receiver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('docs/data/isar_20261017/ship-common-phase-polished/fits.json'))
    parser.add_argument('--candidates', type=Path, default=Path('docs/data/isar_20261017/full-capture-ship-arc-validation/fits.json'))
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--common-phase', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'fits.json').exists():
        raise ValueError('Use a fresh output')
    source = json.loads(args.source.read_text())
    candidates = json.loads(args.candidates.read_text())
    if not source['complete'] or not candidates['complete']:
        raise ValueError('Both input analyses must be complete')
    first, last = source['interval_s']
    selected = [c for c in candidates['candidates']
                if c['last_s'] <= first or c['first_s'] >= last]
    if args.labels:
        selected = [c for c in selected if c['label'] in args.labels]
        if {c['label'] for c in selected} != set(args.labels):
            raise ValueError('Requested candidates are missing or overlap the antenna calibration interval')
    if not selected:
        raise ValueError('No independent candidate intervals')
    trials = [t for t in source['trials'] if t['converged']]
    calibrations = [dict(label='nominal-'+a, assignment=a, antenna_fit=None)
                    for a in sorted({t['assignment'] for t in trials})]
    calibrations += [dict(label=f"{t['assignment']}-cycle-{t['branch_index']}",
                         assignment=t['assignment'], antenna_fit=t['antenna_fit']) for t in trials]
    directory = Path('docs/data/calibration_20261016_sites')
    _, receiver, _ = sites(directory)
    rotation = np.asarray(json.loads((directory/'geometry-prior.json').read_text())['ecef_to_enu'])
    options = assignments(licensed_options(directory))
    code = lambda a: ''.join({'12013':'0', '12015':'1', '12014':'2'}[o['site_id']] for o in a)
    track_path = Path('docs/data/isar_20261013/rio-delta-ais.csv')
    track = read_csv(track_path, reference_height_ellipsoid_m=5., mmsi=563052200)
    fields = []
    field_paths = [Path(f'build/isar-20261015/full-fields/channel-{ch}.npz') for ch in range(5)]
    for path in field_paths:
        with np.load(path) as d:
            fields.append({k:d[k] for k in d.files})
    result = dict(scope=__doc__, complete=False, accepted_calibration=False,
                  source=str(args.source), candidates_source=str(args.candidates),
                  calibration_interval_s=[first,last], calibrations=calibrations, trials=[])
    start = time.perf_counter()
    for candidate in selected:
        begin, end, centre = candidate['first_s'], candidate['last_s'], candidate['centre_sample']
        midpoint = (begin+end)/2
        patches = [crop(d,begin,end,np.arange(centre-2,centre+3)) for d in fields]
        train = [((f.time-begin)//.125).astype(int)%2 == 0 for f in patches]
        testing = [~t for t in train]
        knots = [np.r_[f.data['row_time'][0],np.arange(begin+.5,end,.5),f.data['row_time'][-1]] for f in patches]
        observed = np.asarray([r['absolute_observables'] for r in candidate['channels']])
        covariance = np.asarray([r['absolute_observable_covariance'] for r in candidate['channels']])
        for calibration in calibrations:
            assignment = next(a for a in options if code(a) == calibration['assignment'])
            assignment, rx = frozen_coordinates(assignment,receiver,rotation,calibration['antenna_fit'])
            epoch=EPOCH+midpoint-track.epoch
            crosses_knot=np.any(abs(track.curve.x-epoch)<4.)
            reference=LocalAisReference(track,EPOCH+midpoint,rotation,rx) if crosses_knot else None
            geometry = Flash(observed,covariance,track,EPOCH+midpoint,rotation,rx,assignment,
                             free_antennas=False,ais_reference=reference)
            seeds = [np.zeros(geometry.count), geometry.fit().x]
            summary = Flash(observed,covariance,track,EPOCH+midpoint,rotation,rx,assignment,
                            free_antennas=False,include_acceleration=False,ais_reference=reference)
            seeds.append(summary.fit().x)
            model = PhysicalAmplitudeFit([PhysicalAmplitudeArc(f,t,k,geometry,ch,midpoint)
                for ch,(f,t,k) in enumerate(zip(patches,train,knots))])
            fit, starts = model.fit(seeds)
            lower,upper = geometry.bounds(); n = geometry.count
            value,gradient = model.objective(fit.x)
            active = ((fit.x[:n]<=lower+1e-6)&(gradient[:n]>0)) | ((fit.x[:n]>=upper-1e-6)&(gradient[:n]<0))
            gradient[:n][active] = 0
            record = dict(label=candidate['label'],interval_s=[begin,end],centre_sample=centre,
                calibration=calibration['label'],data_energy=model.energy,
                ais_reference_segments=len(reference.coefficients) if reference else 1,
                ais_numerical_error_m=geometry.ais_representation_error_m,
                independent=dict(parameters=fit.x.tolist(),cost=value*model.energy,objective=value,
                    converged=bool(fit.success and abs(gradient).max()<1e-6),
                    maximum_projected_gradient=float(abs(gradient).max()),termination=str(fit.message),
                    bounds=np.flatnonzero((fit.x[:n]<lower+1e-4)|(fit.x[:n]>upper-1e-4)).tolist(),
                    prediction=model.prediction(fit.x,testing),observables=geometry.prediction(fit.x[:n]).tolist(),
                    starts=[dict(objective=float(s.fun),termination=str(s.message)) for s in starts]))
            if args.common_phase:
                common = CommonPhysicalFit(model)
                branches = common.phase_branches(fit.x)
                for index,branch in enumerate(branches):
                    branch.update(index=index,initial_objective=common.objective(branch['parameters'])[0])
                branches.sort(key=lambda b:b['initial_objective'])
                record['common_trials'] = []
                for branch in branches[:3]:
                    fitted = common.fit(branch['parameters'])
                    p,audit = common.polish(fitted['parameters'])
                    cost = common.objective(p)[0]
                    record['common_trials'].append({**fitted, 'parameters':p.tolist(),
                        'objective':cost,'cost':cost*model.energy,'branch_index':branch['index'],
                        'carrier_cycles':branch['carrier_cycles'],'polishing':audit,
                        'converged':bool(audit['final_kkt_residual']<1e-6 and min(audit.get('curvature_eigenvalues',[-1]))>0),
                        'prediction':common.prediction(p,testing)})
            result['trials'].append(record)
            result['seconds'] = time.perf_counter()-start
            (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))
            print(record['label'],record['calibration'],'cost',round(record['independent']['cost'],2),
                  'prediction',[round(p['test_complex_prediction_fraction'],3) for p in record['independent']['prediction']],
                  'converged',record['independent']['converged'],flush=True)
    result['complete'] = True
    paths = [args.source,args.candidates,track_path,*field_paths,directory/'geometry-prior.json',
             directory/'sites-main.json',directory/'acma-transmitters.json',Path(__file__)]
    paths += [Path(__file__).with_name(name+'.py') for name in
              ('common_physical_arc','physical_amplitude_arc','amplitude_arc','flash_geometry','piecewise_ais','field')]
    result['source_sha256'] = {str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()

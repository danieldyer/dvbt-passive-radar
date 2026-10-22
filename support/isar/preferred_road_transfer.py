"""Repeat shared-origin transfer using only the user's preferred street hypotheses.

Albert Street and Victoria Parade remain plausibility hypotheses, not surveyed
vehicle positions. A whole target is excluded from fitting while ship samples,
remaining preferred targets, priors and fixed/free controls stay matched.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from fit_shared_target_geometry import load_experiment
from road_validation import related_intervals
from scene_dictionary import fingerprint
from shared_origin_transfer import FrozenOriginRoad,fit_control,subset
from validate_shared_geometry import fit_target


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','ship-fit','road-fit','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--alternative-ship-fit',type=Path,
        help='Change only the licensed assignment; retain the main control\'s absolute origin bounds and motion start.')
    p.add_argument('--full-bins',action='store_true',
        help='Retain all saved bins and profile a stationary dictionary; integrate moving responses at original epochs.')
    p.add_argument('--wide-psf',type=Path,
        help='Measured extended pulse export for migrated static dictionaries; used only where required.')
    p.add_argument('--labels',nargs='+',default=['A','O','K','Q'])
    p.add_argument('--test-label',default='A')
    p.add_argument('--fields',type=Path,default=Path('build/isar-20261015/full-fields'))
    p.add_argument('--geometry',type=Path,default=Path('docs/data/calibration_20261013'))
    p.add_argument('--iterations',type=int,default=180)
    args=p.parse_args();started=time.monotonic()
    if args.output.exists():raise ValueError('Use a new experiment directory')
    source=json.loads(args.source.read_text());saved=json.loads(args.road_fit.read_text())
    identity=source['identity'];inherited=identity['legacy_identity']
    if not source['complete'] or not source['fit']['converged'] or source['original_row_maximum_kkt_residual']>1e-7:
        raise ValueError('Require a completed common-origin source')
    for hashes in (inherited['sources'],inherited['implementation_sha256'],identity['implementation_sha256'],
                   source.get('numerical_refinement',{}).get('implementation_sha256',{})):
        for path,digest in hashes.items():
            if fingerprint(Path(path))!=digest:raise ValueError('Source changed: '+path)
    for path in (args.ship_fit,args.road_fit):
        if inherited['sources'].get(str(path))!=fingerprint(path):raise ValueError('Use original source fits')
    legacy,original,bounds,labels,dropped=load_experiment(args.ship_fit,args.road_fit,args.fields,args.geometry,
        identity['labels'],inherited['nodes'])
    assignment_source=args.ship_fit
    if args.alternative_ship_fit:
        main_choice=json.loads(args.ship_fit.read_text());alternative=json.loads(args.alternative_ship_fit.read_text())
        if (main_choice['source_motion']!=alternative['source_motion'] or not alternative['complete']
                or not alternative['fit']['converged']):
            raise ValueError('Use the same ship support/coordinate model and a completed licensed assignment')
        candidate,candidate_original,_,candidate_labels,candidate_dropped=load_experiment(
            args.alternative_ship_fit,args.road_fit,args.fields,args.geometry,identity['labels'],inherited['nodes'])
        if candidate_labels!=labels or candidate_dropped!=dropped:
            raise ValueError('Changing sites must retain identical target hypotheses')
        np.testing.assert_array_equal(candidate.ship.active,legacy.ship.active)
        if (len(candidate.initial)!=len(legacy.initial) or
                np.max(np.abs(np.asarray(source['parameters'])[legacy.antenna_slice]))>1e-12):
            raise ValueError('Require compatible motion coordinates and a zero antenna-offset initial state')
        for a,b in zip(original.bands,candidate_original.bands):
            np.testing.assert_array_equal(a.field.y,b.field.y)
            np.testing.assert_array_equal(a.points,b.points)
            np.testing.assert_array_equal(a.selected,b.selected)
        # Physical motion coordinates and fixed data remain unchanged. Only the
        # nominal antenna geometry and its site-group covariance change.
        legacy,original=candidate,candidate_original
        assignment_source=args.alternative_ship_fit
    if dropped or args.test_label not in args.labels or any(label not in labels for label in args.labels):
        raise ValueError('Require known disjoint targets and one explicitly excluded label')
    road_checks={};exact_roads={}
    if args.full_bins:
        from full_road_likelihood import build
        from geographic_io import read_profiles
        records=read_profiles(args.wide_psf)[1] if args.wide_psf else None
        for label in args.labels:
            index=labels.index(label)
            patches=[Path(saved['input_patches'])/f'{label}-{c}.npz' for c in range(5)]
            fast_road,exact_road,checks=build(legacy.roads[index],patches,records)
            legacy.roads[index]=fast_road;exact_roads[index]=exact_road;road_checks[label]=checks
        print(json.dumps({'stage':'full_road_models','checks':road_checks,
            'seconds':time.monotonic()-started}),flush=True)
    elif args.wide_psf:
        raise ValueError('An extended pulse export is only used by --full-bins')
    calibration=[label for label in args.labels if label!=args.test_label]
    indices=[labels.index(label) for label in calibration];road=legacy.roads[labels.index(args.test_label)]
    if any(related_intervals(road.problem,legacy.roads[i].problem) for i in indices):
        raise ValueError('Calibration contains an overlapping target fragment')
    fast,exact,bounds=subset(legacy,original,bounds,source['parameters'],indices)
    if args.full_bins:
        from shared_origin_geometry import OriginRoad
        exact.roads=[OriginRoad(exact_roads[i]) for i in indices]
        road=exact_roads[labels.index(args.test_label)]
        np.testing.assert_equal(fast.energy,exact.energy)
    identity={'scope':__doc__,'source':str(args.source),'source_sha256':fingerprint(args.source),
        'inherited_identity':source['identity'],'calibration_labels':calibration,'excluded_label':args.test_label,
        'omitted_other_labels':[label for label in labels if label not in args.labels],
        'selection':'User preferred Albert Street (A/O) and Victoria Parade (K/Q), before this transfer test.',
        'limitation':'Target discovery, road assignments and inherited numerical initialization used this recording; conditional transfer is not independent ground-truth validation.',
        'original_rows':True,'yaw_fixed_zero':True,
        'road_observation_model':('Full saved bins, exact original-row moving/static responses, fixed static nuisance.'
            if args.full_bins else 'Legacy Doppler gate, bin-mean moving response, no explicit static nuisance.'),
        'full_bin_road_checks':road_checks,
        'site_assignment_source':str(assignment_source),'site_assignment_sha256':fingerprint(assignment_source),
        'site_assignment':[option['site_id'] for option in json.loads(assignment_source.read_text())['site_assignment']],
        'absolute_origin_bounds_from':str(args.ship_fit),
        'implementation_sha256':{str(path):fingerprint(path) for path in [Path(__file__),
            Path('support/isar/shared_origin_transfer.py'),Path('support/isar/shared_origin_geometry.py'),
            Path('support/isar/fit_physical_profile.py'),Path('support/isar/stationarity.py')]}}
    if args.full_bins:
        files=[Path('support/isar')/name for name in ('full_road_likelihood.py',
            'profiled_road_field.py','road_gate_information.py','integrated_moving_road.py',
            'integrated_stationary_road.py','wide_road_templates.py')]
        identity['implementation_sha256'].update({str(path):fingerprint(path) for path in files})
        identity['additional_sources_sha256']={str(args.wide_psf):fingerprint(args.wide_psf)} if args.wide_psf else {}
    args.output.mkdir(parents=True)
    (args.output/'experiment.json').write_text(json.dumps({'complete':False,'identity':identity},indent=2)+'\n')
    fixed=fit_control(fast,exact,bounds,exact.initial,True,args.iterations,args.output/'fixed.json',identity,calibration)
    if not fixed['fit']['converged']:raise ValueError('Fixed control failed the unchanged stationarity gate')
    fitted=fit_control(fast,exact,bounds,fixed['parameters'],False,args.iterations,args.output/'fitted.json',identity,calibration)
    if not fitted['fit']['converged']:raise ValueError('Fitted control failed the unchanged stationarity gate')
    old=saved['labels'].index(args.test_label)
    starts=[np.asarray(saved['selected_roads'][old]['state']),road.initial]
    transferred=[]
    for case in (fixed,fitted):
        parameters=np.asarray(case['parameters'])
        target=FrozenOriginRoad(road,parameters[exact.range_slice])
        transferred.append(fit_target(target,parameters[exact.antenna_slice],starts,args.iterations))
    error=[sum(c['prediction_error'] for c in case['prediction']) for case in transferred]
    energy=[sum(c['matched_energy'] for c in case['prediction']) for case in transferred]
    stationary=[case['starts'][case['selected_start']]['fit']['converged'] for case in transferred]
    record={'complete':bool(all(stationary)),'identity':identity,'transfer':transferred,'accepted_calibration':False,
        'summary':{'prediction_error_change_percent':100*(error[1]/error[0]-1),
            'matched_energy_change_db':float(10*np.log10(energy[1]/energy[0])),
            'selected_fits_stationary':stationary},'seconds':time.monotonic()-started}
    (args.output/'experiment.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()

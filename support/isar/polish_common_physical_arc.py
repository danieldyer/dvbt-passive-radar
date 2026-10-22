"""Check and polish completed common-phase fits without discarding old trials."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common_physical_arc import CommonPhysicalFit
from fit_common_physical_arc import load_problem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--assignment',nargs='+')
    p.add_argument('--recondition',action='store_true',help='Restart the constrained search using the metric at the current fit')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'fits.json').exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text())
    independent=json.loads(Path(source['independent_source']).read_text())
    intervals=json.loads(Path(independent['independent_arc_source']).read_text())
    if not source['complete']:raise ValueError('Source must be complete')
    output={'scope':__doc__,'complete':False,'accepted_calibration':False,'source':str(args.source),
        'interval_s':source['interval_s'],'trials':[]}
    for record in source['trials']:
        if args.assignment and record['assignment'] not in args.assignment:continue
        model,testing=load_problem({**record,'interval_s':source['interval_s']},intervals)
        common=CommonPhysicalFit(model);parameters=np.array(record['parameters'])
        value,_=common.objective(parameters)
        if not np.isclose(value,record['objective'],atol=1e-11,rtol=1e-10):raise ValueError('Raw likelihood changed')
        reconditioned=None
        if args.recondition:
            reconditioned=common.fit(parameters)
            if reconditioned['objective']<value:
                parameters=reconditioned['parameters']
        fitted,report=common.polish(parameters)
        value,_=common.objective(fitted);g=common.geometry;n=g.count
        result={**record,'parameters':fitted.tolist(),'objective':value,'cost':value*model.energy,
            'polishing':report,'prediction':common.prediction(fitted,testing),
            'antenna_fit':g.antennas.describe(fitted[:18]),'reflector_nuisance_standardised':fitted[18:n].tolist(),
            'predicted_observables':g.prediction(fitted[:n]).tolist(),'common_phase_rad':float(fitted[-1]),
            'converged':bool(report.get('final_kkt_residual',np.inf)<1e-6 and min(report.get('curvature_eigenvalues',[-1]))>0)}
        lower,upper=g.bounds()
        result['bound_indices']=np.flatnonzero((fitted[:n]<lower+1e-4)|(fitted[:n]>upper-1e-4)).tolist()
        if reconditioned is not None:
            result['reconditioned_search']={**reconditioned,'parameters':reconditioned['parameters'].tolist()}
        output['trials'].append(result)
        (args.output/'fits.json').write_text(json.dumps(output,indent=2,allow_nan=False))
        print(record['assignment'],record['branch_index'],'cost',result['cost'],'converged',result['converged'],'report',report,flush=True)
    output['complete']=True
    paths=[args.source,Path(source['independent_source']),Path(independent['independent_arc_source']),Path(__file__),
           Path(__file__).with_name('common_physical_arc.py'),Path(__file__).with_name('stationarity.py')]
    output['source_sha256']={str(path):hashlib.file_digest(path.open('rb'),'sha256').hexdigest() for path in paths}
    (args.output/'fits.json').write_text(json.dumps(output,indent=2,allow_nan=False))


if __name__=='__main__':main()

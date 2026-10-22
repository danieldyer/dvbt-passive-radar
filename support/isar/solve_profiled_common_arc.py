"""Solve existing common-phase branches with an exact residual Jacobian."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common_physical_arc import CommonPhysicalFit
from fit_common_physical_arc import load_problem
from profiled_arc_solver import ProfiledArcSolver


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('docs/data/isar_20261017/ship-common-phase-branches/fits.json'))
    p.add_argument('--assignments',nargs='+',default=['02002'])
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'fits.json').exists():raise ValueError('Use a fresh output')
    source=json.loads(args.source.read_text());independent=json.loads(Path(source['independent_source']).read_text())
    intervals=json.loads(Path(independent['independent_arc_source']).read_text())
    result=dict(scope=__doc__,complete=False,accepted_calibration=False,source=str(args.source),
                interval_s=source['interval_s'],trials=[])
    for record in source['trials']:
        if record['assignment'] not in args.assignments:continue
        model,testing=load_problem({**record,'interval_s':source['interval_s']},intervals)
        common=CommonPhysicalFit(model);solver=ProfiledArcSolver(common)
        seed=np.array(record['parameters']);r,j=solver.evaluate(seed)
        value,gradient=common.objective(seed)
        if not np.isclose(value,r@r,rtol=1e-11):raise ValueError('Residual objective differs')
        np.testing.assert_allclose(gradient,2*j.T@r,rtol=1e-7,atol=1e-9)
        fit=solver.fit(seed)
        polished,audit=common.polish(fit.x)
        value,gradient=common.objective(polished);g=common.geometry;n=g.count;lower,upper=g.bounds()
        final={**record,'parameters':polished.tolist(),'objective':value,'cost':value*model.energy,
            'prediction':common.prediction(polished,testing),'polishing':audit,
            'converged':bool(audit.get('final_kkt_residual',np.inf)<1e-6 and min(audit.get('curvature_eigenvalues',[-1]))>0),
            'bound_indices':np.flatnonzero((polished[:n]<lower+1e-4)|(polished[:n]>upper-1e-4)).tolist(),
            'antenna_fit':g.antennas.describe(polished[:18]),'reflector_nuisance_standardised':polished[18:n].tolist(),
            'predicted_observables':g.prediction(polished[:n]).tolist(),'common_phase_rad':float(polished[-1]),
            'residual_solver':dict(evaluations=fit.nfev,termination=str(fit.message),optimality=float(fit.optimality),
                                   initial_objective=record['objective'],final_objective=float(2*fit.cost))}
        result['trials'].append(final)
        (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        print(record['assignment'],record['branch_index'],'cost',final['cost'],'converged',final['converged'],
              'KKT',audit.get('final_kkt_residual'),'evaluations',fit.nfev,flush=True)
    result['complete']=True
    paths=[args.source,Path(source['independent_source']),Path(independent['independent_arc_source']),Path(__file__)]
    paths += [Path(__file__).with_name(n+'.py') for n in ('profiled_arc_solver','common_physical_arc','physical_amplitude_arc','amplitude_arc','field')]
    result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in paths}
    (args.output/'fits.json').write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()

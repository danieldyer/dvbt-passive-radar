"""Check actual objective curvature and KKT conditions for paired ship fits."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from fit_joint_sea_arcs import prepare_problem
from fit_ship_physical_pair import build_model
from stationarity import polish


def audit_fit(model,parameters,free):
    value,jacobian=model.evaluate(parameters)
    jacobian=jacobian[:,free]
    root=np.linalg.cholesky(jacobian.T@jacobian)
    local=solve_triangular(root.T,np.eye(len(free)),lower=False)
    transform=np.zeros((model.count,len(free)));transform[free]=local
    finite=np.isfinite(model.lower)&np.isfinite(model.upper)
    def objective(z):
        value,gradient=model.objective(parameters+transform@z)
        return value,transform.T@gradient
    z,audit=polish(objective,np.zeros(len(free)),np.eye(len(free)),transform[finite],
        model.lower[finite]-parameters[finite],model.upper[finite]-parameters[finite],
        np.full(len(free),-np.inf),np.full(len(free),np.inf))
    corrected=parameters+transform@z
    audit['strict_local_minimum']=bool(audit.get('final_kkt_residual',np.inf)<1e-6 and
        min(audit.get('curvature_eigenvalues',[-1]))>0)
    audit['maximum_parameter_correction']=float(abs(corrected-parameters).max())
    audit['original_objective']=model.objective(parameters)[0]
    audit['polished_objective']=model.objective(corrected)[0]
    return corrected,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources',nargs='+',type=Path)
    args=parser.parse_args();case=prepare_problem()
    for source in args.sources:
        data=json.loads(source.read_text())
        if not data['complete']:raise ValueError('Complete all declared fits first')
        path=source.with_name('stationarity.json')
        if path.exists():raise ValueError('Audit exists; preserve it')
        model,i=build_model(case,data.get('secondary_origin_bounds_samples',[-.8,2.8]),data['interval']['label'])
        result=dict(scope=__doc__,source=str(source),complete=False,accepted_calibration=False,trials=[])
        for row in data['trials']:
            parameters=np.asarray(row['parameters']);cost=model.objective(parameters)[0]*model.energy
            if abs(cost-row['cost'])>1e-7:raise ValueError('Saved objective does not reproduce')
            fitted,audit=audit_fit(model,parameters,np.asarray(row['audit']['free_indices']))
            prediction=model.prediction(fitted,case['testing'][i])
            result['trials'].append(dict(start_rank=row['start_rank'],parameters=fitted.tolist(),audit=audit,
                cost=audit['polished_objective']*model.energy,prediction=prediction))
            path.write_text(json.dumps(result,indent=2,allow_nan=False))
            print(source.parent.name,row['start_rank'],'KKT',audit.get('final_kkt_residual'),
                'minimum curvature',min(audit.get('curvature_eigenvalues',[-1])),
                'strict',audit['strict_local_minimum'],flush=True)
        result['complete']=True
        files=[source,Path(__file__),Path(__file__).with_name('physical_pair.py'),Path(__file__).with_name('stationarity.py')]
        result['source_sha256']={str(p):hashlib.file_digest(p.open('rb'),'sha256').hexdigest() for p in files}
        path.write_text(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()

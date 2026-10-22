"""Independent-band conditional scene prediction on saved exact statistics.

This deliberately relaxes shared reflectivity. It is a model diagnostic,
not a coherent image, a physical-scatterer proof or a theoretical SNR bound.
Each band's ridge strength is selected using the same inner two-second
blocks. The original outer fold is scored only after selection.
"""
import argparse,json
from pathlib import Path
import numpy as np
from regularized_scene import Observation,FixedRidgePath


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--statistics',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.statistics) as f:
        obs={name:[Observation(f[f'{name}_gram_{c}'],f[f'{name}_rhs_{c}'],float(f[f'{name}_energy_{c}'])) for c in range(5)]
            for name in ['inner','validation','original','outer']}
    records=[];arrays={}
    for c in range(5):
        path=FixedRidgePath([obs['inner'][c]],[1.]);maximum=path.eigenvalue[-1];trials=[];best=None
        for relative in np.logspace(-8,1,19):
            strength=maximum*relative;x=path.solve(strength);validation=obs['validation'][c].error(x)
            trials.append(dict(relative_strength=float(relative),training_explained_fraction=1-obs['inner'][c].error(x)/obs['inner'][c].energy,
                validation_explained_fraction=1-validation/obs['validation'][c].energy))
            if best is None or validation<best[0]:best=(validation,strength,relative)
        validation,strength,relative=best;path=FixedRidgePath([obs['original'][c]],[1.]);x=path.solve(strength)
        arrays[f'coefficient_{c}']=x;original=obs['original'][c].error(x);outer=obs['outer'][c].error(x)
        record=dict(channel=c,trials=trials,strength=float(strength),relative_strength=float(relative),inner_validation_cost=validation,
            inner_validation_energy=obs['validation'][c].energy,outer_cost=outer,outer_energy=obs['outer'][c].energy,
            original_training_explained_fraction=1-original/obs['original'][c].energy,outer_explained_fraction=1-outer/obs['outer'][c].energy)
        records.append(record);print(json.dumps(record),flush=True)
    result=dict(complete=True,scope=__doc__,statistics=str(a.statistics),channels=records,
        inner_validation_explained_fraction=1-sum(r['inner_validation_cost'] for r in records)/sum(r['inner_validation_energy'] for r in records),
        outer_explained_fraction=1-sum(r['outer_cost'] for r in records)/sum(r['outer_energy'] for r in records))
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'scene.npz',**arrays)


if __name__=='__main__':main()

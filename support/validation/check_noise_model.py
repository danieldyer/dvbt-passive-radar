#!/usr/bin/env python3
"""Compare recorded map background with the diagonal carrier-noise prediction."""
import argparse
import json
from pathlib import Path

import numpy as np
from sensitivity import DT, NFFT


def check(cache):
    noise=np.load(cache.with_name(cache.stem+'-carrier-noise.npz'))['noise_variance']
    power=np.load(cache)['reference_power']
    per_row=float(np.sum(noise*power)/NFFT**2)
    records=json.loads(cache.with_name(cache.stem+'-maps.json').read_text())
    checks=[]
    for record in records:
        predicted=per_row*1.5/int(record['cpi_s']/DT)
        observed=record['background_power_proxy']
        checks.append(dict(cpi_s=record['actual_cpi_s'],predicted_map_noise_variance=predicted,
            observed_background_proxy=observed,observed_to_predicted=observed/predicted,
            observed_excess_db=float(10*np.log10(observed/predicted))))
    return dict(source=cache.stem,diagonal_model_range_row_variance=per_row,cpi_checks=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs',nargs='+',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    result=dict(limitations='Observed background includes unresolved returns and interference; estimated diagonal noise is not a calibrated independent floor.',
                excerpts=[check(path) for path in args.inputs])
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

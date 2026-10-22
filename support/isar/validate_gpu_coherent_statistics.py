"""Compare the Vulkan coherent GLRT against the independent double reference."""
import argparse,json,time
from pathlib import Path
import numpy as np
from coherent_amplitude_glrt import statistic
from gpu_coherent_statistics import Statistics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261018);z=(rng.normal(size=(5,10001))+1j*rng.normal(size=(5,10001)))*10**rng.uniform(-3,3,(5,10001))
    z[:,0]=0;z[:,1]=np.arange(1,6)*np.exp(.7j);z[:,2]=[1,-1,1,-1,1]
    worker=Statistics(a.output/'worker')
    try:
        reference,_=statistic(z);power,phase=worker.evaluate(z);relative=np.abs(power-reference)/np.maximum(reference,1e-20)
        phase_score=np.sum(np.maximum(np.real(z*np.exp(-1j*phase)),0)**2,axis=0)
        phase_error=np.abs(phase_score-reference)/np.maximum(reference,1e-20)
        if relative.max()>1e-5 or phase_error.max()>1e-5:raise AssertionError('GPU coherent GLRT differs from the exact CPU maximization')
        expected=.7;aligned_error=float(abs(np.angle(np.exp(1j*(phase[1]-expected)))))
        if aligned_error>1e-5:raise AssertionError('GPU coherent GLRT altered a known common reflector phase')
        rotated,_=worker.evaluate(z*np.exp(.35j))
        gauge=float(max(abs(rotated-power)/np.maximum(reference,1e-20)))
        if gauge>1e-5:raise AssertionError('Common complex phase changed coherent power')
        batched=Statistics(a.output/'batched-worker',max_batch_pixels=997)
        try:
            tiled_power,tiled_phase=batched.evaluate(z)
            np.testing.assert_array_equal(tiled_power,power)
            np.testing.assert_array_equal(tiled_phase,phase)
        finally:batched.close()
        large=np.tile(z,101)[:,:1_000_000];times=[]
        for _ in range(3):
            start=time.perf_counter();worker.evaluate(large);times.append(time.perf_counter()-start)
        report=dict(complete=True,scope=__doc__,maximum_relative_power_error=float(relative.max()),maximum_relative_error_at_returned_phase=float(phase_error.max()),
            known_common_phase_error_degrees=float(np.degrees(aligned_error)),common_phase_gauge_relative_power_error=gauge,
            benchmark_cells=large.shape[1],benchmark_end_to_end_seconds=times)
        report['pixel_batching_complex_phase_and_power_bitwise_equal']=True
    finally:worker.close()
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()

"""Alternate certified scene blocks with an exactly checked shared-path seed.

This offline workflow keeps one physical path and five constant band phases.
Every stage records the original complex-data likelihood. A frozen-carrier
initializer is used only if the exact physical fixed-brightness cost improves;
it is followed by the original positive-brightness/constant-phase solve.
Joint stationarity and final image validation remain separate requirements.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import numpy as np


def run(command,log):
    with log.open('w') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cycles',type=int,default=3);p.add_argument('--phase-iterations',type=int,default=160)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);current=a.motion;records=[];started=time.monotonic()
    for cycle in range(a.cycles):
        previous=json.loads(current.read_text());seed=a.output/f'cycle-{cycle:02d}-path'
        run([sys.executable,'support/isar/seed_frozen_ship_path.py','--motion',str(current),'--output',str(seed)],a.output/f'cycle-{cycle:02d}-path.log')
        seeded=json.loads((seed/'analysis.json').read_text())
        if not seeded['exact_training_cost_improved']:
            records.append(dict(cycle=cycle,path_step_accepted=False,reason='Approximate carrier seed did not improve the exact physical likelihood'))
            break
        support=json.loads(Path(previous['source_scene']).read_text())
        support['phase_parameters']=np.r_[np.zeros(4),seeded['scene_point_phase_radians']].tolist()
        support['band_phase_initial_radians']=np.angle([complex(*z) for z in seeded['physical_gains_real_imag']]).tolist()
        support_path=a.output/f'cycle-{cycle:02d}-support.json';support_path.write_text(json.dumps(support,indent=2)+'\n')
        output=a.output/f'cycle-{cycle:02d}-scene'
        command=[sys.executable,'support/isar/fit_full_ship_point_path.py','--field',previous['field'],
            '--phase-seed',previous['phase_seed'],'--coarse-track',previous['coarse_track'],
            '--resume-motion',str(seed/'analysis.json'),'--scene',str(support_path),
            '--spacing-s',str(previous['path_spacing_s']),'--amplitude-spacing-s',str(previous['amplitude_spacing_s']),
            '--orientation',previous['orientation'],'--noise-objective','--phases-only',
            '--phase-warmup-iterations',str(a.phase_iterations),
            '--max-acceleration-mps2',str(previous['max_acceleration_mps2']),
            '--max-path-correction-m',str(previous['max_path_correction_m']),'--output',str(output)]
        if previous.get('fit_range_offsets'):command+=['--range-offsets',*map(str,previous['fit_range_offsets']),'--interval',*map(str,previous['interval_s'])]
        for key,flag in [('angular_rate_offset_degps','--angular-rate-offset-degps'),('angular_acceleration_offset_degps2','--angular-acceleration-offset-degps2')]:
            if previous.get(key):command +=[flag,str(previous[key])]
        run(command,a.output/f'cycle-{cycle:02d}-scene.log')
        fitted=json.loads((output/'analysis.json').read_text())
        if fitted['objective']>previous['objective']+1e-9:
            raise ValueError('Alternating step increased the original physical training likelihood')
        gain=previous['objective']-fitted['objective']
        record=dict(cycle=cycle,path_step_accepted=True,previous_motion=str(current),motion=str(output/'analysis.json'),
            initial_objective=previous['objective'],objective=fitted['objective'],training_cost_fraction_reduction=gain,
            excluded_explained_fraction=fitted['unused_explained_fraction'],
            phase_block_converged=fitted['phase_warmup']['numerical_convergence'],seconds=time.monotonic()-started)
        records.append(record);current=output/'analysis.json';print(json.dumps(record),flush=True)
        (a.output/'progress.json').write_text(json.dumps(dict(complete=False,scope=__doc__,latest_motion=str(current),cycles=records),indent=2)+'\n')
        if gain<1e-6:break
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,latest_motion=str(current),cycles=records,
        joint_stationarity_proven=False,focused_ship_proven=False,seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':main()

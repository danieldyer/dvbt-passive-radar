"""Compare direct and coarse-initialized local fits at fixed band calibration."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);reference=json.loads(a.reference.read_text())
    curve=BSpline(reference['path_knots_s'],reference['path_coefficients_m'],3)
    beta=np.angle([complex(*z) for z in reference['physical_gains_real_imag']]);records=[]
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(16,11),layout='constrained')
    for row,name in enumerate(['early','middle','late']):
        fits=[]
        for j,method in enumerate(['local-control','chirp-acquisition']):
            path=a.root/f'{name}-coherent-calibration-{method}'/'analysis.json';fit=json.loads(path.read_text())
            phase=np.angle([complex(*z) for z in fit['physical_gains_real_imag']]);error=np.angle(np.exp(1j*((phase-phase[3])-(beta-beta[3]))))
            if max(abs(error))>1e-8:raise ValueError('Local acquisition changed the upstream relative band calibration')
            t=np.linspace(*fit['interval_s'],500);fitted=BSpline(fit['path_knots_s'],fit['path_coefficients_m'],3)
            label=['Direct local fit','Coarse chirp + local fit'][j]
            axes[row,0].plot(t,fitted(t)-curve(t),label=label);axes[row,1].plot(t,fitted(t,1)-curve(t,1),label=label)
            fractions=[100*(1-fit['objective']),100*fit['unused_explained_fraction']]
            axes[row,2].bar(np.arange(2)+(j-.5)*.35,fractions,.35,label=label)
            record=dict(interval=name,method=method,motion=str(path),training_explained_fraction=1-fit['objective'],
                excluded_explained_fraction=fit['unused_explained_fraction'],numerical_convergence=fit['numerical_convergence'],
                maximum_kkt_residual=fit['maximum_kkt_residual'],constant_delay_change_m=fit['physical_path_offset_change_m'],
                maximum_relative_band_phase_error_degrees=float(np.degrees(max(abs(error)))))
            records.append(record);fits.append(record)
        axes[row,0].set(title=f'{name.title()} {fit["interval_s"][0]:.2f}–{fit["interval_s"][1]:.2f} s',ylabel='Physical path change (m)',xlabel='Capture time (s)')
        axes[row,1].set(ylabel='Physical path-rate change (m/s)',xlabel='Capture time (s)')
        axes[row,2].set(ylabel='Explained original crop energy (%)',xticks=[0,1],xticklabels=['Training','Excluded'])
        for ax in axes[row]:ax.legend(fontsize=8)
    fig.suptitle('One shared physical path; upstream relative band phases fixed; one common reflector-phase gauge\nSame original samples, sources, positive brightness basis and constraints; prediction fractions are not target SNR efficiencies')
    fig.savefig(a.output/'calibrated-motion-acquisition.png',dpi=145);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,reference=str(a.reference),comparisons=records,
        adopted_as_continuous_capture_motion=False),indent=2)+'\n')


if __name__=='__main__':main()

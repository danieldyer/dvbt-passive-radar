"""Independent CV-centre recovery control and native short-arc comparison."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from report_measured_scatterer_phases import serial
from test_cartesian_brightness_scene import scenario,independent_echo
from cartesian_brightness_scene import CartesianBrightnessFit


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    ex,model,truth=scenario(refine_points=False,free_centre=True)
    for b in model.bands:
        coefficient=(1+.2*np.sin(np.arange(10)+b.channel)).reshape(2,5)
        b.field.y[:]=independent_echo(model,truth,b.channel,coefficient)
        b.__init__(b.field,b.training,b.knots,b.kinematics,b.points,b.band_phase,b.design,channel=b.channel)
    model=CartesianBrightnessFit(model.bands)
    seed=np.zeros(model.size);seed[4]=.1
    full=model.fit(seed,max_iterations=600)
    full=model.polish(full['parameters'],max_iterations=120)
    strict=model.fit(seed,max_iterations=600,free=np.r_[np.arange(5),np.arange(7,model.size)])
    p=np.asarray(full['parameters']);error=p-truth
    synthetic=dict(scope='Noiseless known-support component test; independent Cartesian distances and numerical symbol quadrature. Estimator seed is coarse, not truth plus a perturbation. Constant band phases and brightness are unknown. This is not blind support recovery in recorded interference.',
        truth=truth,seed=seed,free_centre_fit=full,anchor_forced_to_centre_fit=strict,
        position_error_m=float(np.linalg.norm(error[:2])),velocity_error_mps=float(np.linalg.norm(error[2:4])),
        rotation_centre_offset_error_m=float(np.linalg.norm(error[5:7])),spin_error_degps=float(abs(error[4])),
        unused_prediction=[b.prediction(p) for b in model.bands])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    kin=model.bands[0].kinematics;t=kin.time
    for name,parameters,colour in [('Independent truth',truth,'black'),('Recovered free centre',p,'#4477aa'),('Anchor forced to centre',np.asarray(strict['parameters']),'#ee6677')]:
        position,velocity=kin.point_state(parameters,np.zeros(3),origin=ex.receiver)
        reference=kin.point_state(truth,np.zeros(3),origin=ex.receiver)[0]
        error_position=(position-reference)@ex.rotation.T
        axes[0].plot(t,error_position[:,0],color=colour,label=name+' east')
        axes[0].plot(t,error_position[:,1],color=colour,ls='--',label=name+' north')
        axes[1].plot(t,np.linalg.norm(error_position,axis=1),color=colour,label=name)
    axes[0].set(xlabel='Time (s)',ylabel='Position error (m)',title='Recovered physical anchor trajectory');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Time (s)',ylabel='Position-error magnitude (m)',title='Exact CV centre, turning rigid body');axes[1].legend(fontsize=8)
    fig.savefig(out/'synthetic-recovery.png',dpi=160);plt.close(fig)
    root=Path('docs/data/isar_20261018/native-spatial-validation');real=[]
    for name in ['ais-near-cartesian-cv-4s-checked','ais-near-cv-free-centre-4s']:
        a=json.loads((root/name/'analysis.json').read_text());real.append(dict(source=name,fit=a['fit'],prediction=a['prediction']))
    result=dict(scope=__doc__,synthetic=synthetic,real=real,geographic_solution_accepted=False)
    (out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    rows=[]
    for i,r in enumerate(real):
        fractions=', '.join(format(p['explained_fraction'],'.2%') for p in r['prediction'])
        label='Free rotation centre' if i else 'Anchor is centre'
        rows.append(f'<tr><td>{label}</td><td>{r["fit"]["objective"]:.9f}</td><td>{r["fit"]["maximum_scaled_projected_gradient"]:.3g}</td><td>{r["fit"]["active_bounds"]}</td><td>{fractions}</td></tr>')
    rows=''.join(rows)
    text=f'''<!doctype html><html><meta charset="utf-8"><title>CV rigid-centre validation</title><style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px}}p{{line-height:1.5}}img{{width:100%}}td,th{{padding:8px}}</style><h1>Constant-velocity centre and rotating reflectors</h1><p>Constraining a bright reflector itself to CV is more restrictive than constraining the ship's rotation centre to CV. The corrected physical model is x(t) = p₀ + v₀t + R(ωt)q + [R(ωt) − I − ωtJ]u. Here p₀/v₀ are the anchor's instantaneous state, q is a relative source offset, and u locates the anchor relative to the CV centre. Every channel shares this exact motion; no independent temporal phase correction is allowed.</p><p>At t=0, changing u changes neither anchor position nor velocity. Its acceleration is −ω²u. At zero spin, u is exactly unobservable, as checked by regression. Export to a conventional CV centre plus rotated sources reproduces physical complex waveforms without fitted phase alignment.</p><h2>Independent synthetic control</h2><p>Two known physical sources, independent direct distances and numerical symbol quadrature. Position error <b>{synthetic['position_error_m']:.6g} m</b>, velocity error <b>{synthetic['velocity_error_mps']:.6g} m/s</b>, centre-offset error <b>{synthetic['rotation_centre_offset_error_m']:.6g} m</b>. The seed uses zero corrections and 0.1°/s spin, not the truth. This is a noiseless known-support component test, not a substitute for blind injections into recorded interference.</p><img src="synthetic-recovery.png"><h2>Original native symbols: 158–162 s</h2><table><tr><th>Model</th><th>Training residual / baseline</th><th>Scaled projected gradient</th><th>Bound indices</th><th>Excluded prediction fractions: Seven, SBS, Nine, Ten, ABC</th></tr>{rows}</table><p>The new real fit reaches position, velocity and centre-offset bounds. It does not materially cure the discrepancy and is <b>not accepted as a geographic solution</b>. Optimiser termination alone is insufficient. Next, additional source support is sought from training residuals and checked on excluded data.</p><p><a href="../ais-near-cv-free-centre-4s/phase-histories/phase-histories.png">Frozen native component phase diagnostics</a> · <a href="analysis.json">All measured controls</a></p><p>Literature context: <a href="https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-rsn.2018.5233">Garry and Smith, Passive ISAR part I</a> separates translation and rotation compensation and uses backprojection to handle bistatic geometry on a chosen image plane. The implementation here uses exact physical paths, not an image-axis rotation after formation.</p></html>'''
    (out/'index.html').write_text(text)
    print(json.dumps(serial(synthetic),indent=2))


if __name__=='__main__':main()

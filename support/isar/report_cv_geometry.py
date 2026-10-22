"""Illustrate bistatic CV observability using exact physical path histories.

This is a noiseless geometry control, not an uncertainty or detection claim
about the recorded target. All alternative states match one site's initial
excess path, path rate and path acceleration. Every alternative has exactly
zero Cartesian acceleration. Constant calibration is shown both known and
profiled, because these are different information assumptions.
"""
import argparse,json,struct
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from report_measured_scatterer_phases import serial


def path_derivatives(point,transmitter):
    gradient=np.zeros(3);hessian=np.zeros((3,3))
    for antenna in (transmitter,np.zeros(3)):
        vector=point-antenna;length=np.linalg.norm(vector);unit=vector/length
        gradient+=unit;hessian+=(np.eye(3)-np.outer(unit,unit))/length
    return gradient,hessian


def path_change(old,new,transmitter):
    """Difference of two exact two-leg path lengths, with no Earth-scale loss."""
    change=new-old;result=np.zeros(old.shape[:-1])
    for antenna in (transmitter,np.zeros(3)):
        vector=old-antenna
        result+=(2*np.sum(vector*change,axis=-1)+np.sum(change*change,axis=-1))/(
            np.linalg.norm(vector+change,axis=-1)+np.linalg.norm(vector,axis=-1))
    return result


def equivalent_initial_state(point,velocity,transmitter,displacement_m):
    """A nearby equal-range state matching range rate and curvature as well."""
    g,h=path_derivatives(point,transmitter)
    normal=np.r_[g[:2]/np.linalg.norm(g[:2]),0.];tangent=np.array([-normal[1],normal[0],0.])
    shifted=point+displacement_m*tangent
    correction=brentq(lambda s:path_change(point,shifted+s*normal,transmitter),-2000.,2000.,xtol=1e-11)
    shifted+=correction*normal
    gg,hh=path_derivatives(shifted,transmitter)
    nn=np.r_[gg[:2]/np.linalg.norm(gg[:2]),0.];tt=np.array([-nn[1],nn[0],0.])
    base=nn*((g@velocity)-gg[2]*velocity[2])/np.linalg.norm(gg[:2]);base[2]=velocity[2]
    roots=np.roots([tt@hh@tt,2*base@hh@tt,base@hh@base-velocity@h@velocity])
    candidates=[base+z.real*tt for z in roots if abs(z.imag)<1e-10]
    if not candidates:raise ValueError('No real CV state matches the declared initial curvature')
    return shifted,min(candidates,key=lambda v:np.linalg.norm(v-velocity))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles',type=Path,default=Path('build/isar-20261018/current-native-full/profiles.bin'))
    parser.add_argument('--reference',type=Path,default=Path('docs/data/isar_20261018/native-spatial-validation/ais-near-shared-point-4s/analysis.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    with args.profiles.open('rb') as source:
        if source.read(8)!=b'INJPROF1':raise ValueError('Expected a native profile archive')
        size,=struct.unpack('<I',source.read(4));header=json.loads(source.read(size))
    plan=header['plan'];r=np.asarray(plan['ecef_to_enu']);rx=np.asarray(plan['receiver_ecef_m'])
    tx=(np.asarray(plan['transmitters_ecef_m'])-rx)@r.T
    state=json.loads(args.reference.read_text())['physical_state_enu'];p,v,_=np.asarray(state)
    t=np.linspace(-15,15,2001);old=p+t[:,None]*v
    gradients=np.asarray([path_derivatives(p,a)[0][:2] for a in tx]);unit=gradients/np.linalg.norm(gradients,axis=1)[:,None]
    angle=float(np.degrees(np.arccos(np.clip(np.min(unit@unit.T),-1,1))))
    rf=np.asarray([c['rf_hz'] for c in header['applied_channels']])
    records=[];fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for distance,colour in [(-500,'#4477aa'),(-100,'#66ccee'),(100,'#ee6677'),(500,'#aa3377')]:
        q,w=equivalent_initial_state(p,v,tx[0],distance);new=q+t[:,None]*w
        phases=np.asarray([360*f/299792458.*path_change(old,new,a) for a,f in zip(tx,rf)])
        intercept=np.asarray([360*f/299792458.*path_change(p,q,a) for a,f in zip(tx,rf)])
        varying=phases-intercept[:,None]
        records.append(dict(equal_range_displacement_m=distance,position_enu_m=q,velocity_enu_mps=w,
            velocity_change_mps=float(np.linalg.norm(w-v)),constant_phase_difference_deg=intercept,
            maximum_time_varying_phase_difference_deg=np.max(abs(varying),axis=1)))
        for row,c in enumerate([0,4]):
            axes[row,0].plot(t,phases[c],color=colour,label=f'{distance:+} m; Δv {np.linalg.norm(w-v):.3f} m/s')
            axes[row,1].plot(t,varying[c],color=colour)
    for row,name in enumerate(['Seven / Ornata','ABC / Ridge']):
        for col in range(2):
            axes[row,col].axhline(0,color='#999',lw=.6)
            axes[row,col].set(xlabel='Time relative to epoch (s)',ylabel='Exact phase difference (degrees)',
                title=f'{name}: '+('constant band calibration known' if col==0 else 'unknown constant intercept profiled'))
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Different constant-velocity trajectories with the same initial range, range rate and range acceleration\nKnown-height geometric control; no recorded measurements fitted or corrected')
    fig.savefig(out/'phase-observability.png',dpi=165);plt.close(fig)
    summary=dict(scope=__doc__,reference=str(args.reference),profiles=str(args.profiles),
        position_enu_m=p,velocity_enu_mps=v,rf_hz=rf,horizontal_path_gradients=gradients,
        maximum_gradient_angle_deg=angle,range_gradient_singular_values=np.linalg.svd(unit,compute_uv=False),alternatives=records)
    (out/'analysis.json').write_text(json.dumps(serial(summary),indent=2,allow_nan=False)+'\n')
    table=''.join(f'<tr><td>{a["equal_range_displacement_m"]:+.0f}</td><td>{a["velocity_change_mps"]:.5f}</td><td>{a["maximum_time_varying_phase_difference_deg"][0]:.6f}</td><td>{a["maximum_time_varying_phase_difference_deg"][4]:.4f}</td></tr>' for a in records)
    html=f'''<!doctype html><html><meta charset="utf-8"><title>CV geometric ambiguity</title><style>body{{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 20px}}img{{width:100%}}td,th{{padding:8px;text-align:right}}p{{line-height:1.5}}</style>
<h1>Constant velocity helps; geometry still matters</h1><p>At one epoch a single bistatic pair gives one excess-range ellipse and one velocity projection. In a known-height planar model that is two measurements for four unknown position/velocity coordinates. Constant velocity links epochs: curvature and higher path derivatives add information. It does not make the geometric path history linear, and exact single-pair histories need not retain an infinite planar ambiguity.</p>
<p>For the current working assignment (four Ornata channels, ABC at Ridge), the horizontal path-gradient directions differ by at most <b>{angle:.4f}°</b> at the declared ship reference. This assignment is not a certified antenna calibration.</p>
<p>Each alternative below is computed without noise or radar data. It lies on Seven’s exact equal-range contour and matches the reference initial range rate and range acceleration while maintaining zero Cartesian acceleration. The remaining difference on Seven starts at third order in time.</p>
<img src="phase-observability.png" alt="Exact phase differences for alternative CV trajectories"><table><tr><th>Along contour (m)</th><th>Velocity change (m/s)</th><th>Max varying phase, Seven (°)</th><th>Max varying phase, ABC (°)</th></tr>{table}</table>
<p>The left column retains every constant phase difference: independent, known band calibration supplies this additional information. The right column profiles one constant intercept per band, appropriate only when calibration must be estimated from the same target. It does not change a sample or apply a temporal phase screen. These assumptions must not be interchanged when claiming geographic precision. The figure is an observability illustration, not a measured CRLB or evidence that the actual ship follows this motion.</p><p><a href="analysis.json">Machine-readable states and measurements</a></p></html>'''
    (out/'index.html').write_text(html);print(json.dumps(serial(summary),indent=2))


if __name__=='__main__':main()

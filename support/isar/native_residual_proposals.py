"""Training-only Vulkan residual adjoint, checked against both CPU operators.

This proposes source coordinates, never a scientific power/noise normalisation.
Final acceptance belongs to the unchanged native-symbol likelihood and unused
data. Both real brightness and stationary nuisance are frozen from training
before the residual is formed.
"""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter
from dense_patch_image import DenseImages,trial_geometry
from midpoint_model import discrepancy
from report_measured_scatterer_phases import serial


def propose(native,fast,parameters,points,directory,*,band_phases=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    east=np.arange(-150.,151.,1.);north=np.arange(-200.,201.,1.);xx,yy=np.meshgrid(east,north)
    pixels=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)];views=[];residuals=[]
    for b,a in zip(native,fast):
        fit=b.evaluate_scene(parameters,full=True);mu=fit['prediction'];y=b.field.y.ravel()
        beta=b.nuisance.solve(y[b.selected]-mu[b.selected]);residual=(y-mu-b.design@beta).reshape(b.field.y.shape)
        field=copy.copy(a.field);field.y=residual
        views.append(SimpleNamespace(field=field,training=b.training,kinematics=a.kinematics,
            tx=field.data['transmitter_ecef_m'],rx=field.data['receiver_ecef_m'],origin=0.))
        residuals.append(residual)
    worker=DenseImages(views,pixels,directory/'worker',full_pulse_support=True)
    try:
        m=native[0].motion_size
        geometry=np.stack([trial_geometry(b,parameters[:m],b.training,worker.stride) for b in views])
        numerator=worker.evaluate_geometry(geometry,derivatives=False)[:,0]
    finally:worker.close()
    # Compare representative-epoch GPU contractions with the CPU, and bound
    # the approximation against the original native-symbol forward response.
    probe=np.linspace(0,len(pixels)-1,7,dtype=int);cpu=[];loss=[]
    for b,a,residual in zip(native,views,residuals):
        row=[];kin=a.kinematics
        for i in probe:
            point=pixels[i];pos,vel=kin.point_state(parameters[:m],point)
            h=a.field.trajectory(pos,vel,position_offset=pos-a.field.data['centre_ecef_m'])
            row.append(np.vdot(h[b.training],residual[b.training]))
            pos,vel=b.kinematics.point_state(parameters[:m],point)
            exact=b.field.trajectory(pos,vel,position_offset=pos-b.field.data['centre_ecef_m'])
            loss.append(discrepancy(exact,h))
        cpu.append(row)
    cpu=np.asarray(cpu);gpu=numerator[:,probe]
    relative=np.linalg.norm(gpu-cpu)/np.linalg.norm(cpu)
    if relative>1e-3:raise ValueError(f'Residual GPU/CPU adjoint disagreement {relative:g}')
    if max(v['matched_snr_loss_db'] for v in loss)>.001:raise ValueError('Representative search exceeds its native-template loss budget')
    band_phase=np.array([b.band_phase for b in native]) if band_phases is None else np.asarray(band_phases)
    if band_phase.shape!=(len(native),) or not np.isfinite(band_phase).all():raise ValueError('One physical constant phase per band is required')
    score=abs(np.exp(-1j*band_phase)@numerator)**2;plane=score.reshape(xx.shape)
    peak=(plane==maximum_filter(plane,size=7));order=np.flatnonzero(peak);order=order[np.argsort(-score[order])]
    selected=[]
    for i in order:
        q=pixels[i]
        if min(np.linalg.norm(points[:,:2]-q[:2],axis=1))<20:continue
        if any(np.linalg.norm(q[:2]-pixels[j,:2])<25 for j in selected):continue
        selected.append(int(i))
        if len(selected)==8:break
    candidate=pixels[selected]
    np.savez(directory/'proposals.npz',east_m=east,north_m=north,numerator=numerator,score=plane,
        candidate_points_east_north_up_m=candidate,existing_points_east_north_up_m=points)
    fig,ax=plt.subplots(figsize=(9,9),layout='constrained');db=10*np.log10(np.maximum(plane,1e-30))
    im=ax.imshow(db-db.max(),origin='lower',extent=[east[0],east[-1],north[0],north[-1]],cmap='magma',vmin=-35,vmax=0,aspect='equal')
    ax.scatter(points[:,0],points[:,1],marker='x',color='cyan',label='Existing fitted points')
    for j,q in enumerate(candidate):ax.plot(*q[:2],'+',color='#44ff88');ax.annotate(str(j+1),q[:2],xytext=(4,4),textcoords='offset points',color='#44ff88')
    ax.set(xlabel='East from fitted anchor (m)',ylabel='North (m)',title='Training-only residual source proposals\nFrozen native moving-scene and stationary estimates removed')
    ax.legend();fig.colorbar(im,ax=ax,label='Adjoint proposal score relative to maximum (dB; not an SNR map)')
    fig.savefig(directory/'residual-proposals.png',dpi=160);plt.close(fig)
    result=dict(scope=__doc__,complete=True,gpu_cpu_relative_error=float(relative),
        maximum_native_template_loss_db=max(v['matched_snr_loss_db'] for v in loss),
        maximum_native_phase_discrepancy_deg=max(abs(v['constant_phase_difference_degrees']) for v in loss),
        candidates_east_north_up_m=candidate,training_only=True,scientific_image_or_snr_claim=False)
    (directory/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    return result

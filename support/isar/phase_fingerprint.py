"""Search measured range cells for a separately estimated phase trajectory.

The temporal template is fixed. Each range cell gets one constant phase and
nonnegative real brightness at training epochs. The phase maximum is global;
unused brightness is interpolated from training. No path coefficient or phase
screen is fitted to the searched channel.
"""
import numpy as np

from positive_phase import maximum


def measure(time,values,weight,template,training,testing):
    t,y,w,h,tr,te=map(np.asarray,(time,values,weight,template,training,testing))
    if (t.ndim!=1 or y.ndim!=2 or y.shape[0]!=len(t) or
        any(v.shape!=t.shape for v in (w,h,tr,te)) or np.any(tr&te) or
        np.any(w<0) or not np.isfinite(w).all() or np.any(np.diff(t)<=0)):
        raise ValueError('Matching ordered samples, fixed template and disjoint masks required')
    use=tr&(w>0)&(abs(h)>1e-12);test=te&(w>0)&(abs(h)>1e-12)
    if np.count_nonzero(use)<3:raise ValueError('Insufficient training samples')
    norm=np.sum(w[use]*abs(h[use])**2);scores=[]
    for column in range(y.shape[1]):
        z=y[:,column]
        beta=np.sqrt(w[use])*z[use]*h[use].conj()/abs(h[use])
        phi,_=maximum(beta);unit=h*np.exp(1j*phi)
        amplitude=np.maximum(0.,(z[use]*unit[use].conj()).real)/abs(unit[use])**2
        interpolated=np.interp(t,t[use],amplitude,left=np.nan,right=np.nan)
        valid=test&np.isfinite(interpolated);pred=interpolated*unit
        phase=np.angle(z*unit.conj());phase_weight=w[valid]*abs(pred[valid])**2
        cross=np.sum(w[valid]*pred[valid].conj()*z[valid]);predicted_energy=np.sum(w[valid]*abs(pred[valid])**2)
        observed_energy=np.sum(w[valid]*abs(z[valid])**2)
        coherent=np.sum(w[use]*h[use].conj()*z[use])
        training_error=np.sum(w[use]*abs(z[use]-amplitude*unit[use])**2)
        error=np.sum(w[valid]*abs(z[valid]-pred[valid])**2)
        scores.append(dict(phase_rad=float(phi),coherent_training_score=float(abs(coherent)**2/norm),
            training_error=float(training_error),testing_error=float(error),testing_energy=float(observed_energy),
            testing_explained_fraction=float(1-error/observed_energy) if observed_energy>0 else None,
            testing_phase_rms_deg=float(np.degrees(np.sqrt(np.average(phase[valid]**2,weights=phase_weight)))) if phase_weight.sum()>0 else None,
            testing_phase_bias_deg=float(np.degrees(np.angle(cross))),
            testing_matched_power=float(abs(cross)**2/predicted_energy) if predicted_energy>0 else None,
            testing_coherence_squared=float(abs(cross)**2/(predicted_energy*observed_energy)) if predicted_energy*observed_energy>0 else None,
            phase_residual_rad=phase,testing=valid,predicted_amplitude=interpolated))
    return scores

"""Measured range-cell phase histories, without a fitted scene or phase screen.

The saved cell is already range-compressed and AIS range-migration aligned.
A fixed local complex average improves readability. An explicitly supplied
acquisition rate is removed during averaging and restored afterwards. It does
not become a correction to the measured phase. Carrier phase is converted to
excess-path change using each band's own wavenumber and saved deramp.
"""
import numpy as np


def local_average(time, signal, information, centres, width_s, k, rate_mps, *, acceleration_mps2=0., rate_epoch_s=0.):
    """Information-weighted rectangular complex measurements on fixed windows.

    Return effective epochs, complex values and conditional thermal variance.
    Missing information is excluded; windows never extend beyond the input.
    Adjacent output windows overlap and are therefore correlated.
    """
    t,z,w,c=map(np.asarray,(time,signal,information,centres))
    if t.ndim!=1 or z.shape!=t.shape or w.shape!=t.shape or np.any(np.diff(t)<=0):
        raise ValueError('Matching ordered time, complex signal and information required')
    if width_s<=0 or k<=0 or np.any(w<0) or not np.isfinite(w).all():
        raise ValueError('Positive width/carrier and finite nonnegative information required')
    means=np.full(len(c),np.nan+1j*np.nan);variance=np.full(len(c),np.inf);epochs=c.copy()
    for i,centre in enumerate(c):
        if centre-width_s/2<t[0] or centre+width_s/2>t[-1]:continue
        use=(abs(t-centre)<=width_s/2)&(w>0)&np.isfinite(z)
        weights=w[use];total=weights.sum()
        if total<=0:continue
        epochs[i]=weights@t[use]/total
        dt=t[use]-epochs[i]
        acquisition=rate_mps*dt
        if acceleration_mps2:
            acquisition+=acceleration_mps2*((epochs[i]-rate_epoch_s)*dt+.5*dt**2)
        shifted=z[use]*np.exp(1j*k*acquisition)
        means[i]=weights@shifted/total
        variance[i]=1/(8192*total)
    return epochs,means,variance


def anchored_history(time, values, variance, k, old_path, nominal_path, anchor_index,
                     *, minimum_power_to_noise=4.):
    """Unwrap only the measured connected high-SNR segment at one anchor.

    No integer cycles are inferred through a fade. Other valid segments are
    retained as wrapped phase samples but are not connected to the anchor.
    The threshold is a display-quality flag, not a calibrated detection test.
    """
    t,z,var,old,nominal=map(np.asarray,(time,values,variance,old_path,nominal_path))
    if any(v.shape!=t.shape for v in (z,var,old,nominal)) or not 0<=anchor_index<len(t):
        raise ValueError('Matching history arrays and a valid anchor are required')
    ratio=np.divide(abs(z)**2,var,out=np.zeros_like(var),where=np.isfinite(var)&(var>0))
    good=np.isfinite(z)&(ratio>=minimum_power_to_noise)
    connected=np.zeros(len(t),bool)
    if good[anchor_index]:
        first=last=anchor_index
        while first>0 and good[first-1]:first-=1
        while last+1<len(t) and good[last+1]:last+=1
        connected[first:last+1]=True
    phase=np.full(len(t),np.nan);measured=phase.copy();residual=phase.copy()
    if connected.any():
        p=np.unwrap(np.angle(z[connected]));p-=p[anchor_index-np.flatnonzero(connected)[0]]
        phase[connected]=p
        measured[connected]=(old-old[anchor_index])[connected]-p/k
        residual[connected]=measured[connected]-(nominal-nominal[anchor_index])[connected]
    wrapped=np.angle(z*np.conj(z[anchor_index]))
    return dict(phase_rad=phase,wrapped_phase_rad=wrapped,measured_path_m=measured,
        residual_path_m=residual,nominal_path_m=nominal-nominal[anchor_index],
        power_to_noise=ratio,connected=connected,good=good)


def interpolate_connected(time, values, grid):
    """Interpolate finite connected samples without bridging missing runs."""
    t,v,g=map(np.asarray,(time,values,grid));out=np.full(len(g),np.nan)
    good=np.isfinite(v);edges=np.diff(np.r_[False,good,False].astype(int))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a<2:continue
        use=(g>=t[a])&(g<=t[b-1]);out[use]=np.interp(g[use],t[a:b],v[a:b])
    return out


def range_rate_map(time, values, information, k, rates):
    """Direct periodogram of measured cells; no notch or nuisance projection."""
    t,y,w,r=map(np.asarray,(time,values,information,rates));weight=8192*w
    phasor=np.exp(1j*k*(t-t.mean())[:,None]*r[None,:])
    product=phasor.T@(weight[:,None]*y)
    return abs(product)**2/max(weight.sum(),np.finfo(float).tiny)


def path_increments(time, values, variance, k, old_path, lag, *, minimum_power_to_noise=4.):
    """Local measured path increments; no global phase unwrapping or origin.

    Each increment is ambiguous by one wavelength. Choose the time separation
    so the residual (AIS-deramped) phase change is below pi. Weak endpoints are
    flagged, not interpolated or connected to other pieces.
    """
    t,z,var,path=map(np.asarray,(time,values,variance,old_path))
    if not isinstance(lag,int) or not 0<lag<len(t):raise ValueError('A positive sample lag is required')
    phase=np.angle(z[lag:]*z[:-lag].conj())
    ratio=np.divide(abs(z)**2,var,out=np.zeros_like(var),where=np.isfinite(var)&(var>0))
    good=(ratio[lag:]>=minimum_power_to_noise)&(ratio[:-lag]>=minimum_power_to_noise)&np.isfinite(phase)
    increment=path[lag:]-path[:-lag]-phase/k
    increment[~good]=np.nan
    return (t[lag:]+t[:-lag])/2,increment,good

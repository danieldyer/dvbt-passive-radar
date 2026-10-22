"""Linear Doppler gating with the same operator on data, model and covariance.

Used only by the offline calibration fit. A finite Fourier subspace selects
the associated echo after its reversible seed deramp. Cholesky whitening uses
the complete projected covariance, rather than pretending filtered samples
are independent. The source diagonal covariance is still an empirical model.
"""
import numpy as np
from scipy.linalg import solve_triangular


def gate(time, variance, bandwidth_hz, epoch, duration, selected=None):
    if selected is None:
        selected = np.ones(len(time), dtype=bool)
    harmonics = np.arange(-int(bandwidth_hz*duration), int(bandwidth_hz*duration)+1)
    basis = np.exp(2j*np.pi*(time-epoch)[:, None]*harmonics[None, :]/duration)
    f = basis.conj().T*selected[None, :]/np.sqrt(np.count_nonzero(selected))
    covariance = (f*variance[None, :])@f.conj().T
    factor = np.linalg.cholesky((covariance+covariance.conj().T)/2)
    q = solve_triangular(factor, f, lower=True)
    return q, basis


def profile(y, operator, phase, derivative=None):
    """Variable projection with a general linear observation operator."""
    u0 = np.exp(1j*np.remainder(phase, 2*np.pi))
    u = operator@u0; v = operator@y
    du = None if derivative is None else operator@(1j*u0[:, None]*derivative)
    result = projected_profile(v, u, du)
    return (*result, u0)


def projected_profile(v, u, du=None):
    """Eliminate one complex gain in already whitened observation space."""
    norm = np.vdot(u, u).real
    if norm < 1e-20:
        raise ValueError('Template outside the selected Doppler subspace')
    a = np.vdot(u, v)/norm
    error = np.sqrt(2.)*(a*u-v)
    if du is None:
        return error, a
    da = (du.conj().T@v-a*2*np.real(u.conj()@du))/norm
    jacobian = np.sqrt(2.)*(u[:, None]*da+a*du)
    return error, jacobian, a


def operator_for(d, selected=None):
    if selected is None or np.all(selected):
        return d['operator']
    t = d['time'][selected]
    duration = min(d.get('duration_s',30.), float(t[-1]-t[0]+np.median(np.diff(d['time']))))
    return gate(d['time'], d['variance'], d['bandwidth_hz'], d['seed']['epoch'], duration, selected)[0]


def rebin(data, start, end, step=.02):
    t, y, w = data['row_time'], data['row_value'], data['row_information']
    bin_index = np.floor((t-start)/step).astype(int)
    n = int(np.ceil((end-start)/step))
    good = (bin_index>=0)&(bin_index<n)
    t, y, w, b = t[good], y[good], w[good], bin_index[good]
    weight = np.bincount(b, weights=w, minlength=n)
    count = np.bincount(b, minlength=n)
    norm = np.maximum(weight, 1e-30)
    value = (np.bincount(b, weights=w*y.real, minlength=n)+
             1j*np.bincount(b, weights=w*y.imag, minlength=n))/norm
    time = np.bincount(b, weights=w*t, minlength=n)/norm
    scatter = np.bincount(b, weights=w*w*abs(y-value[b])**2, minlength=n)
    variance = scatter/norm**2*count/np.maximum(count-1, 1)
    good = (count>1)&(variance>0)
    return {'time':time[good], 'value':value[good], 'variance':variance[good]}

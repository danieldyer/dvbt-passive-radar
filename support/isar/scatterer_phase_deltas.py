"""Constant-origin comparisons of measured physical path histories.

Inputs have already had the known acquisition carrier reference restored and
been divided by each band's own wavenumber. No slopes, curvature, geometric
correction or phase screen are estimated here.
"""
import numpy as np


def compare_pair(paths, time, power_to_noise, *, reference_band=2):
    """Compare two candidates on one grid without connecting missing phases.

    Choose a common phase origin by magnitude only in the middle half of the
    available window. A constant is removed from each candidate/band. Return
    D-B within each band and then its difference from the reference band.
    Descriptive RMS uses only the shared finite support, not imputed samples.
    """
    paths=np.asarray(paths,float);time=np.asarray(time,float);strength=np.asarray(power_to_noise,float)
    if (paths.ndim!=3 or paths.shape[0]!=2 or paths.shape[2]!=len(time) or
            strength.shape!=paths.shape or np.any(np.diff(time)<=0) or
            not 0<=reference_band<paths.shape[1]):
        raise ValueError('Two candidates, matching band/time grids and powers required')
    finite=np.all(np.isfinite(paths)&np.isfinite(strength),axis=(0,1))
    middle=(time>=time[0]+(time[-1]-time[0])*.25)&(time<=time[-1]-(time[-1]-time[0])*.25)
    use=finite&middle
    if not np.any(use):
        raise ValueError('No common finite phase origin in the middle half-window')
    score=np.min(strength,axis=(0,1))
    anchor=int(np.argmax(np.where(use,score,-np.inf)))
    centred=paths-paths[:,:,anchor,None]
    pair=centred[1]-centred[0]
    double=pair-pair[reference_band]
    return dict(anchor_index=anchor,anchor_time_s=float(time[anchor]),
        centred_paths_m=centred,pair_path_change_m=pair,double_difference_m=double,
        common_support=finite,
        rms_against_reference_m=np.sqrt(np.mean(double[:,finite]**2,axis=1)),
        peak_to_peak_against_reference_m=np.ptp(double[:,finite],axis=1))

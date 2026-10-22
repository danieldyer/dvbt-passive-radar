"""Restrict exact fixed-grid observations without changing their phase gauge."""
import numpy as np

from fixed_range_field import stationary_profile_design
from motion import C
from shared_echo_cluster import ReferencedAmplitudeArc


def window_band(band,first,last,*,complete_rows=False):
    if complete_rows:
        # Retained bins can straddle a requested boundary. For a new temporal
        # basis confined to that aperture, keep whole bins whose original rows
        # are all inside; never truncate or re-average a recorded observation.
        rows=band.field.data['row_time'];block=band.field.metadata['block_rows']
        start=np.arange(len(band.field.time))*block
        end=np.minimum(start+block,len(rows))-1
        bins=np.flatnonzero((rows[start]>=first)&(rows[end]<last))
        if len(bins)<2 or np.any(np.diff(bins)!=1):raise ValueError('No complete contiguous bins inside window')
        lower=band.field.time[bins[0]]
        upper=np.nextafter(band.field.time[bins[-1]],np.inf)
        field=band.field.windowed(lower,upper)
    else:field=band.field.windowed(first,last)
    indices=np.searchsorted(band.field.data['row_time'],field.data['row_time'])
    np.testing.assert_array_equal(band.field.data['row_time'][indices],field.data['row_time'])
    part=np.floor((field.time-first)/.125).astype(int)%4
    rows=field.data['row_time']
    knots=(np.r_[rows[0],band.knots[(band.knots>rows[0])&(band.knots<rows[-1])],rows[-1]]
        if complete_rows else np.r_[rows[0],np.arange(first+.25,last,.25),rows[-1]])
    sub=ReferencedAmplitudeArc(field,part%2==0,knots,first,last,
        band.known_path[indices],band.known_rate[indices],stationary_design=stationary_profile_design(field))
    return sub,part,indices


def rebase_parameters(parameters,components,old_band,new_band):
    """Exact change of midpoint/scale; envelope and carrier intercepts agree.

    Parameters use the one-band SharedEchoCluster layout: all origin/phase
    pairs followed by all quadratic path pairs. Supplied known paths are
    unchanged on the restricted original rows.
    """
    p=np.asarray(parameters,float).copy()
    if p.shape!=(4*components,):raise ValueError('One-band quadratic parameter layout required')
    q=p[2*components:].reshape(components,2).copy()
    u=(new_band.epoch-old_band.epoch)/old_band.scale
    ratio=new_band.scale/old_band.scale
    intercept=q[:,0]*u+q[:,1]*u*u
    p[:2*components:2]+=intercept*8e6/C
    p[1:2*components:2]-=old_band.field.k*intercept
    p[2*components:2*components+2*components:2]=(q[:,0]+2*q[:,1]*u)*ratio
    p[2*components+1::2]=q[:,1]*ratio*ratio
    return p


def rebase_cluster(source,parameters,target,*,tolerance_m=1e-8):
    """Restrict mixed polynomial/spline paths with exact carrier bookkeeping.

    The observed field and its C0/C1 reference remain fixed. The path's new
    midpoint constant moves into BOTH envelope origin and carrier intercept.
    Known interband increments keep their original gauge. Reject a new basis
    that cannot represent the old path and derivative on the retained rows.
    This maps local cluster coordinates, independently of global sharing.
    """
    parameters=np.asarray(parameters);time=target.base.field.data['row_time']
    rows=np.searchsorted(source.base.field.data['row_time'],time)
    np.testing.assert_array_equal(source.base.field.data['row_time'][rows],time)
    if source.components!=target.components:raise ValueError('Matching component sets required')
    result=np.zeros(target.slices[-1].stop);diagnostics=[]
    for old,new,old_block,new_block in zip(source.arcs,target.arcs,source.slices,target.slices):
        np.testing.assert_array_equal(old.known_path[rows],new.known_path)
        np.testing.assert_array_equal(old.known_rate[rows],new.known_rate)
        p=parameters[old_block];curve=new.base.curve;old_curve=old.base.curve
        offset=float((old_curve(np.array([curve.epoch]))@p[2:])[0])
        wanted=old_curve(time)@p[2:]-offset
        coefficient=np.linalg.lstsq(curve(time),wanted,rcond=None)[0]
        path_error=float(np.max(abs(curve(time)@coefficient-wanted)))
        rate_error=float(np.max(abs(curve(time,1)@coefficient-old_curve(time,1)@p[2:])))
        if path_error>tolerance_m or rate_error>tolerance_m:
            raise ValueError('New temporal basis does not preserve the path and its Doppler derivative')
        result[new_block]=np.r_[p[0]+offset*8e6/C,p[1]-old.field.k*offset,coefficient]
        diagnostics.append(dict(path_max_error_m=path_error,rate_max_error_mps=rate_error,
            envelope_shift_m=offset,carrier_intercept_shift_rad=-old.field.k*offset))
    old_response=source.response_blocks(parameters);new_response=target.response_blocks(result)
    for detail,(old,_),(new,_) in zip(diagnostics,old_response,new_response):
        expected=old[rows];error=float(np.linalg.norm(new-expected)/max(np.linalg.norm(expected),1e-300))
        if error>2e-9:raise ValueError('Carrier/envelope/C0/C1 response changed under window restriction')
        detail['relative_original_row_response_error']=error
    return result,diagnostics

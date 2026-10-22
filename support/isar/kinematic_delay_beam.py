"""Track-before-detect initializer with linked path, rate and acceleration.

The score is the summed conditional Gaussian GLRT plus a stated acceleration
prior. Continuous states propagate exactly under constant interval acceleration.
Quantised-state merging and a finite beam approximate the MAP trajectory;
beam-width convergence is measured rather than assumed. This is an acquisition
initializer, never a substitute for the later complex-data phase likelihood.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def track(time,ranges,rates,power,gate,*,beam=4096,range_resolution_m=2.,rate_resolution_mps=.05,acceleration_sigma=.03,acceleration_limit=.15):
    time,ranges,rates,power,gate=map(np.asarray,[time,ranges,rates,power,gate])
    if power.shape!=(len(time),len(rates),len(ranges)) or gate.shape!=(len(time),2):raise ValueError('Acquisition grids disagree')
    acceleration=np.linspace(-acceleration_limit,acceleration_limit,31)
    def score(i,r,v):
        coords=np.vstack([(v-rates[0])/(rates[1]-rates[0]),(r-ranges[0])/(ranges[1]-ranges[0])])
        return map_coordinates(power[i],coords,order=1,mode='constant',cval=-np.inf,prefilter=False)
    r,v=np.meshgrid(np.arange(*gate[0],5.),rates);r=r.ravel();v=v.ravel();cost=score(0,r,v);at=np.argsort(cost)[-beam:]
    truncated=int(len(r)>beam);maximum_states=len(r)
    r,v,cost=r[at],v[at],cost[at];history=[(r.copy(),v.copy(),np.full(len(r),-1,dtype=int))]
    for i,dt in enumerate(np.diff(time),1):
        nr=(r[:,None]+v[:,None]*dt+.5*acceleration[None]*dt**2).ravel();nv=(v[:,None]+acceleration[None]*dt).ravel()
        parent=np.repeat(np.arange(len(r)),len(acceleration));a=np.tile(acceleration,len(r))
        valid=(nr>=gate[i,0])&(nr<=gate[i,1])&(nv>=rates[0])&(nv<=rates[-1])
        nr,nv,parent,a=nr[valid],nv[valid],parent[valid],a[valid]
        new=cost[parent]+score(i,nr,nv)-.5*(a/acceleration_sigma)**2
        valid=np.isfinite(new);nr,nv,parent,new=nr[valid],nv[valid],parent[valid],new[valid]
        if not len(new):raise ValueError('No kinematically supported acquisition states')
        qr=np.rint((nr-ranges[0])/range_resolution_m).astype(np.int64);qv=np.rint((nv-rates[0])/rate_resolution_mps).astype(np.int64)
        key=qr*(int(np.ceil((rates[-1]-rates[0])/rate_resolution_mps))+3)+qv
        order=np.lexsort((-new,key));first=np.r_[True,np.diff(key[order])!=0];survivors=order[first]
        truncated+=int(len(survivors)>beam);maximum_states=max(maximum_states,len(survivors))
        at=survivors[np.argsort(new[survivors])[-beam:]]
        r,v,cost=nr[at],nv[at],new[at];history.append((r.copy(),v.copy(),parent[at]))
    at=int(np.argmax(cost));value=float(cost[at]);path=[];velocity=[]
    for r,v,parent in reversed(history):path.append(r[at]);velocity.append(v[at]);at=parent[at]
    path=np.asarray(path[::-1]);velocity=np.asarray(velocity[::-1]);innovation=np.diff(path)-.5*np.diff(time)*(velocity[:-1]+velocity[1:])
    return dict(path_m=path,rate_mps=velocity,objective=value,beam=beam,range_resolution_m=range_resolution_m,rate_resolution_mps=rate_resolution_mps,
        maximum_path_rate_consistency_error_m=float(max(abs(innovation))),maximum_interval_acceleration_mps2=float(max(abs(np.diff(velocity)/np.diff(time)))),
        beam_truncated_steps=truncated,maximum_distinct_state_cells=maximum_states)

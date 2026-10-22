"""Test positive time-varying brightness without adding target phase freedom.

Scatterer and band phases stay constant across the aperture. Only nonnegative
piecewise-linear magnitudes vary. Every forward model includes the exact
waveform, physical motion and static-nuisance projection. Training estimates
all coefficients; excluded epochs receive the unchanged predicted envelopes.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from audit_ship_exact_aperture import state
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from positive_envelope_scene import solve
from rigid_patch import RigidPatch,PatchBand,profile_static
from ship_window import Dataset
from report_measured_scatterer_phases import serial


def linear_basis(t,interval,spacing):
    if spacing==0:return np.ones((len(t),1)),np.array([np.mean(interval)])
    nodes=np.linspace(*interval,int(np.ceil((interval[1]-interval[0])/spacing))+1)
    at=np.clip(np.searchsorted(nodes,t,side='right')-1,0,len(nodes)-2);fraction=(t-nodes[at])/(nodes[at+1]-nodes[at])
    b=np.zeros((len(t),len(nodes)));b[np.arange(len(t)),at]=1-fraction;b[np.arange(len(t)),at+1]=fraction
    if b.min()<-1e-12 or b.max()>1+1e-12:raise ValueError('Envelope extrapolation is forbidden')
    return b,nodes


def statistics(h,y,root,mask,basis):
    """Exact projected Gaussian normal equations, without a huge dictionary."""
    n,p,k=len(h),h.shape[2],basis.shape[1];q=root*mask;denom=np.vdot(q,q).real
    data=profile_static(y,root,mask)*mask[:,None]
    gram_time=np.einsum('trp,trq->tpq',h.conj(),h)*mask[:,None,None]
    gram=np.einsum('tpq,tk,tl->pkql',gram_time,basis,basis,optimize=True).reshape(p*k,p*k)
    mean=np.einsum('t,trp,tk->rpk',q,h,basis,optimize=True).reshape(h.shape[1],p*k)
    gram-=mean.conj().T@mean/denom
    rhs=np.einsum('trp,tr,tk->pk',h.conj(),data,basis,optimize=True).ravel()
    return (gram+gram.conj().T)/2,rhs,float(np.vdot(data,data).real)


def fit(normals,initial,points,knots):
    free=[0,1,2,4];groups=np.repeat(np.arange(points),knots);energy=sum(n[2] for n in normals)
    def objective(parameters,full=False):
        beta=np.zeros(5);beta[free]=parameters[:4];phi=parameters[4:];cost=0.;gradient=np.zeros_like(parameters);solutions=[]
        for c,(gram,rhs,power) in enumerate(normals):
            r=solve(gram,rhs,power,groups,phi+beta[c]);solutions.append(r);cost+=r['cost'];gradient[4:]+=r['phase_gradient']
            if c in free:gradient[free.index(c)]+=sum(r['phase_gradient'])
        return (cost/energy,gradient/energy,solutions) if full else (cost/energy,gradient/energy)
    r=minimize(objective,initial,jac=True,method='L-BFGS-B',options=dict(maxiter=400,gtol=1e-8,ftol=1e-12,maxcor=30))
    value,gradient,solutions=objective(r.x,True)
    return r.x,solutions,dict(objective=value,maximum_gradient=float(max(abs(gradient))),message=str(r.message),iterations=r.nit,
        numerical_convergence=bool(r.success and max(abs(gradient))<1e-6),nnls_numerical_ranks=[s['numerical_rank'] for s in solutions],
        active_positive_magnitudes=[s['active_amplitudes'] for s in solutions])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True);p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--spacing-s',nargs='+',type=float,default=[0.,5.,2.]);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(a.motion.read_text());motion=JointMotion(Path(d['motion']));r=motion.result;mid=d['mid_s'];start=time.monotonic()
    win=Window(a.windows/f'window-{mid:07.2f}',r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds']);dataset=Dataset()
    exact=dataset.window_from_description(win.result);points=np.asarray(d['points']);pose=np.asarray(d['parameters']);bands=[]
    for c,old in enumerate(exact['fast']):
        s=state(win,np.asarray(r['parameters']),c);kin=RigidPatch(win.time[c],mid,s['position'],s['velocity'],exact['local_to_ecef'],motion.tx[3],motion.rx,s['frame'],s['frame_rate'])
        band=PatchBand(old.field,kin,points,old.training,motion.tx[c],motion.rx);h,_=band.design(pose)
        bands.append((band,h))
    constant=d['scene_response_comparison']['phase_coherent'];g=np.asarray(constant['gain_real_imag']);g=g[:,0]+1j*g[:,1]
    initial=np.r_[np.angle(g[[0,1,2,4]]),constant['point_phase_rad']];reports=[]
    for spacing in a.spacing_s:
        training=[];testing=[];bases=[]
        for band,h in bands:
            b,nodes=linear_basis(band.field.time,win.result['interval_s'],spacing);bases.append(b)
            training.append(statistics(h,band.field.y,band.field.root_information,band.training,b))
            testing.append(statistics(h,band.field.y,band.field.root_information,~band.training,b))
        parameters,solutions,result=fit(training,initial,len(points),len(nodes));unused=[];arrays={}
        for c,((gram,rhs,power),solution,(band,h),b) in enumerate(zip(testing,solutions,bands,bases)):
            coefficient=solution['coefficient'];cost=float(power-2*np.vdot(coefficient,rhs).real+np.vdot(coefficient,gram@coefficient).real)
            unused.append(dict(channel=c,rss=cost,energy=power,explained_fraction=1-cost/power))
            arrays[f'coefficient_{c}']=coefficient.reshape(len(points),len(nodes));arrays[f'time_s_{c}']=band.field.time
            arrays[f'envelope_{c}']=b@solution['amplitude'].reshape(len(points),len(nodes)).T
        result.update(spacing_s=spacing,knots_s=nodes,parameters=parameters,unused=unused,
            unused_explained_fraction=1-sum(x['rss'] for x in unused)/sum(x['energy'] for x in unused),seconds=time.monotonic()-start)
        reports.append(result);np.savez_compressed(a.output/f'envelopes-{spacing:g}s.npz',**arrays)
        (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,motion=str(a.motion),models=reports)),indent=2)+'\n')
        print(json.dumps(serial(result)),flush=True)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,motion=str(a.motion),models=reports)),indent=2)+'\n')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,9),layout='constrained')
    for result in reports:axes[0].plot(range(5),[100*v['explained_fraction'] for v in result['unused']],'.-',label=f'{result["spacing_s"]:g} s basis' if result['spacing_s'] else 'Constant brightness')
    axes[0].legend();axes[0].set(xticks=range(5),xticklabels=['Seven','SBS','Nine','Ten','ABC'],ylabel='Excluded data energy predicted (%)')
    last=reports[-1];data=np.load(a.output/f'envelopes-{last["spacing_s"]:g}s.npz')
    for j in range(len(points)):axes[1].plot(data['time_s_3'],data['envelope_3'][:,j],label=f'Point {j+1}')
    axes[1].set(xlabel='Capture time (s)',ylabel='Predicted Ten magnitude');axes[1].legend(ncol=5,fontsize=8)
    fig.suptitle('Time-varying brightness with one constant phase per scatterer and band\nFixed physical motion; all envelopes/phases fitted on training epochs; no time-varying phase freedom')
    fig.savefig(a.output/'envelope-transfer.png',dpi=140);plt.close(fig)
if __name__=='__main__':main()

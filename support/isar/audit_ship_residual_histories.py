"""Test a shared physical residual path on training-selected ship histories.

The fitted correction is a diagnostic, not an adopted motion update. A single
path in metres drives every carrier; complex scatterer amplitudes are nuisance
constants. Two temporal folds and points excluded from the fit test transfer.
Point histories are correlated, so this is not an independent-noise likelihood
or a certificate of geographic identifiability. Original data remain unchanged.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from dense_patch_image import DenseImages
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from report_measured_scatterer_phases import serial
from spline_motion import basis
from gpga import Isolation


def common_path(time,values,information,training,wavenumber,fit_points,spacing=2.):
    """Concentrate histories with a common carrier-scaled, non-affine path."""
    interval=[min(t[0] for t in time),max(t[-1] for t in time)]
    b=[basis(t,interval,spacing) for t in time]
    w=[i*s/np.mean(i[s]) for i,s in zip(information,training)]
    affine=[np.c_[np.ones(len(t)),t-np.mean(interval)] for t in time]
    projection=np.linalg.solve(sum(x.T@(i[:,None]*x) for x,i in zip(affine,w)),
        sum(x.T@(i[:,None]*h) for x,i,h in zip(affine,w,b)))
    b=[h-x@projection for h,x in zip(b,affine)]
    z=[v[:,fit_points] for v in values]
    energy=sum(np.sum(i[:,None]*abs(v)**2) for i,v in zip(w,z))
    def objective(p):
        score=0.;gradient=np.zeros_like(p)
        for h,i,v,k in zip(b,w,z,wavenumber):
            rotated=v*np.exp(1j*k*(h@p))[:,None]
            amplitude=np.einsum('t,tj->j',i,rotated)/i.sum()
            score-=i.sum()*np.sum(abs(amplitude)**2)/energy
            gradient+=h.T@(2*i*k*np.einsum('tj,j->t',rotated,amplitude.conj()).imag/energy)
        return float(score),gradient
    fit=minimize(objective,np.zeros(b[0].shape[1]),jac=True,method='L-BFGS-B',
        options=dict(maxiter=500,gtol=1e-8,ftol=1e-13,maxcor=30))
    return [h@fit.x for h in b],dict(objective=fit.fun,maximum_gradient=float(max(abs(fit.jac))),message=str(fit.message),iterations=fit.nit,
        converged=bool(fit.success and max(abs(fit.jac))<1e-6),fit_points=fit_points,spacing_s=spacing)


def metrics(z,w,selected):
    a=np.einsum('t,ctj->cj',w*selected,z)
    return np.sum(abs(a)**2,axis=0)/(w*selected).sum()


def spatial_phase(time,values,information,training,wavenumber,levers,fit_points):
    """Local first-order translation/yaw test; not final image geometry.

    Unwrapped training-only low-pass histories initialize a complex objective.
    The angular term uses each ray's own bistatic spatial derivative. No
    band-specific phase history is free. Constant phase per point is nuisance.
    """
    interval=[min(t[0] for t in time),max(t[-1] for t in time)]
    p=[basis(t,interval,2.,fixed_midpoint=True)[:,:-1] for t in time]
    h=[basis(t,interval,10.,fixed_midpoint=True)[:,:-1] for t in time]
    n=p[0].shape[1];d=[];v=[];weight=[];design=[]
    for tc,z,w,s,k,l,pc,hc in zip(time,values,information,training,wavenumber,levers,p,h):
        kernels=Isolation(tc,w,s,.6);smooth,_=kernels.apply(z[:,fit_points])
        at=np.asarray(fit_points)
        dc=k*np.concatenate([np.broadcast_to(pc[:,None,:],(len(tc),len(at),n)),hc[:,None,:]*l[:,at,None]],axis=2)
        phase=np.unwrap(np.angle(smooth),axis=0)
        # Weighted nuisance constant per point; no phase slope is removed.
        wc=w[:,None]*s[:,None]*abs(smooth)**2
        dc0=dc-np.sum(wc[:,:,None]*dc,axis=0)[None]/wc.sum(axis=0)[None,:,None]
        phase-=np.sum(wc*phase,axis=0)[None]/wc.sum(axis=0)
        d.append(dc0.reshape(-1,dc.shape[-1]));v.append(-phase.ravel());weight.append(wc.ravel())
        design.append(dc)
    root=np.sqrt(np.concatenate(weight));seed=np.linalg.lstsq(np.concatenate(d)*root[:,None],np.concatenate(v)*root,rcond=1e-10)[0]
    energy=sum(np.sum(w[s,None]*abs(z[s][:,fit_points])**2) for w,s,z in zip(information,training,values))
    def objective(q):
        score=0.;gradient=np.zeros_like(q)
        for dc,z,w,s in zip(design,values,information,training):
            w=w*s;rot=z[:,fit_points]*np.exp(1j*np.einsum('tjp,p->tj',dc,q))
            a=np.sum(w[:,None]*rot,axis=0)/w.sum()
            score-=w.sum()*np.sum(abs(a)**2)/energy
            gradient+=2*np.einsum('t,tj,tjp->p',w,np.imag(rot*a.conj()),dc)/energy
        return float(score),gradient
    # Broad search bounds belong to this diagnostic only. They are not an
    # accepted physical yaw estimate and are never used to rotate a video.
    bounds=[(-15.,15.)]*n+[(-.15,.15)]*h[0].shape[1]
    fits=[minimize(objective,np.clip(initial,*np.asarray(bounds).T),jac=True,method='L-BFGS-B',bounds=bounds,
        options=dict(maxiter=700,gtol=1e-8,ftol=1e-13,maxcor=30)) for initial in [np.zeros_like(seed),seed]]
    best=min(fits,key=lambda r:r.fun);path=[pc@best.x[:n] for pc in p];angle=[hc@best.x[n:] for hc in h]
    corrected=[z*np.exp(1j*k*(pc[:,None]+ac[:,None]*l)) for z,k,pc,ac,l in zip(values,wavenumber,path,angle,levers)]
    return corrected,path,angle,dict(objective=best.fun,maximum_raw_gradient=float(max(abs(best.jac))),message=str(best.message),
        maximum_angle_degrees=float(max(np.max(abs(a)) for a in angle)*180/np.pi),scope='First-order spatial phase diagnostic, not a final physical trajectory',
        starting_trials=[dict(objective=r.fun,iterations=r.nit) for r in fits])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--windows',type=Path,required=True)
    p.add_argument('--points',type=Path,required=True);p.add_argument('--centres',type=float,nargs='+')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    motion=JointMotion(a.motion);result=motion.result;records=json.loads(a.points.read_text())['windows'];worker=None;reports=[]
    try:
        for record in records:
            mid=record['mid_s']
            if a.centres and min(abs(mid-c) for c in a.centres)>.1:continue
            directory=a.windows/f'window-{mid:07.2f}'
            win=Window(directory,result['interval_s'],result['path_spacing_seconds'],result['cross_spacing_seconds'])
            points=np.asarray(record['refined_positions']);hist=[];checks=[];levers=[]
            for train in [True,False]:
                archive=directory/('training' if train else 'unused')/'model.bin'
                if worker is None:worker=DenseImages.reopen(archive,derivatives=8)
                else:worker.replace_archive(archive)
                worker.training=train;worker.set_pixels(points)
                geo,_,_=win.geometry(np.asarray(result['parameters']),worker,motion.gauge_point,result['transmitter_groups'],motion.receiver_local,antennas=bool(result['parameter_counts']['antenna']))
                image=worker.evaluate_geometry(geo,derivatives=False)[:,0];rows=worker.histories()
                checks.append(float(np.linalg.norm(rows.sum(axis=1)-image)/np.linalg.norm(image)));hist.append(rows)
                if train:
                    for c,tc in enumerate(win.time):
                        frame=geo[c,:len(tc),2:5,:3].transpose(0,2,1).astype(float)
                        offsets=np.einsum('tij,pj->tpi',frame,points)
                        gradient=np.zeros_like(offsets)
                        for ray in [0,1]:
                            r=geo[c,:len(tc),ray,:3,None].transpose(0,2,1)+offsets
                            gradient+=r/np.linalg.norm(r,axis=2)[:,:,None]
                        dp=np.einsum('tij,pj->tpi',frame,np.c_[-points[:,1],points[:,0],np.zeros(len(points))])
                        levers.append(np.sum(gradient*dp,axis=2))
            rows=hist[0]+hist[1]
            # Independent OFDM symbols have different physical epochs. Never
            # align row indices or silently interpolate their carrier phases.
            t=win.time;w=[win.context[f'information_{c}'] for c in range(5)]
            z=[np.divide(rows[c,:len(t[c])],8192*w[c][:,None],out=np.zeros((len(t[c]),len(points)),complex),where=w[c][:,None]>0) for c in range(5)]
            train=[s&(i>0) for s,i in zip(win.training,w)];test=[(~s)&(i>0) for s,i in zip(win.training,w)]
            k=2*np.pi*motion.rf/299792458.;fit_points=list(range(3))
            path,detail=common_path(t,z,w,train,k,fit_points)
            corrected=[v*np.exp(1j*kc*q)[:,None] for v,kc,q in zip(z,k,path)]
            before=sum(metrics(v[None],i,s) for v,i,s in zip(z,w,test));after=sum(metrics(v[None],i,s) for v,i,s in zip(corrected,w,test));gain=10*np.log10(after/before)
            spatial,sp,angle,sp_detail=spatial_phase(t,z,w,train,k,levers,fit_points)
            sp_gain=10*np.log10(sum(metrics(v[None],i,s) for v,i,s in zip(spatial,w,test))/before)
            arrays={'points':points}
            for c in range(5):
                for key,value in dict(time_s=t[c],information=w[c],training=train[c],before=z[c],after=corrected[c],diagnostic_path_m=path[c],
                    spatial_after=spatial[c],spatial_path_m=sp[c],spatial_angle_rad=angle[c],angular_path_derivative_m=levers[c]).items():arrays[f'{key}_{c}']=value
            np.savez_compressed(a.output/f'window-{mid:07.2f}.npz',**arrays)
            report=dict(mid_s=mid,row_sum_relative_errors=checks,diagnostic_fit=detail,unused_power_gain_db_by_point=gain,
                path_rms_m=float(np.std(np.concatenate(path))),spatial_phase_fit=sp_detail,spatial_unused_power_gain_db_by_point=sp_gain,accepted_as_motion=False)
            reports.append(report);(a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,scope=__doc__,motion=str(a.motion),windows=reports)),indent=2)+'\n')
            plot(a.output,mid,t,z,corrected,w,train,path,gain)
            plot(a.output,mid,t,z,spatial,w,train,sp,sp_gain,suffix='-spatial')
            print(json.dumps(serial(report)),flush=True)
    finally:
        if worker is not None:worker.close()
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,scope=__doc__,motion=str(a.motion),windows=reports)),indent=2)+'\n')


def plot(out,mid,t,z,corrected,w,train,path,gain,suffix=''):
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(17,12),layout='constrained',sharex=True)
    names=['Seven','SBS','Nine','Ten','ABC'];kernels=[Isolation(tc,wc,np.ones(len(tc),bool),.6) for tc,wc in zip(t,w)]
    # Smoothing is for display only, explicitly uses all epochs; the fit above
    # has no access to validation values. Each point gets one constant origin.
    for j in range(6):
        ax=axes[j//3,j%3]
        for c in range(5):
            smooth,_=kernels[c].apply(z[c][:,j:j+1]);smooth=smooth[:,0]
            phase=np.unwrap(np.angle(smooth));phase-=np.average(phase,weights=w[c])
            ax.plot(t[c],np.degrees(phase),label=names[c],lw=.9)
        ax.set(title=f'Point {j+1} · unused gain {gain[j]:+.2f} dB',ylabel='Residual phase (deg)')
    for j in range(3):
        ax=axes[2,j]
        for c in range(5):
            smooth,_=kernels[c].apply(corrected[c][:,j:j+1]);phase=np.unwrap(np.angle(smooth[:,0]));phase-=np.average(phase,weights=w[c])
            ax.plot(t[c],np.degrees(phase),lw=.9)
        ax.set(title=f'Point {j+1} after diagnostic common path',ylabel='Residual phase (deg)')
        axes[3,j].plot(t[0],path[0]);axes[3,j].set(xlabel='Capture time (s)',ylabel='Diagnostic path (m)')
    axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle(f'{mid-15:.2f}–{mid+15:.2f} s · measured known-waveform matched histories\nFit points 1–3 and training epochs only; points 4–6 excluded; display smoothing 0.6 Hz; no correction adopted')
    fig.savefig(out/f'window-{mid:07.2f}{suffix}.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()

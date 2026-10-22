"""Joint continuous ship-path and static multiband calibration autofocus.

This offline estimator shares one cubic physical path across overlapping
apertures. Optional transmitter displacements and one constant response per
band are shared over the whole capture, rather than rephasing each video
frame. Unused epochs are never used to estimate any of these parameters.
The objective is image concentration, not a claim of a Gaussian likelihood
or absolute geographic identifiability. Original-symbol BP remains final.
"""
import argparse,json,struct,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,LinearConstraint,nnls
from scipy.interpolate import BSpline

from dense_patch_image import DenseImages,entropy
from imaging import geometry_from_state,sites,motion_state
from ais import read_csv,utc_seconds
from motion import C,enu_basis,excess_path
from spline_motion import basis,SplineMotion,knots
from report_measured_scatterer_phases import serial


def antenna_reference(point,tx,rx):
    return -(point-tx)/np.linalg.norm(point-tx)-(tx-rx)/np.linalg.norm(tx-rx)


class Window:
    def __init__(self,directory,interval,spacing,cross_spacing=15.):
        self.directory=Path(directory);self.cross_spacing=cross_spacing;self.result=json.loads((self.directory/'analysis.json').read_text())
        with np.load(self.directory/'geometry-context.npz') as source:self.context=dict(source)
        self.time=[self.context[f'time_{c}'] for c in range(5)]
        self.basis=[basis(t,interval,spacing,fixed_midpoint=True,reference_time=157.)[:,:-1] for t in self.time]
        self.rate=[basis(t,interval,spacing,derivative=1)[:,:-1] for t in self.time]
        self.cross_basis=[basis(t,interval,cross_spacing,fixed_midpoint=True,reference_time=157.)[:,:-1] for t in self.time]
        self.cross_rate=[basis(t,interval,cross_spacing,derivative=1)[:,:-1] for t in self.time]
        self.training=[self.context[f'training_{c}'] for c in range(5)]

    def geometry(self,parameters,model,gauge_point,groups,rotation,*,antennas=False):
        npath=self.basis[0].shape[1];ncross=self.cross_basis[0].shape[1];count=npath+ncross;offset=parameters[count:count+3*len(set(groups))].reshape(-1,3)@rotation.T if antennas else None
        packet=[];phase_references=[];k_values=[]
        for c,t in enumerate(self.time):
            n=len(t);data=self.context;q=data[f'direction_{c}'];qd=data[f'direction_rate_{c}']
            path=self.basis[c]@parameters[:npath];rate=self.rate[c]@parameters[:npath]
            up=data[f'frame_{c}'][n//2,:,2];vector=np.cross(up,q);norm=np.linalg.norm(vector,axis=1)
            cross_direction=vector/norm[:,None];vr=np.cross(up,qd)
            cross_direction_rate=(vr-cross_direction*np.sum(cross_direction*vr,axis=1)[:,None])/norm[:,None]
            cross=self.cross_basis[c]@parameters[npath:count];cross_rate=self.cross_rate[c]@parameters[npath:count]
            state=dict(position_ecef_m=data[f'position_{c}']+path[:,None]*q+cross[:,None]*cross_direction,
                velocity_ecef_mps=data[f'velocity_{c}']+path[:,None]*qd+rate[:,None]*q+cross[:,None]*cross_direction_rate+cross_rate[:,None]*cross_direction,
                body_to_ecef=data[f'frame_{c}'],frame_velocity=data[f'frame_rate_{c}'])
            tx=data[f'tx_{c}'];rx=data[f'rx_{c}'];old_tx=tx
            if antennas:tx=tx+offset[groups[c]]
            selected=(self.training[c] if model.training else ~self.training[c])&(data[f'information_{c}']>0)
            base,details=geometry_from_state(state,tx,rx,float(data[f'rf_hz_{c}']),0.,selected.astype(float))
            base[:,9,0]=(details['path_m']-data[f'observer_{c}'])*8e6/C
            k=2*np.pi*float(data[f'rf_hz_{c}'])/C
            reference=(excess_path(gauge_point[None],tx,rx)[0]-excess_path(gauge_point[None],old_tx,rx)[0])[0] if antennas else 0.
            phase=k*(details['path_m']-data[f'carrier_{c}']-reference)
            base[:,9,1]=np.cos(phase);base[:,9,2]=np.sin(phase);base[:,5,3]=k
            geo=np.zeros((model.stride,10+8*model.derivatives,4),'<f4');geo[:n,:10]=base
            geo[:n,10,:3]=q;geo[:n,11,:3]=qd;geo[:n,19,:3]=q
            geo[:n,26,:3]=cross_direction;geo[:n,27,:3]=cross_direction_rate;geo[:n,35,:3]=cross_direction
            if antennas:
                ref_gradient=antenna_reference(gauge_point,tx,rx)
                baseline=(tx-rx)/np.linalg.norm(tx-rx)
                for j in range(3):
                    at=42+8*j;direction=rotation[:,j]
                    geo[:n,at,:3]=direction;geo[:n,at,3]=1.
                    geo[:n,at+1,:3]=baseline;geo[:n,at+1,3]=ref_gradient@direction
            packet.append(geo);phase_references.append(float(reference));k_values.append(k)
        return np.asarray(packet),np.asarray(phase_references),np.asarray(k_values)


def curve_seed(windows,interval,spacing):
    """A numerical initializer, not final merging of independent estimates."""
    designs=[];values=[]
    for w in windows:
        a=w.result;t=np.linspace(*a['interval_s'],121)
        # Values have one unobservable constant per local aperture. Only
        # rates initialize a single globally anchored continuous path.
        design=basis(t,interval,spacing,derivative=1)[:,:-1]
        local=basis(t,a['interval_s'],a['path_knot_seconds'],derivative=1)@np.asarray(a['path_parameters_m'])
        designs.append(design);values.append(local)
    return np.linalg.lstsq(np.concatenate(designs),np.concatenate(values),rcond=None)[0]


class Joint:
    def __init__(self,windows,interval,spacing,gains,*,coherent=False,antennas=False,independent_objective='sum-power'):
        self.windows=windows;self.interval=interval;self.spacing=spacing;self.gains=np.asarray(gains)
        self.coherent,self.antennas=coherent,antennas;self.independent_objective=independent_objective;self.groups=[0,0,0,0,1];self.path_count=windows[0].basis[0].shape[1];self.cross_count=windows[0].cross_basis[0].shape[1];self.count=self.path_count+self.cross_count
        prior=json.loads(Path('docs/data/calibration_20261016_sites/geometry-prior.json').read_text())
        self.rotation=np.asarray(prior['ecef_to_enu']).T
        base=windows[0].result['source_model'];track=read_csv(Path('docs/data/isar_20261013/rio-delta-ais.csv'),reference_height_ellipsoid_m=5.,mmsi=563052200)
        tx0,rx0,_=sites(Path(base['geometry_directory']));t=np.array([157.])
        self.gauge_point=SplineMotion(base).apply(t,motion_state(track,utc_seconds(base['nominal_start_utc'])+t,base),tx0[3],rx0)['position_ecef_m'][0]
        self.offset_count=6 if antennas else 0;self.phase_count=4 if coherent else 0;self.free=[0,1,2,4]
        self.calls=0;self.seconds=0.;self.cache=[None,None];self.worker=None;self.loaded=None

    def score(self,parameters,training=True,derivatives=True,save_images=None):
        started=time.monotonic();gradient=np.zeros(len(parameters));value=0.;statistics=[]
        gains=self.gains.copy()
        if self.coherent:gains[self.free]*=np.exp(1j*parameters[self.count+self.offset_count:])
        for w in self.windows:
            directory=w.directory/('training' if training else 'unused');path=directory/'model.bin'
            if self.worker is None:self.worker=DenseImages.reopen(path,derivatives=8);self.loaded=path
            elif self.loaded!=path:self.worker.replace_archive(path);self.loaded=path
            worker=self.worker;worker.training=training
            geometry,reference,k=w.geometry(parameters,worker,self.gauge_point,self.groups,self.rotation,antennas=self.antennas)
            images=worker.evaluate_geometry(geometry,derivatives=False)[:,0]
            if self.coherent:
                z=np.einsum('c,cp->p',gains.conj(),images);h=entropy(z);power=abs(z)**2
                cot=2*z*(1/power.sum()-power/np.sum(power**2))
                weight=gains[:,None]*cot[None,:]
                if derivatives:
                    for j,c in enumerate(self.free):gradient[self.count+self.offset_count+j]+=2*np.real(np.vdot(cot,-1j*gains[c].conjugate()*images[c]))
            else:
                normal=np.asarray([8192*np.sum(w.context[f'information_{c}'][w.training[c] if training else ~w.training[c]]) for c in range(5)])
                if self.independent_objective=='sum-power':
                    power=np.sum(abs(images)**2/normal[:,None],axis=0);total=power.sum();squared=(power**2).sum()
                    h=float(2*np.log(total)-np.log(squared));weight=2*images/normal[:,None]*(1/total-power/squared)[None,:]
                else:
                    power=abs(images)**2;total=power.sum(axis=1);squared=(power**2).sum(axis=1)
                    h=float(np.mean(2*np.log(total)-np.log(squared)))
                    weight=2*images*(1/total[:,None]-power/squared[:,None])/len(images)
            value+=h;statistics.append(dict(mid_s=w.result['mid_s'],entropy=h,individual=[entropy(z) for z in images]))
            if derivatives:
                row=worker.adjoint(weight,active_derivatives=7 if self.antennas else 4)
                for c in range(5):
                    n=len(w.time[c]);g=row[c,:n]
                    gradient[:self.path_count]+=w.basis[c].T@g[:,0]+w.rate[c].T@g[:,1]
                    gradient[self.path_count:self.count]+=w.cross_basis[c].T@g[:,2]+w.cross_rate[c].T@g[:,3]
                    if self.antennas:
                        at=self.count+3*self.groups[c];gradient[at:at+3]+=g[:,4:7].sum(axis=0)
            if save_images is not None:
                destination=Path(save_images);destination.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination/f'window-{w.result["mid_s"]:07.2f}.npz',individual=images,combined=np.einsum('c,cp->p',gains.conj(),images),
                    gauge_reference_m=reference,native_gains=gains*np.exp(1j*k*reference))

        self.calls+=1;self.seconds+=time.monotonic()-started
        return dict(value=value/len(self.windows),gradient=gradient/len(self.windows),windows=statistics)


def physical_constraints(model):
    """Convex-hull bounds on spline derivative control coefficients.

    They bound correction rate/acceleration for the entire continuous curve,
    including between evaluation epochs, rather than sampling a penalty.
    The path bound applies to the base radial correction plus its refinement.
    """
    matrix=[];lower=[];upper=[];size=model.count+model.offset_count+model.phase_count
    base=SplineMotion(model.windows[0].result['source_model']).path_coefficients
    for spacing,offset,count,initial,limits in [
            (model.spacing,0,model.path_count,base,(3.,.15)),
            (model.windows[0].cross_spacing,model.path_count,model.cross_count,None,(3.,.1))]:
        grid=knots(model.interval,spacing);n=len(grid)-4
        for order,limit in enumerate(limits,1):
            d=BSpline(grid,np.eye(n),3).derivative(order).c[:n-order]
            row=np.zeros((len(d),size));row[:,offset:offset+count]=d[:,:-1]
            bias=d@initial if initial is not None else 0.
            matrix.append(row);lower.extend(np.broadcast_to(-limit-bias,len(d)));upper.extend(np.broadcast_to(limit-bias,len(d)))
    return np.concatenate(matrix),np.asarray(lower),np.asarray(upper)


def kkt_residual(q,gradient,bounds,matrix=None,lower=None,upper=None):
    normals=[]
    for j,(lo,hi) in enumerate(bounds):
        if q[j]<lo+1e-6:normals.append(-np.eye(1,len(q),j)[0])
        if q[j]>hi-1e-6:normals.append(np.eye(1,len(q),j)[0])
    violation=0.
    if matrix is not None:
        value=matrix@q;violation=float(max(0.,max(lower-value),max(value-upper)))
        normals.extend(-matrix[value<lower+1e-6]);normals.extend(matrix[value>upper-1e-6])
    if normals:
        a=np.asarray(normals);a/=np.linalg.norm(a,axis=1)[:,None]
        multiplier=nnls(a.T,-gradient,maxiter=10*len(a))[0];residual=gradient+a.T@multiplier
    else:residual=gradient
    return dict(maximum_kkt_residual=float(max(abs(residual))),maximum_constraint_violation=violation)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--windows',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--spacing-s',type=float,default=3.);p.add_argument('--cross-spacing-s',type=float,default=15.);p.add_argument('--iterations',type=int,default=60);p.add_argument('--coherent',action='store_true');p.add_argument('--antennas',action='store_true')
    p.add_argument('--physical-bounds',action='store_true')
    p.add_argument('--independent-objective',choices=['sum-power','mean-entropy'],default='sum-power')
    p.add_argument('--initial',type=Path);p.add_argument('--limit',type=int);a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False)
    paths=sorted(a.windows.glob('window-*/analysis.json'));paths=paths[:a.limit] if a.limit else paths
    interval=[3.2720443125000007,601.8136398125001];windows=[Window(f.parent,interval,a.spacing_s,a.cross_spacing_s) for f in paths]
    source=json.loads(Path('docs/data/isar_20261017/coherent-ship-hull-north-up/analysis.json').read_text());gains=np.asarray([complex(*z) for z in source['gains_real_imag']])
    model=Joint(windows,interval,a.spacing_s,gains,coherent=a.coherent,antennas=a.antennas,independent_objective=a.independent_objective)
    initial=np.r_[curve_seed(windows,interval,a.spacing_s),np.zeros(model.cross_count+model.offset_count+model.phase_count)]
    if a.initial:
        old=json.loads(a.initial.read_text());previous=np.asarray(old['parameters']);initial[:model.path_count]=previous[:model.path_count]
        if old.get('parameter_counts',{}).get('cross')==model.cross_count:initial[model.path_count:model.count]=previous[model.path_count:model.count]
        if old.get('parameter_counts',{}).get('antenna')==model.offset_count and model.offset_count:
            old_basis=np.asarray(old.get('antenna_offset_basis_ecef',enu_basis(-37.81040454194318,144.98744506389454)))
            old_offset=previous[model.count:model.count+model.offset_count].reshape(-1,3)
            initial[model.count:model.count+model.offset_count]=(old_offset@old_basis.T@model.rotation).ravel()
        if old.get('parameter_counts',{}).get('phase')==model.phase_count and model.phase_count:initial[-model.phase_count:]=previous[-model.phase_count:]
    scale=np.r_[np.ones(model.path_count),np.full(model.cross_count,20.),np.full(model.offset_count,10.),np.ones(model.phase_count)]
    bounds=[(-100.,100.)]*model.path_count+[(-200.,200.)]*model.cross_count+([(-50.,50.),(-50.,50.),(-25.,25.)]*2 if a.antennas else [])+[(-4*np.pi,4*np.pi)]*model.phase_count
    result=dict(complete=False,windows=[str(w.directory) for w in windows],interval_s=interval,path_spacing_seconds=a.spacing_s,cross_spacing_seconds=a.cross_spacing_s,
        base_model=windows[0].result['source_model'],parameter_counts=dict(path=model.path_count,cross=model.cross_count,antenna=model.offset_count,phase=model.phase_count),
        coherent_objective=a.coherent,independent_objective=a.independent_objective,antennas_fitted=a.antennas,transmitter_groups=model.groups,
        antenna_offset_basis_ecef=model.rotation,antenna_offset_frame='ENU at surveyed receiver',
        calibration_gains_real_imag=np.c_[gains.real,gains.imag],gauge_point_ecef_m=model.gauge_point,trace=[])
    def save():(out/'analysis.json').write_text(json.dumps(serial(result),indent=2,allow_nan=False)+'\n')
    cache=[None,None]
    def evaluate(q):
        if cache[0] is None or not np.array_equal(cache[0],q):cache[:]=[q.copy(),model.score(q*scale)]
        return cache[1]
    def objective(q):
        r=evaluate(q);return r['value'],r['gradient']*scale
    scaled_bounds=[(lo/s,hi/s) for (lo,hi),s in zip(bounds,scale)];constraints=[];matrix=low=high=None
    if a.physical_bounds:
        raw,low,high=physical_constraints(model);matrix=raw*scale[None,:]
        constraints=[LinearConstraint(matrix,low,high)]
        seed=initial/scale
        projected=minimize(lambda q:(.5*np.sum((q-seed)**2),q-seed),np.clip(seed,*np.asarray(scaled_bounds).T),jac=True,
            method='SLSQP',bounds=scaled_bounds,constraints=constraints,options=dict(maxiter=300,ftol=1e-11))
        violation=max(0.,max(low-matrix@projected.x),max(matrix@projected.x-high))
        if not projected.success or violation>1e-7:raise ValueError(f'Physical initializer projection failed: {projected.message}; violation={violation}')
        initial=projected.x*scale
        result['physical_bounds']=dict(path_rate_correction_mps=3.,path_acceleration_correction_mps2=.15,
            transverse_rate_correction_mps=3.,transverse_acceleration_correction_mps2=.1,
            scope='Bounds on spline derivative control coefficients, valid over continuous time. No penalty on image pixels or observations.')
    before=evaluate(initial/scale);result['training_before']=before;save()
    def callback(q):
        r=evaluate(q);record=dict(iteration=len(result['trace'])+1,objective=r['value'],maximum_gradient=float(max(abs(r['gradient']*scale))),elapsed_s=model.seconds)
        result['trace'].append(record);result['parameters']=q*scale;save();print(json.dumps(record),flush=True)
    fitted=minimize(objective,initial/scale,jac=True,method='SLSQP' if a.physical_bounds else 'L-BFGS-B',bounds=scaled_bounds,callback=callback,
        constraints=constraints,options=dict(maxiter=a.iterations,ftol=2e-8) if a.physical_bounds else dict(maxiter=a.iterations,maxls=12,maxcor=20,ftol=2e-8,gtol=1e-5))
    parameters=fitted.x*scale;result.update(parameters=parameters,training_after=evaluate(fitted.x),message=fitted.message,
        maximum_gradient=float(max(abs(fitted.jac))),calls=model.calls,seconds=model.seconds,
        stationarity=kkt_residual(fitted.x,fitted.jac,scaled_bounds,matrix,low,high))
    result['unused_before']=model.score(initial,training=False,derivatives=False)
    result['unused_after']=model.score(parameters,training=False,derivatives=False,save_images=out/'unused-images')
    result['complete']=True;save();model.worker.close();model.worker=None

if __name__=='__main__':main()

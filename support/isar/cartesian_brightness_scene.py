"""Multiscatterer CV fit with constant band-calibration nuisance parameters.

The midpoint phase gauge is an invertible coordinate change: it removes
stiff constant carrier terms while keeping every physical delay and phase.
Only four constant relative band phases are unknown. No parameter may supply
a band-specific phase history, time warp, velocity or target trajectory.
"""
import numpy as np
from scipy.optimize import minimize,LinearConstraint
from amplitude_arc import qr_nnls
from rigid_brightness_scene import RigidBrightnessBand,RigidBrightnessFit


class CartesianBrightnessBand(RigidBrightnessBand):
    def __init__(self,*args,channel,reference_channel=3,fixed_calibration=False,reference_rf_hz=None,reference_transmitter=None,**kwargs):
        super().__init__(*args,**kwargs)
        self.channel,self.reference_channel=channel,reference_channel
        self.fixed_calibration=fixed_calibration
        self.reference_k=None if reference_rf_hz is None else 2*np.pi*reference_rf_hz/299792458.
        self.reference_transmitter=None if reference_transmitter is None else np.asarray(reference_transmitter)
        if fixed_calibration and (self.reference_k is None or self.reference_k<=0 or self.reference_transmitter is None or self.reference_transmitter.shape!=(3,)):
            raise ValueError('Fixed physical calibration needs one shared reference frequency and transmitter')
        self.calibration_start=self.size;self.size+=0 if fixed_calibration else 4
        self.calibration_index=None if fixed_calibration or channel==reference_channel else self.calibration_start+channel-(channel>reference_channel)

    def responses(self,parameters,derivatives=True):
        raw,change=super().responses(parameters,derivatives)
        antenna=self.reference_transmitter if self.fixed_calibration else self.field.data['transmitter_ecef_m']
        k=self.reference_k if self.fixed_calibration else self.field.k
        delta,gradient=self.kinematics.midpoint_path_change(parameters,antenna)
        phi=k*delta+(parameters[self.calibration_index] if self.calibration_index is not None else 0.)
        phase=np.exp(1j*phi)
        result=[h*phase for h in raw]
        if derivatives:
            for j,(h,d) in enumerate(zip(result,change)):
                d*=phase;d[:,:,:2]+=1j*h[:,:,None]*(k*gradient)
                if self.calibration_index is not None:d[:,:,self.calibration_index]+=1j*h
        return result,change

class CartesianBrightnessFit(RigidBrightnessFit):
    def settings(self,position_bound_m=60.):
        n=len(self.bands[0].points);m=self.bands[0].motion_size;extra=self.bands[0].calibration_start-m-n
        calibration=self.size-self.bands[0].calibration_start
        scale=np.r_[25.,25.,.1,.1,.1,np.full(m-5,50.),np.full(n,.3),np.full(extra,1.),np.full(calibration,.3)]
        limit=np.r_[250.,250.,1.,1.,.5,np.full(m-5,250.),np.full(n,np.inf),np.full(extra,position_bound_m),np.full(calibration,np.inf)]
        return scale,limit

    def polish(self,initial,*,max_iterations=120,position_bound_m=60.):
        """Invertible mean-information coordinates; unchanged likelihood/bounds.

        Small modes are capped only in the optimiser's step coordinates.
        No statistical loading, prior, truncated mode or fitted data transform
        is introduced. Every physical parameter remains free within its bounds.
        """
        initial=np.asarray(initial);scale,limit=self.settings(position_bound_m);blocks=[]
        for b in self.bands:
            v=b.evaluate_scene(initial,full=True)
            a=b.nuisance.remove(v['design'][b.selected]);a=np.r_[a.real,a.imag]
            j=v['jacobian'];j=np.r_[j.real,j.imag]
            blocks.append(j-a@np.linalg.lstsq(a,j,rcond=1e-12)[0])
        j=np.concatenate(blocks)*scale/np.sqrt(self.energy)
        _,singular,vh=np.linalg.svd(j,full_matrices=False)
        if not len(singular) or not np.isfinite(singular).all() or singular[0]<=0:
            raise ValueError('The frozen scene has no observable mean-response direction')
        transform=scale[:,None]*vh.T/np.maximum(singular,singular[0]/1e4)
        bounded=np.isfinite(limit);constraint=LinearConstraint(transform[bounded],-limit[bounded]-initial[bounded],limit[bounded]-initial[bounded])
        def objective(q):
            value,gradient=self.objective(initial+transform@q)
            return value,gradient@transform
        found=minimize(objective,np.zeros(self.size),jac=True,method='SLSQP',constraints=[constraint],
            options=dict(maxiter=max_iterations,ftol=2e-14))
        p=initial+transform@found.x;result=self.assess(p,position_bound_m=position_bound_m)
        result.update(success=bool(found.success and result['maximum_scaled_projected_gradient']<1e-7),
            iterations=found.nit,message=str(found.message),optimizer='SLSQP, invertible mean-information coordinates',
            initial_scaled_mean_singular_values=singular,optimizer_singular_value_condition_cap=1e4)
        return result

    def information(self,parameters):
        result=super().information(parameters)
        if not self.bands[0].fixed_calibration:
            result['joint_constant_band_calibration']=result.pop('fixed_band_calibration')
            result.pop('unknown_constant_band_phases')
        result['motion_coordinates']=['east_position_m','north_position_m','east_velocity_mps','north_velocity_mps','counterclockwise_spin_degps']
        if self.bands[0].motion_size==7:result['motion_coordinates']+=['anchor_from_rotation_centre_east_m','anchor_from_rotation_centre_north_m']
        result['scope']='Conditional local mean information after source phases, relative point positions, stationary profiles and all real brightness coefficients are profiled. Constant band phases are either declared fixed physical calibration or explicit nuisance parameters, as labelled. No AIS position/velocity prior or active search-bound information is counted. Height, transmitter/receiver priors, source support and noise covariance remain conditional.'
        return result

    def physical_calibration(self,parameters):
        """Undo the exact midpoint/source-coordinate phase gauges for export."""
        fixed=self.bands[0].fixed_calibration
        if any(b.fixed_calibration!=fixed for b in self.bands):raise ValueError('Use one physical calibration hypothesis across all bands')
        phases=[]
        for b in self.bands:
            delta,_=b.kinematics.midpoint_path_change(parameters,b.field.data['transmitter_ecef_m'])
            phases.append(b.band_phase if fixed else b.band_phase+b.field.k*delta+(parameters[b.calibration_index] if b.calibration_index is not None else 0.))
        ref=self.bands[0].reference_channel
        reference=self.bands[ref]
        delta,_=reference.kinematics.midpoint_path_change(parameters,reference.reference_transmitter if fixed else reference.field.data['transmitter_ecef_m'])
        gauge=(reference.reference_k if fixed else reference.field.k)*delta
        m=self.bands[0].motion_size
        point_phase=np.asarray(parameters)[m:m+len(self.bands[0].points)].copy()+gauge
        if self.bands[0].position_gauge is not None:
            n=len(point_phase);offset=np.asarray(parameters)[m+n:self.bands[0].calibration_start].reshape(-1,2)
            point_phase[1:]+=np.sum(self.bands[0].position_gauge[1:]*offset,axis=1)
        return np.asarray(phases)-(0. if fixed else gauge),point_phase

"""One envelope-origin nuisance per band, shared by vehicles and ship.

The origin is an absolute excess-path offset in metres in both forward models.
It changes the envelope only: each target/band already has an independent
unknown complex gain, so a constant carrier phase supplies no extra constraint.
This control changes the forward model, never the observed samples or weights.
"""
import numpy as np

from shared_target_geometry import SharedGeometry, complex_gain, path_partials


class OriginRoad:
    def __init__(self, original):
        self.original=original
        self.initial=original.initial.copy()
        self.nmotion,self.antennas,self.energy=original.nmotion,original.antennas,original.energy
        self.training,self.validation=original.training,original.validation
        self.nband=len(self.training)

    def template(self,motion,antenna,origins,channel,field):
        source=self.original
        position,jacobian,_=source.problem.model.positions(motion,field.time,source.problem.sites)
        mapping=source.mappings[channel];offset=(mapping@antenna).reshape(2,3)
        path,partial=path_partials(position,jacobian,source.transmitters[channel]+offset[0],offset[1])
        derivative=np.c_[partial[:,:self.nmotion],partial[:,self.nmotion:]@mapping]
        u,du=field.template(path,derivative,range_bias=origins[channel],bias_derivative=True)
        result=np.zeros((len(u),self.nmotion+self.antennas.count+self.nband),complex)
        result[:,:-self.nband]=du[:,:-1]
        result[:,-self.nband+channel]=du[:,-1]
        return u,result

    def evaluate(self,motion,antenna,origins,*,include_motion_prior=True):
        size=self.nmotion+self.antennas.count+self.nband
        value,gradient,curvature=0.,np.zeros(size),np.zeros((size,size))
        information=[];coefficients=[]
        for c,field in enumerate(self.training):
            u,du=self.template(motion,antenna,origins,c,field)
            residual,jacobian,amplitude,mean=complex_gain(field.y,u,du)
            value+=float(np.vdot(residual,residual).real)
            gradient+=2*np.real(jacobian.conj().T@residual)
            curvature+=2*np.real(jacobian.conj().T@jacobian)
            information.append(np.sqrt(2)*np.r_[mean.real,mean.imag]);coefficients.append(amplitude)
        data_value=value
        if include_motion_prior:
            problem=self.original.problem
            residual=np.concatenate(problem.auxiliary(motion))
            derivative=np.concatenate(problem.auxiliary(motion,jacobian=True))
            value+=.5*float(residual@residual)
            gradient[:self.nmotion]+=derivative.T@residual
            curvature[:self.nmotion,:self.nmotion]+=derivative.T@derivative
        return {'value':value,'data_value':data_value,'motion_prior_value':value-data_value,
            'gradient':gradient,'gauss_newton':curvature,'coefficients':coefficients,
            'expected_real_jacobian':np.concatenate(information)}

    def predict(self,motion,antenna,origins):
        result=[]
        for c,(train,test) in enumerate(zip(self.training,self.validation)):
            u,du=self.template(motion,antenna,origins,c,train)
            _,_,amplitude,_=complex_gain(train.y,u,du)
            v,_=self.template(motion,antenna,origins,c,test)
            predicted=amplitude*v;norm=float(np.vdot(predicted,predicted).real)
            result.append({'prediction_error':float(np.vdot(test.y-predicted,test.y-predicted).real),
                'data_energy':float(np.vdot(test.y,test.y).real),
                'matched_energy':float(abs(np.vdot(predicted,test.y))**2/norm) if norm else 0.,
                'coefficient_real_imag':[float(amplitude.real),float(amplitude.imag)]})
        return result


class SharedOriginGeometry(SharedGeometry):
    def __init__(self,original):
        super().__init__(original.ship,[OriginRoad(r) for r in original.roads],
                         initial_ship=original.initial[:original.ship_count])
        self.initial=original.initial.copy()
        self.range_slice=self.ship.range_slice

    def evaluate(self,parameters):
        if self.last is not None and np.array_equal(parameters,self.last['parameters']):return self.last
        ship=self.ship.evaluate(parameters[:self.ship_count])
        value=ship['value']*self.ship.energy
        gradient=np.zeros(len(parameters));curvature=np.zeros((len(parameters),len(parameters)))
        gradient[:self.ship_count]=ship['gradient']*self.ship.energy
        curvature[:self.ship_count,:self.ship_count]=ship['gauss_newton']*self.ship.energy
        diagnostics=[];indices=np.arange(len(parameters))
        for road,section in zip(self.roads,self.road_slices):
            fitted=road.evaluate(parameters[section],parameters[self.antenna_slice],parameters[self.range_slice])
            columns=np.r_[indices[section],indices[self.antenna_slice],indices[self.range_slice]]
            value+=fitted['value'];gradient[columns]+=fitted['gradient']
            curvature[np.ix_(columns,columns)]+=fitted['gauss_newton']
            diagnostics.append({k:fitted[k] for k in ('data_value','motion_prior_value')})
        self.evaluations+=1
        self.last={'parameters':np.array(parameters,copy=True),'value':value/self.energy,
            'gradient':gradient/self.energy,'gauss_newton':curvature/self.energy,
            'scene_conditions':ship['scene_conditions'],'road_values':diagnostics,
            'raw_ship_value_including_one_antenna_prior':ship['value']*self.ship.energy,
            'evaluations':self.evaluations}
        return self.last

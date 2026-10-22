"""Exact off-grid complex-measure image likelihood with a total-mass prior.

For fixed locations/motion the coefficient problem is a convex complex
LASSO. Spatial and motion derivatives differentiate the same physical
waveforms and profile those coefficients. The total-mass penalty prevents
unbounded opposite-phase coefficient pairs from being a free improvement.
Global support discovery/certification belongs to the acquisition driver.
"""
import copy
import numpy as np
from complex_lasso import ComplexLasso
from rigid_patch import profile_static


class ContinuousSparseScene:
    def __init__(self,bands,basis,rates,spectrum,strength,*,height_m=20.,kkt_tolerance=1e-8,gap_tolerance=1e-8):
        self.bands=bands;self.basis=basis;self.rates=rates;self.spectrum=np.asarray(spectrum)
        self.strength=strength;self.height=height_m;self.kkt_tolerance=kkt_tolerance;self.gap_tolerance=gap_tolerance
        self.data=[profile_static(b.field.y,b.field.root_information,b.training,b.groups)*b.training[:,None] for b in bands]
        self.energy=float(sum(np.vdot(y,y).real for y in self.data));self.warm=None

    def shifted_bands(self,path):
        result=[]
        for original,basis,rate in zip(self.bands,self.basis,self.rates):
            band=copy.copy(original);band.base_path=original.base_path+basis@path;band.base_rate=original.base_rate+rate@path
            result.append(band)
        return result

    def evaluate(self,parameters,full=False):
        path=parameters[:3];xy=parameters[3:].reshape(-1,2);points=np.c_[xy,np.full(len(xy),self.height)]
        if not len(points):raise ValueError('Need at least one acquired atom')
        bands=self.shifted_bands(path);columns=[];derivatives=[];gram=0.;rhs=0.
        for c,(band,gain,y) in enumerate(zip(bands,self.spectrum,self.data)):
            h=np.empty((*y.shape,len(points)),complex);d=np.empty((4,*h.shape),complex)
            for j,point in enumerate(points):
                h[:,:,j],d[0,:,:,j],d[1,:,:,j],d[2,:,:,j],d[3,:,:,j]=band.waveform(
                    np.zeros(band.b.shape[1]),c,True,point-band.reference_point,spatial_derivatives=True)
            h*=gain;d*=gain
            projected=profile_static(h,band.field.root_information,band.training,band.groups)*band.training[:,None,None]
            a=projected.reshape(-1,len(points));gram=gram+a.conj().T@a;rhs=rhs+a.conj().T@y.ravel()
            columns.append(h);derivatives.append(d)
        solver=ComplexLasso(gram,rhs,self.energy)
        initial=self.warm if self.warm is not None and len(self.warm)==len(points) else None
        coefficient,certificate=solver.solve(self.strength,initial,kkt_tolerance=self.kkt_tolerance,gap_tolerance=self.gap_tolerance,maximum_iterations=30000)
        if not certificate['numerical_convergence']:raise ValueError('Continuous-scene coefficient solve missed its convex certificate')
        self.warm=coefficient;gradient_path=np.zeros(3);gradient_points=np.zeros((len(points),2));raw=[];error=0.
        for band,h,d,y,basis,rate in zip(bands,columns,derivatives,self.data,self.basis,self.rates):
            prediction=h@coefficient;raw.append(prediction)
            residual=(profile_static(prediction,band.field.root_information,band.training,band.groups)*band.training[:,None])-y
            error+=float(np.vdot(residual,residual).real)
            dr=d[0]@coefficient;dv=d[1]@coefficient
            gr=2*np.real(np.sum(residual.conj()*dr,axis=1));gv=2*np.real(np.sum(residual.conj()*dv,axis=1))
            gradient_path+=basis.T@gr+rate.T@gv
            for axis in range(2):gradient_points[:,axis]+=2*np.real(coefficient*np.einsum('tr,trp->p',residual.conj(),d[2+axis],optimize=True))
        penalty=float(2*self.strength*sum(abs(coefficient)));cost=error+penalty
        algebra=float(self.energy-2*np.vdot(coefficient,rhs).real+np.vdot(coefficient,gram@coefficient).real+penalty)
        if abs(cost-algebra)>1e-9*self.energy:raise ValueError('Physical off-grid likelihood failed original-row closure')
        gradient=np.r_[gradient_path,gradient_points.ravel()]/self.energy
        report=dict(cost=cost,energy=self.energy,unpenalized_cost=error,penalty=penalty,coefficient=coefficient,
            points_body_m=points,raw_prediction=raw,coefficient_optimization={k:v for k,v in certificate.items() if k!='trace'},
            original_row_closure_error=float(abs(cost-algebra)/self.energy))
        return (cost/self.energy,gradient,report) if full else (cost/self.energy,gradient)

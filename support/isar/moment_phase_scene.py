"""Original-epoch sufficient moments for common polynomial-path acquisition.

Only the analytic phase weights are interpolated, with their errors checked
at every original epoch and derivative. Observations are never resampled.
Stationary nuisance is profiled at each trial. The raw-row implementation
remains the independent reference; image priors are injected separately.
"""
import numpy as np
from numpy.polynomial.chebyshev import chebvander,chebinterpolate,chebval
from profiled_phase_scene import ProfiledPhaseScene
from rigid_patch import profile_static


class MomentPhaseScene(ProfiledPhaseScene):
    def __init__(self,design,data,root,groups,selected,basis,wavenumber,strength,spectrum=None,*,degree=64,projected_gram=None,operator_moments=None,image_solver=None,progress=None):
        self.design=design;self.data=data;self.root=root;self.groups=groups;self.selected=selected;self.basis=basis
        self.k=np.asarray(wavenumber);self.strength=float(strength);self.spectrum=np.ones(len(design),complex) if spectrum is None else np.asarray(spectrum,complex)
        self.degree=degree;self.image_solver=image_solver;self.energy=0.;self.gram=0.;self.mean_moments=[];self.score_moments=[];self.u=[]
        self.parameters=basis[0].shape[1]
        for c,(h,y,r,g,mask,b) in enumerate(zip(design,data,root,groups,selected,basis)):
            u=b[:,0];expected=np.column_stack([u**j for j in range(1,self.parameters+1)])
            if abs(u).max()>1+1e-12 or b.shape[1]!=self.parameters or max(abs((b-expected).ravel()))>1e-12:
                raise ValueError('Moment acquisition requires a declared monomial path basis on [-1,1]')
            self.u.append(u);vander=chebvander(u,degree)
            q=[]
            for group in np.unique(g):
                value=r*mask*(g==group);norm=np.linalg.norm(value)
                if norm:q.append(value/norm)
            q=np.asarray(q)
            if operator_moments is None:
                weight=(q[:,None,:]*vander.T[None,:,:]).reshape(-1,len(u))
                moments=(weight@h.reshape(len(u),-1)).reshape(len(q),degree+1,h.shape[1],h.shape[2])
                moments=moments.transpose(1,0,2,3).reshape(degree+1,-1,h.shape[2])
            else:
                moments=np.asarray(operator_moments[c])
                if moments.shape!=(degree+1,len(q)*h.shape[1],h.shape[2]):raise ValueError('Original-epoch operator moments changed dimensions')
            self.mean_moments.append(moments)
            projected=profile_static(y,r,mask,g)*mask[:,None];self.energy+=float(np.vdot(projected,projected).real)
            score=np.einsum('trp,tr->tp',h,projected.conj(),optimize=True).conj()
            self.score_moments.append(vander.T@score)
            if projected_gram is None:
                a=h[mask].reshape(-1,h.shape[2]);raw=a.conj().T@a
            else:raw=np.asarray(projected_gram[c])+moments[0].conj().T@moments[0]
            self.gram=self.gram+abs(self.spectrum[c])**2*raw
            if progress:progress(dict(channel=c,degree=degree,modes=h.shape[2]))
        if self.energy<=0 or strength<=0:raise ValueError('Invalid moment phase likelihood')

    def phase_coefficients(self,parameters,channel):
        k=self.k[channel];u=self.u[channel];b=self.basis[channel]
        def values(nodes):
            curve=sum(value*nodes**(j+1) for j,value in enumerate(parameters))
            return np.exp(-1j*k*curve)
        coefficients=chebinterpolate(values,self.degree)
        exact=np.exp(-1j*k*(b@parameters));error=float(max(abs(chebval(u,coefficients)-exact)))
        derivatives=[]
        for j in range(self.parameters):
            derivative=chebinterpolate(lambda nodes:(-1j*k*nodes**(j+1))*values(nodes),self.degree)
            error=max(error,float(max(abs(chebval(u,derivative)-(-1j*k*b[:,j])*exact)))/max(k,1.))
            derivatives.append(derivative)
        if error>1e-11:raise ValueError(f'Analytic phase weights exceed their original-epoch precision gate: {error}')
        return coefficients,np.asarray(derivatives),error

    def normal_with_derivatives(self,parameters):
        gram=self.gram.copy();rhs=np.zeros(len(gram),complex);db=np.zeros((self.parameters,len(gram)),complex);mean=[];derivatives=[];maximum_error=0.
        for c,(moments,score) in enumerate(zip(self.mean_moments,self.score_moments)):
            phase,dp,error=self.phase_coefficients(parameters,c);maximum_error=max(maximum_error,error);gain=self.spectrum[c]
            m=gain*np.einsum('d,drp->rp',phase,moments,optimize=True)
            dm=gain*np.einsum('jd,drp->jrp',dp,moments,optimize=True)
            gram-=m.conj().T@m;rhs+=gain.conjugate()*(phase.conj()@score);db+=gain.conjugate()*(dp.conj()@score)
            mean.append(m);derivatives.append(dm)
        return (gram+gram.conj().T)/2,rhs,db,mean,derivatives,maximum_error

    def objective(self,parameters,full=False):
        gram,rhs,db,mean,dm,error=self.normal_with_derivatives(parameters)
        coefficient,penalty,details=self.solve_image(gram,rhs)
        cost=float(self.energy-2*np.vdot(coefficient,rhs).real+np.vdot(coefficient,gram@coefficient).real+penalty)
        gradient=-2*np.real(db@coefficient.conj())
        for m,derivative in zip(mean,dm):
            value=m@coefficient;change=np.einsum('jrp,p->jr',derivative,coefficient)
            gradient-=2*np.real(change@value.conj())
        report=dict(coefficient=coefficient,cost=cost,energy=self.energy,maximum_phase_weight_error=error,**details)
        return (cost/self.energy,gradient/self.energy,report) if full else (cost/self.energy,gradient/self.energy)

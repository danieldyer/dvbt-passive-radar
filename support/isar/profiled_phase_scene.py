"""Joint common-path phase acquisition with the complex image eliminated.

For each trajectory trial, solve the stated ridge-regularized image exactly
and profile the original stationary nuisance. This is a phase-only coarse
acquisition model: envelope/geometry updates require subsequent validation
against the full physical waveform. All bands share one physical path.
"""
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from rigid_patch import profile_static


class ProfiledPhaseScene:
    def __init__(self,design,data,root,groups,selected,basis,wavenumber,strength,spectrum=None,*,image_solver=None):
        self.design=design;self.data=data;self.root=root;self.groups=groups;self.selected=selected;self.basis=basis
        self.k=np.asarray(wavenumber);self.strength=float(strength);self.spectrum=np.ones(len(design),complex) if spectrum is None else np.asarray(spectrum,complex)
        self.image_solver=image_solver
        if strength<=0 or len(design)!=len(data) or len(design)!=len(self.spectrum):raise ValueError('Invalid coherent image-acquisition model')
        self.projected=[];self.q=[];self.gram=0.;self.energy=0.
        for c,(h,y,r,g,mask) in enumerate(zip(design,data,root,groups,selected)):
            projected=profile_static(y,r,mask,g)*mask[:,None];self.projected.append(projected)
            self.energy+=float(np.vdot(projected,projected).real)
            a=h[mask].reshape(-1,h.shape[2]);self.gram=self.gram+abs(self.spectrum[c])**2*(a.conj().T@a)
            columns=[]
            for group in np.unique(g):
                q=r*mask*(g==group);norm=np.linalg.norm(q)
                if norm:columns.append(q/norm)
            self.q.append(np.asarray(columns))
        if self.energy<=0:raise ValueError('No original training-data energy')

    def normal(self,parameters):
        gram=self.gram.copy();rhs=np.zeros(len(gram),complex);phases=[]
        for c,(h,y,q,b) in enumerate(zip(self.design,self.projected,self.q,self.basis)):
            phase=self.spectrum[c]*np.exp(-1j*self.k[c]*(b@parameters));phases.append(phase)
            mean=np.einsum('gt,t,trp->grp',q,phase,h,optimize=True).reshape(-1,h.shape[2])
            gram-=mean.conj().T@mean
            rhs+=np.einsum('trp,tr,t->p',h.conj(),y,phase.conj(),optimize=True)
        return (gram+gram.conj().T)/2,rhs,phases

    def solve_image(self,gram,rhs):
        if self.image_solver is not None:return self.image_solver(gram,rhs,self.energy,self.spectrum)
        matrix=gram.copy();matrix.flat[::len(matrix)+1]+=self.strength
        coefficient=cho_solve(cho_factor(matrix,lower=True),rhs)
        penalty=float(self.strength*np.vdot(coefficient,coefficient).real)
        return coefficient,penalty,dict(maximum_scene_normal_residual=float(abs(matrix@coefficient-rhs).max()/self.energy))

    def objective(self,parameters,full=False):
        gram,rhs,phases=self.normal(parameters);coefficient,penalty,details=self.solve_image(gram,rhs)
        cost=float(self.energy-2*np.vdot(coefficient,rhs).real+np.vdot(coefficient,gram@coefficient).real+penalty)
        gradient=np.zeros_like(parameters);records=[]
        direct=0.
        for c,(h,y,phase,b,root,groups,mask) in enumerate(zip(self.design,self.projected,phases,self.basis,self.root,self.groups,self.selected)):
            raw=np.einsum('trp,p->tr',h,coefficient)*phase[:,None]
            prediction=profile_static(raw,root,mask,groups)*mask[:,None];residual=prediction-y
            error=float(np.vdot(residual,residual).real);direct+=error
            time_gradient=2*np.real(np.sum(residual.conj()*(-1j*self.k[c]*raw),axis=1))
            gradient+=b.T@time_gradient
            records.append(dict(cost=error,energy=float(np.vdot(y,y).real)))
        direct+=penalty
        if abs(direct-cost)>1e-8*self.energy:raise ValueError('Profiled scene and original-row likelihood disagree')
        report=dict(coefficient=coefficient,cost=cost,energy=self.energy,channels=records,**details)
        return (cost/self.energy,gradient/self.energy,report) if full else (cost/self.energy,gradient/self.energy)

    def score(self,parameters,coefficient,selected):
        cost=0.;energy=0.
        for c,(h,y,r,g,mask,b) in enumerate(zip(self.design,self.data,self.root,self.groups,selected,self.basis)):
            raw=np.einsum('trp,p->tr',h,coefficient)*(self.spectrum[c]*np.exp(-1j*self.k[c]*(b@parameters)))[:,None]
            observed=profile_static(y,r,mask,g)*mask[:,None]
            residual=profile_static(raw,r,mask,g)*mask[:,None]-observed
            cost+=float(np.vdot(residual,residual).real);energy+=float(np.vdot(observed,observed).real)
        return dict(cost=cost,energy=energy,explained_fraction=1-cost/energy)

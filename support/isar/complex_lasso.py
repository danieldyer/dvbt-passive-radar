"""Complex image LASSO with explicit KKT and convex primal/dual certificates.

The uniform Laplace prior is statistical regularization. ADMM rho and one
global invertible numerical scale affect convergence only. No cell-wise
normalization, source-position fitting or phase screen is introduced.
"""
import numpy as np
from scipy.linalg import eigh


class ComplexLasso:
    def __init__(self,gram,rhs,energy):
        self.gram=np.asarray(gram);self.rhs=np.asarray(rhs);self.energy=float(energy)
        self.eigenvalue,self.vector=eigh((self.gram+self.gram.conj().T)/2)
        self.largest=float(self.eigenvalue[-1]);self.linear_scale=float(max(abs(rhs)))
        if self.largest<=0 or self.linear_scale<=0 or energy<=0:raise ValueError('Need a nonzero original-data image likelihood')
        if self.eigenvalue[0]<-1e-10*self.largest:raise ValueError('Image normal matrix is not positive semidefinite')
        self.amplitude_scale=self.linear_scale/self.largest

    def ridge_step(self,rhs,rho):
        eigen=self.eigenvalue/self.largest
        return self.vector@((self.vector.conj().T@rhs)/(eigen+rho))

    def certificate(self,x,strength):
        gx=self.gram@x;correlation=self.rhs-gx;magnitude=abs(x)
        residual=np.maximum(abs(correlation)-strength,0.)
        active=magnitude>0
        residual[active]=abs(correlation[active]-strength*x[active]/magnitude[active])
        quadratic=float(np.vdot(x,gx).real);linear=float(np.vdot(x,self.rhs).real)
        rss=self.energy-2*linear+quadratic;penalty=strength*float(magnitude.sum())
        primal=.5*rss+penalty
        feasible=min(1.,strength/max(float(max(abs(correlation))),1e-300))
        dual=feasible*(self.energy-linear)-.5*feasible**2*rss
        gap=primal-dual
        if gap<-1e-9*max(primal,self.energy):raise ValueError('Complex LASSO primal value is below its feasible dual')
        return dict(maximum_scaled_kkt=float(max(residual)/self.linear_scale),primal_objective=float(primal),
            feasible_dual_objective=float(dual),primal_dual_gap=float(gap),gap_relative_to_data_energy=float(gap/self.energy),
            gap_relative_to_fitted_energy=float(gap/max(abs(linear),self.energy*np.finfo(float).eps)),
            nonzero_coefficients=int(np.count_nonzero(active)))

    def solve(self,strength,initial=None,*,maximum_iterations=10000,kkt_tolerance=1e-6,gap_tolerance=1e-5):
        if strength<=0:raise ValueError('Use a positive complex Laplace-prior strength')
        n=len(self.rhs);scale=self.amplitude_scale;linear=self.rhs/self.linear_scale;prior=strength/self.linear_scale
        z=np.zeros(n,complex) if initial is None else np.asarray(initial,complex).copy()/scale
        dual=np.zeros(n,complex);rho=.05;trace=[];report=None
        for iteration in range(maximum_iterations):
            rhs=linear+rho*(z-dual)
            x=self.ridge_step(rhs,rho)
            old=z.copy();value=x+dual;magnitude=abs(value)
            factor=np.maximum(0.,1-np.divide(prior/rho,magnitude,out=np.full(n,np.inf),where=magnitude>0))
            z=factor*value;dual+=x-z
            if iteration%10==0 or iteration+1==maximum_iterations:
                primal=float(np.linalg.norm(x-z));dual_residual=float(rho*np.linalg.norm(z-old))
                report=self.certificate(z*scale,strength)
                trace.append(dict(iteration=iteration+1,rho=rho,primal_residual=primal,dual_residual=dual_residual,
                    maximum_scaled_kkt=report['maximum_scaled_kkt'],relative_gap=report['gap_relative_to_fitted_energy']))
                if report['maximum_scaled_kkt']<kkt_tolerance and report['gap_relative_to_fitted_energy']<gap_tolerance:break
                previous=rho
                # Standard residual balancing changes the numerical
                # penalty only; it does not change the Laplace prior.
                if primal>10*dual_residual:rho=min(rho*2,1e3)
                elif dual_residual>10*primal:rho=max(rho/2,1e-7)
                if rho!=previous:dual*=previous/rho
        report.update(strength=float(strength),strength_relative_to_null=float(prior),iterations=iteration+1,
            numerical_convergence=bool(report['maximum_scaled_kkt']<kkt_tolerance and report['gap_relative_to_fitted_energy']<gap_tolerance),
            kkt_tolerance=kkt_tolerance,gap_tolerance=gap_tolerance,
            amplitude_coordinate_scale=scale,gram_coordinate_scale=self.largest,trace=trace)
        return z*scale,report


class ModeNormal:
    def __init__(self,basis,gram):self.basis=basis;self.gram=gram
    def __matmul__(self,value):return self.basis@(self.gram@(self.basis.conj().T@value))


class LowRankComplexLasso(ComplexLasso):
    """The same physical-pixel Laplace prior with a modal data operator.

    The sparse prior remains on physical image coefficients. It is not
    moved to singular-vector coordinates. The orthogonal complement is
    retained analytically in each ADMM solve, avoiding a dense pixel Gram.
    """
    def __init__(self,basis,gram,rhs,energy):
        self.basis=np.asarray(basis);self.modal_gram=np.asarray(gram)
        if max(abs((self.basis.conj().T@self.basis-np.eye(self.basis.shape[1])).ravel()))>1e-10:
            raise ValueError('Physical spatial modes must be orthonormal')
        self.eigenvalue,self.vector=eigh((gram+gram.conj().T)/2)
        self.largest=float(self.eigenvalue[-1]);self.rhs=self.basis@rhs;self.energy=float(energy)
        self.linear_scale=float(max(abs(self.rhs)));self.amplitude_scale=self.linear_scale/self.largest
        self.gram=ModeNormal(self.basis,self.modal_gram)
        if self.largest<=0 or self.linear_scale<=0 or energy<=0 or self.eigenvalue[0]<-1e-10*self.largest:
            raise ValueError('Invalid modal complex image likelihood')

    def ridge_step(self,rhs,rho):
        projected=self.basis.conj().T@rhs;eigen=self.eigenvalue/self.largest
        solved=self.vector@((self.vector.conj().T@projected)/(eigen+rho))
        return (rhs-self.basis@projected)/rho+self.basis@solved

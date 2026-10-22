"""Constant interband response times one shared complex spatial scene.

Only small sufficient statistics enter the alternating conditional least
squares solve. No image power, local normalization or validation data enter.
"""
import numpy as np
from scipy.optimize import minimize,nnls
from scipy.linalg import solve_triangular


def solve(gram,rhs,energy,*,reference_band=3,tolerance=1e-12,max_iterations=500):
    gram,rhs=np.asarray(gram),np.asarray(rhs);energy=np.asarray(energy)
    independent=np.asarray([np.linalg.solve(g,b) for g,b in zip(gram,rhs)])
    u,s,vh=np.linalg.svd(independent,full_matrices=False)
    initial=u[:,0]
    seeds=[initial/initial[reference_band],np.ones(len(gram),complex)]
    trials=[]
    for gain in seeds:
        last=np.inf
        for iteration in range(max_iterations):
            matrix=np.einsum('c,cij->ij',abs(gain)**2,gram)
            vector=np.einsum('c,ci->i',gain.conj(),rhs)
            scene=np.linalg.solve(matrix,vector)
            numerator=np.einsum('i,ci->c',scene.conj(),rhs)
            denominator=np.einsum('i,cij,j->c',scene.conj(),gram,scene).real
            gain=numerator/denominator
            phase=gain[reference_band];gain/=phase;scene*=phase
            prediction_energy=np.einsum('c,i,cij,j->',abs(gain)**2,scene.conj(),gram,scene).real
            cross=np.einsum('c,i,ci->',gain.conj(),scene.conj(),rhs).real
            cost=float(energy.sum()-2*cross+prediction_energy)
            if abs(last-cost)<=tolerance*max(float(energy.sum()),1.):break
            last=cost
        trials.append(dict(cost=cost,gain=gain,scene=scene,iterations=iteration+1,
            prediction_energy=float(prediction_energy)))
    return min(trials,key=lambda r:r['cost'])


def phase_coherent(gram,rhs,energy,*,reference_band=3):
    """Common reflector phases, with independent nonnegative band magnitudes.

    One constant phase per band applies to the entire scene. Each reflector
    has one common physical phase, and may be bright or absent in any band.
    Unlike an arbitrary complex coefficient per band/point, this constraint
    cannot rotate each observation into alignment. Magnitudes are exact NNLS
    nuisance parameters in the original Gaussian metric.
    """
    gram,rhs=np.asarray(gram),np.asarray(rhs);energy=np.asarray(energy)
    nc,npnt=rhs.shape;free=[c for c in range(nc) if c!=reference_band];total=energy.sum()
    baseline=solve(gram,rhs,energy,reference_band=reference_band)
    independent=np.asarray([np.linalg.solve(g,b) for g,b in zip(gram,rhs)])
    seeds=[np.r_[np.angle(baseline['gain'][free]),np.angle(baseline['scene'])],
        np.r_[np.angle(independent[free,0]*independent[reference_band,0].conjugate()),np.angle(independent[reference_band])]]
    def evaluate(parameters,full=False):
        beta=np.zeros(nc);beta[free]=parameters[:nc-1];phi=parameters[nc-1:]
        coefficient=[];cost=0.;gradient=np.zeros(nc-1+npnt)
        for c in range(nc):
            phase=np.exp(1j*(beta[c]+phi));matrix=(phase.conj()[:,None]*gram[c]*phase[None,:]).real
            target=(phase.conj()*rhs[c]).real;root=np.linalg.cholesky(matrix)
            amplitude=nnls(root.T,solve_triangular(root,target,lower=True))[0]
            z=phase*amplitude;coefficient.append(z);r=gram[c]@z-rhs[c]
            cost+=energy[c]-2*np.vdot(z,rhs[c]).real+np.vdot(z,gram[c]@z).real
            local=2*np.imag(z.conj()*r);gradient[nc-1:]+=local
            if c!=reference_band:gradient[free.index(c)]+=local.sum()
        if full:return dict(cost=float(cost),gain=np.exp(1j*beta),point_phase_rad=phi,
            coefficient=np.asarray(coefficient),gradient=gradient/total)
        return float(cost/total),gradient/total
    trials=[]
    for initial in seeds:
        fitted=minimize(evaluate,initial,jac=True,method='BFGS',options={'gtol':1e-10,'maxiter':500})
        r=evaluate(fitted.x,True);r.update(iterations=fitted.nit,maximum_gradient=float(np.max(abs(r['gradient']))),
            converged=bool(np.max(abs(r['gradient']))<1e-8))
        trials.append(r)
    return min(trials,key=lambda r:r['cost'])

"""Bounded equality-constrained quadratic in whitened optimization coordinates.

The positive-definite root is supplied directly; no squared condition number
from normal equations is introduced. This is a convex initializer, not a new
penalty, measurement or bound in the physical data model.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import LinearConstraint,minimize


def bounded_step(root,equality,target,centre,lower,upper,feasible):
    """Minimize ||root @ (step-centre)||² under equality and box bounds."""
    root=np.asarray(root);inverse=solve_triangular(root,np.eye(len(centre)))
    mean=root@centre;initial=root@feasible
    magnitude=max(1.,float(np.linalg.norm(initial-mean)),float(np.linalg.norm(mean)))
    equal=equality@inverse;scale=np.linalg.norm(equal,axis=1)
    if np.any(scale==0):raise ValueError('Zero equality constraint')
    constraints=[LinearConstraint(equal/scale[:,None],target/scale/magnitude,target/scale/magnitude)]
    finite=np.isfinite(lower)|np.isfinite(upper)
    bound=inverse[finite];norm=np.linalg.norm(bound,axis=1)
    constraints.append(LinearConstraint(bound/norm[:,None],lower[finite]/norm/magnitude,upper[finite]/norm/magnitude))
    mean=mean/magnitude
    def objective(z):
        residual=z-mean
        return .5*float(residual@residual),residual
    fitted=minimize(objective,initial/magnitude,jac=True,method='SLSQP',constraints=constraints,
        options={'maxiter':300,'ftol':1e-13})
    step=inverse@(fitted.x*magnitude)
    equality_error=float(np.max(abs(equality@step-target)))
    bound_error=float(max(0.,np.max(lower-step),np.max(step-upper)))
    return step,dict(success=bool(fitted.success),status=int(fitted.status),message=str(fitted.message),
        iterations=int(fitted.nit),equality_error=equality_error,bound_error=bound_error,
        cost=float(np.linalg.norm(root@(step-centre))**2))

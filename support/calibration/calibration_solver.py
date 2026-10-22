"""Bounded damped Gauss--Newton for the offline calibration's sparse Jacobian.

Variable-projection residuals and their derivatives come from the physical
model. The solver changes only the linear algebra: sparse normal products,
column scaling and a positive Levenberg diagonal. Every accepted step must
reduce the original objective; bounds are enforced before evaluating a trial.
"""
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.sparse import csr_array


def solve(residual,start,bounds,max_nfev=300,callback=None):
    low,high=map(np.asarray,bounds)
    x=np.clip(np.asarray(start,float),low,high)
    r=residual(x);cost=float(r@r/2);nfev=1;njev=0
    damping=1e-3;growth=2.;success=False;message='Maximum evaluations reached'
    stable=0
    while nfev<max_nfev:
        j=csr_array(residual(x,True));njev+=1
        gradient=np.asarray(j.T@r).ravel()
        normal=(j.T@j).toarray()
        scale=np.sqrt(np.maximum(np.diag(normal),1e-16))
        h=normal/scale[:,None]/scale[None,:];g=gradient/scale
        active=((x<=low+1e-10)&(g>0))|((x>=high-1e-10)&(g<0))
        projected=g.copy();projected[active]=0
        optimality=float(np.max(abs(projected)))
        if optimality<1e-8:
            success=True;message='Projected scaled gradient tolerance';break
        h[active,:]=0;h[:,active]=0;h[active,active]=1;g[active]=0
        accepted=False
        while nfev<max_nfev:
            delta=np.linalg.solve(h+damping*np.eye(len(x)),-g)/scale
            trial=np.clip(x+delta,low,high);step=trial-x
            prediction=-gradient@step-.5*step@normal@step
            if prediction<=0 or not np.isfinite(prediction):
                damping*=growth;growth*=2
                if damping>1e16:break
                continue
            error=residual(trial);nfev+=1
            next_cost=float(error@error/2);change=cost-next_cost
            ratio=change/prediction
            if ratio>1e-4 and np.isfinite(next_cost):
                stable=stable+1 if change<1e-8*max(cost,1.) else 0
                x,r,cost=trial,error,next_cost
                damping*=max(1/3,1-(2*ratio-1)**3);growth=2.
                accepted=True
                if callback:callback(OptimizeResult(x=x,cost=cost,nfev=nfev,optimality=optimality))
                break
            damping*=growth;growth*=2
            if damping>1e16:break
        if stable>=3:
            success=True;message='Three accepted steps below relative objective tolerance';break
        if not accepted:
            message='No decreasing bounded Gauss--Newton step';break
    return OptimizeResult(x=x,fun=r,cost=cost,nfev=nfev,njev=njev,success=success,
                          message=message,optimality=optimality)

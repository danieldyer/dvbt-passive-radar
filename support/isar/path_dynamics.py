"""Exact global acceleration bounds for a cubic physical path spline.

The second derivative is a degree-one B-spline. Its control values are its
values at the distinct knots, so bounding them bounds acceleration everywhere
without sampling assumptions. Projection is an initializer only; the final
complex likelihood retains the same explicit linear constraints.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import LinearConstraint,minimize,nnls


def acceleration_matrix(grid):
    n=len(grid)-4;derivative=BSpline(grid,np.eye(n),3).derivative(2)
    return derivative.c[:len(derivative.t)-derivative.k-1]


def project_initial(grid,coefficients,limit,anchor):
    coefficients=np.asarray(coefficients);matrix=acceleration_matrix(grid)
    value=matrix@(coefficients-coefficients[0]);origin=BSpline(grid,np.eye(len(coefficients)),3)(anchor)
    result=minimize(lambda x:(float(x@x/2),x),np.zeros(len(coefficients)),jac=True,method='SLSQP',
        constraints=[LinearConstraint(matrix,-limit-value,limit-value),LinearConstraint(origin[None],0.,0.)],
        options=dict(maxiter=300,ftol=1e-13))
    projected=coefficients+result.x;actual=matrix@(projected-projected[0]);violation=float(max(0,np.max(abs(actual))-limit))
    if not result.success or violation>1e-8:raise ValueError('Could not construct a dynamically feasible path initializer')
    return projected,dict(maximum_coefficient_change_m=float(max(abs(result.x))),maximum_acceleration_mps2=float(max(abs(actual))),constraint_violation_mps2=violation)


def constraint(grid,base,parameters,limit,correction_limit=None,anchor=162.,free_path_offset=False):
    base=np.asarray(base)
    matrix=acceleration_matrix(grid);n=len(base)-1;value=matrix@(base-base[0]);a=np.zeros((len(matrix),parameters));a[:,:n]=matrix[:,:n]
    lower,upper=-limit-value,limit-value
    if correction_limit is not None:
        origin=BSpline(grid,np.eye(len(base)),3)(anchor);coordinate=np.zeros((len(base),parameters))
        coordinate[:,:n]=np.eye(len(base))[:,:n]-origin[None,:n]
        if free_path_offset:coordinate[:,n]=1.
        a=np.r_[a,coordinate];lower=np.r_[lower,np.full(len(base),-correction_limit)];upper=np.r_[upper,np.full(len(base),correction_limit)]
    return LinearConstraint(a,lower,upper)


def certificate(parameters,gradient,bound):
    value=bound.A@parameters;violation=float(max(0,np.max(bound.lb-value),np.max(value-bound.ub)))
    active=[]
    for row,v,lo,hi in zip(bound.A,value,bound.lb,bound.ub):
        if v-lo<1e-7:active.append(row)
        if hi-v<1e-7:active.append(-row)
    if active:
        a=np.asarray(active);a=a/np.linalg.norm(a,axis=1)[:,None];multipliers=nnls(a.T,gradient,maxiter=20*len(active))[0]
        residual=gradient-a.T@multipliers
    else:residual=gradient
    return dict(maximum_kkt_residual=float(max(abs(residual))),maximum_constraint_violation=violation,active_linear_constraints=len(active))

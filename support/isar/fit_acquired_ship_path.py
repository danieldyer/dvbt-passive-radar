"""Coarse radar delay/rate seed with exact continuous-time kinematic bounds.

This uses the acquired stronger branch, not the AIS antenna's absolute range.
A convex Huber fit to coarse delay/rate measurements initializes subsequent
complex-data phase estimation. It is not an adopted fine carrier trajectory.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize,LinearConstraint
from spline_motion import basis,knots
from joint_ship_autofocus import kkt_residual


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acquisition',type=Path,required=True);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--path-offset-m',type=float,default=450.);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    d=np.load(a.acquisition);t=d['centres_s'];ranges=d['ranges_m'];rates=d['rates_mps'];power=d['power'];prior=d['predicted'];observations=[]
    for z,pred in zip(power,prior):
        allowed=(abs(ranges[None]-pred[0]-a.path_offset_m)<200)&(abs(rates[:,None]-pred[1])<2)
        v,r=np.unravel_index(np.argmax(np.where(allowed,z,-np.inf)),z.shape)
        def shift(line,i):
            if not 0<i<len(line)-1:return 0.
            denominator=line[i-1]-2*line[i]+line[i+1]
            return float(np.clip(.5*(line[i-1]-line[i+1])/denominator,-.5,.5)) if denominator<0 else 0.
        observations.append([ranges[r]+shift(z[v],r)*(ranges[1]-ranges[0]),rates[v]+shift(z[:,r],v)*(rates[1]-rates[0]),float(z[v,r])])
    observations=np.asarray(observations);description=json.loads(a.motion.read_text());interval=description['interval_s'];spacing=15.
    b=basis(t,interval,spacing);db=basis(t,interval,spacing,derivative=1);offset=14500.
    design=np.r_[b/20.,db/.15];y=np.r_[(observations[:,0]-offset)/20.,observations[:,1]/.15]
    def objective(p):
        residual=design@p-y;gradient=np.clip(residual,-2.,2.)
        cost=np.where(abs(residual)<=2,.5*residual**2,2*abs(residual)-2).sum()
        return float(cost),design.T@gradient
    grid=knots(interval,spacing);n=len(grid)-4;operators=[];limit=[]
    for order,bound in [(1,6.),(2,.15)]:
        matrix=BSpline(grid,np.eye(n),3).derivative(order).c[:n-order]
        operators.append(matrix);limit.extend(np.full(len(matrix),bound))
    matrix=np.r_[tuple(operators)];limit=np.asarray(limit);constraints=[LinearConstraint(matrix,-limit,limit)]
    seed=np.linalg.lstsq(design,y,rcond=None)[0]
    projection=minimize(lambda q:(.5*np.sum((q-seed)**2),q-seed),np.zeros_like(seed),jac=True,method='SLSQP',constraints=constraints,options=dict(maxiter=500,ftol=1e-10))
    if not projection.success:raise ValueError('Coarse-path initial feasibility projection failed')
    result=minimize(objective,projection.x,jac=True,method='SLSQP',constraints=constraints,options=dict(maxiter=1000,ftol=1e-10))
    coefficients=result.x+offset;stationarity=kkt_residual(result.x,result.jac,[(-np.inf,np.inf)]*n,matrix,-limit,limit)
    report=dict(complete=True,scope=__doc__,source_acquisition=str(a.acquisition),interval_s=interval,path_spacing_seconds=spacing,
        path_coefficients_m=coefficients.tolist(),path_offset_gate_m=a.path_offset_m,range_gate_halfwidth_m=200.,rate_gate_halfwidth_mps=2.,
        nominal_coarse_range_sigma_m=20.,nominal_coarse_rate_sigma_mps=.15,huber_delta_sigma=2.,
        objective=float(result.fun),message=str(result.message),stationarity=stationarity,numerical_convergence=bool(result.success and stationarity['maximum_kkt_residual']<1e-5),
        adopted_as_fine_motion=False)
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output/'observations.npz',time_s=t,observations=observations,prediction=np.c_[b@coefficients,db@coefficients])
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(15,11),layout='constrained',sharex=True)
    axes[0].plot(t,observations[:,0],'.',label='Acquired stronger branch');axes[0].plot(t,b@coefficients,label='Radar delay/rate fit');axes[0].plot(t,prior[:,0]+a.path_offset_m,'--',label='Earlier path plus constant 450 m');axes[0].legend();axes[0].set_ylabel('Excess path (m)')
    axes[1].plot(t,observations[:,1],'.');axes[1].plot(t,db@coefficients);axes[1].set_ylabel('Path rate (m/s)')
    axes[2].plot(t,observations[:,0]-b@coefficients,'.');axes[2].set(xlabel='Capture time (s)',ylabel='Coarse range residual (m)')
    fig.suptitle('Data-derived coarse path: measured delay and Doppler remain linked\nConvex Huber fit; exact spline control bounds |rate| ≤ 6 m/s and |acceleration| ≤ 0.15 m/s²; initialization only')
    fig.savefig(a.output/'coarse-radar-path.png',dpi=140);plt.close(fig);print(json.dumps(report),flush=True)
if __name__=='__main__':main()

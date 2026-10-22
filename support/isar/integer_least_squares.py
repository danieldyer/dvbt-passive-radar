"""Small-dimensional integer least squares with a complete bounded search.

Minimize (n-a)' Q^-1 (n-a), n integer. Integer column operations reduce the
whitened lattice, then a shrinking sphere enumerates its K nearest points.
This is the reduction/search principle used in LAMBDA; it is not MLAMBDA.
The supplied metric need not be a statistical covariance. A node limit raises
an error, rather than silently returning an incomplete 'optimal' result.
"""
import heapq
import math

import numpy as np
from scipy.linalg import solve_triangular


class SearchLimitError(RuntimeError):
    pass


def _positive_qr(basis):
    q,r=np.linalg.qr(basis)
    sign=np.where(np.diag(r)<0,-1.,1.)
    return q*sign,r*sign[:,None]


def reduce_basis(basis,delta=.75):
    """LLL reduction by integer column subtraction and interchange."""
    basis=np.asarray(basis,float);n=basis.shape[1]
    transform=np.eye(n,dtype=object);k=1;steps=0
    q,r=_positive_qr(basis)
    while k<n:
        steps+=1
        if steps>50000:raise SearchLimitError('Lattice reduction did not finish')
        for j in range(k-1,-1,-1):
            multiple=int(np.rint(r[j,k]/r[j,j]))
            if multiple:
                transform[:,k]-=multiple*transform[:,j]
                if max(abs(x) for x in transform.flat)>2**50:
                    raise ArithmeticError('Integer basis exceeds reliable float conversion')
                q,r=_positive_qr(basis@np.asarray(transform,float))
        mu=r[k-1,k]/r[k-1,k-1]
        if r[k,k]**2 >= (delta-mu**2)*r[k-1,k-1]**2:
            k+=1
        else:
            transform[:,[k-1,k]]=transform[:,[k,k-1]]
            q,r=_positive_qr(basis@np.asarray(transform,float));k=max(1,k-1)
    return transform,q,r,steps


def nearest_integers(mean,metric,count=1,*,max_nodes=2000000):
    """Return K best integer vectors and costs, or explicitly fail the search.

    Completeness is for this positive-definite quadratic metric, subject to
    floating-point arithmetic; it says nothing about an underlying nonlinear
    likelihood. Tied optima may be returned in any order.
    """
    mean=np.asarray(mean,float);metric=np.asarray(metric,float)
    if mean.ndim!=1 or not len(mean) or metric.shape!=(len(mean),len(mean)):
        raise ValueError('Expected a vector and square metric of the same dimension')
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(metric)):
        raise ValueError('Finite inputs required')
    if not isinstance(count,int) or count<1 or max_nodes<1:
        raise ValueError('Positive count and node limit required')
    if not np.allclose(metric,metric.T,rtol=1e-12,atol=0):
        raise ValueError('Metric must be symmetric')
    try:factor=np.linalg.cholesky((metric+metric.T)/2)
    except np.linalg.LinAlgError as error:raise ValueError('Metric must be positive definite') from error
    basis=solve_triangular(factor,np.eye(len(mean)),lower=True)
    transform,q,r,steps=reduce_basis(basis)
    shift=np.array([int(np.rint(x)) for x in mean],dtype=object)
    target=q.T@(basis@(mean-np.asarray(shift,float)))
    n=len(mean);babai=np.zeros(n,dtype=object)
    for k in range(n-1,-1,-1):
        babai[k]=int(np.rint((target[k]-r[k,k+1:]@np.asarray(babai[k+1:],float))/r[k,k]))
    heap=[];present=set()
    def insert(z):
        key=tuple(int(x) for x in z)
        if key in present:return
        error=r@np.asarray(z,float)-target;cost=float(error@error)
        item=(-cost,key)
        if len(heap)<count:heapq.heappush(heap,item);present.add(key)
        elif cost < -heap[0][0]:
            removed=heapq.heapreplace(heap,item);present.remove(removed[1]);present.add(key)
    insert(babai)
    distance=1
    while len(heap)<count:
        for axis in range(n):
            for sign in (-1,1):
                seed=babai.copy();seed[axis]+=sign*distance;insert(seed)
        distance+=1
    visited=0;z=np.zeros(n,dtype=object)
    tolerance=128*np.finfo(float).eps*max(1.,float(target@target),-heap[0][0])
    def search(k,partial):
        nonlocal visited
        radius=-heap[0][0]+tolerance-partial
        if radius<0:return
        centre=(target[k]-r[k,k+1:]@np.asarray(z[k+1:],float))/r[k,k]
        half=math.sqrt(radius)/r[k,k]
        low,high=math.ceil(centre-half),math.floor(centre+half)
        left=math.floor(centre);right=left+1
        while left>=low or right<=high:
            if right>high or (left>=low and centre-left<=right-centre):
                value=left;left-=1
            else:value=right;right+=1
            visited+=1
            if visited>max_nodes:raise SearchLimitError(f'Integer search exceeded {max_nodes} nodes')
            new=partial+(r[k,k]*(value-centre))**2
            if new>-heap[0][0]+tolerance:continue
            z[k]=value
            if k:search(k-1,new)
            else:insert(z)
    search(n-1,0.)
    records=[]
    for _,key in heap:
        integer=transform@np.asarray(key,dtype=object)+shift
        difference=solve_triangular(factor,np.asarray(integer,float)-mean,lower=True)
        records.append(dict(integer=[int(x) for x in integer],cost=float(difference@difference)))
    records.sort(key=lambda row:(row['cost'],row['integer']))
    return dict(candidates=records,complete=True,visited_nodes=visited,reduction_steps=steps,
        transform=[[int(x) for x in row] for row in transform],
        scope='K best points of the supplied quadratic metric, not global nonlinear likelihood optima')

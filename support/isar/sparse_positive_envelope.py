"""Block-banded positive envelopes with an exact stationary-nuisance term.

Linear amplitude bases overlap only adjacent knots. Their normal matrix is
block tridiagonal; profiling a stationary range vector adds a low-rank term.
Keeping these two structures avoids a dense (points * knots)^2 matrix. The
bounded quadratic solve has an explicit projected-gradient certificate.
"""
import numpy as np
from scipy.sparse import bsr_matrix
from scipy.optimize import minimize,nnls,OptimizeResult
from scipy.linalg import cholesky,solve_triangular,LinAlgError,eigh
from scipy.sparse.linalg import LinearOperator,cg
from scipy.sparse.csgraph import connected_components
from rigid_patch import profile_static


def direct_positive_quadratic(multiply,linear):
    """Exact small positive-definite NNLS fallback; no statistical loading.

    The factor changes only the numerical representation of the quadratic.
    For singular matrices, a rank-revealing square root removes machine-
    precision null modes. The ORIGINAL operator still certifies KKT: a null
    component in the linear term cannot silently become an accepted optimum.
    There is no statistical eigenvalue cutoff, loading or relaxed tolerance.
    """
    gram=np.column_stack([multiply(q) for q in np.eye(len(linear))])
    gram=(gram+gram.T)/2
    try:
        lower=cholesky(gram,lower=True);design=lower.T
        target=solve_triangular(lower,linear,lower=True)
    except LinAlgError:
        eigenvalue,vector=eigh(gram)
        tolerance=64*np.finfo(float).eps*len(linear)*max(max(abs(eigenvalue)),1e-300)
        if eigenvalue[0]<-tolerance:raise ValueError('Projected amplitude Gram is not positive semidefinite')
        keep=eigenvalue>tolerance
        if not np.any(keep):return None
        design=np.sqrt(eigenvalue[keep,None])*vector[:,keep].T
        target=(vector[:,keep].T@linear)/np.sqrt(eigenvalue[keep])
    return nnls(design,target,maxiter=10000)[0]


def block_positive_quadratic(local,mean,linear,blocks,initial,multiply,max_sweeps=500):
    """Exact block-coordinate NNLS for disjoint temporal scene supports.

    The only between-block coupling is the profiled global stationary term.
    Each diagonal block retains that term with its GLOBAL normalization.
    No independent per-window clutter subtraction, loading or prior is added.
    Original full-quadratic KKT remains the stopping/acceptance criterion.
    """
    factors=[]
    for index in blocks:
        block=local(index)-np.real(mean[:,index].conj().T@mean[:,index])
        block=(block+block.T)/2
        try:
            lower=cholesky(block,lower=True);design=lower.T
            factors.append((index,design,lambda b,l=lower:solve_triangular(l,b,lower=True)))
        except LinAlgError:
            eigen,vector=eigh(block);tolerance=64*np.finfo(float).eps*len(index)*max(max(abs(eigen)),1e-300)
            if eigen[0]<-tolerance:raise ValueError('Temporal NNLS block is not positive semidefinite')
            keep=eigen>tolerance
            if not np.any(keep):
                factors.append((index,None,None));continue
            root=np.sqrt(eigen[keep]);basis=vector[:,keep]
            factors.append((index,root[:,None]*basis.T,lambda b,v=basis,r=root:(v.T@b)/r))
    x=np.maximum(np.asarray(initial),0).copy();common=mean@x
    for sweep in range(max_sweeps):
        for index,design,target in factors:
            old=x[index].copy();outside=common-mean[:,index]@old
            rhs=linear[index]+np.real(mean[:,index].conj().T@outside)
            x[index]=0. if design is None else nnls(design,target(rhs),maxiter=10000)[0]
            common+=mean[:,index]@(x[index]-old)
        gradient=2*(multiply(x)-linear)
        projected=np.where(x>1e-10,gradient,np.minimum(gradient,0))
        if max(abs(projected))<2e-9:break
    return x,sweep+1


def polish_positive_quadratic(multiply,linear,initial,preconditioner=None,max_iterations=100):
    """Primal-feasible active-set solve of the unchanged NNLS quadratic.

    Optimize the existing free face BEFORE admitting a violated inactive
    constraint. Adding it while the old face is nonstationary can give that
    new zero-valued variable a negative unconstrained solution, force a zero
    step, and repeat forever. This ordering is essential for warm starts.
    """
    x=np.maximum(np.asarray(initial),0).copy()
    for iteration in range(max_iterations):
        gradient=multiply(x)-linear;free=x>1e-10
        projected=np.where(free,gradient,np.minimum(gradient,0))
        if max(abs(projected))<2e-9:break
        inactive=np.where(~free,gradient,np.inf)
        if np.max(abs(gradient[free]),initial=0)<2e-9 and inactive.min()<-2e-9:
            free[np.argmin(inactive)]=True
        index=np.flatnonzero(free)
        def restricted(v):
            full=np.zeros_like(x);full[index]=v;return multiply(full)[index]
        matrix=LinearOperator((len(index),len(index)),matvec=restricted,dtype=float)
        inverse=None if preconditioner is None else preconditioner(index)
        value,code=cg(matrix,linear[index],x0=x[index],M=inverse,rtol=2e-11,atol=1e-12,maxiter=3000)
        if code<0:raise ValueError('Positive-envelope normal solve failed')
        if code>0:break
        direction=value-x[index];negative=direction<0
        step=min(1.,float(np.min(-x[index][negative]/direction[negative],initial=1.)))
        x[index]=np.maximum(0,x[index]+step*direction);x[x<1e-12]=0
    return x,iteration


class Normal:
    def __init__(self,h,y,root,selected,basis,groups=None):
        n,r,p=h.shape;k=basis.shape[1];self.points=p;self.knots=k
        groups=np.zeros(n,int) if groups is None else np.asarray(groups)
        projected=profile_static(y,root,selected,groups)*selected[:,None]
        # Retain views only for a bounded, rarely needed direct-design solve.
        # It avoids subtractive cancellation in stationary normal equations.
        self.design_context=(h,projected,root,selected,basis,groups) if k*p<=512 else None
        self.energy=float(np.vdot(projected,projected).real)
        time_gram=np.einsum('trp,trq->tpq',h.conj(),h)*selected[:,None,None]
        columns=basis.tocsc();diag=np.empty((k,p,p),complex);off=np.zeros((k-1,p,p),complex)
        for j in range(k):
            rows=columns.indices[columns.indptr[j]:columns.indptr[j+1]];weights=columns.data[columns.indptr[j]:columns.indptr[j+1]]
            diag[j]=np.einsum('tpq,t->pq',time_gram[rows],weights**2)
            if j+1<k:
                adjacent=np.asarray(basis[rows,j+1].toarray()).ravel()
                off[j]=np.einsum('tpq,t->pq',time_gram[rows],weights*adjacent)
        values=[];indices=[];pointers=[0]
        for j in range(k):
            if j:values.append(off[j-1].conj().T);indices.append(j-1)
            values.append((diag[j]+diag[j].conj().T)/2);indices.append(j)
            if j+1<k:values.append(off[j]);indices.append(j+1)
            pointers.append(len(values))
        self.local=bsr_matrix((np.asarray(values),np.asarray(indices),np.asarray(pointers)),shape=(k*p,k*p))
        self.local_csr=self.local.tocsr()
        graph=self.local_csr.copy();graph.eliminate_zeros();graph.data=np.ones(graph.nnz,np.int8)
        components,labels=connected_components(graph,directed=False)
        self.temporal_blocks=[np.flatnonzero(labels==i) for i in range(components) if np.any(self.local.diagonal().real[labels==i]>0)]
        means=[]
        for group in np.unique(groups):
            q=np.asarray(root)*selected*(groups==group);denom=float(q@q)
            if denom:
                mean=basis.T@(h.reshape(n,-1)*(q/np.sqrt(denom))[:,None])
                means.append(np.asarray(mean).reshape(k,r,p).transpose(1,0,2).reshape(r,k*p))
        self.mean=np.concatenate(means,axis=0);self.denom=1.
        self.rhs=np.asarray(basis.T@np.einsum('trp,tr->tp',h.conj(),projected)).ravel()
        self.diagonal=self.local.diagonal().real-np.sum(abs(self.mean)**2,axis=0)/self.denom
        if self.diagonal.min()<-1e-10*max(self.diagonal.max(),1e-30):raise ValueError('Invalid projected envelope Gram diagonal')

    def multiply(self,value):return self.local@value-self.mean.conj().T@(self.mean@value)/self.denom

    def design_nnls(self,rot,scale,power):
        if self.design_context is None:return None
        h,y,root,selected,basis,groups=self.design_context
        # Original whitened rows and exactly the same stationary projection.
        # No factorization or truncation of an almost-singular Gram matrix.
        design=h[:,:,None,:]*basis.toarray()[:,None,:,None]
        design=profile_static(design,root,selected,groups)[selected].reshape(-1,len(rot))
        design*=rot[None,:]/scale[None,:]
        target=y[selected].ravel()/power
        return nnls(np.r_[design.real,design.imag],np.r_[target.real,target.imag],maxiter=10000)[0]

    def solve(self,phase,initial=None,phase_groups=None):
        phase=np.asarray(phase)
        groups=np.tile(np.arange(self.points),self.knots) if phase_groups is None else np.asarray(phase_groups).ravel()
        if groups.shape!=self.rhs.shape or groups.min()<0 or groups.max()>=len(phase):raise ValueError('Each envelope coefficient needs a declared reflector-phase identity')
        rot=np.exp(1j*phase[groups]);target=(rot.conj()*self.rhs).real
        # Strictly positive numerical scaling is an invertible coordinate
        # change, never a loading term or a penalty in the likelihood.
        scale=np.sqrt(np.maximum(self.diagonal,16*np.finfo(float).eps*max(self.diagonal.max(),1e-30)))
        power=max(float(np.max(abs(target/scale))),1e-30);linear=target/scale/power
        def multiply(v):return (rot.conj()*self.multiply(rot*(v/scale))).real/scale
        def objective(v):
            product=multiply(v);return float(v@product-2*v@linear),2*(product-linear)
        x0=np.zeros(len(scale)) if initial is None else np.asarray(initial).ravel()*scale/power
        block_solve=len(x0)>512 and len(self.temporal_blocks)>1 and max(map(len,self.temporal_blocks))<=512
        if block_solve:
            def local(index):
                block=self.local_csr[index][:,index].toarray()
                return np.real((rot[index].conj()/scale[index])[:,None]*block*(rot[index]/scale[index])[None,:])
            value,sweeps=block_positive_quadratic(local,self.mean*(rot/scale)[None,:],linear,self.temporal_blocks,x0,multiply,max_sweeps=10)
            result=OptimizeResult(x=value,nit=sweeps,message='Exact temporal-block NNLS with global stationary coupling')
        else:
            result=minimize(objective,x0,jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(x0),
                options=dict(maxiter=3000,gtol=2e-9,ftol=1e-14,maxcor=30,maxls=30))
        preconditioner=None
        if block_solve:
            def preconditioner(index):
                # The unprojected quadratic is block diagonal. After its
                # inverse, only the low-rank stationary downdate remains;
                # diagonal scaling alone cannot condition those modes.
                lookup=np.full(len(linear),-1,int);lookup[index]=np.arange(len(index));factors=[]
                for block in self.temporal_blocks:
                    chosen=block[lookup[block]>=0]
                    if not len(chosen):continue
                    gram=local(chosen)
                    eigen,vector=eigh((gram+gram.T)/2)
                    floor=max(eigen[-1],1e-300)*1e-12
                    inverse=(vector/np.maximum(eigen,floor)[None,:])@vector.T
                    factors.append((lookup[chosen],inverse))
                def inverse_local(v):
                    out=np.zeros_like(v)
                    for at,inverse in factors:out[at]=inverse@v[at]
                    return out
                return LinearOperator((len(index),len(index)),matvec=inverse_local,dtype=float)
        x,polish_iterations=polish_positive_quadratic(multiply,linear,result.x,preconditioner,1000 if block_solve else 100)
        gradient=objective(x)[1];projected=np.where(x>1e-10,gradient,np.minimum(gradient,0))
        direct_polish=False
        if max(abs(projected))>=1e-6 and len(x)<=2048:
            candidate=direct_positive_quadratic(multiply,linear)
            if candidate is not None:
                x=candidate;direct_polish=True
            gradient=objective(x)[1];projected=np.where(x>1e-10,gradient,np.minimum(gradient,0))
            if max(abs(projected))>=1e-6:
                candidate=self.design_nnls(rot,scale,power)
                if candidate is not None:x=candidate;direct_polish=True
        amplitude=x*power/scale;coefficient=rot*amplitude;residual=self.multiply(coefficient)-self.rhs
        gradient=objective(x)[1];projected=np.where(x>1e-10,gradient,np.minimum(gradient,0))
        kkt=float(max(abs(projected)));cost=float(self.energy-2*np.vdot(coefficient,self.rhs).real+np.vdot(coefficient,self.multiply(coefficient)).real)
        return dict(cost=cost,coefficient=coefficient,amplitude=amplitude,phase_gradient=np.bincount(groups,2*np.imag(coefficient.conj()*residual),minlength=len(phase)),
            maximum_scaled_kkt=kkt,iterations=result.nit,normal_polish_iterations=polish_iterations,direct_cholesky_nnls_polish=direct_polish,numerical_convergence=bool(kkt<1e-6),message=str(result.message))

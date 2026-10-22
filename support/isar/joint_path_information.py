"""Data-only information for shared paths, with all scene nuisance retained.

Use expected proper-complex mean derivatives, never a residual Hessian or
motion-prior rows. Scale columns before eliminating nuisance so a numerical
null is not promoted into information. Only estimable functionals get bounds.
"""
import numpy as np


def expected_jacobian(model,parameters,transform=None):
    """Stack band mean Jacobians, eliminating each band's linear nuisance."""
    if transform is not None:
        transform=np.asarray(transform,float)
        if transform.ndim!=2 or transform.shape[0]!=model.count:raise ValueError('Matching affine parameter transform required')
    rows=[];norm2=np.zeros(model.count if transform is None else transform.shape[1])
    for cluster,index in zip(model.clusters,model.indices):
        b=cluster.base;local=np.asarray(parameters)[index];value=cluster.evaluate(local)
        mean=np.sqrt(2)*value['mean_derivative']
        if transform is None:
            matrix=np.zeros((len(mean),model.count));matrix[:,index]=mean
        else:matrix=mean@transform[index]
        rows.append(matrix)
        k=len(b.knots);original=[]
        for (_,derivative),amplitude in zip(cluster.response_blocks(local),value['amplitude'].reshape(cluster.components,k)):
            original.append(b.reduced(derivative*(b.amplitude_basis@amplitude)[:,None,None])[b.selected])
        raw=np.concatenate(original,axis=1)
        if transform is None:norm2[index]+=2*np.sum(abs(raw)**2,axis=0)
        else:norm2+=2*np.sum(abs(raw@transform[index])**2,axis=0)
    return np.concatenate(rows),np.sqrt(norm2)


def functional_bounds(matrix,original_norm,interest,functionals,*,rank_cutoff=1e-8):
    """Schur projection and SVD bounds; null directions stay unmeasured."""
    matrix=np.asarray(matrix,float);norm=np.asarray(original_norm,float);interest=np.asarray(interest,int)
    functionals=np.asarray(functionals,float)
    if (matrix.ndim!=2 or norm.shape!=(matrix.shape[1],) or functionals.ndim!=2 or
            functionals.shape[1]!=len(interest) or len(set(interest))!=len(interest) or
            np.any(interest<0) or np.any(interest>=matrix.shape[1]) or not 0<rank_cutoff<1):
        raise ValueError('Matching mean Jacobian, original column norms, target coordinates and functionals required')
    nuisance=np.setdiff1d(np.arange(matrix.shape[1]),interest)
    scale=np.where(norm>0,norm,1.)
    other=matrix[:,nuisance]/scale[nuisance]
    u,s,_=np.linalg.svd(other,full_matrices=False)
    supported=s>rank_cutoff*max(1.,s.max(initial=0.));q=u[:,supported]
    target=matrix[:,interest];residual=(target-q@(q.T@target))/scale[interest]
    _,singular,right=np.linalg.svd(residual,full_matrices=residual.shape[0]<residual.shape[1])
    # A tall observation matrix already retains every right null vector with
    # the thin SVD. Only a wide matrix needs full_matrices; never allocate an
    # observation-count-squared left factor for real recorded data.
    padded=np.r_[singular,np.zeros(len(interest)-len(singular))]
    keep=padded>rank_cutoff*max(1.,padded.max(initial=0.))
    g=functionals/scale[interest]
    null=g@right[~keep].T
    estimable=np.linalg.norm(null,axis=1)<=1e-8*np.linalg.norm(g,axis=1)
    value=g@right[keep].T
    deviation=np.sqrt(np.sum((value/padded[keep])**2,axis=1));deviation[~estimable]=np.nan
    return dict(standard_deviation=deviation,estimable=estimable,nuisance_rank=int(supported.sum()),
        target_rank=int(keep.sum()),target_dimension=len(interest),normalized_singular_values=padded,
        rank_cutoff=rank_cutoff,
        scope='Conditional data-only proper-complex information after static, active brightness and every other nonlinear parameter. No prior, active bound or external guide supplies information. Unknown interference, branch ambiguity and point/geometry errors remain outside this bound.')


def paths(model,parameters,components=(1,4),channel=0,transform=None):
    matrix,norm=expected_jacobian(model,parameters,transform)
    indices=[model.local_path_indices[channel][s] for s in components]
    interest=np.concatenate(indices)
    curve=model.curve;time=np.linspace(curve.epoch-curve.scale,curve.epoch+curve.scale,201)
    functionals=[];edges=np.r_[0,np.cumsum([len(index) for index in indices])]
    for i,s in enumerate(components):
        basis=model.clusters[channel].arcs[s].base.curve(time)
        g=np.zeros((len(time),len(interest)));g[:,edges[i]:edges[i+1]]=basis;functionals.append(g)
    if len(functionals)==2:functionals.append(functionals[1]-functionals[0])
    count=len(functionals)
    functionals=np.concatenate(functionals)
    if transform is not None:
        projected=functionals@np.asarray(transform)[interest]
        interest=np.flatnonzero(np.any(projected!=0,axis=0));functionals=projected[:,interest]
    bounds=functional_bounds(matrix,norm,interest,functionals)
    bounds.update(time_s=time,components=list(components),
        labels=[str(s) for s in components]+(['difference'] if len(components)==2 else []),
        standard_deviation_by_functional_m=bounds.pop('standard_deviation').reshape(count,len(time)),
        estimable_by_functional=bounds.pop('estimable').reshape(count,len(time)))
    return bounds

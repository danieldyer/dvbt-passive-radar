"""Original-epoch matched images, separate from sparse coefficient estimates."""
import numpy as np


def matched_statistics(model,spatial_basis,raw_point_information,parameters):
    """Compute A^H P y and diag(A^H P A), with exact nuisance profiling.

    The modal approximation is that of the validated data-independent
    operator. No reflectivity estimate, fitted coefficient or L1 prior
    enters this diagnostic. Band gains are the declared coherent spectrum.
    """
    score=[];information=[];v=np.asarray(spatial_basis)
    for c,(moments,data) in enumerate(zip(model.mean_moments,model.score_moments)):
        phase,_,_=model.phase_coefficients(parameters,c);gain=model.spectrum[c]
        mean=gain*np.einsum('d,drp->rp',phase,moments,optimize=True)
        rhs=gain.conjugate()*(phase.conj()@data)
        b=v@rhs;m=mean@v.conj().T
        q=abs(gain)**2*np.asarray(raw_point_information[c])-np.sum(abs(m)**2,axis=0)
        if np.min(q)<-1e-10*max(np.max(q),1.):raise ValueError('Negative profiled point information')
        score.append(b);information.append(np.maximum(q,0.))
    score.append(sum(score));information.append(sum(information))
    return np.asarray(score),np.asarray(information)


def save_review(output,model,spatial_basis,points,raw_point_information,initial,final,*,truth=None,scale_db=(0.,55.),initial_spectrum=None):
    from matplotlib import pyplot as plt
    from pathlib import Path
    output=Path(output);cases=[];final_spectrum=model.spectrum.copy()
    for j,parameters in enumerate([initial,final]):
        model.spectrum=initial_spectrum if j==0 and initial_spectrum is not None else final_spectrum
        cases.append(matched_statistics(model,spatial_basis,raw_point_information,parameters))
    model.spectrum=final_spectrum
    np.savez_compressed(output/'matched-statistics.npz',points_body_m=points,initial_score=cases[0][0],initial_information=cases[0][1],
        final_score=cases[1][0],final_information=cases[1][1])
    x=np.unique(points[:,0]);y=np.unique(points[:,1]);names=['Seven','SBS','Nine','Ten','ABC','All five: coherent fixed spectrum']
    fig,axes=plt.subplots(6,2,figsize=(15,14),layout='constrained',sharex=True,sharey=True)
    for col,(score,information) in enumerate(cases):
        power=np.divide(abs(score)**2,information,out=np.zeros_like(information),where=information>0)
        for c in range(6):
            ax=axes[c,col];image=ax.imshow(10*np.log10(np.maximum(power[c].reshape(len(y),len(x)),1e-30)),origin='lower',
                extent=[x[0],x[-1],y[0],y[-1]],cmap='magma',vmin=scale_db[0],vmax=scale_db[1],aspect='equal')
            ax.set(title=f"{names[c]} · {'initial' if col==0 else 'after common-path fit'}",ylabel='Body y (m)')
            if truth is not None:
                p=np.asarray(truth['points_body_m']);ax.plot(p[:,0],p[:,1],'+',color='#00ff00',markersize=4)
    for ax in axes[-1]:ax.set_xlabel('Body x (m); physical coordinates, not a display rotation')
    fig.colorbar(image,ax=axes.ravel().tolist(),shrink=.8,label='Conditional whitened matched power (dB); no image prior applied')
    kind='SYNTHETIC; green marks truth, not supplied to fitter' if truth else 'RECORDED DATA; no accepted full-capture hull reconstruction'
    fig.suptitle(kind+'\nOriginal training epochs and fixed scale; sparse coefficients do not enter these matched images')
    fig.savefig(output/'matched-before-after.png',dpi=145);plt.close(fig)

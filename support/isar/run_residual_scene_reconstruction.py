"""Exercise blind support reconstruction at a saved local physical motion.

Synthetic controls use the same retained kernels and epochs with explicitly
given motion, but do not give the reconstruction the target locations. This
isolates image/support recovery before any joint autofocus extension.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from residual_scene_reconstruction import reconstruct
from rigid_patch import profile_static


def simulate(bands,beta,pixels,kind,point_snr_db,*,spectral_model='pointwise-positive'):
    """Deterministic known-model control; truth is never supplied to pursuit."""
    rng=np.random.default_rng(90517)
    desired=np.array([[-90.,-45.,20.]]) if kind=='point' else np.array([
        [-180.,-65.,20.],[-130.,-65.,20.],[-80.,-65.,20.],[-30.,-65.,20.],[20.,-60.,20.],
        [40.,-45.,20.],[20.,-30.,20.],[-30.,-25.,20.],[-80.,-25.,20.],[-130.,-25.,20.],[-180.,-25.,20.]])
    indices=[int(np.argmin(np.linalg.norm(pixels-point,axis=1))) for point in desired];points=pixels[indices]
    phases=rng.uniform(-np.pi,np.pi,len(points));magnitudes=rng.uniform(.4,1.2,(5,len(points)))
    if spectral_model=='shared':magnitudes=np.broadcast_to(magnitudes[0],magnitudes.shape).copy()
    elif spectral_model!='pointwise-positive':raise ValueError('Unknown controlled scattering spectrum')
    waveform=[[b.waveform(np.zeros(b.b.shape[1]),c,False,point-b.reference_point) for point in points] for c,b in enumerate(bands)]
    norms=[]
    for j in range(len(points)):
        value=0.
        for c,b in enumerate(bands):
            h=profile_static(waveform[c][j],b.field.root_information,np.ones(len(b.t),bool),b.groups)
            value+=magnitudes[c,j]**2*np.vdot(h,h).real
        norms.append(value)
    scale=np.sqrt(10**(point_snr_db/10)/np.asarray(norms));simulated=[]
    for c,b in enumerate(bands):
        other=copy.copy(b);other.field=copy.copy(b.field)
        signal=sum(h*(scale[j]*magnitudes[c,j]*np.exp(1j*(beta[c]+phases[j]))) for j,h in enumerate(waveform[c]))
        noise=(rng.normal(size=signal.shape)+1j*rng.normal(size=signal.shape))/np.sqrt(2)
        background=b.field.root_information[:,None]*(.1+.2j)*(b.groups[:,None]+1)
        other.field.y=signal+noise+background;simulated.append(other)
    truth=dict(kind=kind,points_body_m=points.tolist(),grid_aligned=True,motion_given=True,support_given_to_estimator=False,
        source_phase_radians=phases.tolist(),per_point_integrated_snr_db=point_snr_db,noise='Independent proper-complex unit noise plus an exact stationary nuisance',
        source_magnitudes_by_band=(scale[None,:]*magnitudes).tolist(),
        spectral_model=spectral_model,interpretation='Known-motion, known-calibration support/image recovery only; forward kernels and epochs retained from recording')
    return simulated,truth


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--bounds-m',type=float,nargs=4,default=[-200.,80.,-80.,-20.])
    p.add_argument('--spacing-m',type=float,default=.75);p.add_argument('--maximum-sources',type=int,default=20)
    p.add_argument('--proposal-candidates',type=int,default=8)
    p.add_argument('--amplitude-spacing-s',type=float,help='Declared visibility basis; zero means constant throughout the aperture')
    p.add_argument('--synthetic',choices=['point','hull']);p.add_argument('--point-snr-db',type=float,default=30.)
    a=p.parse_args();fit,_,scene,_=open_scene(a.motion)
    if scene[0].points.ndim!=2:raise ValueError('Use a completed local scene to supply the physical motion and band calibration')
    if a.spacing_m<=0:raise ValueError('Need positive physical grid spacing')
    west,east,south,north=a.bounds_m
    x=np.arange(west,east+a.spacing_m*.1,a.spacing_m);y=np.arange(south,north+a.spacing_m*.1,a.spacing_m)
    xx,yy=np.meshgrid(x,y);pixels=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,20.)]
    beta=np.angle([complex(*z) for z in fit['physical_gains_real_imag']]);bands=[s.point for s in scene];truth=None
    if a.synthetic:
        bands,truth=simulate(bands,beta,pixels,a.synthetic,a.point_snr_db)
    if a.amplitude_spacing_s is not None:
        from audit_ship_scattering_envelopes import linear_basis
        from scipy.sparse import csr_matrix
        if a.amplitude_spacing_s<0:raise ValueError('Negative amplitude-basis spacing')
        for b in bands:
            basis,nodes=linear_basis(b.t,fit['interval_s'],a.amplitude_spacing_s)
            b.amplitude=csr_matrix(basis);b.amplitude_nodes=nodes
    result=reconstruct(bands,pixels,beta,a.output,maximum_sources=a.maximum_sources,proposal_candidates=a.proposal_candidates)
    result.update(motion=str(a.motion),scope=__doc__,grid_bounds_m=a.bounds_m,grid_spacing_m=a.spacing_m,truth=truth,
        selected_support_is_physical_truth=False,amplitude_spacing_s=fit['amplitude_spacing_s'] if a.amplitude_spacing_s is None else a.amplitude_spacing_s)
    if truth and result['selected_sources']:
        selected=np.asarray(result['points_body_m']);true=np.asarray(truth['points_body_m'])
        result['nearest_true_point_distance_m']=[float(np.linalg.norm(selected-point,axis=1).min()) for point in true]
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['selected_sources']:
        with np.load(a.output/'scene.npz') as source:data=dict(source)
        from matplotlib import pyplot as plt
        energy=data['component_excluded_predicted_energy'];joint=energy.sum(axis=0)
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained',sharex=True,sharey=True)
        power=np.r_[energy,joint[None]];maximum=max(30.,float(10*np.log10(power.max())))
        for c,ax in enumerate(axes.ravel()):
            image=np.zeros(len(pixels));image[result['selected_grid_indices']]=power[c]
            im=ax.imshow(10*np.log10(np.maximum(image.reshape(len(y),len(x)),1e-30)),origin='lower',extent=[x[0],x[-1],y[0],y[-1]],
                cmap='magma',vmin=0,vmax=maximum,aspect='equal')
            if truth:ax.plot(true[:,0],true[:,1],'+',color='#00e5ff',markersize=5)
            ax.set(title=['Seven','SBS','Nine','Ten','ABC','Joint phase-constrained model'][c],xlabel='Body coordinate x (m)',ylabel='Body coordinate y (m)')
        fig.colorbar(im,ax=axes.ravel().tolist(),label='Fitted source component energy (dB, conditional model; not calibrated detection SNR)',shrink=.7)
        fig.suptitle('Sparse inverse-scene estimate; shared source phases; support count chosen by inner blocked prediction\nPoint locations/support are estimates, not validated hull geometry; native data images remain separate')
        fig.savefig(a.output/'inverse-scene.png',dpi=145);plt.close(fig)
    print(json.dumps(dict(selected_sources=result['selected_sources'],outer=result['outer'],nearest_true_point_distance_m=result.get('nearest_true_point_distance_m'))),flush=True)


if __name__=='__main__':main()

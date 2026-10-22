"""Exact recorded-aperture point/hull controls alongside measured ship data.

Synthetic data use the actual per-symbol waveform kernels, information,
carrier references, independent OFDM epochs, spatial rays and physical state.
The exact same range whitening, stationary projection and Vulkan image
operator form all three cases. Relative-peak panels compare PSF shape only;
they do not compare SNR, and no synthetic image replaces measured data.
"""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
from dense_patch_image import DenseImages
from joint_ship_autofocus import Window
from joint_ship_motion import JointMotion
from ship_window import Dataset
from report_measured_scatterer_phases import serial


def state(window,parameters,c):
    d=window.context;n=window.basis[c].shape[1];nc=window.cross_basis[c].shape[1]
    q=d[f'direction_{c}'];qd=d[f'direction_rate_{c}'];up=d[f'frame_{c}'][len(q)//2,:,2]
    v=np.cross(up,q);norm=np.linalg.norm(v,axis=1);cross=v/norm[:,None]
    vr=np.cross(up,qd);cross_rate=(vr-cross*np.sum(cross*vr,axis=1)[:,None])/norm[:,None]
    p=window.basis[c]@parameters[:n];pd=window.rate[c]@parameters[:n]
    x=window.cross_basis[c]@parameters[n:n+nc];xd=window.cross_rate[c]@parameters[n:n+nc]
    return dict(position=d[f'position_{c}']+p[:,None]*q+x[:,None]*cross,
        velocity=d[f'velocity_{c}']+pd[:,None]*q+p[:,None]*qd+xd[:,None]*cross+x[:,None]*cross_rate,
        frame=d[f'frame_{c}'],frame_rate=d[f'frame_rate_{c}'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--windows',type=Path,required=True);p.add_argument('--points',type=Path,required=True)
    p.add_argument('--centres',type=float,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);motion=JointMotion(a.motion);r=motion.result
    records=json.loads(a.points.read_text())['windows'];dataset=Dataset();reports=[]
    for mid in a.centres:
        started=time.monotonic();rec=min(records,key=lambda rec:abs(rec['mid_s']-mid))
        if abs(rec['mid_s']-mid)>.1:raise ValueError('Missing point association')
        source=a.windows/f'window-{mid:07.2f}';win=Window(source,r['interval_s'],r['path_spacing_seconds'],r['cross_spacing_seconds'])
        exact=dataset.window_from_description(win.result)
        archived=np.load(source/'training-images.npz');pixels=archived['pixels'];u=archived['forward_m'];v=archived['port_m']
        shape=(len(v),len(u));directory=a.output/f'window-{mid:07.2f}';directory.mkdir()
        points=np.asarray(rec['refined_positions'])
        # Vessel-shaped truth is a control, not a constraint on measured data.
        association=np.c_[(pixels[1,:2]-pixels[0,:2])/(u[1]-u[0]),(pixels[len(u),:2]-pixels[0,:2])/(v[1]-v[0])]
        origin=np.asarray(win.result['association_origin_east_north_m'])
        hull_body=np.array([[-130,-20],[-65,-20],[0,-20],[65,-20],[130,-15],[143,0],
            [130,15],[65,20],[0,20],[-65,20],[-130,20]])
        hull=np.c_[hull_body@association.T+origin,np.full(len(hull_body),20.)]
        scenes={'point':points[:1],'hull':hull}
        actual=DenseImages.reopen(source/'training/model.bin',derivatives=8);actual.training=True
        geo,_,_=win.geometry(np.asarray(r['parameters']),actual,motion.gauge_point,r['transmitter_groups'],motion.receiver_local,antennas=bool(r['parameter_counts']['antenna']))
        try:measured=actual.evaluate_geometry(geo,derivatives=False)[:,0]
        finally:actual.close()
        cases={'measured':measured};closure=[]
        for label,scene in scenes.items():
            bands=[]
            for c,band in enumerate(exact['fast']):
                np.testing.assert_allclose(band.field.time,win.time[c],atol=2e-8,rtol=0)
                f=band.field;physical=state(win,np.asarray(r['parameters']),c);y=np.zeros_like(f.y)
                for j,point in enumerate(scene):
                    pos=physical['position']+np.einsum('tij,j->ti',physical['frame'],point)
                    vel=physical['velocity']+np.einsum('tij,j->ti',physical['frame_rate'],point)
                    h=f.trajectory(pos,vel,transmitter=motion.tx[c],receiver=motion.rx,position_offset=pos-f.data['centre_ecef_m'])
                    amplitude=motion.gains[c]*(1 if label=='point' else np.exp(.7j*j))
                    y+=amplitude*h
                field=copy.copy(f);field.y=y;sim=copy.copy(band);sim.field=field;bands.append(sim)
            worker=DenseImages(bands,pixels,directory/label)
            worker.close();worker=DenseImages.reopen(directory/label/'model.bin',derivatives=8);worker.training=True
            try:
                image=worker.evaluate_geometry(geo,derivatives=False)[:,0]
                cases[label]=image
                if label=='point':
                    # The peak at a known point must close against the CPU
                    # matched norm, including the static-nuisance projection.
                    worker.set_pixels(scene)
                    at=worker.evaluate_cached(derivatives=False)[:,0,0]
                    for c,band in enumerate(bands):
                        y=band.project(band.field.y)
                        energy=np.vdot(y[band.training],y[band.training]).real
                        expected=energy/motion.gains[c].conjugate()
                        # The GPU uses a fixed phase gauge for antenna shifts.
                        ratio=abs(at[c])/abs(expected)
                        closure.append(float(ratio))
                        if not .999<ratio<1.001:raise ValueError(f'Synthetic known-point matched norm failed: {c} {ratio}')
            finally:worker.close()
        gains=np.asarray([complex(*z) for z in r['calibration_gains_real_imag']]);counts=r['parameter_counts']
        if counts['phase']:gains[[0,1,2,4]]*=np.exp(1j*np.asarray(r['parameters'][-counts['phase']:]))
        combined={name:np.r_[value,np.einsum('c,cp->p',gains.conj(),value)[None]].reshape(6,*shape) for name,value in cases.items()}
        np.savez_compressed(directory/'images.npz',pixels=pixels,forward_m=u,port_m=v,point_truth=points[:1],hull_truth=hull,**combined)
        plot(directory,mid,combined,u,v)
        report=dict(mid_s=mid,synthetic_point_matched_norm_ratio=closure,seconds=time.monotonic()-started,
            scope=__doc__,motion=str(a.motion),accepted_real_scene=False)
        reports.append(report);(a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=False,windows=reports)),indent=2)+'\n')
        print(json.dumps(serial(report)),flush=True)
    (a.output/'analysis.json').write_text(json.dumps(serial(dict(complete=True,windows=reports)),indent=2)+'\n')


def plot(directory,mid,cases,u,v):
    import matplotlib;matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(6,3,figsize=(17,16),layout='constrained',sharex=True,sharey=True)
    for j,(name,images) in enumerate(cases.items()):
        for c,ax in enumerate(axes[:,j]):
            power=abs(images[c])**2;power/=power.max()
            im=ax.imshow(10*np.log10(np.maximum(power,1e-12)),extent=[u[0],u[-1],v[0],v[-1]],origin='lower',cmap='magma',vmin=-35,vmax=0,aspect='equal')
            ax.set(title=['Seven','SBS','Nine','Ten','ABC','Five-band coherent'][c]+' · '+name,xlabel='Forward coordinate (m)',ylabel='Port coordinate (m)')
    fig.colorbar(im,ax=list(axes.ravel()),shrink=.7,label='Power relative to each panel peak (dB); shape comparison only')
    fig.suptitle(f'{mid-15:.2f}–{mid+15:.2f} s · exact recorded-aperture controls\nSame physical motion, kernels, epochs, whitening and nuisance projection; noiseless synthetic point and 286 × 40 m hull')
    fig.savefig(directory/'comparison.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()

"""Known-point aperture information on the exact retained ship epochs."""
import argparse,json,time
from pathlib import Path
import numpy as np
from full_scene_context import open_scene
from spatial_aperture_information import information
from rigid_patch import profile_static


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--motion',type=Path,required=True)
    selection=p.add_mutually_exclusive_group(required=True)
    selection.add_argument('--native-review',type=Path)
    selection.add_argument('--intervals',type=float,nargs='+')
    p.add_argument('--x-axis',choices=['capture-time','duration'],default='capture-time')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    fit,motion,bands,saved=open_scene(a.motion);intervals=[]
    if a.native_review:
        for f in sorted(a.native_review.glob('aperture-*/analysis.json')):intervals.append(json.loads(f.read_text())['interval_s'])
        if len(intervals)!=20:raise ValueError('Use the completed twenty-aperture native review')
    else:
        if len(a.intervals)%2:raise ValueError('Supply complete interval pairs')
        intervals=list(zip(a.intervals[::2],a.intervals[1::2]))
    records=[]
    for first,last in intervals:
        values=[];derivatives=[];projected=[];projected_derivatives=[]
        for c,band in enumerate(bands):
            b=band.point;lo,hi=np.searchsorted(b.t,[first,last]);v=b.waveform_window(lo,hi)
            h,_,_,de,dn=v.waveform(np.zeros(b.b.shape[1]),c,False,np.zeros(3),True)
            d=np.stack([de,dn],axis=-1);values.append(h);derivatives.append(d)
            mask=np.ones(len(v.t),bool)
            projected.append(profile_static(h,v.field.root_information,mask,v.groups))
            projected_derivatives.append(profile_static(d,v.field.root_information,mask,v.groups))
        record=dict(interval_s=[first,last],cases={})
        for name,hh,dd,coherent in [('coherent_native',values,derivatives,True),('independent_band_phases',values,derivatives,False),
            ('coherent_local_stationary_nuisance',projected,projected_derivatives,True)]:
            fisher=information(hh,dd,coherent=coherent);eigen,axes=np.linalg.eigh(fisher)
            if eigen.min() < -1e-9*max(eigen.max(),1e-300):raise ValueError('Spatial information became indefinite')
            lengths=[None if x<=0 else float(np.sqrt(2/x)) for x in eigen]
            record['cases'][name]=dict(unit_snr_position_fisher_per_m2=fisher.tolist(),eigenvalues=eigen.tolist(),
                local_one_radian_ambiguity_lengths_m=lengths,principal_axes_body_xy=axes.T.tolist())
        records.append(record);print(json.dumps(record),flush=True)
    from matplotlib import pyplot as plt
    t=np.mean(intervals,axis=1) if a.x_axis=='capture-time' else np.diff(intervals,axis=1)[:,0]
    fig,axes=plt.subplots(2,1,figsize=(15,9),layout='constrained',sharex=True)
    for name,label in [('coherent_native','Known interband phase'),('independent_band_phases','Independent unknown band phases'),
        ('coherent_local_stationary_nuisance','Known interband phase + local stationary nuisance')]:
        values=np.array([r['cases'][name]['local_one_radian_ambiguity_lengths_m'] for r in records],float)
        axes[0].semilogy(t,values[:,0],'.-',label=label);axes[1].semilogy(t,values[:,1],'.-',label=label)
    for ax in axes:ax.legend(fontsize=9)
    axes[0].set(ylabel='Weak spatial direction: local ambiguity length (m)')
    axes[1].set(ylabel='Strong spatial direction: local ambiguity length (m)',xlabel='Capture time (s)' if a.x_axis=='capture-time' else 'Coherent aperture duration (s)')
    fig.suptitle('Known flat point: conditional finite-field local spatial information; exact epochs, motion and waveform kernels\nUnit integrated SNR; RMS position CRLB = ambiguity length / sqrt(2 SNR); not a hull-resolution or global-ambiguity guarantee')
    fig.savefig(a.output/'spatial-aperture-information.png',dpi=145);plt.close(fig)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),
        recorded_data_rescaled=False,point_body_m=fit['point_body_m'],apertures=records,seconds=time.monotonic()-started),indent=2)+'\n')


if __name__=='__main__':main()

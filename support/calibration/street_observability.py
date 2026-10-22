"""Angular coverage and optimistic known-trajectory information for street trials."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from road_fit import load_trial
from temporal_bands import TemporalBandProblem


def projected_information(j,selected,nuisance):
    q=j[:,selected]
    if len(nuisance):
        u,s,_=np.linalg.svd(j[:,nuisance],full_matrices=False)
        u=u[:,s>s[0]*1e-10]
        q=q-u@(u.T@q)
    scale=np.tile([30.,30.,75.],3)
    q=q/scale
    singular=np.linalg.svd(q,compute_uv=False)
    posterior=np.linalg.inv(q.T@q+np.eye(9)/100)
    return {'site_information_singular_values_per_m':singular.tolist(),
            'posterior_coordinate_sigma_with_10m_prior':np.sqrt(np.diag(posterior)).reshape(3,3).tolist(),
            'posterior_covariance_m2_with_10m_prior':posterior.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('fit','histories','patches','templates','routes','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();saved=json.loads(args.fit.read_text())
    joint=load_trial(saved,args.routes,args.histories,args.patches,args.templates,allow_time_dependent=True)
    p=np.array(saved['reduced_state'])
    model=TemporalBandProblem(joint,saved['amplitude_block_seconds'])
    expected=model.expected_signal(p,all_band_powers=True)
    j=expected.residual(p,True,False,False)
    sites=np.arange(9)
    calibration=np.arange(joint.beta_offset,model.count)
    optimistic={
        'known_paths_phases_and_range_origins':projected_information(j,sites,[]),
        'known_paths_unknown_phase_and_range_origins':projected_information(j,sites,calibration),
        'unknown_paths_phase_and_range_origins':projected_information(j,sites,np.arange(9,model.count))}
    fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    site_names=['Ornata','Eyre','Ridge'];records=[]
    for index,source in enumerate(joint.problems):
        time=source.model.seed['epoch']+np.linspace(-source.model.half,source.model.half,101)
        local=np.r_[p[9+index*joint.n:9+(index+1)*joint.n],p[:9]]
        x,_,transmitters=source.model.positions(local,time,source.sites,True)
        colour=plt.get_cmap('tab10')(index)
        for site in range(3):
            a=x-transmitters[site];d=source.model.receiver_enu_m-transmitters[site]
            az=np.degrees(np.angle(np.exp(1j*(np.arctan2(a[:,1],a[:,0])-np.arctan2(d[1],d[0])))))
            el=np.degrees(np.arctan2(a[:,2],np.linalg.norm(a[:,:2],axis=1))-
                          np.arctan2(d[2],np.linalg.norm(d[:2])))
            axes[site].plot(az,el,color=colour,lw=2,label=saved['labels'][index])
            axes[site].annotate(saved['labels'][index],(az[-1],el[-1]),fontsize=8,color=colour)
            records.append({'label':saved['labels'][index],'site':site,
                'azimuth_departure_degrees':az.tolist(),'elevation_departure_degrees':el.tolist()})
    for ax,name in zip(axes,site_names):
        ax.plot(0,0,'+',color='black',ms=10)
        ax.set(title=name,xlabel='Azimuth from direct receiver direction (degrees)',
               ylabel='Elevation from direct receiver direction (degrees)')
    fig.suptitle('The preferred road hypotheses as seen from each transmitter\nBlack cross: direct receiver direction; tracks span only a narrow departure-angle sector')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output.with_suffix('.png'),dpi=150);plt.close(fig)
    spans=[]
    for site in range(3):
        selected=[r for r in records if r['site']==site]
        spans.append([float(np.ptp(np.concatenate([r[key] for r in selected])))
                      for key in ('azimuth_departure_degrees','elevation_departure_degrees')])
    result={'source_fit':str(args.fit),'angular_coverage':records,
        'azimuth_elevation_span_degrees_by_site':spans,'information_cases':optimistic,
        'scope':'Conditional sensitivity analysis of the preferred-road hypotheses. Expected samples use every independently measured band power but an ideal shared-phase point response, including bands the actual fit rejects. The optimistic case treats complete target trajectories as exactly known. Common target phase and positive band magnitudes remain nuisance parameters within each block. This is not a certified experimental CRLB: the covariance model, road positions, point-scattering hypothesis and receiver are conditional assumptions.',
        'interpretation':'Reported posterior standard deviations include an explicit 10 m coordinate prior. Values close to 10 m show that this prior supplies most of the position information. The known-path cases are optimistic comparisons, not independent target anchors.'}
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2))
    print('angle spans',spans,flush=True)
    for name,data in optimistic.items():print(name,np.round(data['posterior_coordinate_sigma_with_10m_prior'],2),flush=True)


if __name__=='__main__':main()

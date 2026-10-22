"""Independent stationary-scene phase control over the whole recording.

Fixed 0.6–5 and 5–10 km regions exclude the vessel. No correction is applied.
A disjoint 30–90 s reference is a diagnostic matched pattern, not a CRLB claim.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
from matplotlib import pyplot as plt
from extract_full_aperture import load
from field import average

out=Path('docs/data/isar_20261017/full-capture-static-phase');out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(5,2,figsize=(15,12),layout='constrained');report=[]
for c in range(5):
 a=load(Path(f'build/isar-20261015/full-geographic-aperture/channel-{c}'));meta=json.loads(a['metadata_json']);coordinate=(meta['first_range_sample']+np.arange(a['moments'].shape[2]))*299792458/8e6
 time=[];values=[];information=[]
 for first in range(0,len(a['time_s']),4096):
  last=min(first+4096,len(a['time_s']));w=a['information'][first:last]*a['valid'][first:last];y,weight=average(a['moments'][0,first:last],w,64)
  t=average(a['time_s'][first:last],w,64)[0];fallback=average(a['time_s'][first:last],np.ones(len(w)),64)[0];time.append(np.where(weight>0,t,fallback));values.append(y);information.append(weight)
 t=np.concatenate(time);y=np.concatenate(values);weight=np.concatenate(information);channel=[]
 for j,(lo,hi) in enumerate([(600.,5000.),(5000.,10000.)]):
  select=(coordinate>=lo)&(coordinate<hi);rows=(t>=30)&(t<90)&(weight>0);reference=np.average(y[rows][:,select],axis=0,weights=weight[rows]);z=y[:,select]
  energy=np.vdot(reference,reference).real;gain=z@reference.conj()/energy;coherence=abs(gain)**2*energy/np.maximum(np.sum(abs(z)**2,axis=1),1e-40)
  good=weight>0;phase=np.angle(gain);origin=np.angle(np.sum(np.exp(1j*phase[rows])*weight[rows]));relative=np.angle(np.exp(1j*(phase-origin)))
  resultant=abs(np.sum(weight[good]*np.exp(1j*relative[good]))/weight[good].sum());sigma=np.sqrt(max(0.,-2*np.log(resultant)))
  axes[c,j].plot(t,np.degrees(relative),'.',ms=1);axes[c,j].set(ylim=(-180,180),ylabel=['Seven','SBS','Nine','Ten','ABC'][c]+' phase (deg)')
  if c==0:axes[c,j].set_title(f'{lo/1000:g}–{hi/1000:g} km fixed stationary region')
  if c==4:axes[c,j].set_xlabel('Capture time (s)')
  channel.append(dict(range_m=[lo,hi],circular_phase_rms_degrees=float(np.degrees(sigma)),minimum_pattern_coherence_squared=float(coherence[good].min()),median_pattern_coherence_squared=float(np.median(coherence[good])),
    time_s=t.tolist(),phase_radians=relative.tolist(),pattern_coherence_squared=coherence.tolist()))
 report.append(dict(channel=c,regions=channel));print(c,[(r['circular_phase_rms_degrees'],r['median_pattern_coherence_squared']) for r in channel],flush=True)
fig.suptitle('Stationary-reference phase over the entire recording\nFixed range regions and independent reference interval; only a constant phase origin removed; no correction applied')
fig.savefig(out/'phase-control.png',dpi=140);plt.close(fig);(out/'analysis.json').write_text(json.dumps(dict(channels=report,correction_applied=False,scope=__doc__),indent=2)+'\n')

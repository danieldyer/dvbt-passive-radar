"""Unused-epoch coherence at physical pixels selected by the coherent detector.

Incoherent range peaks need not lie at the correct synthetic-band phase;
selecting only their centres is not a valid test of coherent image formation.
Both selectors are retained here to make that distinction directly testable.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
from matplotlib import pyplot as plt
from report_measured_scatterer_phases import serial

root=Path('docs/data/isar_20261017/full-capture-site-comparison');result=json.loads((root/'analysis.json').read_text())
source=json.loads(Path(result['motion']).read_text());bank=np.load(root/'image-bank.npz');n=source['parameter_counts']['path']+source['parameter_counts']['cross']
gains=np.asarray([complex(*z) for z in source['calibration_gains_real_imag']]);windows=sorted(Path('docs/data/isar_20261017/overlapping-ship-paths').glob('window-*/analysis.json'))
comparisons={r['assignment']:r for r in result['comparisons']};summary={}
for assignment in ['00002','10002']:
 r=comparisons[assignment];selection=r['options'];training=bank['training'][:,selection,np.arange(5)];unused=bank['unused'][:,selection,np.arange(5)];info=bank['normal']
 g=gains.copy();g[[0,1,2,4]]*=np.exp(1j*np.asarray(r['phases']));records=[]
 for k,p in enumerate(windows):
  with np.load(p.parent/'geometry-context.npz') as context:
   train_info=np.asarray([8192*np.sum(context[f'information_{c}'][context[f'training_{c}']]) for c in range(5)])
  a=np.load(p.parent/'training-images.npz');pixels=a['pixels'];i=unused[k];j=training[k]
  coherent_train=abs(np.einsum('c,cp->p',g.conj(),j))**2/np.sum(abs(g)**2*train_info)
  score_incoherent=np.sum(abs(j)**2/train_info[:,None],axis=0);points={}
  for name,score in [('coherent',coherent_train),('incoherent',score_incoherent)]:
   allowed=np.ones(len(pixels),bool);chosen=[]
   for count in range(6):
    at=int(np.argmax(np.where(allowed,score,-np.inf)));allowed&=np.sum((pixels[:,:2]-pixels[at,:2])**2,axis=1)>=35**2
    powers=abs(i[:,at])**2/info[k];combined=abs(np.vdot(g,i[:,at]))**2/np.sum(abs(g)**2*info[k])
    phase=np.degrees(np.angle(i[:,at]*i[3,at].conjugate()*g.conj()*g[3]))
    chosen.append(dict(pixel_index=at,point=pixels[at],individual_power=powers,combined_power=combined,
     gain_over_best_band_db=10*np.log10(combined/max(powers)),coherent_fraction_of_independent_power=combined/powers.sum(),phase_residual_degrees=phase))
   points[name]=chosen
  records.append(dict(mid_s=float(json.loads(p.read_text())['mid_s']),selectors=points))
 summary[assignment]=records
 phase=np.asarray([v['selectors']['coherent'][0]['phase_residual_degrees'] for v in records]);gain=np.asarray([v['selectors']['coherent'][0]['gain_over_best_band_db'] for v in records]);eta=np.asarray([v['selectors']['coherent'][0]['coherent_fraction_of_independent_power'] for v in records])
 print(assignment,'unused gain median/10/90:',np.median(gain),np.percentile(gain,[10,90]),'median efficiency',np.median(eta),flush=True)
 fig,ax=plt.subplots(3,1,figsize=(14,10),layout='constrained');time=[v['mid_s'] for v in records]
 for c in [0,1,2,4]:ax[0].plot(time,phase[:,c],'.',label=['Seven','SBS','Nine','Ten','ABC'][c])
 ax[0].set(ylabel='Unused phase residual (deg)',ylim=(-180,180));ax[0].legend(ncol=4)
 ax[1].plot(time,gain,'.-',label='Coherent peak selected on training');ax[1].plot(time,[v['selectors']['incoherent'][0]['gain_over_best_band_db'] for v in records],'.-',alpha=.5,label='Incoherent peak selected on training');ax[1].axhline(0,color='grey');ax[1].set_ylabel('Unused matched-power gain (dB)');ax[1].legend()
 ax[2].plot(time,eta,'.-');ax[2].set(ylabel='Fraction of independent-band power',xlabel='Capture time (s)',ylim=(0,1))
 fig.suptitle(f'Static calibration {assignment}: coherent and incoherent pixel selection differ\nOnly training pixels select candidates; both selectors use identical unused epochs and frozen phase calibration')
 fig.savefig(root/f'coherent-peaks-{assignment}.png',dpi=150);plt.close(fig)
 if assignment=='00002':
  updated=dict(source);parameters=source['parameters'].copy();at=n+source['parameter_counts']['antenna'];parameters[at:]=r['phases'];updated.update(parameters=parameters,
    phase_calibration_derivation=str(root/'analysis.json'),phase_calibration_scope='Four constant phases fitted on training images over all 39 windows; site assignment retained at 00002 because the best-site difference is inconclusive.')
  Path('docs/data/isar_20261017/bank-calibrated-motion-preview.json').write_text(json.dumps(updated,indent=2)+'\n')
(root/'coherent-peak-analysis.json').write_text(json.dumps(serial(summary),indent=2)+'\n')

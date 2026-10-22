"""Compare unchanged data and image coordinates under two physical motions."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES=['Seven','SBS','Nine','Ten','ABC','Coherent fixed-spectrum sum']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=[dict(np.load(path)) for path in [a.baseline,a.candidate]]
    np.testing.assert_array_equal(data[0]['information'],data[1]['information']);np.testing.assert_array_equal(data[0]['gains'],data[1]['gains'])
    np.testing.assert_allclose(data[0]['chart_to_body'],data[1]['chart_to_body'],rtol=0,atol=1e-10)
    east,xi,xj=np.intersect1d(data[0]['along_m'],data[1]['along_m'],return_indices=True)
    north,yi,yj=np.intersect1d(data[0]['port_m'],data[1]['port_m'],return_indices=True)
    keep_e=abs(east)<=130;keep_n=abs(north)<=200;east=east[keep_e];north=north[keep_n]
    if len(east)!=261 or len(north)!=401:raise ValueError('The comparison needs the complete reviewed one-metre N/E grid')
    power=[d['power'][:,y[keep_n]][:,:,x[keep_e]] for d,x,y in zip(data,[xi,xj],[yi,yj])]
    db=[10*np.log10(np.maximum(v,1e-30)) for v in power];high=max(float(v.max()) for v in db);low=high-45
    records=[]
    for name,values in zip(['reviewed-local','regularised-global'],db):
        fig,axes=plt.subplots(2,3,figsize=(12,12),layout='constrained',sharex=True,sharey=True)
        for c,ax in enumerate(axes.ravel()):
            im=ax.imshow(values[c],origin='lower',extent=[east[0],east[-1],north[0],north[-1]],vmin=low,vmax=high,cmap='magma',interpolation='nearest')
            ax.set(xlabel='East from reference scatterer (m)',ylabel='North (m)',title=NAMES[c])
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.6,label='Matched-image power (dB, same information scale)')
        fig.suptitle(name.replace('-',' ')+'; identical samples, information and coordinates');fig.savefig(a.output/f'{name}.png',dpi=130);plt.close(fig)
        records.append(dict(name=name,peak_power_db_by_channel=[float(v.max()) for v in values]))
    fig,ax=plt.subplots(1,2,figsize=(10,8),layout='constrained',sharex=True,sharey=True)
    for x,label,v in zip(ax,['Reviewed local motion','Regularised global motion'],db):
        im=x.imshow(v[5],origin='lower',extent=[east[0],east[-1],north[0],north[-1]],vmin=low,vmax=high,cmap='magma',interpolation='nearest');x.set(title=label,xlabel='East (m)',ylabel='North (m)')
    fig.colorbar(im,ax=ax,shrink=.6,label='Matched-image power (dB)');fig.savefig(a.output/'coherent-comparison.png',dpi=150);plt.close(fig)
    report=dict(scope='Observed-scene image comparison, not an isolated target SNR measurement. No registration, fitted phase screen, adaptive power scaling or pixel-dependent band selection.',scale_db=[low,high],records=records,
        peak_change_db=[b-a for a,b in zip(records[0]['peak_power_db_by_channel'],records[1]['peak_power_db_by_channel'])])
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Motion focus comparison</title><style>body{font:17px system-ui;max-width:1400px;margin:2em auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Focus comparison</h1><p>Identical original samples, coherent interval, information, band calibration and N/E pixels. Only the physical motion changes. These are observed-scene powers, not isolated target SNR.</p><img src="coherent-comparison.png"><h2>Reviewed local motion</h2><img src="reviewed-local.png"><h2>Regularised global motion</h2><img src="regularised-global.png"><pre>'+json.dumps(report,indent=2)+'</pre>')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

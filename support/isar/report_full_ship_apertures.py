"""Fixed-scale native image overview; no image estimates or normalizations."""
import argparse,html,json
from pathlib import Path
import numpy as np
from local_ship_motion import load_motion
from motion import enu_basis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-apertures',type=int,default=20)
    a=p.parse_args();records=[]
    for path in sorted(a.native.glob('aperture-*/analysis.json')):
        record=json.loads(path.read_text())
        if record['complete']:records.append((path.parent,record))
    if len(records)!=a.expected_apertures:raise ValueError('Finish every requested native aperture before publishing the whole-record overview')
    a.output.mkdir(parents=True,exist_ok=False);motion=load_motion(a.motion)
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(4,5,figsize=(17,16),layout='constrained',sharex=True,sharey=True)
    items=[]
    for ax,(directory,record) in zip(axes.ravel(),records):
        first,last=record['interval_s'];mid=(first+last)/2
        with np.load(directory/'images.npz') as source:
            east,north=source['east_m'],source['north_m'];power=source['power'][5]
        state=motion.state(np.array([mid]));ais=motion.track.sample(np.array([motion.epoch+mid]),max_gap_s=400.)
        local=enu_basis(ais['latitude'][0],ais['longitude'][0]);offset=(ais['position_ecef_m'][0]-state['position_ecef_m'][0])@local
        im=ax.imshow(10*np.log10(np.maximum(power,1e-30)),extent=[east[0],east[-1],north[0],north[-1]],
            origin='lower',cmap='magma',vmin=3,vmax=55,aspect='equal')
        ax.plot(offset[0],offset[1],'+',color='#00e5ff',markersize=5)
        ax.set(title=f'{first:.2f}–{last:.2f} s',xlabel='East (m)',ylabel='North (m)');ax.tick_params(labelsize=7)
        import os
        relative=os.path.relpath(directory,a.output)
        items.append(dict(interval_s=[first,last],images=relative+'/images.png',statistics=relative+'/coherent-statistics.png',
            metadata=relative+'/analysis.json',ais_offset_ne_m=[float(offset[1]),float(offset[0])]))
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.65,label='Fixed calibrated complex-sum power / conditional thermal variance (dB)')
    fig.suptitle('Twenty complete 30 s native apertures; one continuous physical path and fixed band calibration\nNorth up; cyan is interpolated AIS; same absolute 3–55 dB scale; motion checkpoint, not accepted resolved hull imagery',fontsize=12)
    fig.savefig(a.output/'whole-capture-coherent-overview.png',dpi=145);plt.close(fig)
    options=''.join(f'<option value="{i}">{x["interval_s"][0]:.2f}–{x["interval_s"][1]:.2f} s</option>' for i,x in enumerate(items))
    page='''<!doctype html><meta charset="utf-8"><title>Full-capture native ship image review</title>
<style>body{font:17px/1.5 system-ui;max-width:1500px;margin:35px auto;padding:0 20px}img{max-width:100%;height:auto}.note{padding:15px;background:#fff1d6}button,select{font:inherit;padding:5px}a{color:#075985}</style>
<h1>Full-capture native ship image review</h1>
<p class="note">Motion checkpoint only. All five original complex images and their calibrated complex sum are formed before power detection. These are complete thirty-second apertures, with fixed north/east axes, constant band calibration and a fixed colour scale. A resolved hull and a uniquely localized physical trajectory remain unproved. No fitted source image replaces the observations.</p>
<p>Times cover the available native recording. The final aperture overlaps its predecessor to retain a complete thirty-second integration. Cyan marks the interpolated AIS antenna, which is a prior rather than measured target truth.</p>
<a href="whole-capture-coherent-overview.png"><img src="whole-capture-coherent-overview.png"></a>
<h2>Five independent bands and coherent combinations</h2>
<p><button id="previous">Previous</button> <select id="aperture">OPTIONS</select> <button id="next">Next</button> · <a id="metadata">Numerical metadata</a></p>
<img id="bands"><img id="statistics">
<script>const items=ITEMS;const choose=document.getElementById('aperture');function show(){const x=items[Number(choose.value)];document.getElementById('bands').src=x.images;document.getElementById('statistics').src=x.statistics;document.getElementById('metadata').href=x.metadata;}function step(n){choose.value=Math.max(0,Math.min(items.length-1,Number(choose.value)+n));show();}choose.onchange=show;document.getElementById('previous').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);show();</script>'''
    page=page.replace('OPTIONS',options).replace('ITEMS',json.dumps(items))
    (a.output/'index.html').write_text(page)
    (a.output/'analysis.json').write_text(json.dumps(dict(complete=True,scope=__doc__,motion=str(a.motion),
        fixed_overview_scale_db=[3,55],apertures=items,accepted_as_resolved_hull=False),indent=2)+'\n')


if __name__=='__main__':main()

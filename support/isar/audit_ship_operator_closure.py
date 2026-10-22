"""Compare data-side recorded-field imaging against original-symbol BP.

Tests the actual recorded observations, not only synthetic template Jacobians.
Both projected and unprojected fixed-field scores are recorded separately.
"""
import argparse,json
from pathlib import Path
import numpy as np
from field import interpolate,average
from motion import C,excess_path
from ship_window import Dataset
from render_ship_window import WindowMotion
from rigid_patch import profile_static
from report_measured_scatterer_phases import serial


def audit(dataset,result,out):
    first,last=result['interval_s'];w=dataset.window(first,last,model=result['source_model']);motion=WindowMotion(result)
    images=np.load(Path(out).parent/f'window-{motion.mid:07.2f}'/'images.npz')
    xx,yy=np.meshgrid(images['east_m'],images['north_m']);power=images['power'][:5].sum(axis=0)
    allowed=(abs(xx)<120)&(yy>-80)&(yy<240);points=[]
    for j in range(4):
        i=np.unravel_index(np.argmax(np.where(allowed,power,-np.inf)),xx.shape);points.append([xx[i],yy[i],20.]);allowed&=(xx-xx[i])**2+(yy-yy[i])**2>40**2
    points=np.asarray(points);records=[]
    for c,(native,fast) in enumerate(zip(w['native'],w['fast'])):
        a=dataset.raw[c];meta=json.loads(a['metadata_json']);f=fast.field;t=f.time
        s=motion.state(t);row_state=motion.state(native.field.data['row_time'])
        keep=(a['time_s']>=first)&(a['time_s']<last);rt=a['time_s'][keep];raw_state=motion.state(rt);weight=a['information'][keep]*a['valid'][keep]
        record=dict(channel=c,points=[])
        for point in points:
            pos=s['position_ecef_m']+s['body_to_ecef']@point;vel=s['velocity_ecef_mps']+s['frame_velocity']@point
            h=f.trajectory(pos,vel,transmitter=fast.tx,receiver=fast.rx,position_offset=pos-f.data['centre_ecef_m'])
            root=f.root_information;all_rows=root>0;py=profile_static(f.y,root,all_rows);ph=profile_static(h,root,all_rows)
            score=np.vdot(h,f.y);projected=np.vdot(h,py);retained=np.vdot(ph,ph).real/np.vdot(h,h).real
            pos=raw_state['position_ecef_m']+raw_state['body_to_ecef']@point;vel=raw_state['velocity_ecef_mps']+raw_state['frame_velocity']@point
            path,g=excess_path(pos,fast.tx,fast.rx);omega=-f.k*.001024*np.sum(g*vel,axis=1);pieces=[]
            for start in range(0,len(rt),512):
                stop=min(start+512,len(rt));index=np.flatnonzero(keep)[start:stop]
                zero,one=interpolate(a['moments'][:,index[0]:index[-1]+1],path[start:stop]*8e6/C-meta['first_range_sample'],np.array([0]))
                pieces.append((zero[:,0]-1j*omega[start:stop]*one[:,0])*np.exp(1j*f.k*path[start:stop]))
            original=np.sum(np.concatenate(pieces)*weight)/weight.sum()
            record['points'].append(dict(point=point,original_bp=[original.real,original.imag],field_bp=[(score/root.dot(root)).real,(score/root.dot(root)).imag],
                field_to_original_magnitude_db=20*np.log10(abs(score/root.dot(root)/original)),field_to_original_phase_deg=np.degrees(np.angle(score/original)),
                static_projected_to_original_magnitude_db=20*np.log10(abs(projected/root.dot(root)/original)),static_template_retention_db=10*np.log10(retained),
                predicted_path_rate_mps=[float(np.min(np.sum(g*vel,axis=1))),float(np.max(np.sum(g*vel,axis=1)))]))
        records.append(record)
    Path(out).write_text(json.dumps(serial(dict(interval=[first,last],points=points,channels=records)),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--fits',type=Path,required=True);p.add_argument('--images',type=Path,required=True);p.add_argument('--centres',nargs='+',type=float,required=True);a=p.parse_args()
    d=Dataset()
    for mid in a.centres:audit(d,json.loads((a.fits/f'window-{mid:07.2f}'/'analysis.json').read_text()),a.images/f'operator-closure-{mid:07.2f}.json')
if __name__=='__main__':main()

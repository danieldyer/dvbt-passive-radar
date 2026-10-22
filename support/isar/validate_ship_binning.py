"""Independent native-symbol synthesis checks for short coherent means.

Tests all five carriers, off-centre targets, nonuniform information, C0/C1,
rotating-view geometry and differential phase after shared-time bookkeeping.
"""
from pathlib import Path
import json,tempfile,subprocess
import numpy as np
from numpy.polynomial.legendre import leggauss
from motion import C,excess_path
from field import average
from imaging import geometry_from_state,write_input,read_image
from channel_update import ChannelUpdates


def state(t):
    angle=.004*np.asarray(t)+.00001*np.asarray(t)**2;rate=.004+.00002*np.asarray(t);c,s=np.cos(angle),np.sin(angle)
    b=np.zeros((len(t),3,3));bd=np.zeros_like(b);b[:,0,0]=b[:,1,1]=c;b[:,0,1]=-s;b[:,1,0]=s;b[:,2,2]=1
    bd[:,0,0]=bd[:,1,1]=-s*rate;bd[:,0,1]=-c*rate;bd[:,1,0]=c*rate
    return dict(position_ecef_m=np.array([-5000.,-6500.,10.])+np.asarray(t)[:,None]*[-2.,4.,0.],
        velocity_ecef_mps=np.broadcast_to([-2.,4.,0.],(len(t),3)),body_to_ecef=b,frame_velocity=bd)


def main():
    period=.001088;t=np.arange(512)*period;block=4;weight=1+.2*np.sin(17*t);tx=np.array([30000.,1000.,700.]);rx=np.array([0.,0.,10.])
    u,w=leggauss(32);u/=2;w/=2;points=np.array([[0.,0.,20.],[600.,-250.,20.],[-300.,300.,20.]])
    records=[]
    with tempfile.TemporaryDirectory(prefix='ship-binning-validation-',dir='build') as temporary:
        root=Path(temporary)
        for point_index,point in enumerate(points):
            errors=[]
            for c,rf in enumerate([177.5e6,184.5e6,191.625e6,219.5e6,226.5e6]):
                s=state(t);pos=s['position_ecef_m']+s['body_to_ecef']@point;vel=s['velocity_ecef_mps']+s['frame_velocity']@point
                path,gradient=excess_path(pos,tx,rx);centre=excess_path(s['position_ecef_m'],tx,rx)[0];k=2*np.pi*rf/C
                first=int(np.floor(path.min()*8e6/C))-80;ranges=192;coordinate=path*8e6/C-first
                pulse=np.sinc(.81*(np.arange(ranges)[None,:]-coordinate[:,None]));omega=-k*.001024*np.sum(gradient*vel,axis=1)
                tone=np.exp(1j*omega[:,None]*u);m0=tone@w;m1=tone@(u*w)
                raw=pulse*np.exp(-1j*k*path[:,None]);moments=np.stack([m0[:,None]*raw,m1[:,None]*raw])
                g,_=geometry_from_state(s,tx,rx,rf,first,weight);source=root/'native.bin';target=root/'native-image.bin'
                write_input(source,g,moments,np.array([[*point,0.]]),rf);subprocess.run(['build/release/support/isar/isar_backprojection',str(source),str(target)],check=True)
                a=read_image(target,1)[0];native=(a[0]+1j*a[1])/a[2];source.unlink();target.unlink()
                reference=centre+2.+.4*t+.01*t*t
                phase=np.exp(1j*k*reference);binned=[]
                for moment in moments:binned.append(average(moment*phase[:,None],weight,block)[0])
                tm,info=average(t,weight,block);bs=state(tm);bg,_=geometry_from_state(bs,tx,rx,rf,first,info)
                ref_at_mean=excess_path(bs['position_ecef_m'],tx,rx)[0]+2.+.4*tm+.01*tm*tm
                rotation=np.exp(1j*k*(excess_path(bs['position_ecef_m'],tx,rx)[0]-ref_at_mean))
                bg[:,9,1]=rotation.real;bg[:,9,2]=rotation.imag;bg[:,5,3]=k;bg[:,6,3]=c;bg[:,7,3]=1.;bg[:,8,3]=1.
                worker=ChannelUpdates(np.array([[*point,0.]]),root/f'worker-{point_index}-{c}')
                try:z,information=worker.update(bg,np.asarray(binned))
                finally:worker.close()
                actual=z[c,0]/information[c];ratio=actual/native;error=dict(point=point.tolist(),channel=c,
                    coherent_power_change_db=float(20*np.log10(abs(ratio))),phase_error_degrees=float(np.degrees(np.angle(ratio))))
                errors.append(error);print(json.dumps(error),flush=True)
                if abs(error['coherent_power_change_db'])>.01 or abs(error['phase_error_degrees'])>.02:raise AssertionError('Coherent means exceed their error budget')
            phase=np.asarray([r['phase_error_degrees'] for r in errors]);relative=phase-phase[3]
            if max(abs(relative))>.025:raise AssertionError('Binning corrupts differential interband phase')
            records.append(dict(channels=errors,maximum_differential_phase_error_degrees=float(max(abs(relative)))))
    Path('docs/data/isar_20261017/ship-binning-validation.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':main()

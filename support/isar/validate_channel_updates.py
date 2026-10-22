"""Verify signed per-band updates against separate original-symbol images."""
from pathlib import Path
import argparse,tempfile,subprocess,json
import numpy as np
from channel_update import ChannelUpdates
from test_rigid_motion import fixture
from imaging import geometry_from_state,write_input,read_image

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--executable',default='build/release/support/isar/isar_backprojection');args=p.parse_args()
rng=np.random.default_rng(731);pixels=np.array([[0.,0.,20.,0.],[35.,-20.,20.,0.],[-40.,24.,20.,0.]])
geometry=[];samples=[];reference=[];information=[];ranges=192
with tempfile.TemporaryDirectory(prefix='channel-update-',dir='build') as temporary:
 root=Path(temporary)
 for c in range(5):
  f=fixture(c);d=f.data;n=min(31,len(d['row_time']));s=dict(position_ecef_m=d['centre_ecef_m'][:n],velocity_ecef_mps=d['centre_velocity_ecef_mps'][:n],
   body_to_ecef=d['body_to_ecef'][:n],frame_velocity=np.gradient(d['body_to_ecef'],d['row_time'],axis=0,edge_order=2)[:n])
  first=int(np.floor(d['centre_path_m'][:n].min()*8e6/299792458.))-80
  g,_=geometry_from_state(s,d['transmitter_ecef_m'],d['receiver_ecef_m'],f.metadata['rf_hz'],first,d['row_information'][:n])
  raw=(rng.normal(size=(2,n,ranges))+1j*rng.normal(size=(2,n,ranges))).astype('<c8')
  # Each band includes entering and leaving data. The independent reference
  # forms each positive subset separately with the original operator.
  sign=np.where(np.arange(n)%3==0,-1.,1.);total=0.;weight=0.
  for direction in [-1.,1.]:
   use=sign==direction;source=root/f'{c}-{direction}.bin';target=root/f'{c}-{direction}-image.bin'
   write_input(source,g[use],raw[:,use],pixels,f.metadata['rf_hz']);subprocess.run([args.executable,str(source),str(target)],check=True)
   a=read_image(target,len(pixels));total+=direction*(a[:,0]+1j*a[:,1]);weight+=direction*a[:,2]
  reference.append(total);information.append(weight)
  g[:,5,3]=f.k;g[:,6,3]=c;g[:,7,3]=sign;geometry.append(g);samples.append(raw)
 g=np.concatenate(geometry);raw=np.concatenate(samples,axis=1);order=rng.permutation(len(g));g=g[order];raw=raw[:,order]
 worker=ChannelUpdates(pixels,root/'worker',active_channels=False,executable=args.executable)
 try:actual,weight=worker.update(g,raw)
 finally:worker.close()
 reference=np.asarray(reference);info=np.asarray(information)[:,0]
 error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference));info_error=float(np.max(abs(weight-info))/max(abs(info)))
 print(json.dumps(dict(relative_complex_error=error,relative_information_error=info_error)),flush=True)
 if error>4e-4 or info_error>1e-6:raise AssertionError('Signed channel updates change the image')
 # Nonconsecutive and reordered active IDs exercise the compact transport.
 # Accumulate signed packets directly, without allocating empty band images.
 worker=ChannelUpdates(pixels,root/'active-worker',active_channels=True,executable=args.executable)
 compact=np.zeros_like(actual);compact_info=np.zeros(5)
 try:
  for channels in [[4,0],[3],[1,2]]:
   use=np.isin(g[:,6,3],channels);worker.accumulate(g[use],raw[:,use],compact,compact_info)
 finally:worker.close()
 error=float(np.linalg.norm(compact-reference)/np.linalg.norm(reference));info_error=float(np.max(abs(compact_info-info))/max(abs(info)))
 transfer_ratio=worker.transferred_bytes/(3*5*len(pixels)*8)
 print(json.dumps(dict(active_relative_complex_error=error,active_relative_information_error=info_error,active_transport_ratio=transfer_ratio)),flush=True)
 if error>4e-4 or info_error>1e-6:raise AssertionError('Active channel updates change the image')
 if transfer_ratio!=1/3:raise AssertionError('Inactive images were transferred')

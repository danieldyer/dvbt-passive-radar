"""Recorded-data end-to-end derivative check for the joint fit."""
import json
from pathlib import Path
import numpy as np
from joint_ship_autofocus import Window,Joint,curve_seed

root=Path('docs/data/isar_20261017/overlapping-ship-paths');interval=[3.2720443125000007,601.8136398125001]
w=[Window(p.parent,interval,3.,15.) for p in sorted(root.glob('window-*/analysis.json'))[:2]]
a=json.loads(Path('docs/data/isar_20261017/coherent-ship-hull-north-up/analysis.json').read_text())
gains=np.asarray([complex(*z) for z in a['gains_real_imag']]);m=Joint(w,interval,3.,gains,coherent=True,antennas=True)
p=np.r_[curve_seed(w,interval,3.),np.zeros(m.cross_count+m.offset_count+m.phase_count)]
a=m.score(p);records=[]
for name,j,h in [('path',5,.002),('cross_path',m.path_count+1,4.),('transmitter_height',m.count+2,4.),('band_phase',m.count+m.offset_count,.002)]:
 d=np.eye(1,len(p),j)[0]*h
 finite=(m.score(p+d,derivatives=False)['value']-m.score(p-d,derivatives=False)['value'])/(2*h)
 analytic=a['gradient'][j];relative=abs(finite-analytic)/max(abs(finite),abs(analytic),1e-7)
 record=dict(name=name,analytic=analytic,finite=finite,relative_error=relative);records.append(record);print(json.dumps(record),flush=True)
 if relative>.02 and abs(finite-analytic)>2e-6:raise AssertionError(f'Joint {name} gradient disagrees')
Path('docs/data/isar_20261017/joint-gradient-validation.json').write_text(json.dumps(records,indent=2)+'\n')

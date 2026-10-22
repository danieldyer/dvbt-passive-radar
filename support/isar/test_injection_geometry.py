import json
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np

from render_injected_point import fitted_state
from test_recovery_experiment import problem


class InjectionGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executable=Path('build/release/support/validation/recorded_injection_probe')
        if not cls.executable.exists():raise unittest.SkipTest('Build the recorded_injection_probe validation target')

    def test_cpp_point_paths_match_python_for_all_motion_families(self):
        ex=problem();times=np.linspace(2.,12.,21)
        state=np.asarray(ex.state);state[2]=np.array([.01,-.007,0.])
        plan=dict(interval_s=[1.,15.],midpoint_s=ex.midpoint,receiver_ecef_m=ex.receiver.tolist(),
            transmitters_ecef_m=ex.transmitters.tolist(),ecef_to_enu=ex.rotation.tolist(),state_enu=state.tolist(),
            gains_real_imag=[[1.,0.]]*5)
        rigid=dict(rigid_parameters=[.07,-.003,.0004,.2,-.01],reference_transmitter_ecef_m=ex.transmitters[2].tolist())
        results=[]
        with tempfile.TemporaryDirectory(prefix='injection-geometry-',dir='build') as directory:
            for variant in [{},rigid,{**rigid,'translation_direction_ecef':[.15,-.22,.09]}]:
                record={**plan,**variant};path=Path(directory)/'plan.json';path.write_text(json.dumps(record))
                actual=np.asarray(json.loads(subprocess.check_output([str(self.executable),'--paths',str(path),*[str(t) for t in times]],text=True)))[:,1:]
                model=dict(motion_model='rigid-patch',**rigid,translation_direction_ecef=variant.get('translation_direction_ecef')) if variant else {}
                pose=fitted_state(times,state,ex.midpoint,ex.rotation,ex.receiver,model)['position_ecef_m']
                expected=np.column_stack([np.linalg.norm(pose-a,axis=1)+np.linalg.norm(pose-ex.receiver,axis=1)-np.linalg.norm(a-ex.receiver) for a in ex.transmitters])
                np.testing.assert_allclose(actual,expected,atol=2e-8,rtol=0.)
                results.append(actual)
            # Angular motion cannot move a point at the frame origin.
            record={**plan,**rigid,'rigid_parameters':[.07,-.003,.0004,-.4,.03]};path.write_text(json.dumps(record))
            other=np.asarray(json.loads(subprocess.check_output([str(self.executable),'--paths',str(path),*[str(t) for t in times]],text=True)))[:,1:]
            np.testing.assert_array_equal(other,results[1])


if __name__=='__main__':unittest.main()

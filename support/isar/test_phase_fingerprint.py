import unittest
import numpy as np

from phase_fingerprint import measure


class PhaseFingerprintTests(unittest.TestCase):
    def test_fixed_trajectory_matches_brightness_changes_and_phase_gauges(self):
        t=np.arange(160)*.0125;k=4.7;h=np.exp(-1j*k*(.37*(t-1)+.09*(t-1)**2))
        amplitude=np.column_stack([.4+.2*t,.8-.1*t]);phi=np.array([.8,-2.3])
        y=h[:,None]*amplitude*np.exp(1j*phi);training=np.arange(len(t))%2==0
        result=measure(t,y,np.ones(len(t))*1e4,h,training,~training)
        for i,r in enumerate(result):
            self.assertLess(abs(np.angle(np.exp(1j*(r['phase_rad']-phi[i])))),1e-12)
            self.assertLess(r['testing_error'],1e-20)
            self.assertLess(r['testing_phase_rms_deg'],1e-10)

    def test_search_cannot_change_template_or_use_test_phases(self):
        t=np.arange(160)*.0125;h=np.exp(-2j*t);y=h[:,None]*np.exp(.4j);train=np.arange(len(t))%2==0
        before=measure(t,y,np.ones(len(t)),h,train,~train)[0]
        y[~train]*=np.exp(.9j)
        after=measure(t,y,np.ones(len(t)),h,train,~train)[0]
        self.assertEqual(before['phase_rad'],after['phase_rad'])
        self.assertEqual(before['training_error'],after['training_error'])
        self.assertGreater(after['testing_phase_rms_deg'],50.)

    def test_neighbour_with_matching_motion_is_separated_from_bright_wrong_motion(self):
        t=np.arange(240)/120;k=4.7;h=np.exp(-1j*k*(.4*(t-1)+.1*(t-1)**2))
        y=np.column_stack([3*h*np.exp(.7j+1.5j*(t-1)),.5*h*np.exp(-.4j)])
        train=np.floor(t/.125).astype(int)%2==0
        result=measure(t,y,np.ones(len(t))*100,h,train,~train)
        self.assertGreater(result[0]['coherent_training_score'],result[1]['coherent_training_score'])
        self.assertGreater(result[0]['testing_phase_rms_deg'],20.)
        self.assertLess(result[1]['testing_phase_rms_deg'],1e-10)


if __name__=='__main__':unittest.main()

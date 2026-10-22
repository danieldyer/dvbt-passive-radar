"""Profile an unknown image with one constant calibration phase per RF band.

The reference band fixes the global complex-image gauge. No per-epoch phase
is added. Since each band's nuisance projection is complex linear, a
constant unit phase leaves its normal matrix unchanged.
"""
import numpy as np


class ConstantBandCalibration:
    def __init__(self,model,reference_channel=3):
        self.model=model;self.spectrum=model.spectrum.copy()
        self.channels=[c for c in range(len(self.spectrum)) if c!=reference_channel]
        self.path_parameters=model.parameters

    def objective(self,parameters,full=False):
        n=self.path_parameters
        if np.shape(parameters)!=(n+len(self.channels),):raise ValueError('Calibration/path parameter dimensions differ')
        correction=np.zeros(len(self.spectrum));correction[self.channels]=parameters[n:]
        self.model.spectrum=self.spectrum*np.exp(1j*correction)
        value,gradient,report=self.model.objective(parameters[:n],True)
        coefficient=report['coefficient'];extra=[]
        for c in self.channels:
            phase,_,_=self.model.phase_coefficients(parameters[:n],c)
            rhs=self.model.spectrum[c].conjugate()*(phase.conj()@self.model.score_moments[c])
            extra.append(-2*np.imag(np.vdot(coefficient,rhs))/self.model.energy)
        gradient=np.r_[gradient,extra]
        report.update(band_phase_corrections_radians=correction.tolist())
        return (value,gradient,report) if full else (value,gradient)

"""Correlated location prior for five transmitter phase centres and one receiver.

Transmitters on the same licensed tower share its survey uncertainty, while
each frequency may have a distinct antenna offset. Whitening this covariance
avoids redundant common-site/individual-centre optimization parameters.
"""
import numpy as np
from scipy.linalg import block_diag


def spatial_phase_mapping(position, body_to_ecef, transmitters, receiver, wavenumbers, antennas, reference):
    """Linear antenna-error map to horizontal phase slope relative to one band.

    The derivative is at the ship reference point and one epoch. Constant band
    phases have been eliminated. This is an identifiability/plausibility
    diagnostic; a physical imaging correction still needs the full path model.
    """
    mappings = []
    for c, (tx, k) in enumerate(zip(transmitters, wavenumbers)):
        hessians = []
        for phase_centre in (tx, receiver):
            vector = position-phase_centre
            distance = np.linalg.norm(vector)
            unit = vector/distance
            hessians.append((np.eye(3)-np.outer(unit, unit))/distance)
        mappings.append(k*body_to_ecef[:, :2].T@np.concatenate(hessians, axis=1)@antennas.mapping(c))
    mappings = np.array(mappings)
    return mappings-mappings[reference]


def exact_spatial_phase(position, body_to_ecef, transmitters, receiver, wavenumbers, antennas, parameters, reference):
    """Exact phase-gradient change at the reference point for finite offsets."""
    slopes = []
    for c, (tx, k) in enumerate(zip(transmitters, wavenumbers)):
        delta = antennas.offsets(parameters, c)
        change = np.zeros(3)
        for antenna, offset in zip((tx, receiver), delta):
            original, shifted = position-antenna, position-antenna-offset
            change += shifted/np.linalg.norm(shifted)-original/np.linalg.norm(original)
        slopes.append(-k*body_to_ecef[:, :2].T@change)
    slopes = np.array(slopes)
    return slopes-slopes[reference]


class Antennas:
    def __init__(self, ecef_to_enu, channel_sites, *, site_sigma=(30., 30., 75.),
                 antenna_sigma=(5., 5., 10.), receiver_sigma=(15., 15., 7.5)):
        self.rotation = np.asarray(ecef_to_enu)
        self.channel_sites = np.asarray(channel_sites)
        shared = self.channel_sites[:, None] == self.channel_sites[None, :]
        tx_covariance = np.kron(shared, np.diag(np.asarray(site_sigma)**2))
        tx_covariance += np.kron(np.eye(len(channel_sites)), np.diag(np.asarray(antenna_sigma)**2))
        self.factor = block_diag(np.linalg.cholesky(tx_covariance), np.diag(receiver_sigma))
        self.count = self.factor.shape[1]
        self.site_sigma, self.antenna_sigma, self.receiver_sigma = site_sigma, antenna_sigma, receiver_sigma

    def mapping(self, channel):
        rows = np.r_[np.arange(channel*3, channel*3+3), np.arange(self.count-3, self.count)]
        return block_diag(self.rotation.T, self.rotation.T)@self.factor[rows]

    def offsets(self, parameters, channel):
        return (self.mapping(channel)@parameters).reshape(2, 3)

    def describe(self, parameters):
        physical = self.factor@parameters
        return {'transmitter_offsets_enu_m': physical[:-3].reshape(-1, 3).tolist(),
            'receiver_offset_enu_m': physical[-3:].tolist(), 'standardised_parameters': np.asarray(parameters).tolist(),
            'channel_sites': self.channel_sites.tolist(), 'ecef_to_enu': self.rotation.tolist(),
            'site_prior_sigma_enu_m': list(self.site_sigma), 'antenna_prior_sigma_enu_m': list(self.antenna_sigma),
            'receiver_prior_sigma_enu_m': list(self.receiver_sigma),
            'prior': 'Correlated Gaussian site + independent per-channel phase-centre offsets; independent receiver location. Declared location uncertainties, not calibration measurements.'}

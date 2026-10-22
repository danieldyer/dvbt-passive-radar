"""One metadata source for physical site priors and the allowed assignments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METADATA = ROOT/'recordings/live_mt_dandenong.json'
SITE_ORDER = ('ornata', 'eyre', 'ridge')
CHANNEL_ORDER = ('seven', 'sbs', 'nine', 'ten', 'abc')


def configuration(geometry=None):
    """New geometries freeze their configuration; old searches use current policy.

    This does not move antennas in historical geometries. It only limits which
    assignments may be searched when an old directory has no policy snapshot.
    """
    data = (geometry or {}).get('site_configuration')
    if data is None:
        data = json.loads(METADATA.read_text())
    if tuple(s['id'] for s in data['transmitters']) != SITE_ORDER:
        raise ValueError('Transmitter order must be Ornata, Eyre, Ridge')
    if tuple(s['id'] for s in data['signals']) != CHANNEL_ORDER:
        raise ValueError('Signal order must be Seven, SBS, Nine, Ten, ABC')
    for signal in data['signals']:
        candidates = signal['candidate_transmitter_ids']
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError('Empty or repeated transmitter candidate')
        if not set(candidates) <= set(SITE_ORDER):
            raise ValueError('Unknown transmitter candidate')
        if signal['transmitter_id'] not in candidates:
            raise ValueError('Nominal transmitter must belong to the candidate set')
    return data


def candidate_indices(geometry=None):
    return tuple(tuple(SITE_ORDER.index(site) for site in s['candidate_transmitter_ids'])
                 for s in configuration(geometry)['signals'])


def candidate_acma_ids(geometry=None):
    data = configuration(geometry)
    ids = {s['id']: s['acma_site_id'] for s in data['transmitters']}
    return tuple(tuple(ids[site] for site in s['candidate_transmitter_ids']) for s in data['signals'])


def nominal_indices(geometry=None):
    return tuple(SITE_ORDER.index(s['transmitter_id']) for s in configuration(geometry)['signals'])


def validate_assignment(assignment, geometry=None):
    candidates = candidate_indices(geometry)
    if len(assignment) != len(candidates) or any(site not in allowed for site, allowed in zip(assignment, candidates)):
        raise ValueError('Assignment lies outside the configured primary/backup candidate set')

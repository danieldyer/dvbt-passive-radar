"""Canonical AIS motion priors with explicit UTC, units, gaps and heading.

CSV columns: utc, latitude_deg, longitude_deg. Optional: sog_knots, cog_deg,
true_heading_deg, rot_deg_per_min, mmsi, position_sigma_m, heading_sigma_deg.
The CSV is a normalised import format, not an autodetected provider schema.
In particular MarineTraffic API SPEED is knots*10 and must be converted first.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline

from motion import WGS84_A, WGS84_E2, ecef, enu_basis


def utc_seconds(text):
    stamp = datetime.fromisoformat(text.strip().replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('AIS timestamps need an explicit UTC offset or Z suffix')
    return stamp.timestamp()


def number(row, key):
    value = (row.get(key) or '').strip()
    if not value:
        return np.nan
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f'Non-finite {key}; use an empty field for unavailable data')
    return result


@dataclass
class AisTrack:
    seconds: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    speed: np.ndarray
    course: np.ndarray
    heading: np.ndarray
    turn_rate: np.ndarray
    position_sigma: np.ndarray
    heading_sigma: np.ndarray
    mmsi: int | None
    reference_height_ellipsoid_m: float

    def __post_init__(self):
        if len(self.seconds) < 2 or np.any(np.diff(self.seconds) <= 0):
            raise ValueError('Need at least two strictly increasing AIS timestamps')
        if not np.isfinite(self.reference_height_ellipsoid_m):
            raise ValueError('An explicit finite ellipsoidal reference height is required')
        self.epoch = self.seconds[0]
        times = self.seconds-self.epoch
        coordinates = np.column_stack([self.latitude, np.unwrap(self.longitude)])
        slopes = np.gradient(coordinates, times, axis=0)
        measured = np.isfinite(self.speed)&np.isfinite(self.course)
        n = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(self.latitude)**2)
        m = WGS84_A*(1-WGS84_E2)/(1-WGS84_E2*np.sin(self.latitude)**2)**1.5
        h = self.reference_height_ellipsoid_m
        slopes[measured, 0] = (self.speed*np.cos(self.course)/(m+h))[measured]
        slopes[measured, 1] = (self.speed*np.sin(self.course)/((n+h)*np.cos(self.latitude)))[measured]
        self.velocity_is_reported = measured
        self.curve = CubicHermiteSpline(times, coordinates, slopes, extrapolate=False)

    def sample(self, seconds, *, max_gap_s):
        seconds = np.atleast_1d(seconds).astype(float)
        if not np.isfinite(max_gap_s) or max_gap_s <= 0 or not np.all(np.isfinite(seconds)):
            raise ValueError('Invalid interpolation times or maximum gap')
        if np.any(seconds < self.seconds[0]) or np.any(seconds > self.seconds[-1]):
            raise ValueError('AIS extrapolation is not permitted')
        interval = np.clip(np.searchsorted(self.seconds, seconds, side='right')-1, 0, len(self.seconds)-2)
        dt = self.seconds[interval+1]-self.seconds[interval]
        fraction = (seconds-self.seconds[interval])/dt
        interpolated = (fraction > 0)&(fraction < 1)
        if np.any(interpolated&(dt > max_gap_s)):
            raise ValueError('Requested aperture crosses an AIS gap beyond the explicit limit')
        latitude, longitude = self.curve(seconds-self.epoch).T
        rates = self.curve(seconds-self.epoch, 1)
        h = self.reference_height_ellipsoid_m
        n = WGS84_A/np.sqrt(1-WGS84_E2*np.sin(latitude)**2)
        m = WGS84_A*(1-WGS84_E2)/(1-WGS84_E2*np.sin(latitude)**2)**1.5
        local_velocity = np.column_stack([rates[:, 1]*(n+h)*np.cos(latitude), rates[:, 0]*(m+h), np.zeros(len(seconds))])
        start, stop = self.heading[interval], self.heading[interval+1]
        delta = np.angle(np.exp(1j*(stop-start)))
        heading = np.remainder(start+fraction*delta, 2*np.pi)
        heading = np.where(fraction == 0, start, np.where(fraction == 1, stop, heading))
        return {'seconds': seconds, 'latitude': latitude, 'longitude': longitude,
                'position_ecef_m': ecef(latitude, longitude, h),
                'velocity_ecef_mps': np.einsum('nij,nj->ni', enu_basis(latitude, longitude), local_velocity),
                'true_heading_radians': heading,
                'bracketing_interval_s': dt,
                'policy': 'Interpolated motion prior, not a carrier-phase-accurate trajectory; altitude, attitude and time offset require explicit treatment.'}


def read_csv(path, *, reference_height_ellipsoid_m, mmsi=None):
    with Path(path).open(newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        if not {'utc', 'latitude_deg', 'longitude_deg'}.issubset(reader.fieldnames or []):
            raise ValueError('Expected canonical AIS columns utc, latitude_deg, longitude_deg')
        rows = list(reader)
    ids = {int(row['mmsi']) for row in rows if row.get('mmsi', '').strip()}
    if len(ids) > 1 or (mmsi is not None and ids and ids != {mmsi}):
        raise ValueError('AIS file contains another vessel or mixed MMSIs')
    times = np.array([utc_seconds(row['utc']) for row in rows])
    values = {key: np.array([number(row, key) for row in rows]) for key in
              ('latitude_deg', 'longitude_deg', 'sog_knots', 'cog_deg', 'true_heading_deg',
               'rot_deg_per_min', 'position_sigma_m', 'heading_sigma_deg')}
    lat, lon = values['latitude_deg'], values['longitude_deg']
    if not np.all(np.isfinite(lat)&np.isfinite(lon)) or np.any(abs(lat) >= 90) or np.any(abs(lon) > 180):
        raise ValueError('Invalid or unavailable AIS coordinates')
    speed = values['sog_knots']
    if np.any(np.isfinite(speed)&((speed < 0)|(speed >= 102.3))):
        raise ValueError('Invalid/unavailable AIS speed; use an empty field for unavailable data')
    for key in ('cog_deg', 'true_heading_deg'):
        a = values[key]
        a[a == (360 if key == 'cog_deg' else 511)] = np.nan
        if np.any(np.isfinite(a)&((a < 0)|(a >= 360))):
            raise ValueError('Invalid angle or incorrect provider units')
    for key in ('position_sigma_m', 'heading_sigma_deg'):
        a = values[key]
        if np.any(np.isfinite(a)&(a <= 0)):
            raise ValueError('Supplied uncertainty must be positive')
    return AisTrack(times, np.radians(lat), np.radians(lon), speed*(1852/3600),
                    np.radians(values['cog_deg']), np.radians(values['true_heading_deg']),
                    np.radians(values['rot_deg_per_min'])/60, values['position_sigma_m'],
                    np.radians(values['heading_sigma_deg']), next(iter(ids), mmsi), reference_height_ellipsoid_m)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv', type=Path)
    p.add_argument('--mmsi', type=int)
    p.add_argument('--reference-height-ellipsoid-m', type=float, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    track = read_csv(args.csv, reference_height_ellipsoid_m=args.reference_height_ellipsoid_m, mmsi=args.mmsi)
    result = {'source': str(args.csv), 'mmsi': track.mmsi, 'positions': len(track.seconds),
              'first_utc': datetime.fromtimestamp(track.seconds[0], timezone.utc).isoformat(),
              'last_utc': datetime.fromtimestamp(track.seconds[-1], timezone.utc).isoformat(),
              'maximum_gap_s': float(np.diff(track.seconds).max()),
              'true_heading_reports': int(np.isfinite(track.heading).sum()),
              'reported_velocity_count': int(track.velocity_is_reported.sum()),
              'reference_height_ellipsoid_m': track.reference_height_ellipsoid_m,
              'status': 'Motion prior only; radar association and phase-accurate compensation are unvalidated.'}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

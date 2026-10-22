"""North-up plot of supplied AIS fixes, with interpolation explicitly labelled."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ais import read_csv, utc_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ais', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    session = json.loads(args.session.read_text())
    start = utc_seconds(session['time_alignment']['file_creation_utc'])
    stop = start+session['duration_s']
    track = read_csv(args.ais, reference_height_ellipsoid_m=5.)
    local = ZoneInfo('Australia/Melbourne')
    label = lambda t: datetime.fromtimestamp(float(t), local).strftime('%H:%M:%S')
    latitude, longitude = np.degrees(track.latitude), np.degrees(track.longitude)
    # Straight lines connect actual reports. Do not silently bridge the six-minute
    # gap with the phase-trajectory interpolator used by the imaging pipeline.
    lat = lambda t: np.interp(t, track.seconds, latitude)
    lon = lambda t: np.interp(t, track.seconds, longitude)
    fig, (ax, course_ax) = plt.subplots(1, 2, figsize=(12, 8),
        gridspec_kw={'width_ratios': [1.3, 1]}, layout='constrained')
    for a, b in zip(track.seconds[:-1], track.seconds[1:]):
        if b < start or a > stop:
            continue
        lo, hi = max(a, start), min(b, stop)
        gap = b-a > 180
        ax.plot(lon([lo, hi]), lat([lo, hi]), '--' if gap else '-',
                color='#8a8f98' if gap else '#1464a5', lw=2.5)
        if gap:
            ax.plot(lon([hi, b]), lat([hi, b]), ':', color='#b3b6bc', lw=1.5)
            ax.scatter(lon(b), lat(b), marker='o', s=35, color='#8a8f98')
            ax.annotate(label(b)+' (next report)', (lon(b), lat(b)),
                        xytext=(10, -2), textcoords='offset points', fontsize=9)
    observed = (track.seconds >= start)&(track.seconds <= stop)
    ax.scatter(longitude[observed], latitude[observed], color='#1464a5', s=35, zorder=3)
    indices = np.flatnonzero(observed)
    for i in indices:
        if i and track.seconds[i]-track.seconds[i-1] < 30:
            continue
        ax.annotate(label(track.seconds[i]), (longitude[i], latitude[i]),
                    xytext=(12, -2), textcoords='offset points', fontsize=9)
    for stamp, name, marker in ((start, 'Capture start ≈ '+label(start), 's'),
                                (stop, 'Capture end ≈ '+label(stop), 'X')):
        ax.scatter(lon(stamp), lat(stamp), marker=marker, color='#242b35', s=65, zorder=5)
        ax.annotate(name, (lon(stamp), lat(stamp)), xytext=(-12, -22),
                    textcoords='offset points', ha='right', fontsize=9)
    imaging = start+np.array([142., 172.])
    ax.plot(lon(imaging), lat(imaging), color='#df7924', lw=7, solid_capstyle='round',
            label='ISAR preview support (142–172 s)', zorder=4)
    ax.annotate('North', xy=(.88, .92), xytext=(.88, .82), xycoords='axes fraction',
                ha='center', arrowprops={'arrowstyle': '-|>', 'color': '#242b35'}, fontsize=11)
    ax.set_aspect(1/np.cos(np.radians(latitude[observed].mean())))
    ax.set(xlabel='Longitude (° E)', ylabel='Latitude (°)',
           title='North-up geographic ground track\nSolid: fixes joined; dashed: >180 s AIS gap')
    ax.ticklabel_format(useOffset=False)
    ax.margins(x=.5, y=.15)
    ax.legend(loc='lower left', fontsize=8)
    cog = np.degrees(np.unwrap(track.course))
    time = (track.seconds-start)/60
    course_ax.plot(time, cog, ':', color='#8a8f98', label='Connecting reports')
    course_ax.scatter(time[observed], cog[observed], color='#1464a5', label='Reported course over ground')
    course_ax.axvspan(142/60, 172/60, color='#df7924', alpha=.2)
    course_ax.axvspan((track.seconds[6]-start)/60, (stop-start)/60, color='#dfe2e8', alpha=.6)
    course_ax.set(xlim=(0, session['duration_s']/60), ylim=(-50, 8),
                  xlabel='Time from nominal capture start (minutes)',
                  ylabel='Course (degrees clockwise from north)',
                  title='Initially north, turning northwest\nCourse is not a true-heading measurement')
    course_ax.set_yticks([0, -15, -30, -45], ['000° / N', '345°', '330°', '315° / NW'])
    course_ax.legend(loc='lower left', fontsize=9)
    course_ax.text(.04, .12, 'No AIS position report from\n19:39:11 to 19:45:43.\nThe capture ends inside this gap.',
                   transform=course_ax.transAxes, fontsize=10)
    fig.suptitle('MAERSK RIO DELTA · 13 October 2026 · Melbourne time (AEDT)', fontsize=16)
    fig.supxlabel('Capture UTC comes from filesystem chronology (uncertain by several seconds).\n'
                  'ISAR videos use ship-relative axes: forward → right, port → up. They are not north-up.', fontsize=10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix('.pdf'))


if __name__ == '__main__':
    main()

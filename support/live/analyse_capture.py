#!/usr/bin/env python3
"""Measure timing, oracle validity and the actual matched-filter apertures.

The analysis reads archive metadata and saved carrier weights. It never changes
the scientific profiles, fills gaps, or estimates a motion trajectory.
"""

import argparse
import collections
import json
import math
import struct
from pathlib import Path

import numpy as np
from read_archive import exact, read_header


def fwhm(weights, spacing, size=65536):
    power = np.abs(np.fft.rfft(weights, n=size)) ** 2
    power /= power[0]
    below = np.flatnonzero(power < 0.5)
    if not len(below):
        return float("nan")
    j = int(below[0])
    crossing = j - 1 + (0.5 - power[j - 1]) / (power[j] - power[j - 1])
    return 2 * crossing / (size * spacing)


def windows(rows, interval, integration, update, erasures=False):
    """Replay the archive's policy, retaining unknown times only for new files."""
    required = math.ceil(integration / interval)
    count = 0
    last = -math.inf
    for j, row in enumerate(rows):
        if j and (
            row[6] != rows[j - 1, 6] or abs(row[1] - rows[j - 1, 1] - interval) > interval * 0.01
        ):
            count = 0
        if not row[5] and not erasures:
            count = 0
            continue
        count += 1
        if count < required or row[1] - last + interval * 1e-8 < update:
            continue
        selected = rows[j - required + 1 : j + 1]
        weights = np.where(selected[:, 5], selected[:, 7], 0)
        if not np.any(weights):
            continue
        effective = weights.sum() ** 2 / np.dot(weights, weights) * interval
        yield j, effective, fwhm(weights, interval)
        last = row[1]


def analyse(path, integration=3, update=1):
    channels = collections.defaultdict(list)
    batches = collections.defaultdict(list)
    range_weights = collections.defaultdict(list)
    tags = collections.Counter()
    clock = []
    with path.open("rb") as source:
        header = read_header(source)
        erasures = header.get("profile_row_policy") == "fixed_grid_zero_information_erasures"
        if header["archive_mode"] not in ("profiles", "both"):
            raise ValueError("This analysis needs a profile archive")
        while True:
            tag = exact(source, 4).decode("ascii")
            length, size = struct.unpack("<IQ", exact(source, 12))
            if length > 1024 * 1024 or size > 64 * 1024 * 1024:
                raise ValueError("Oversized archive record")
            meta = json.loads(exact(source, length))
            tags[tag] += 1
            if tag == "PROF":
                c = meta["channel"]
                if (
                    len(meta["rows"]) != meta["count"]
                    or meta["size"] not in (1024, 3072, 4096)
                    or meta["moments"] != 2
                ):
                    raise ValueError("Invalid profile dimensions")
                channels[c].extend(
                    (*r[:6], meta["segment"], meta["information"], meta["reference_origin"])
                    for r in meta["rows"]
                )
                skip = (2 * meta["count"] * meta["size"] + meta["stationary_carriers"]) * 8
                tail = (
                    meta["response_carriers"] * 8
                    + meta["noise_carriers"] * 4
                    + meta.get("cancellation_variance_carriers", 0) * 8
                )
                if (
                    skip + tail != size
                    or meta["response_carriers"] != 6817
                    or meta["noise_carriers"] != 6817
                ):
                    raise ValueError("Missing archived template/noise weights")
                source.seek(skip, 1)
                response = np.frombuffer(exact(source, 6817 * 8), dtype="<c8")
                variance = np.frombuffer(exact(source, 6817 * 4), dtype="<f4")
                source.seek(meta.get("cancellation_variance_carriers", 0) * 8, 1)
                q = abs(response.astype(np.complex128)) ** 2 / variance
                if not np.isfinite(q).all() or np.any(q <= 0):
                    raise ValueError("Invalid archived carrier information")
                # An ideal point target's expected range response is the
                # Fourier transform of the physical matched-template weights.
                width_m = fwhm(q, 8e6 / 8192) * 299792458
                range_weights[c].append(q)
                batches[c].append(
                    [
                        meta["rows"][0][1],
                        meta["rows"][-1][1],
                        meta["information"],
                        float(np.median(variance)),
                        width_m,
                    ]
                )
            else:
                if source.tell() + size > path.stat().st_size:
                    raise ValueError("Truncated archive")
                source.seek(size, 1)
                if tag == "CLCK":
                    clock.extend(meta["segments"])
            if tag == "DONE":
                if source.read(1):
                    raise ValueError("Trailing archive data")
                break
    arrays = {}
    results = []
    for c, rows in sorted(channels.items()):
        a = np.asarray(rows, dtype=np.float64)
        signal = header["signals"][c]
        interval = (8192 + 8192 / signal["guard_denominator"]) / 8e6
        b = np.asarray(batches[c])
        q = np.asarray(range_weights[c])
        admitted = []
        required = math.ceil(integration / interval)
        for j, effective, doppler_width in windows(a, interval, integration, update, erasures):
            # The profiles divide by information; the focus multiplies by it.
            # Their product recovers the sum of the actual carrier weights.
            selected = a[j - required + 1 : j + 1]
            indices = np.searchsorted(b[:, 0], selected[selected[:, 5] > 0, 1], side="right") - 1
            ids, counts = np.unique(indices, return_counts=True)
            carrier_weights = np.sum(q[ids] * counts[:, None], axis=0)
            admitted.append(
                [j, effective, doppler_width, fwhm(carrier_weights, 8e6 / 8192) * 299792458]
            )
        w = np.asarray(admitted)
        arrays[f"{signal['id']}_rows"] = a
        arrays[f"{signal['id']}_batches"] = b
        arrays[f"{signal['id']}_windows"] = w
        arrays[f"{signal['id']}_range_weights"] = q
        valid = a[:, 5] > 0
        required = math.ceil(integration / interval)
        quantile = lambda values: dict(
            zip(
                ("min", "p10", "median", "p90", "max"),
                map(float, np.percentile(values, [0, 10, 50, 90, 100])),
            )
        )
        result = {
            "id": signal["id"],
            "name": signal["name"],
            "channel": c,
            "rows": len(a),
            "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()),
            "segments": len(np.unique(a[:, 6])),
            "first_profile_s": float(a[0, 1]),
            "last_profile_s": float(a[-1, 1]),
            "interval_s": interval,
            "window_rows": required,
            "integration_s": required * interval,
            "nominal_window_span_s": (required - 1) * interval,
            "timestamp_step_error_ns_max": float(np.max(abs(np.diff(a[:, 1]) - interval)) * 1e9),
            "information": quantile(a[:, 7]),
            "batch_range_fwhm_m": quantile(b[:, 4]),
            "range_fwhm_m": quantile(w[:, 3]),
            "admitted_maps": len(w),
            "effective_integration_s": quantile(w[:, 1]),
            "doppler_fwhm_hz": quantile(w[:, 2]),
        }
        results.append(result)
    return {
        "archive": str(path.resolve()),
        "header": header,
        "records": dict(tags),
        "row_policy": "fixed_grid_zero_information_erasures"
        if erasures
        else "legacy_complete_valid_rows",
        "clock_segments": len(clock),
        "integration_requested_s": integration,
        "analysis_update_s": update,
        "ideal_rectangular_doppler_fwhm_hz": 0.885892941 / integration,
        "channels": results,
    }, arrays


def plots(report, arrays, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signals = report["channels"]
    fig, axes = plt.subplots(len(signals), 1, figsize=(13, 11), sharex=True, layout="constrained")
    for s, ax in zip(signals, axes):
        a = arrays[s["id"] + "_rows"]
        b = arrays[s["id"] + "_batches"]
        level = 10 * np.log10(b[:, 2] / np.median(a[:, 7]))
        ax.step(b[:, 0], level, where="post", label="Saved inverse-noise information")
        bad = a[a[:, 5] == 0]
        ax.scatter(
            bad[:, 1],
            np.full(len(bad), -18),
            color="red",
            marker="|",
            s=30,
            label="Invalid oracle row",
        )
        ax.set_ylabel(s["name"] + "\ndB")
        ax.grid(alpha=0.2)
        ax.set_ylim(-20, 30)
    axes[0].legend(loc="upper left")
    axes[-1].set_xlabel("Time from first captured sample (s)")
    fig.suptitle("Continuous ADC capture; changing matched-filter weights and oracle failures")
    fig.savefig(output / "weights_and_validity.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, layout="constrained")
    for s in signals:
        a = arrays[s["id"] + "_rows"]
        w = arrays[s["id"] + "_windows"]
        b = arrays[s["id"] + "_batches"]
        times = a[w[:, 0].astype(int), 1]
        axes[0].plot(times, w[:, 1], ".-", ms=2, label=s["name"])
        axes[1].plot(times, w[:, 2], ".-", ms=2)
        axes[2].plot(times, w[:, 3], ".-", ms=2)
    axes[0].axhline(
        report["integration_requested_s"], color="black", ls="--", label="Requested CPI"
    )
    axes[1].axhline(report["ideal_rectangular_doppler_fwhm_hz"], color="black", ls="--")
    axes[0].set_ylabel("Equivalent weighted\naperture (s)")
    axes[0].legend(ncol=6)
    axes[1].set_ylabel("Doppler power\nFWHM (Hz)")
    axes[2].set_ylabel("Range power\nFWHM (m)")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time from first captured sample (s)")
    fig.suptitle("Fixed 2,758-row CPI; the weighted point-spread function is not fixed")
    fig.savefig(output / "aperture_and_resolution.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    s = next(s for s in signals if s["id"] == "nine")
    a = arrays["nine_rows"]
    w = arrays["nine_windows"]
    required = s["window_rows"]
    dt = s["interval_s"]
    for k, name in (
        (int(np.argmin(w[:, 1])), "Most concentrated weights"),
        (int(np.argmax(w[:, 1])), "Most uniform weights"),
    ):
        j = int(w[k, 0])
        selected = a[j - required + 1 : j + 1]
        weights = np.where(selected[:, 5], selected[:, 7], 0)
        times = np.arange(required) * dt
        axes[0].plot(times, weights / weights.sum() / dt, label=f"{name}, ending {a[j, 1]:.2f} s")
        power = abs(np.fft.fftshift(np.fft.fft(weights, n=131072))) ** 2 / weights.sum() ** 2
        f = np.fft.fftshift(np.fft.fftfreq(131072, dt))
        pick = abs(f) <= 2
        axes[1].plot(
            f[pick], 10 * np.log10(np.maximum(power[pick], 1e-6)), label=f"FWHM {w[k, 2]:.3f} Hz"
        )
    axes[0].set_xlabel("Position inside the same three-second CPI (s)")
    axes[0].set_ylabel("Normalized temporal weight (1/s)")
    axes[1].set_xlabel("Doppler mismatch (Hz)")
    axes[1].set_ylabel("Point-target power response (dB)")
    axes[1].set_ylim(-40, 1)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Nine: exact scalar-weight Doppler responses, before motion and front-end phase errors"
    )
    fig.savefig(output / "weighted_point_spread.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--integration", type=float, default=3)
    p.add_argument("--update", type=float, default=1)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report, arrays = analyse(args.archive, args.integration, args.update)
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output / "timing_and_weights.npz", **arrays)
    plots(report, arrays, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "header"}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarise a capture_soak.py log and its associated capture diagnostics."""

import argparse
import json
import math
import statistics
from pathlib import Path


def records(path):
    with path.open() as source:
        for line in source:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # A currently running writer can leave an incomplete last line.


def stats(values):
    x = sorted(v for v in values if isinstance(v, (int, float)))
    if not x:
        return None
    return {"min": x[0], "median": statistics.median(x), "max": x[-1], "mean": statistics.mean(x)}


def tail_trend(pairs, seconds=300):
    """Report a descriptive slope of minute medians; no IID confidence claim."""
    if seconds <= 0:
        raise ValueError("Trend window must be positive")
    valid = [(t, value) for t, value in pairs if isinstance(value, (int, float))]
    if not valid:
        return None
    end = valid[-1][0]
    bins = {}
    for t, value in valid:
        if end - seconds <= t <= end:
            bins.setdefault(
                min(math.ceil(seconds / 60) - 1, int((t - end + seconds) / 60)), []
            ).append((t, value))
    points = [
        (statistics.mean(t for t, _ in group), statistics.median(v for _, v in group))
        for group in bins.values()
        if len(group) >= 20
    ]
    if len(points) < 3:
        return None
    centre_t = statistics.mean(t for t, _ in points)
    centre_v = statistics.mean(v for _, v in points)
    slope = sum((t - centre_t) * (v - centre_v) for t, v in points) / sum(
        (t - centre_t) ** 2 for t, _ in points
    )
    return {
        "slope_per_minute": 60 * slope,
        "minute_medians": points,
        "first_to_last_minute_delta": points[-1][1] - points[0][1],
        "window_seconds": seconds,
        "interpretation": "descriptive trend, correlated observations; not a confidence bound",
    }


def analyze(telemetry, capture, status_file=None):
    data = list(records(telemetry / "telemetry.jsonl"))
    status = [
        r
        for r in records(status_file or capture / "diagnostics.jsonl")
        if r["type"] == "status" and r["phase"] == "RUNNING"
    ]
    summary_path = capture / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    result = {
        "telemetry_seconds": data[-1]["elapsed_s"],
        "capture_complete": summary is not None,
        "capture_summary": summary,
        "temperature_c": {},
        "rapl_watts": {},
        "throttle_count_delta": {},
        "cpu_frequency_khz": {},
        "gpu_frequency_mhz": {},
        "capture_metrics": {},
        "channels": {},
        "final_five_minutes": {},
    }
    for field, dest in [
        ("temperature_c", "temperature_c"),
        ("cpu_frequency_khz", "cpu_frequency_khz"),
        ("gpu_frequency_mhz", "gpu_frequency_mhz"),
    ]:
        for key in data[0][field]:
            result[dest][key] = stats(r[field].get(key) for r in data)
    for key in data[0]["temperature_c"]:
        if "coretemp" in key:
            result["final_five_minutes"][key] = tail_trend(
                [(r["elapsed_s"], r["temperature_c"].get(key)) for r in data]
            )
    for key in ["headroom", "queue_ms"]:
        result["final_five_minutes"][key] = tail_trend(
            [(r["elapsed_s"], r.get(key)) for r in status]
        )
    for key, domain in data[0]["rapl"].items():
        result["rapl_watts"][key] = dict(
            name=domain["name"], **(stats(r["rapl"].get(key, {}).get("watts") for r in data) or {})
        )
    for key, value in data[0]["throttle_counts"].items():
        last = data[-1]["throttle_counts"].get(key)
        result["throttle_count_delta"][key] = (
            last - value if last is not None and value is not None else None
        )
    for key in [
        "headroom",
        "recent_headroom",
        "queue_ms",
        "archive_queued_bytes",
        "missing",
        "discarded",
        "unknown_gaps",
        "rf_rms_dbfs",
    ]:
        result["capture_metrics"][key] = stats(r.get(key) for r in status)
    if summary or status:
        for channel in (summary if summary else status[-1])["channels"]:
            name = channel["id"]
            observations = [c for r in status for c in r["channels"] if c["id"] == name]
            result["channels"][name] = {
                "pilot_snr_db": stats(c["pilot_snr_db"] for c in observations),
                "rs_good": channel["rs_good"],
                "rs_bad": channel["rs_bad"],
                "accepted_fraction": channel["rs_good"]
                / max(1, channel["rs_good"] + channel["rs_bad"]),
            }
    return result, data, status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--status-file", type=Path, help="Optional status-only extract of diagnostics"
    )
    parser.add_argument("--title", default="PL64 live three-channel soak — profiles to 100 km")
    parser.add_argument(
        "--bin-seconds",
        type=float,
        default=60,
        help="Plot aggregation interval; does not change statistics",
    )
    args = parser.parse_args()
    if not math.isfinite(args.bin_seconds) or args.bin_seconds <= 0:
        parser.error("--bin-seconds must be finite and positive")
    result, data, status = analyze(args.telemetry, args.capture, args.status_file)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    t = [r["elapsed_s"] / 60 for r in data]
    import numpy as np

    def band(ax, times, values, label):
        period = args.bin_seconds / 60
        groups = {}
        for t, value in zip(times, values):
            if isinstance(value, (int, float)):
                groups.setdefault(int(t / period), []).append((t, value))
        points = [
            (np.mean([x[0] for x in g]), *np.percentile([x[1] for x in g], [10, 50, 90]))
            for g in groups.values()
        ]
        if not points:
            return
        x, lo, median, hi = np.asarray(points).T
        (line,) = ax.plot(x, median, label=label)
        ax.fill_between(x, lo, hi, color=line.get_color(), alpha=0.15)

    for key in data[0]["temperature_c"]:
        if "Package id 0" in key or key.endswith("/Core 8"):
            band(axes[0, 0], t, [r["temperature_c"].get(key) for r in data], key.split("/")[-1])
    axes[0, 0].set_ylabel("Temperature (°C)")
    axes[0, 0].legend(fontsize=8)
    for key in ["intel-rapl:0", "intel-rapl:0:0", "intel-rapl:0:1"]:
        if key in data[0]["rapl"]:
            band(
                axes[0, 1], t, [r["rapl"][key]["watts"] for r in data], data[0]["rapl"][key]["name"]
            )
    axes[0, 1].set_ylabel("RAPL domain power (W)")
    axes[0, 1].legend()
    for key in ["cpu0", "cpu4"]:
        if key in data[0]["cpu_frequency_khz"]:
            band(axes[1, 0], t, [(r["cpu_frequency_khz"][key] or 0) / 1e6 for r in data], key)
    axes[1, 0].set_ylabel("Sampled CPU frequency (GHz)")
    axes[1, 0].legend()
    for key in data[0]["gpu_frequency_mhz"]:
        if "cur" in key:
            band(axes[1, 1], t, [r["gpu_frequency_mhz"][key] for r in data], key)
    axes[1, 1].set_ylabel("Requested GPU frequency (MHz)")
    axes[1, 1].legend(fontsize=8)
    for ax in axes[:2].flat:
        ax.set_xlabel("Time including warmup (min)")
    s = [r["elapsed_s"] / 60 for r in status]
    band(axes[2, 0], s, [r["headroom"] for r in status], "Cumulative measured DSP headroom")
    axes[2, 0].axhline(1, color="black", linestyle="--")
    axes[2, 0].set_ylim(bottom=0.9)
    axes[2, 0].set_ylabel("Processing headroom (×)")
    axes[2, 0].legend(fontsize=8)
    band(axes[2, 1], s, [r["queue_ms"] for r in status], "Receive queue")
    axes[2, 1].set_ylabel("Receive queue (ms)")
    for name in result["channels"]:
        values = [
            (r["elapsed_s"] / 60, c) for r in status for c in r["channels"] if c["id"] == name
        ]
        band(axes[3, 0], [x[0] for x in values], [x[1]["pilot_snr_db"] for x in values], name)
    axes[3, 0].set_ylabel("Pilot SNR (dB)")
    axes[3, 0].legend()
    for key in ["missing", "discarded", "unknown_gaps"]:
        axes[3, 1].plot(s, [r[key] for r in status], label=key)
    axes[3, 1].set_ylabel("Cumulative samples / gap events")
    axes[3, 1].legend()
    for ax in axes[2:].flat:
        ax.set_xlabel("Capture time (min)")
    figure.suptitle(
        f"{args.title}\n{args.bin_seconds:g}-second medians and 10–90% bands; RAPL domains overlap and are not wall-socket power"
    )
    figure.savefig(args.output / "soak.png", dpi=150)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Read-only Linux telemetry around a capture command; capture owns its duration.

Usage: capture_soak.py --output NEW_DIRECTORY -- COMMAND [ARGS...]
No governor/power settings are changed. RAPL measures domains, not wall power.
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def read(path):
    try:
        return path.read_text().strip()
    except (OSError, ValueError):
        return None


def number(path):
    try:
        return int(read(path))
    except (TypeError, ValueError):
        return None


def sensors():
    root = Path("/sys")
    temperatures = []
    for hw in sorted((root / "class/hwmon").glob("hwmon*")):
        for entry in sorted(hw.glob("temp*_input")):
            label = read(entry.with_name(entry.name.replace("_input", "_label")))
            temperatures.append((f"{hw.name}/{read(hw / 'name')}/{label or entry.stem}", entry))
    energy = []
    # Resolve symlinks to avoid counting a nested domain twice.
    paths = {
        p.resolve()
        for domain in (root / "class/powercap").iterdir()
        for p in domain.glob("**/energy_uj")
    }
    for entry in sorted(paths):
        energy.append(
            (
                entry.parent.name,
                read(entry.parent / "name"),
                entry,
                number(entry.parent / "max_energy_range_uj"),
            )
        )
    return temperatures, energy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("capture command required")
    args.output.mkdir(parents=True, exist_ok=False)
    temps, domains = sensors()
    cpus = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
    gpu = sorted(Path("/sys/class/drm").glob("card*/gt_*_freq_mhz"))
    config = {
        "command": command,
        "started_unix_s": time.time(),
        "uname": list(os.uname()),
        "rapl_is_wall_power": False,
        "rapl_domains": [
            {"id": key, "name": name, "path": str(path), "range_uj": span}
            for key, name, path, span in domains
        ],
        "temperatures": {key: str(path) for key, path in temps},
        "power_constraints": {
            str(p): read(p)
            for _, _, energy, _ in domains
            for p in energy.parent.glob("constraint_*")
        },
        "cpu_governors": {cpu.name: read(cpu / "cpufreq/scaling_governor") for cpu in cpus},
        "cpuinfo": read(Path("/proc/cpuinfo")),
    }
    (args.output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    previous = {}
    with (
        (args.output / "capture.log").open("x") as log,
        (args.output / "telemetry.jsonl").open("x") as data,
    ):
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)

        def stop(signum, _frame):
            process.send_signal(signum)

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        start = time.monotonic()
        while True:
            now = time.monotonic()
            row = {
                "elapsed_s": now - start,
                "unix_s": time.time(),
                "pid": process.pid,
                "temperature_c": {},
                "rapl": {},
                "cpu_frequency_khz": {},
                "throttle_counts": {},
                "gpu_frequency_mhz": {},
                "loadavg": read(Path("/proc/loadavg")),
                "proc_stat": read(Path("/proc/stat")),
                "network_counters": read(Path("/proc/net/dev")),
                "meminfo": read(Path("/proc/meminfo")),
                "process_stat": read(Path(f"/proc/{process.pid}/stat")),
                "process_io": read(Path(f"/proc/{process.pid}/io")),
                "disk_free_bytes": os.statvfs(args.output).f_bavail
                * os.statvfs(args.output).f_frsize,
            }
            for key, path in temps:
                value = number(path)
                row["temperature_c"][key] = value / 1000 if value is not None else None
            for key, name, path, span in domains:
                value = number(path)
                watts = None
                if value is not None:
                    if key in previous:
                        before, stamp = previous[key]
                        delta = value - before
                        if delta < 0 and span:
                            delta += span
                        if delta >= 0:
                            watts = delta * 1e-6 / (now - stamp)
                    previous[key] = value, now
                row["rapl"][key] = {"name": name, "energy_uj": value, "watts": watts}
            for cpu in cpus:
                row["cpu_frequency_khz"][cpu.name] = number(cpu / "cpufreq/scaling_cur_freq")
                for path in (cpu / "thermal_throttle").glob("*throttle_count"):
                    row["throttle_counts"][f"{cpu.name}/{path.name}"] = number(path)
            for path in gpu:
                row["gpu_frequency_mhz"][f"{path.parent.name}/{path.name}"] = number(path)
            row["returncode"] = process.poll()
            data.write(json.dumps(row) + "\n")
            data.flush()
            if row["returncode"] is not None:
                break
            time.sleep(max(0, 1 - (time.monotonic() - now)))
    (args.output / "complete.json").write_text(
        json.dumps(
            {
                "returncode": process.returncode,
                "elapsed_s": time.monotonic() - start,
                "finished_unix_s": time.time(),
            }
        )
        + "\n"
    )
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())

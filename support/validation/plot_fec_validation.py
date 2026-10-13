#!/usr/bin/env python3
"""Plot known-truth qualification results; zero counts are marked, not hidden."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with path.open() as stream:
        return [{k: v if k in ("model", "receiver") else float(v)
                 for k, v in row.items()} for row in csv.DictReader(stream)]


def curve(axis, rows, label, numerator="packet_errors", denominator="packets", **style):
    rows = sorted(rows, key=lambda r: r["esn0_db"])
    x = np.array([r["esn0_db"] for r in rows])
    count = np.array([r["rejected"] + r["wrong_accepted"] if numerator == "packet_errors"
                      else r[numerator] for r in rows])
    total = np.array([r[denominator] for r in rows])
    # Open downward triangles show zero errors at the observation resolution.
    # They are not binomial confidence bounds: codeword errors can be correlated.
    line, = axis.semilogy(x, np.where(count > 0, count / total, np.nan), label=label, **style)
    zero = count == 0
    axis.scatter(x[zero], 1 / total[zero], marker="v", facecolors="white",
                 edgecolors=line.get_color(), s=32, zorder=5)


def finish(figure, axes, path, ylabel="True post-RS packet error rate"):
    for index, axis in enumerate(np.asarray(axes).ravel()):
        axis.set_xlabel("Data-carrier Es/N0 (dB)")
        axis.set_ylabel(ylabel if isinstance(ylabel, str) else ylabel[index])
        axis.set_ylim(1e-9 if "bit" in axis.get_ylabel().lower() else 1e-6, 1.2)
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    figure.text(.5, .015, "Open downward triangles: zero observed errors (shown at 1/N; not a confidence bound).",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, 1))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rates = ["1/2", "2/3", "3/4", "5/6", "7/8"]
    awgn = read(args.data / "awgn.csv")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, qam in zip(axes, [4, 16, 64]):
        for rate, name in enumerate(rates):
            curve(axis, [r for r in awgn if r["qam"] == qam and r["rate"] == rate], name)
        axis.set_title(f"{qam}-QAM, known channel, all code rates")
    finish(figure, axes, args.output / "awgn_all_modes.png")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, qam in zip(axes, [4, 16, 64]):
        for name, file in [("AWGN", "awgn-detail"), ("F1", "f1"), ("P1", "p1"), ("Bursts", "burst")]:
            rows = read(args.data / f"{file}.csv")
            if file == "burst":
                rows += read(args.data / "burst-tail.csv")
            curve(axis, [r for r in rows if r["qam"] == qam], name)
        axis.set_title(f"{qam}-QAM 3/4, known channel")
    finish(figure, axes, args.output / "channel_models.png")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, file in [("Known channel", "f1"), ("Old single-symbol equaliser", "gpu-f1"),
                       ("New frame equaliser", "frame-f1")]:
        rows = [r for r in read(args.data / f"{file}.csv") if r["qam"] == 64]
        curve(axes[0], rows, name)
        curve(axes[1], rows, name, "trellis_errors", "trellis_bits")
    axes[0].set_title("64-QAM 3/4, F1: known-payload packet errors")
    axes[1].set_title("64-QAM 3/4, F1: post-Viterbi bit errors")
    finish(figure, axes, args.output / "equaliser_comparison.png",
           ["True post-RS packet error rate", "Post-Viterbi bit error rate"])


if __name__ == "__main__":
    main()

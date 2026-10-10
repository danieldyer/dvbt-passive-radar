# Saved sensitivity measurements, 6 October 2026

These small artifacts accompany [the report](../../sensitivity_audit_20261006.md).
They describe three independently processed 32-second excerpts of the C31
recording, starting at 50, 140 and 265 seconds. The report displays ±200 Hz;
internal pilot-ghost diagnostics retain wider coverage.

| Artifact | Meaning |
|---|---|
| `excerpt-*.json` | Per-batch residual, coherent direct-template and phase-replay measurements |
| `excerpt-*-maps.json` | Final 1 s and 10 s cancellation and pilot-ghost measurements |
| `excerpt-*-windows.json` | Recorded background/leakage comparisons using four windows |
| [matched-filter.json](matched-filter.json) | Exact finite-template white-noise bound, 919 symbols, frozen response, fixed delay |
| [matched-filter-coloured.json](matched-filter-coloured.json) | Same comparison with first excerpt's held-out diagonal carrier variance |
| [covariance-bound.json](covariance-bound.json) | Diagnostic carrier/common-mode statistics and conditional complex/real bounds |
| `excerpt-*-carrier-noise.npz` | Physical carrier variance, separate training/evaluation estimates and fitted response |
| [noise-model-check.json](noise-model-check.json) | Observed map background versus diagonal prediction; explicitly fails as a complete model |
| [doppler-bound.json](doppler-bound.json) | Exact local window-information bounds and disjoint train/validation/evaluation comparisons |
| `excerpt-*-doppler-noise.npz` | Doppler frequencies and averaged, first-half and second-half background profiles |
| [response-injection.csv](response-injection.csv) | Complex echo retention after injecting before production response fitting |
| [theory.json](theory.json) | Window metrics, DC projection, ICI and conditional motion/CPI models |
| [tests.log](tests.log), [pipeline-tests.log](pipeline-tests.log) | Nine support tests and three relevant production regressions |
| [production-source-sha256.json](production-source-sha256.json) | 76 unchanged production source/shader hashes, compared with the preceding profiling baseline |

The large spectral archives and complex range caches remain under
`build/sensitivity-20261006`; they are not duplicated here.
[Reproduction commands](../../../support/validation/README.md) describe the
isolated Python environment and optional C++ support probe.

SNR gains in the JSON are conditional calculations, not calibrated real-target
detection gains. Doppler covariance results assume a constant-amplitude on-bin
echo in the retained complex map. They overlap window losses in the finite-
waveform tables and must not be added independently. Noise proxies can include
wanted returns. Minimum-phase echo self-subtraction is measured separately.

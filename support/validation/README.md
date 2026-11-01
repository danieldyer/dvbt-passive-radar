# Capture qualification

Build the explicit `live_pipeline_benchmark`, `profile_benchmark`,
`channelizer_benchmark`, `archive_benchmark` or `fft_benchmark` targets.
The pipeline benchmark uses one fixed production decoder policy:

```
live_pipeline_benchmark FIXTURE.dvblive --seconds 8 --warmups 12 --repetitions 8 \
  --workers 5 --slots 9 --archive NEW_DIRECTORY
```

`fixtures.json` records the retained private RF file identities and expected
lower-three-channel packet gate. Fixture payloads are ignored rather than committed.
Input must include channelised I/Q. The benchmark reads actual saved channels
through the production decoder/profile path, runs equivalent upstream kernels on
resident synthetic wideband input, and performs real profile writes. This is a
coupled workload, not an end-to-end replay of RF acquisition/clock recovery.
Reports include CPU worker seconds, GPU stage time, wall time including reads,
end drains/reset/writes, payload checksums, RS counts and profile-validity counts.
Short-file replay pays a drain per replay; also qualify sustained live capture.
`--replicate-channel INDEX` copies one saved signal to every receiver to stress
all workers with the same marginal RF fixture (matching guard intervals required).
This changes only the benchmark input, never decoder policy or work limits.
For the retained lower-three fixture, index 2 is Nine. Record this field with
results; replicated inputs do not represent a physical multistatic capture.
`--keep-last` retains one profile archive for renderer validation; otherwise
benchmark output payloads are deleted after each repetition.

`capture_soak.py --output NEW_DIRECTORY -- COMMAND ...` logs Linux temperatures,
RAPL domain energy, clocks, thermal throttle counters, process/load and disk stats
without changing governors/power settings. `analyze_capture_soak.py` combines them
with capture diagnostics. RAPL domain powers overlap and are not wall power.
A thermal plateau needs evidence, not just an absence of throttling. Install
`requirements.txt` in an isolated Python environment for the offline plots; the
on-target logger itself uses only the Python standard library.

The optional `archive_format_benchmark` compares current binary framing with the
typed MCAP/FlatBuffers proposal in `schema/range_profiles.fbs`. Generate its C++
header and `.bfbs` with FlatBuffers 25.12.19 and set `PROFILE_SCHEMA_INCLUDE_DIR`,
`FLATBUFFERS_INCLUDE_DIR`, `MCAP_INCLUDE_DIR` (MCAP C++ 2.1.1). These are study-only
dependencies; there is no selectable production MCAP writer. Inspect the report
before choosing migration/CRC settings. Tests and production fixtures have one
archive format, independently parsed by archive tests.

Store generated measurements under ignored `build/production-qualification/`.
Keep experiments outside the source tree and promote only validated improvements.

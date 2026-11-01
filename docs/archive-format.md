# Range-profile archives and PL64 navigation

The navigation service in the current `pl64-nix` checkout writes uncompressed
MCAP C++ 2.1.1 containing FlatBuffers 25.12.19 messages. Its embedded `.bfbs` schema
is `pl64.ins.fb.Record`, on `/navigation/events`. It retains monotonic arrival,
UTC, estimated measurement time/uncertainty and native counters. Both services
use `std::chrono::steady_clock` on Linux; a shared host clock domain is therefore
available. The Nix checkout was inspected read-only and has not been modified.

The present capture writer remains the single `DVBTLIV1` implementation. It
preserves complex binary32 C0/C1 profiles, per-row ADC/window phase coordinates,
per-transmitter timing offsets/rates, common clock segments, channel RF/NCO
settings, decoder erasures, inverse-noise information and receiver response/noise
spectra. Display operations never enter this file. MCAP is a proposed replacement,
not an extra selectable writer or a format migration claimed as complete.

The proposed typed profile payload is in `schema/range_profiles.fbs`. Production
migration must also define/retain configuration, clock trajectory, input gaps,
device epoch, host/device anchors, timing uncertainty and quality records, rather
than putting all data into one opaque byte field. Independent navigation and
radar channels can coexist in one MCAP or use separate MCAPs with common clock
identifiers. Capture-native sample counters and correction trajectories must
remain authoritative: MCAP nanosecond indexes cannot replace them or turn an
uncalibrated device epoch into UTC.

The current uSDR driver exports sample-counter timestamps, but does not implement
the UHD-style host/device clock query anchor. MCAP encapsulation alone will not
solve that physical alignment. A future combined service needs an explicit,
uncertainty-qualified mapping from the uSDR counter to host/navigation time,
including stream/receiver latency. Host arrival time must remain distinguishable
from the sample acquisition time. This is an integration requirement, not a
reason to modify saved complex phase or infer UTC from a filename.

## Measurements

`archive_format_benchmark` serially writes twelve 952-row/3072-lag, two-moment
batches, with 6817 response/noise entries and per-row metadata. Equivalent load is
three channels at 1.088 ms row cadence. FlatBuffers builders reuse capacity; files
are uncompressed. All writes and short-write errors are checked. First-batch
FlatBuffer verification is included. Output files are removed after measurement.
These are synchronous format costs, not a measured coupled throughput loss.
Files are flushed/closed but **not fsynced** in this benchmark: navigation's
periodic durability flush latency is excluded and still needs coupled testing.

| Format | PL64 CPU ms/input s | M5 CPU ms/input s | File bytes |
|---|---:|---:|---:|
| Current binary records |23.2|88.4|563439000|
| Typed MCAP, chunk+data CRC |139.8|191.0|563232481|
| Typed MCAP, chunk CRC |91.6|143.8|563232481|
| Typed MCAP, no CRC (diagnostic only) |43.3|95.1|563232481|

Typed MCAP is 0.037% smaller in this test: file-size overhead is negligible and
fixed-width row structs offset MCAP record/index overhead. FlatBuffers assembly
costs about 10.1–10.3 CPU ms/input s on PL64. The additional chunk CRC is about 48 CPU ms/s;
computing a second data-section CRC adds another 48 CPU ms/s. MCAP's portable
slicing-by-eight CRC implementation is the main incremental cost. Mac file-write
CPU cost differs substantially from PL64; compare formats within each host.

A single profile message is about 47 MB, so it becomes one oversized chunk even
with navigation's 64 KiB chunk target. This bounds message/index overhead; writing
one MCAP message per complex cell would be inappropriate. Keep the existing
bounded asynchronous archive queue and write each owned FlatBuffer directly from
the archive worker, avoiding an extra producer-side serialization copy. A
production comparison must include the same CRC/durability settings and a warmed
coupled capture/soak, because archiver CPU and memory traffic share the passive
PL64 power budget with FEC and Vulkan.

## Standard-library CRC acceleration

A separate experiment replaced only MCAP's internal IEEE CRC32 update helper
with the standard zlib API, converting between MCAP's raw CRC state and zlib's
finalized state using the required XORs. On PL64, the installed zlib-ng 2.3.3
native API was also tested. No production writer or Nix package was changed.
Offset 0–63, empty, partial and streamed inputs agree, and every byte of all four
complete archive variants is identical to the original MCAP implementation.

| CRC implementation | PL64 binary | PL64 dual CRC | PL64 chunk CRC | M5 binary | M5 dual CRC | M5 chunk CRC |
|---|---:|---:|---:|---:|---:|---:|
| Standard zlib |22.8|93.7|68.1|89.8|96.2|95.0|
| Native zlib-ng |22.5|57.3|50.5|—|—|—|

Values are CPU ms/input s, with unchanged integrity coverage. Native zlib-ng
reduces the PL64 dual-CRC increment over current framing to **34.8 CPU ms/s**,
versus about 117 with MCAP's portable helper. FlatBuffers assembly still costs
about 10.6 CPU ms/s; serialization/copying accounts for much of the remainder.
This is an actionable integration option, not a measured coupled speedup. A
future migration should use a maintained accelerated CRC implementation before
considering reduced integrity coverage. Turning all CRCs off is diagnostic only.
The helper experiment remains outside production; promote a replacement archive
format once, with the same durability policy and a warmed coupled soak.

The results support MCAP/FlatBuffers compatibility with negligible size overhead
and modest accelerated compute overhead. They do not establish unchanged live
throughput or storage durability. No second permanent archive backend was added.

References: [MCAP specification](https://mcap.dev/spec),
[FlatBuffers MCAP registry](https://mcap.dev/spec/registry),
[MCAP FlatBuffers schema guidance](https://mcap.dev/guides/getting-started/flatbuffers),
[FlatBuffers layout](https://flatbuffers.dev/internals/).

Raw measurements: `build/production-qualification/archive-format-m5.json` and
`build/production-qualification/archive-format-pl64.json`. Accelerated CRC
measurements and byte-equivalence evidence are under
`build/production-qualification/crc-study/`.

# Archive-to-video export

Run `render_capture.py CAPTURE_DIRECTORY --output NEW_DIRECTORY` using the
Python dependencies in `requirements.txt`; build `live_profiles_preview` first.
The default is a 3 s coherent window, ±100 Hz, 100 km and at least 30 fps, aligned
to a whole number of OFDM symbols. `--integration` accepts 0.5–10 s. Display DC
projection is applied after archive ingestion; capture profiles are preserved.

Range FFTs are computed once per archived row on Vulkan and retained in a bounded
GPU ring. The unchanged focus operator evaluates each full sliding window with
original information weights and explicit zero-information erasures. Gaps and
acquisition resets never form one aperture. Peak pooling also runs on Vulkan;
only display pixels cross the process pipe. The fixed-scale pass uses a compact
float16 dB spool, discarded after successful export. `--resume` resumes encoding
from a completed spool; `--collect-only` isolates GPU formation measurements.

ProRes (`--encoding prores`, default) and H.264 (`--encoding h264`) use macOS
VideoToolbox with software fallback disabled. Each frame is independently encoded
for scrubbing. `--encode-workers` limits parallel channel encoders. Fixed scale
limits are derived once per channel from the whole recording, not per-frame.
JSON sidecars retain the source fingerprint, scientific coordinates, commands,
colour limits and stage elapsed times. `validate_capture_videos.py` verifies
cadence, coherent windows, channels and output frame counts.

`video_export.py` handles spooling and hardware export; `render_video.py` contains
shared frame drawing/framing helpers, with no alternate DSP executable.

On macOS, build the exact-seek checker with
`swiftc -module-cache-path build/cache/swift support/video/check_quicktime.swift -o build/check_quicktime`,
then pass `--quicktime-check build/check_quicktime` to the validator along with
`--archive CAPTURE.dvblive` and `--output REPORT.json`. It checks distant and
adjacent seeks in both directions using AVFoundation.

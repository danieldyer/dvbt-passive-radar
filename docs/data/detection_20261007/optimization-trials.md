# Continuation optimization trials

- Exact short-prefix convolution: independent finite CAF test passed all guards,
  fractional origin and both complex Doppler moments. Full-pipeline result
  124.80 ms/s, CAF 19.00 ms/s; no reliable improvement over 115.61–129.62 ms/s
  with CAF 17.38–21.09. Removed rather than retain extra shaders/plans/buffers.
- Cached within-symbol gain: compute once per map shape/frequency grid, rather
  than transcendental functions for every range cell. Positive scalar only.
- Eight-by-eight extraction transpose: coalesces frequency-major reads and
  range-major writes; independent direct backprojection still passes.
- Larger focusing batch: 1024 rather than 256 Doppler frequencies per range-IFFT
  submission, same total padded-bin work at the default CPI/extent, +24 MiB scratch.
- Resident CAF batch: 1024 symbols did not provide a repeatable throughput
  benefit. Reverted to the compact 128-symbol tile.
- Bounded asynchronous focusing: retained. Owned correlation profiles permit
  reception/FEC to continue while GPU focus and CPU calibration run; one job
  per channel, explicit finish/drain, caller-thread consumers and exceptions.
- Temporal information weights: retained for correctness under changing noise
  levels. One small weight upload and existing-pass multiplications; no added FFT.

# Portable 0.2.7 fixture adaptation — draft

This draft changes build admission only. It is not yet a frozen or executed test
packet. Do not run fixture copying, hashing, compilation or tests during a timed
qualification.

After the timing hold is released:

- Keep all existing GPU fixture source and numeric oracle bodies unchanged:
  133 numeric cases, 96 lifetime guards, eight format families and T1–4.
- Bind the manifest to the raw 0.2.7 custom-port model and manifest `799413c6…`.
  Its six actual staging/direct/recovery/registration fragments remain LF-equal.
- Call `scripts/gpu_fixture_admission.py` from the explicit build path. Require
  `fresh-portable-v027-V6b-native-build-v1`, four raw compiled source pins,
  complete reconstructed sources, and all four custom static libraries rebuilt
  from source. Never translate an older private build into this schema.
- Compile a new fixture main with CUDA13.2, SM120a and MSVC14.44, linking the ten
  current native objects and four current libraries from that single fresh build.
- Keep the existing numeric validators unchanged. The initial comparison with a
  closed earlier numeric result must require all 132 historical FNV values equal;
  any difference is a retained failure pending an explicit source diagnosis.
- Add the unchanged Q6/mixed-format oracle and 1,248-row matrix only after the
  source-copy hold is released. Link its IQ/GGML inputs to the same new portable
  build; no old object or private absolute path belongs in the published packet.

These synthetic tests exercise actual helpers, registration and GPU math with
synthetic epoch/worker/47-buffer adapters. Real Model cancellation, maintenance,
quality, 1M capacity and throughput remain separate loaded qualification gates.

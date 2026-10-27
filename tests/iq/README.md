# Q6 and mixed-format numeric oracle

This is the same source and expected-row table used for the private 1,248-row
numeric matrix. The source fixes Q6_K's F16 scale initializer at byte208. It
checks token and grouped/dense GPU paths against independent GGML CPU decoding
and double-precision reference arithmetic. It does not load model weights.

First build the recipe's complete pinned native source. From an MSVC14.44 x64
developer environment, compile the fixture against that fresh build:

```powershell
python -B scripts/test_iq.py --build --native-build .work/build-native/BUILD.json
```

Run it separately when no inference process owns the GPU:

```powershell
python -B scripts/test_iq.py --run
```

All eight processes must naturally exit0 with the owned-Job/held-handle proof.
Each of the four selector settings runs102 self rows (`14,18,21,22`) and210 mixed
rows. The selectors are default, `STRATA_DS4_NO_LIGHT=1`, `STRATA_DS4_NO_TC=1`,
and both. Every expected row and tolerance must be present, finite and `ok`;
each output must end in exactly one `ALL OK`. The aggregate must total1,248.
Raw per-process output and cleanup reports are retained below the new
`.work/iq-test-01/owned-matrix-01` directory. No retry or model/API request occurs.

The GPU staging fixture is separate (`scripts/test_gpu.py`):133 numeric cases
and96 guards across eight format families with T1–4. Real Model cancellation,
1M context, quality and throughput require the loaded qualification commands.

The new portable IQ wrapper is source-prepared. Its final small-file admission
checks and any public CLI execution must be recorded separately; a private
numeric result does not claim that a public command has already run.

# Source overlays

The native build starts from the pinned public Strata and llama.cpp commits in
`pins/source-locks.json`, then applies only the small files stored here. It does
not reuse precompiled custom-engine or GGML objects/libraries. CUDA toolkit
libraries such as cuBLAS remain external vendor dependencies.

`strata/` contains the earlier tested custom Strata engine plus the opt-in GPU
expert miss staging, retained direct RAM DMA, and post-cancellation recovery.
The parent is an older custom Strata snapshot, not stock Strata 0.2.6.
`ggml/` contains the seven modified CPU/base files and matching modified/new
public headers used by its static GGML libraries. These are from the retained Xolotl dependency; this is not an
unmodified clean llama.cpp build. GGML CUDA is disabled for these libraries.

Each overlay keeps its original copyright and attribution. Strata is MIT
licensed (`LICENSE.Strata`); GGML is MIT licensed (`LICENSE.GGML`). Ported Project
Maya notices and licenses remain in `strata/src/dsv41/provenance/`, and the
Xolotl indexer attribution remains in `strata/third_party/xolotl-indexer/`.
Unmodified upstream source and its license files are obtained by checkout.

Files are stored with LF line endings. `pins/source-overlays.json` records the
portable file hash and the original local source byte hash separately. The
normalization changes line endings only. No weights, API keys, profiles, build
products, binaries, or private machine configuration are included.

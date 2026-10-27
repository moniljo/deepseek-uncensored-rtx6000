# Custom v0.2.7 recipe update

**The custom v0.2.7 default is selected after native nonregression, five cancellation modes, owned near-1M API qualification and managed local/public verification passed.** The preceding recipe and its measured results remain in commit
`29506b51122ff32ad01102e0749038a094a1195f` and the existing experiment history.
Preserve a rollback installation; selection here follows the separately scoped gates below.

## Source and build identity

| Input | Exact reference |
|---|---|
| Strata tag checkout | `5e8a4cca7d690ac60ca37eb09e7d0e010099ad6d` |
| Release executable-code ancestor | `a42cc3d7df70d1940de82cd0882f03e1f5dfc741` |
| Corrected custom port manifest | `799413c6b137a5457becf10c90ce471d57018cedbb467708066c041b7f7003d0` |
| GGML parent | `3cf03257f219afbe7334045ff7c6a06ac68c627d`, plus the fifteen locked overlays |
| Complete raw Strata source | 5,185 files; `fda1cc136f00fe3ebe5cc17bcdaa27f19d5f6390bf7df78ba62dca8c71aaa4f6` |
| Fresh development image reference | `4b3a7f1a29b1d860eb79363544acbe19d9b758345f4ec04dbad1a9db0609dd8b` |

This is v0.2.7 **plus** the retained V6b performance/recovery port, not an
unmodified official binary. The native inputs are forty-six Strata overlays and
fifteen GGML overlays. They preserve the tested raw bytes. The upstream Windows
checkout uses repository-local CRLF policy with upstream Git-attribute
exceptions; the overlays are copied without conversion. Package `* -text`
protects those checked-in overlay bytes.

The public source preparation must reproduce the complete source digest above.
The native build targets SM120a using CUDA 13.2, MSVC 14.44/compiler 19.44, Release,
Ninja, `/Z7`, tests enabled, and parallelism 6. It rebuilds the native and GGML
objects and libraries from source. Source reconstruction and equivalent build
options can connect the development qualification to these portable inputs.
They do not mean that the private development image was produced by this public
CLI, or that binaries built under different paths must have identical bytes.

The fresh portable receipt has schema
`fresh-portable-v027-V6b-native-build-v1`. It records four raw compiled source
hashes, the complete reconstruction digest, actual recipe pins and executable
hash. Old portable or incremental build schemas are refused by the new native
admission; their fields are not relabeled as new provenance.

`pins/update-027.json` is an immutable source-preparation snapshot, created before
loaded qualification. Its `candidate_qualification_status=PENDING` and
`new_default_claimed=false` fields mean that source preparation itself did not
qualify or select a deployment. The preparation-time API status in
`pins/api-runtime-overlays.json` has the same scope. Current results and selection
come from the later closed [native](../evidence/update-027-native.json),
[cancellation](../evidence/update-027-recovery.json),
[owned API](../evidence/update-027-api.json), and
[managed deployment](../evidence/update-027-deployment.json) evidence. Those later
results do not rewrite the earlier preparation record. The build reads the
active `pins/source-locks.json`, which selects the v0.2.7 tag above; earlier
source-lock snapshots are retained only as history.

## Compatibility and retained settings

The exact seven Mix-256G shards, lossless decoded-F32 sidecars and separately
pinned DSpark drafter remain unchanged. The profile keeps total context 1,048,576,
all six routed trunk experts, CPU 16, DSpark on/block 3/confidence 0.65, mmap off,
SSD cache 0, C 65,280 / A 4,096 / AI 128 and the 192 GiB registration cap. It explicitly
sets `DS4_WARM=0` and `DS4_COPY_PIN=0` for the custom ownership path.

Source review found that official pack-info acceptance alone did not establish
working Q6 expert kernels. The candidate includes the required grouped-switch
and float-decoder Q6 support and the retained-prefix checkpoint boundary fix.
The first full-port compile exposed duplicate legacy HC definitions; the
corrected source removes those duplicates. A successful compile or synthetic
numeric fixture is not a loaded 1M or speed qualification.

Native benchmarks keep the separately pinned older frontend with 60-second
grace. Prepare it with `scripts/prepare_runtime.py --benchmark --prepare`.
The API runtime is a separate complete v0.2.7 archive with the twelve retained UI
assets, custom headers and the same bounded grace. The newer upstream UI feature
set is not claimed. Two retained fonts are fetched from exact public commit
paths with SHA checks instead of being bundled as binaries.

## Benchmark and cancellation contract

The workload, corpus, thirteen original FIRST grades, existing dictionary-v2
oracle, timing clock, token-ID accounting and owned Job cleanup are retained.
Each arm uses two fresh 999,764-token seeds, thirteen followups per wave and one
unrelated graded interlude in a single process. No retries, output edits or
regrades are allowed. Raw 26 results and the original-twelve timing metric remain
visible; primary quality is the already-declared first eleven plus dictionary-v2,
twelve per wave.

The update nonregression policy was declared before candidate inference: both
waves need at least 95% of the same-day same-weight control decode rate and at
most 105% of its fresh TTFT, primary 24 and the applicable native, cancellation and
API gates. An observed speed improvement must be reported separately. The older
discovery threshold of 5% speedup and the older different-weight XQ TTFT target
are historical fields, not this update policy.

One initial candidate attempt remains failed: its wrapper required the older
run's fixed RAM-registration byte total. The engine completed its seed and first
followup and exited normally, but that wrapper stopped before quality grading.
No grade is inferred or repaired. The corrected observer checks the actual
startup domain/slot/all-CPU-copy inventory and all forty registered arenas. It
anchors exact bytes/fingerprint after the first normal seed, before graded
requests, and requires that same ledger through every later answer and physical
unregister/re-register/exit transition. It does not claim an independently
observed exact byte sum or fingerprint before the seed.

Arena size can vary with actual admitted GPU placement. A layer's CPU arena is
its copied-expert count plus spare, multiplied by its role-layout stride. The
native domain reconstruction checks those extents before publishing the startup
record. An old run's aggregate is not a model-weight invariant.

All five loaded cancellation modes remain separate required gates: repeated
setup cancellation, ordinary decode, active prefill, early rearm and initial
prefill. FIRST recovery must preserve quality/EOS, direct work with no bounce or
CPU fallback, at least 95% of its matching BEFORE decode rate, the normal pool
budget, physical lifetime and natural native 0/V3 cleanup. Source or fixture
results do not substitute for those loaded tests. The updated owned API qualification and managed/default verification passed separately, with their scopes recorded below.

The candidate explicitly pins `regex==2026.9.10`, required by the tokenizer and present in the Python 3.13 preparation/API environment. The historical Python 3.12 benchmark environment used 2026.9.29; complete frozen-ID parity remains mandatory rather than assuming these versions are interchangeable. The optional upstream `truststore` package was absent in the observed environments; TLS still verifies through the existing certifi/default fallback. No new trust-store mode is silently asserted as part of qualification.

## Closed native/recovery evidence

See [the paired native summary](../evidence/update-027-native.json) and [five-mode cancellation summary](../evidence/update-027-recovery.json). The candidate measured 36.6547/38.1305 original-twelve ID intervals/s versus 37.1561/38.7642 for the same-day incumbent; fresh TTFT was 351.162/352.900 s versus 347.321/346.128 s. Decode was 1.35%/1.63% slower and TTFT 1.11%/1.96% higher, within the predeclared 95/105 limits. Raw 25/26 remains a failure; corrected-primary 24/24 and all native/recovery gates passed. The [updated owned API summary](../evidence/update-027-api.json) records 16 passed checks and seven inference requests. Its FIRST request used 999,797 uncached prompt tokens and returned the exact answer with 97 completion tokens, natural stop, one `[DONE]` and valid usage. First visible content took 357.044 s; the complete stream took 357.174 s. That visible-content latency includes reasoning and is not the native benchmark first-token metric. The same held native remained live through the checks; source, library and payload pins were unchanged. Natural native/frontend exit 0, empty owned Job/V3 cleanup and a 19.464-second shutdown passed. The [managed verification](../evidence/update-027-deployment.json) subsequently passed 15 local and 14 public checks with the selected image and preserved configuration, key, learned profile and tunnel settings. Three fresh public samples matched the local boot, model and 1M context. This establishes the selected custom update. It does not attest the native through a kernel handle, repeat the managed near-1M request, or prove each individual connector; the separate owned API qualification covers native identity and near-1M inference. The update remained within its declared nonregression limits, and no speedup or all-raw-grades pass is claimed.

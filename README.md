# DeepSeek Mix on one 96 GB Blackwell GPU

**This tested custom V6b build is the current verified default on the measured machine.** With near-1M contexts it measured **38.49–39.16 actual ID intervals/s**, versus the same-weight stock control's 35.22–35.33. Cancellation/recovery, fresh near-1M API and managed local/public API checks passed. The raw Q02 score remains a disclosed **25/26 failure**; its separately predeclared corrected-primary 24/24 and paired decode gates passed. This is a custom Strata recipe, not an official upstream 0.2.6 feature or a universal performance claim.

The recipe retains the exact pinned orcarouter Mix-256G weights, all six selected trunk experts, DSpark on3, and a total context limit of **1,048,576 tokens**. It distributes source, tests, small routing metadata and a licensed public-text benchmark corpus. It distributes **no weights, binaries, keys, private configurations or raw conversations**.

## Current same-weight result

[Q02 independent evidence](evidence/qualification-v6b-q02.json) compares the fresh V6b public-source native build against the completed same-machine stock81 control, using the same verified model-pack namespace, input hashes and settings. Each wave has a fresh 999,764-token seed and thirteen graded followups. Both waves and an unrelated graded interlude ran in one native process: 29 GEN requests, no RESET or replay.

| Wave | Stock original-12 ID/s | V6b original-12 ID/s | Observed change | Stock TTFT | V6b TTFT | Raw grades | Corrected primary |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 35.2192 | 38.4915 | +9.291% | 334.862 s | 329.164 s | 13/13 | 12/12 |
| 2 | 35.3267 | 39.1636 | +10.861% | 328.409 s | 330.404 s | 12/13 | 12/12 |

The frozen decode metric is `sum(emitted IDs - 1) / sum(last-ID time - first-ID time)` over the **original twelve** tasks. It includes reasoning/control/EOS IDs, includes the legacy task even when its grade fails, and excludes the separate corrected-v2 task. It is not total request tokens divided by wall time. TTFT is reported separately: wave 2 was about 2 seconds slower than stock, despite faster decode. Outputs differed in length; these are not identical-output latency comparisons.

The raw report remains `passed=false`, `measurement-complete-quality-failed`, command exit 1. Wave 2's legacy account-balance answer used attribute access; the frozen worker accepts dictionary access and rejected it before execution (0/11). The original prompt did not specify that representation. The already-frozen dictionary-contract-v2 task passed 11/11 in both waves. The **predeclared** primary quality set is the first eleven original tasks plus that v2 task: **24/24 passed**. No raw grade, oracle, output or report was rewritten. This does not mean all raw tests passed.

Both fresh seeds reused 0 tokens; all 26 followups ended naturally and reused exactly 999,764. Native generation 1 work, physical unregister-to-zero before maintenance, new retained registration, and final exit-zero were checked. All 29 answer records reported 40 registered ranges, zero bounce and zero GPU-miss fallback. Native exit 0, V3 owned cleanup and no shutdown escalation passed. The separately scoped fresh-install API and managed-default checks also passed below; neither relabels the raw Q02 score.

## Hardware and dependencies

The measured machine is Windows x64 with an **RTX PRO 6000 Blackwell Max-Q 96 GB/300 W**, Ryzen 9 7950X3D (16 cores/32 threads), nominal 256 GiB RAM (**255.2 GiB reported physical**) and NVMe storage. Selected settings are CPU 16, DSpark on/block 3/confidence 0.65, expert mmap off, SSD cache 0, prefill chunk 65,280 and repeated-prefill minimum 32,768. CPU 32/SMT and removing MTP were measured slower on this machine.

Fresh source build, packing, recipe setup and the API-observer campaign use Python 3.13.15. Historical/private Stop and the paired benchmark controller use Python 3.12.12; receipts must preserve that scope and pair controls consistently. Native tools are CUDA 13.2, MSVC 14.44/compiler 19.44, CMake and Ninja, targeting `120a`.

Original pinned downloads total **354,355,210,720 bytes**. Lossless compatibility sidecars add **33,443,020,800 bytes**: **387,798,231,520 bytes** before build files, temporary packing, logs and Windows paging space. Original GGUFs remain read-only. The earlier full conversion audited 40,768,634,880 values with zero different float bits. This recipe attempt verified original/sidecar SHA-256 values and reused those audited sidecars; it did not repeat the whole conversion. Its fresh compiled codec sample checked 1,048,576 values.

Inspect available RAM, Windows commit/limit and storage before loading. Do not blindly copy a page-file size from another machine. Extra commit capacity does not replace physical RAM required by page-locked weights. Smaller-memory hardware is not validated by these measurements.

## How GPU misses and recovery work

The custom prefill pipeline uses chunk-major processing, selected attention/decode kernels and dense/Engram weight caches. It compacts temporary buffers after prefill and grows the resident GPU expert pool. A later unrelated long prompt can rebuild the large workspace through a checked worker epoch. These changes preserve weights, expert count and context length.

GPU-miss staging moves nonresident expert computation to the existing GPU kernels for admitted DSpark windows of 1..4 tokens, with at most 24 unique experts. It uses a bounded roughly 559 MiB GPU staging arena and two roughly 23.3 MiB pinned bounce buffers. The selected direct path avoids bounce packing when the current copied-RAM slot has proven full-span registration coverage. A private copy stream, readiness event and compute-stream wait protect ownership. No new expert arithmetic is introduced, but CPU/GPU output bit identity is not claimed.

`DS4_GPU_MISSES=1` and `DS4_GPU_MISS_DIRECT=1` select the path. `DS4_PREFILL_REGISTER_RAM_MB=196608` is a **192 GiB host-registration cap**, interpreted as MiB despite the MB name. Actual observed retention was 40 ranges/184,926,136,320 bytes. This keeps substantial existing **host RAM page-locked**; it is not 192 GiB of VRAM and does not duplicate that model payload. Every stage resolves current slot/domain/role identity, including after cache rotation. Registrations retire only after the required worker, mailbox, copy and compute drains, before maintenance can invalidate sources.

A healthy ordinary decode Stop retains its valid context. A cancelled new prefill never publishes fresh retained pins. V6b repairs the following short request through a held epoch, compact layout, pool admission and re-registration before token mutation. Recovery reserves 512 MiB below the last normal healthy pool budget. An already-large pool is checked/shrunk first; an unknown normal budget must reach validated initial targets. The normal anchor is not overwritten by the reduced recovery pool, so repeated cancellation does not ratchet the budget downward. Existing actual free-memory floors and rollback still apply.

## Validation and limitations

| Scope | Closed result |
|---|---|
| Fresh native/GGML source build | Pass; no old private objects copied |
| CPU policy tests | 230 checks pass |
| Fresh-library CUDA fixture | 133 numeric cases, 96 guards; all 132 inherited numeric hashes match |
| Five private cancellation modes on this build | Pass: 39 rows, 34 FIRST grades, 6 actual cancellations |
| Public two-wave Q02 | Raw 25/26 failure retained; corrected-primary 24/24 and paired >=5% each wave pass |
| Actual portable public active-prefill cancellation | Attempt 02 passes actual Stop, FIRST direct recovery, 131K reuse and normal cleanup; failed attempt 01 retained |
| FIRST near-1M API/auth/SSE/tools/disconnect | API05 passes 16 checks with the declared long-only reasoning preset; earlier attempts remain failed |
| Managed production/default change | Applied and verified: 15 local/14 public checks; three public samples match the local instance |

See [build/fixture evidence](evidence/build-and-fixture.json), [setup twice](evidence/recovery-setup-stop-twice.json), [decode Stop](evidence/recovery-decode-stop.json), [active-prefill Stop](evidence/recovery-active-prefill-stop.json) and [early/initial edges](evidence/recovery-edge-stops.json). The synthetic CUDA fixture uses real registration/copies/expert kernels but synthetic maintenance adapters; it does not alone prove loaded lifecycle or performance. The initial-prefill edge used an explicitly separate-process BEFORE control, not an in-process warmup.

Recovery cost remains in FIRST TTFT. Setup-twice recovery took 7.09/6.19 seconds; active 5.53, early 8.76 and initial 6.94 seconds. FIRST answers were direct/zero-bounce, but cancellation does not promise instant first-token latency or a universal speed gain.

The frontend's module-level `ENGINE_QUIT_WAIT_S` gives native QUIT 60 seconds, then retains the existing 5-second terminate/20-second kill fallback. Qualification requires normal native 0 and rejects escalation. The five private cancellation runs closed in under 20 seconds. The failed public cancellation attempt closed normally after a recorded 20.432-second native wait. Neither observation establishes the cause of the older untraced shutdown failure. The longer wait is teardown headroom, not an inference optimization.

The [first portable cancellation attempt](evidence/public-cancel-attempt01.json) did not send STOP because its binary-decoded log view preserved CRLF while a strict readiness expression expected LF. The request completed normally; cancellation and post-cancel recovery were not exercised. The correction normalizes only that log view, with LF/CRLF CPU trigger tests. It changes no native code, regex, timer, grader or full-Q execution block. The failed receipt remains failed.

The [fresh portable rerun](evidence/public-cancel-active02.json) passed its selected active-prefill mode: actual Stop at 62,486/999,764 input progress, zero-ID cancellation, seven FIRST grades and natural EOS, including the immediate corrected code answer's 11/11 vectors. BEFORE/FIRST measured 36.4374/42.2655 ID intervals/s with different 645/499-ID outputs. FIRST TTFT was 14.425 s, including 5.361 s of recovery. It restored 99.3511% of the normal pool anchor, used 40 direct ranges with zero bounce/fallback, then passed a fresh 131K seed and three exact-prefix reuses. Native 0, physical exit-zero and V3 cleanup passed without escalation. This one public mode supplements the separate five-mode private campaign; it is not a new full-quality or API qualification.

The [API-attempt history](evidence/api-attempts01-03.json) retains three failures. The first two loaded 1M capacity but never reached controlled inference: their observer lacked an independently held native handle. A separately proven HMAC framing defect and a post-start Windows access-denied query were corrected in the private observer, using a restricted creator handle acquired before resume; no DACL, privilege or model change was made. Attempt 03 then passed ownership, launch identity, loaded-library and normal-exit checks and actually read 999,771 uncached tokens. Its FIRST stream still failed a strict assertion whose reason the old checker did not retain. Neither its answer nor complete API qualification is claimed.

The [diagnosed fourth attempt](evidence/api-attempt04-and05-preset.json) remains failed: it returned the correct sum as an arithmetic expression instead of only the requested integer. Natural stop, terminal SSE, uncached usage and owned native 0 passed; the exact-format assertion was not relaxed. The separate [API05 qualification](evidence/api05-qualified.json) passed with the same prompt/answer and the declared long-only client preset `--long-thinking-preset high-budget256`: high effort maps to native effort level 75, with a **separate 256-token thinking budget and 1,024-token total cap**. Effort 75 is not a token budget. All short protocol requests remain unchanged. This does not change the installation's global 32,768-token reasoning budget or model settings.

API05 actually processed **999,797 uncached prompt tokens** and 64 completion tokens, returning the exact six-digit answer with natural stop, one `[DONE]` and valid usage. It passed 16 checks/seven API inference requests, both synthetic tool dialects without execution, and active-content disconnect/idle recovery. Original-native identity, loaded-library paths, unchanged pins and normal native 0/V3 cleanup passed. First visible answer content took 315.857 s; this includes reasoning and is not the native benchmark's first-token metric. Its 65 data frames and 4,346 heartbeats are not token counts.

The [default-deployment verification](evidence/default-deployment-verified.json) passed 15 local and 14 public checks with the selected executable, 1M context and preserved model, CPU/DSpark settings, learned profile, key and tunnel configuration. Three fresh public samples matched the same local instance. It did **not** perform a second managed near-1M request, attest the native through a kernel handle, or prove each connector separately; the owned API05 run supplies the separate native/near-1M proof. The first post-deployment verifier failed before HTTP because of its own bind/model-tag assumptions and remains recorded. Runtime receipts do not attest a Git publication action.

## Reproduce, inspect, and retain failures

Follow [setup](docs/setup.md), [portable benchmark instructions](bench/README.md) and the [experiment history](docs/experiments.md). The latter includes earlier Opt1..18/epoch/cache/attention work, official 0.2.6 comparison, CPU/DSpark controls, all GPU-miss variants and preparation failures. The [sanitized catalog](evidence/experiment-catalog.json) indexes raw receipt hashes without distributing private logs or million-token requests.

Important retained negatives include V1/V2 bounce decode 13.66/14.09 versus stock 29.78 ID/s, V3/96 at 21.25, and the V5 post-cancel 36.18->24.94 screen. V4 overlap 31.24 was effectively tied with the simpler V3/192 at 31.23 and was not selected. Earlier V3 full-Q also failed raw 25/26; its separate corrected 24/24 assessment did not rewrite that failure. The older Xolotl 32.60 ID/s reference used different, smaller XQ weights and is not a paired Mix baseline.

The package has 60 LF native overlays under 3 MB and four separate LF runtime overlays. Some tested benchmark/Python/JSON/profile files use CRLF. Preserve `.gitattributes` as `* -text`; globally forcing LF can invalidate raw SHA pins even when code or JSON still behaves the same. Check actual staged/index and committed-export bytes before publication.

`scripts/checkout_source.py`, `build_native.ps1`, `build_codec.ps1`, `download_models.py`, `prepare_model.py`, `prepare_runtime.py`, `configure.py`, `serve.py`, `test_gpu.py`, `check_api.py` and `check_package.py` implement the workflow. Model downloads use exact public revisions/SHA pins. The initial 61,440-byte routing profile is placement metadata, not weights or KV state; each installation gets a private mutable copy. Existing learned 43-row production profiles must not be replaced by benchmark/fresh-install metadata.

The benchmark text is a deterministic compressed public Python-source snapshot. Its [notice](bench/data/CORPUS-NOTICE.md), [inventory](bench/data/retained-public-archive.inventory.json) and [license](bench/data/PYTHON-LICENSE.txt) preserve copyright, Debian-specific provenance and exact bytes. It is not represented as an unmodified stock Python installation. Model downloads have their own repository licenses/access terms.

Keep downloaded weights, sidecars, builds, receipts, logs and credential-bearing connection/configuration files in ignored `.work/`. The source recipe is not a credential or binary distribution. Preserve failures and use a fresh output directory after diagnosis; do not reduce a guard or relabel a failed report to obtain a passing result.

# Experiment history

This is a compact reading guide to 1,176 preserved receipt records, not that many independent trials. The [sanitized receipt index](../evidence/experiment-catalog.json) contains portable trial IDs, original receipt SHA-256 values, literal outcomes and selected numerical fields. It omits prompts, answers, token-ID vectors, config bodies, personal paths, service hostnames, raw logs, binaries and weights. Copied upstream/community result trees are excluded.

Build/source proposals, CPU checks, isolated CUDA fixtures, loaded-model quality and deployment admission are separate scopes. Some older reports use `passed` for completed measurement; their FIRST-result grade counts below remain literal. Missing gates are unavailable, not false. No older failure is changed by a later correction.

## Long-context Mix campaign

All rows below are the custom lossless Mix/Q6-compatible campaign on the 96 GB card, not the retained XQ model. TTFT uses the fresh 999,764-token seed. Decode is the original 12-task actual-ID interval rate; the separate pre-existing corrected-v2 task is excluded from that rate. These were successive builds/settings, so a row-to-row change does not isolate one optimization.

| Trial | Fresh TTFT (s) | Original-12 ID intervals/s | FIRST joint grades | Outcome | Receipt SHA prefix |
|---|---:|---:|---:|---|---|
| full1m-01-latecopy-split3 | 681.03 | 27.6811 | 13/13 | completed measurement | `995286474786` |
| full1m-02-opt8 | 593.43 | 32.2945 | 13/13 | completed measurement | `be56c75235b3` |
| full1m-03-opt9-cold | 482.73 | 23.5767 | 12/13 | completed measurement; raw quality failure retained | `1e39aa35cfec` |
| full1m-04-opt10-direct96-r5 | 480.33 | 31.3278 | 13/13 | completed measurement | `295f01bb2a6a` |
| full1m-05-opt12-direct96-fusion | 474.66 | 34.2832 | 12/13 | completed measurement; raw quality failure retained | `ddcac0e16a0e` |
| full1m-06-opt12-a4-i128-noprof | 477.01 | 35.3603 | 13/13 | completed measurement | `851a17530263` |
| full1m-07-opt13 | 446.73 | 35.1788 | 13/13 | completed measurement | `51cd1e148b27` |
| full1m-08-v026-q6-auto | 842.53 | - | 1/1 recorded | failed/incomplete; no full decode score | `257c3100f2a8` |
| full1m-09-v026-q6-pinfix-auto | 843.21 | 22.0475 | 12/13 | completed measurement; raw quality failure retained | `f037d22df554` |
| full1m-10-opt15 | 378.50 | 36.0337 | 12/13 | completed measurement; raw quality failure retained | `566da3e3366f` |
| full1m-11-opt16 | 340.74 | 34.5792 | 13/13 | completed measurement | `2aa094296925` |
| full1m-12-opt17 | 335.22 | 34.9319 | 13/13 | completed measurement | `99f1f4030d62` |
| full1m-13-opt18-repeat3, wave 1 | 335.01 | 34.2818 | 13/13 | two fresh waves, preserved independent review | `87a4c15cc03c` |
| full1m-13-opt18-repeat3, wave 2 | 329.84 | 35.5984 | 13/13 | two fresh waves, preserved independent review | `87a4c15cc03c` |

Earlier Opt1-18 source, build and short-probe receipts remain indexed, including index-score/native-FP4 work, exact TopK, expert/RAM-store changes, cold expert caches, sparse attention variants, query RoPE, Engram/dense caches, MoE combining, and the epoch/shrink/regrow/rearm primitives. They are not all full-model benchmarks. The index also preserves compiler and fixture-initialization failures rather than silently replacing them.

Component ideas without a stable whole-path win were not treated as successes: alternative attention group sizes, XOR shared layouts, split/slab attention, pre-unpacked global KV and reduced-precision tensor-core studies have distinct source/fixture outcomes. Component speedups are not asserted as whole-model speedups.

## Interactive prefill and decode controls

The previous short-warmup path used compact buffers for an unrelated 85K prompt. Lowering only the repeated-prefill threshold to 32,768 reduced the controlled 85K fresh-seed TTFT from 256.25 to 78.80 seconds. The same-process optional 1M regression seed was 327.22 seconds; its four long followups plus the short/85K tasks passed 9/9 FIRST grades. That subset is not a new full-13 qualification.

| Controlled 85K decode setting | Actual three-suffix ID intervals/s | FIRST grades |
|---|---:|---:|
| DSpark on3, CPU16 | 28.7249 | 4/4 |
| No loaded MTP, CPU16 | 20.5191 | 4/4 |
| DSpark on3, CPU32 / SMT1 | 18.0636 | 4/4 |

The no-MTP and SMT32 variants were slower and were not adopted. The SMT32 trial used all 32 logical threads/16 physical cores; it is distinct from the unrun 24-thread idea that would have selected only 12 physical pairs.

## GPU-miss variants 1-4

All variants kept the same Mix weights and 1M capacity. The 85K screen had three fixed suffixes and four FIRST grades including the initial short answer.

| Variant | Actual ID intervals/s | FIRST grades |
|---|---:|---:|
| Stock81 same-day screen | 29.7806 | 4/4 |
| V1, same-stream bounce | 13.6581 | 4/4 |
| V2, private-stream bounce | 14.0862 | 4/4 |
| V3, 96 GiB registration cap | 21.2509 | 4/4 |
| V3, 192 GiB registration cap | 31.2264 | 4/4 |
| V4, resident/miss overlap | 31.2378 | 4/4 |

V1/V2 were substantial regressions; V3/96 remained slower. V4 differed from V3/192 by only about 0.04%, so the simpler V3 was selected. Its same-day full-1M original-12 rates were 39.2539 and 39.1027 versus stock81 35.2192 and 35.3267 (+11.46%, +10.69%). TTFT was 330.10/335.67 versus 334.86/328.41 seconds.

The original strict qualification **failed: 25/26 FIRST grades passed, with one legacy failure**. One legacy account-balance response used attribute access although the worker only accepts dictionary access; the legacy prompt did not specify the representation. The already-frozen corrected-v2 prompt did and passed both waves. A separate symmetric first-11-plus-v2 calculation passed 24/24 per arm and measured +13.60%/+7.94%; it did not replace the original score or authorize the old promotion.

All three later V3 Stop campaigns passed their loaded cancellation/first-resume/131K recovery/cleanup gates. They also exposed a real limitation: the first short request after setup or active-prefill cancellation used slower bounce copies until a successful long phase restored direct registrations. Zero CPU-expert fallback did not mean zero bounce copying. This motivates the separate V5 recovery campaign; old negative results remain unchanged.

## V5 preparation and reproducibility failures

- The exact native recovery policy passed 162 compiled CPU checks. The inherited/additive CUDA fixture passed 132 numerical cases and 64 guard/lifetime checks, using actual registration and GPU consumption. Synthetic epoch/47-buffer adapters are explicitly labelled; these tests alone do not prove loaded recovery speed or default eligibility.
- The first isolated Windows source checkout failed on deep Git paths. Long-path handling was corrected only in the new checkout; the failed attempt is retained.
- The first model-pack reuse attempt lacked its output parent directory. Attempt 2 passed after the wrapper correction, including an actual compiled lossless-codec test over 1,048,576 values. This is a codec/packing gate, not an inference-quality result.
- A frozen first runtime observer rejected the layout grammar before model loading. Its replacement and failure receipt remain separate; no model result is inferred from that preflight refusal.
- Earlier fixture preparation failures include the missing Windows `advapi32` link dependency and a C++ ambiguous-lookup call later fixed by one fully qualified function name. Failed builds and corrected builds remain separate.

The closed V5 failure is recorded below. V6b stability and full-1M results are separate gates; no V5 success or deployment is inferred from source/fixture results.

## Retained Xolotl history is a different model

The catalog includes retained source22-27 and maintenance evidence (attention24, decode25, maya26, compact/release27, internet/cleanup/launcher work). Those runs use the retained XQ representation, not this Mix payload. Their historical throughput is useful context but is not a same-weight comparison or evidence that a Mix change met a paired threshold.

Private source catalog snapshot SHA-256: `aac5b2e03a6a4b87124305b1886dcc333617e89a524f64fcda766674b87e9ed8`. Full receipt hashes, literal gates and trial IDs are in the linked JSON; prefixes in the table are only for scanning.

## Closed V5 recovery failure and V6b correction

The first loaded V5 setup-Stop recovery test failed deployment admission while all four FIRST grades and normal owned cleanup/protected-file gates passed. It compacted to a pool of 83,155,322,880 bytes, leaving 3,216.4 MiB free before registration. After 30 successful arena registrations, free memory was 3,037.2 MiB, below the unchanged 3,072 MiB floor. Registration correctly rolled back to zero; the following short request used slower GPU bounce copies. This was a headroom/performance failure, not a fabricated quality failure or unsafe retained-pin result.

V6b leaves an extra 512 MiB in the recovery-only pool budget. It reuses the checked held-epoch shrink when a current pool is already large, verifies immutable targets when no normal healthy reference exists, and preserves the last normal target to prevent repeated-cancel ratcheting. Normal/full-1M math and registration floors are unchanged. A fresh full V6b build passed 230 CPU checks. The real CUDA fixture passed 133 numeric cases and 96 guards; all 132 inherited numeric hashes match. The later loaded, full-1M, API and deployment results are recorded below in their separate scopes. The earlier V5 failure is retained.

## V6b two-setup cancellation result

The [independent closed result](../evidence/recovery-setup-stop-twice.json) covers two actual setup cancellations in one process: each cancelled a 999,764-token request before activation, returned zero generated IDs and published no fresh retained pins from the aborted phase. Both immediate short code responses passed 11/11 vectors and natural EOS. BEFORE/FIRST1/FIRST2 measured 34.3365 / 45.3288 / 42.6257 ID intervals/s, with 501 / 473 / 553 IDs. TTFT was 3.5140 / 15.4167 / 12.5182 seconds, including recovery; output sequences differ.

Both recovered pools were 82,591,211,520 bytes, 99.3505% of the unchanged normal 83,131,176,960-byte anchor. No second 512 MiB ratchet occurred. All ten answer records had 40 ranges and zero bounce, nine FIRST grades passed, and three later suffixes reused exactly 131,072 tokens. Native exit 0, final physical zero and V3 cleanup passed. The other closed modes, full-1M, API and default checks appear below as separate gates. The earlier V5 short-screen negative, 36.18 to 24.94 ID/s, is not selected or rewritten.

## Build, packing and frontend scope

This recipe ships source and small licensed metadata, never weights or binaries. Full original SHA checks and pack metadata generation passed by reusing audited sidecars; this attempt did not repeat the earlier 40,768,634,880-value conversion audit. The fresh compiled codec matched 1,048,576 independently decoded values. Separate Python runtime preparation applies four pinned overlays to a source archive; `--help` passed.

The host native-QUIT wait changed from 20 to 60 seconds before unchanged 5/20-second terminate/kill fallback. One traced normal decode close took 16.682 seconds, below 20: it does not causally prove that the old untraced failure was a timeout. The old failed run remains failed; the new setting supplies headroom for variable huge-RAM teardown. Normal qualification must not use escalation.

The catalog snapshot is a receipt index, not 1,176 independent trials or a live status feed. New V6b evidence is separately linked. API and deployment rows below use closed receipts; publication is a separate action.

The [closed ordinary decode cancellation](../evidence/recovery-decode-stop.json) passed 7/7 FIRST grades. BEFORE/FIRST were 34.9098/48.7941 ID/s with 227/287 emitted IDs and 2.5350/1.7504 s TTFT. It cancelled after 18 actual IDs, preserved the healthy 40-range context, and required no recovery re-registration. All 8 answer records had zero bounce; final physical zero/native 0/V3 passed. The traced engine.close took 16.682 s with no escalation, below the prior 20 s limit; no causal timeout proof is claimed.

Toolchain scope is explicit: fresh source build, packing and the API-observer campaign use Python 3.13.15; historical/private Stop and paired benchmark controllers use Python 3.12.12. Actual interpreter provenance belongs to each report.

## Closed V6b active-prefill cancellation

The [active-prefill review](../evidence/recovery-active-prefill-stop.json) passed 7/7 FIRST grades after an actual zero-ID cancellation at 62,486/999,764 input progress, with positive generation 1 prefill work and physical retirement to zero. BEFORE/FIRST were 35.4479/40.5270 ID intervals/s, 542/434 IDs and 3.5630/14.7795 s TTFT; both code 11/11 and natural EOS. Recovery took 5.527 s, retained 40 ranges with zero bounce, and restored 99.3499% of the normal anchor. Three followups reused 131,072 tokens; normal native 0, V3 and final physical zero passed. Close took 17.821 s without escalation, again below 20 s. Other edge modes/full-1M/API/default gates remain separate.

## Closed V6b early-rearm and initial-prefill edges

The [edge review](../evidence/recovery-edge-stops.json) passed. Early rearm Stop exercised C/A 1024 and a pool above healthy-minus-512-MiB. Checked shrink reached 63.341 GB before regrowth to 82.591 GB, 99.3505% of the 83.131 GB normal anchor. BEFORE/FIRST were 34.3848/46.0493 ID/s, 501/420 IDs, 3.5534/17.6450 s TTFT, including 8.761 s recovery. Seven FIRST grades, zero bounce, three exact 131072 reuses, native 0/V3 and final physical zero passed.

Initial-prefill Stop had no previous GEN or healthy/decode owner. At 62486/999764 progress it cancelled, retired to zero, then checked the immutable initial targets with saved normal anchor 0. FIRST was 434 IDs at 34.3194 ID/s and 13.1983 s TTFT, including 6.943 s recovery, code 11/11/EOS. The 34.3365 ID/s BEFORE comes from the predeclared separate setup-control process; it is not an in-process BEFORE. Four FIRST grades, 40 ranges/zero bounce, three exact 131072 reuses, native 0/V3 and final physical zero passed. Shutdowns 16.811/18.137 s were normal without escalation, both below 20 s. All five private modes pass; the public-adapter, full-1M, API and default results are separately recorded below.

## Portable qualification preflight and retained shutdown failure

A publication byte check found that forcing LF through Git attributes would rewrite tested CRLF benchmark/Python/JSON/profile metadata and invalidate raw SHA pins. The root attribute was changed only to `* -text`, preserving each file's tested bytes across checkouts. The 60 native overlays already use LF and remain unchanged. This is a packaging correction, not a benchmark/source/pin change or a performance result; no native rebuild is required. Actual index blobs and the committed export must still be checked before publication.

The first public full-Q adapter attempt failed **before any GEN**, not before model startup: native had loaded, the strict same-day baseline contrast rejected relocation of the profile option, and the owned native closed normally in 19.814 s. No wave, quality or decode-rate result exists for that attempt. MF03 replaces only the profile value at its original argument position; the original strict baseline contrast and all quality/rate policies remain unchanged. Its separately preserved source/CPU proof passes. The subsequent closed Q02 result is recorded below.

An earlier V6b decode attempt remains overall failed: seven FIRST grades passed but normal-exit/cleanup admission failed with reported native return 1. That run has no detailed shutdown trace. Later normal traces below 20 s do not establish its cause; the 60 s grace supplies headroom without retroactively passing it. Both failure records and the value-only fix are indexed.

## Closed public V6b Q02: raw failure and predeclared primary pass

[Independent Q02 evidence](../evidence/qualification-v6b-q02.json) preserves raw report SHA `27020650...` and literal `passed=false`/`measurement-complete-quality-failed`/command exit 1. Wave 1 passed 13/13 and wave 2 passed 12/13, retaining the legacy account-balance attribute-access rejection (0/11). The already-frozen dictionary-contract-v2 case passed 11/11 in both waves. The predeclared first-eleven-plus-v2 primary passed 24/24; it does not replace the raw 25/26 result. No regrade or oracle change occurred.

| Wave | Stock original-12 ID/s | V6b original-12 ID/s | Change | Stock TTFT (s) | V6b TTFT (s) | Raw | Primary |
|---|---:|---:|---:|---:|---:|---:|---:|
|1|35.219164|38.491455|+9.291223%|334.862150|329.163734|13/13|12/12|
|2|35.326693|39.163643|+10.861336%|328.409236|330.403563|12/13|12/12|

Both original-12 rates include the legacy row and exclude separate v2, with actual emitted-ID intervals/seconds recomputed. Both 14-input tuples matched stock per wave. All 29 GEN counts/output hashes/native DONE counts, two fresh zero-reuse seeds, 27 naturally finished graded/interlude responses, 26 exact 999,764-token followup reuses, generation 1 cumulative-difference work and physical retirement/new retention/final zero were verified. All 29 answer records had 40 registered ranges and zero bounce/fallback. Native 0/V3/no escalation passed, shutdown 17.275 s. No CPU/GPU token-bit identity or universal speed claim is made.

The paired candidate used the fresh public native build with the same verified existing pack namespace as stock. The selected actual portable cancellation mode, fresh-install API05 and managed deployment are separately recorded below. The raw failed command is not relabeled an all-tests pass, even though the predeclared primary Q gate is true.

## First portable cancellation attempt: STOP not exercised

The [closed first attempt](../evidence/public-cancel-attempt01.json), raw SHA `87974569...`, remains failed. All three BEFORE grades passed with natural EOS, but `STOP_written=false`: the long request read 999,764 tokens with reuse 0 and completed its requested one ID with `length` finish after 331.078 s. Neither cancellation nor FIRST-after-cancellation recovery was exercised. Native 0, final physical zero, source protection and V3 cleanup passed. The recorded native QUIT wait was 20.432 s without escalation; this is a trace of this run, not a retrospective explanation of the older untraced failure.

The public binary log reader preserved CRLF; the private text reader had normalized line endings. The unchanged strict readiness expression matched zero raw CRLF lines and one normalized line. MF04 changes only that observer's returned view to normalize CRLF to LF. Actual interrupt-function CPU tests on LF and CRLF each write one STOP and drain a synthetic zero-ID cancellation. All seven canonical cancellation functions, regexes, clocks, native code, graders, full-Q execution blocks and source provenance remain unchanged. Q02 needs no retiming from this log-view-only correction. The failed first attempt is retained separately from the fresh result below.

## Portable active-prefill cancellation rerun

The [closed second attempt](../evidence/public-cancel-active02.json), raw SHA `33651997...`, passed its selected public mode. It sent STOP at 62,486/999,764 prompt progress with no generated IDs, drained actual native `cancel`, physically retired the old owner and published no fresh retention from the cancelled phase. Seven FIRST grades passed with natural EOS; three code responses passed 11/11 vectors.

| Matched short response | Emitted IDs | Actual ID intervals/s | TTFT (s) |
|---|---:|---:|---:|
| BEFORE corrected code | 645 | 36.437385 | 3.524805 |
| FIRST after Stop | 499 | 42.265465 | 14.425204 |

The input hash and 168-token size matched, but output hashes and lengths differed. The observed rate ratio was 1.15995 against the declared 0.95 floor; this is not identical-output latency or universal-speed evidence. Recovery's 5.361 s remains inside FIRST TTFT (3.121 s compaction and 2.195 s registration). The held recovery preserved issued sequence 946 and token/checkpoint guards, compacted to C/A 1024 with AI 128, and restored 82,675,722,240 bytes: 99.3511% of the saved normal anchor, within the recovery cap.

All eight answer records had 40 registered ranges and zero bounce/fallback. A subsequent fresh 131,072-token seed reached generation 2/attempt 2, with positive native work and three natural-EOS suffixes reusing exactly 131,072 tokens. Physical unregister-before-maintenance and final exit-zero, source protection, separately held identity, empty owned Job and native 0 passed. Native QUIT waited 18.161 s without escalation. This selected portable mode supplements the five private modes; it does not erase attempt 01 or change raw Q02 grades. Subsequent API/default checks have their own receipts below.

## Closed API attempts 01-03: keep component passes and failures separate

The [sanitized API history](../evidence/api-attempts01-03.json) preserves these three **failed** attempts. These three failed attempts do not provide an API/default or publication pass; subsequent API05 and default verification are separate.

| Attempt | Closed observation | Limitation | Raw report SHA prefix |
|---|---|---|---|
| 01 | Native READY with capacity 1,048,576; no controlled inference | No independently held native; generic error erased the first cause. Only frontend exit was proven before forced owned-Job cleanup. | `af4edfe8fdef` |
| 02 | Same capacity; no controlled inference | Loaded-health guard refused without a held native. An owned native query recorded access denied (Win32 5). Forced owned-Job cleanup is not natural native 0. | `4249ea37744d` |
| 03 | Ownership, launch HMAC, selected DLL paths and normal native 0/V3 passed; actual FIRST API input read 999,771 tokens with reuse 0 | Six read-only auth/context checks passed, but the FIRST stream failed a strict assertion. The old checker discarded its precise reason and parsed-answer metadata. | `f1aec9d1b493` |

A separate source defect in attempt 01 omitted the eight-byte big-endian serialized-argv length from the launch HMAC. Three synthetic comparisons against the actual runtime proved the mismatch and corrected parity; it is not asserted to be that attempt's unrecorded first cause. The corrected framing in 02 did not resolve native-handle observation.

The private observer was then changed to receive a restricted read capability (`0x101410`) from the native's actual creator before resume. A signed request/ACK binds the nonce, executable, creation time, kernel PID and existing owned Job. The same retained handle supplies later identity and module queries; no DACL, administrator or SeDebug change and no global PID search is used. The actual runtime's native constructor, lifeline, nested Job, requests and model remain unchanged. This is qualification instrumentation, not an official engine feature or an additional user-facing serving requirement.

The first non-model capability probe passed its handle/child-exit components but failed an immediate Job-active sample while a console helper lingered. A separate probe with a bounded empty-Job wait passed before final cleanup. Attempt 03 then demonstrated the loaded handle/attestation/module/normal-exit path, without retroactively passing attempts 01/02.

For 03, native prompt time was 312.668 s and the 11 generated IDs took 453 ms; the frontend reported stop without cancellation. All 40 registered ranges were direct with zero bounce/fallback and retired physically to zero. Shutdown completed normally in 19.481 s. That short 11-ID answer is not a throughput benchmark, and the API wrapper/input differs from the paired native benchmark. HTTP 200, 13 data frames and 4,348 heartbeat comments do not establish an exact answer, one valid usage record or complete SSE termination. The failed predicate remains unknown in the original receipt; no engine, protocol or quality cause is invented.

The public synthetic prompt was checked independently without tokenization: its 3,434,689-byte prefix matches the frozen public corpus, followed by the exact 130-byte frozen `audit-sum` question. The unique ORCHID and CEDAR codes are 642,913 and 185,726, whose sum is correctly 828,639. No wrong expected value or generator contract was found. This does not reveal or validate the answer actually produced in 03.

The next checker revision records preassert answer hash/length/exact-match, finish/DONE and usage metadata, plus a bounded redacted exception reason, while keeping every assertion and request unchanged. The private observer may save bounded failed-response events only for that known public synthetic input, using the same consumer; these private raw events must never be published. The later closed 04 result is recorded below. Attempt 03's missing predicate is not reconstructed from a later response.

## API04 format failure and the declared API05 client mode

The [fourth attempt](../evidence/api-attempt04-and05-preset.json), raw report SHA `7f5a56d7...`, remains **failed**. Its 24-character content contained the correct sum but also an arithmetic expression, violating the unchanged instruction to return only the integer. The expected answer was not wrong, and no grader was broadened. The private raw response is not published.

The preassert summary recorded one `[DONE]`, one `stop`, and one usage object: 999,771 prompt tokens, 11 completion tokens, zero cached tokens and total 999,782. Those observed values independently satisfy the existing usage conditions, but the original row had already failed its exact-answer assertion. Ownership/HMAC/selected-library checks, native 0, physical exit-zero, empty owned Job/V3, unchanged source/configuration pins and no live escalation passed. Owned shutdown took 20.549 s; this is the whole owner-close observation, not a newly isolated native-QUIT interval or a diagnosis of older failures. There is no complete API qualification from 04.

Before attempt 05, a new **long-only client preset** was declared: `high-budget256` sends high effort, a separate 256-token thinking budget and a 1,024-token total generation cap. Actual DeepSeek OpenAI normalization and the pack's Jinja renderer were checked on tiny text: high maps to the same effort level 75 used by the native benchmark. **Effort 75 is not a 75-token budget.** The input archive/question and exact expected integer stay unchanged, as do all response assertions, native math, weights and the 1M context. The installation's global 32,768-token reasoning budget is not changed. All short protocol/tool requests retain their original settings; the bounded disconnect request also stays unchanged.

The old default nonthinking long request remains available with its prior total cap of 256. The new explicit mode is a distinct declared experiment, not a regrade or a claim that 04 passed. Public source checks passed 54 CPU cases and the private observer retained its 38 CPU checks; those CPU results alone are not an API runtime pass. The subsequent closed API05 and managed-default results follow below.


## API05 qualification and verified selected default

The [closed API05 evidence](../evidence/api05-qualified.json), raw owner report SHA `1aa1b021...` and API receipt `579b5706...`, passed **16/16 checks and seven API inference requests** in one fresh process. The declared FIRST-long preset was high effort (native level 75), a separate 256-token thinking budget and 1,024-token total generation cap. It processed 999,797 prompt tokens with cache 0, generated 64 completion tokens including reasoning, and returned the exact bare six-digit integer with one natural stop and one `[DONE]`. This is a pass for that declared mode, not a regrade of the nonthinking format failure in 04 or a reconstruction of 03's lost predicate.

First visible answer content was observed at 315.857 s, with the whole long stream at 315.978 s. These include prompt processing and reasoning and are not the native benchmark's first-token or weighted decode metric. The 65 SSE data frames, 4,346 heartbeat comments and two content frames are not token counts. Both forced synthetic tool dialects passed without execution; actual-content disconnect, observed busy-to-idle recovery and the natural followup passed. The original native remained held before/after API calls, launch HMAC and selected loaded libraries matched, source/configuration pins and payload stats remained unchanged, and native 0/empty owned Job/V3/no live escalation passed. Owned shutdown took 16.444 s. Initial routing metadata was a private fresh 40-row/61,440-byte profile, separate from the learned production namespace.

The [guarded default promotion and managed verification](../evidence/default-deployment-verified.json) subsequently passed. The promotion preserves exact model weights, 1M context, CPU/DSpark arguments, key, learned profile namespace and tunnel configuration. The managed verification passed **15 local and 14 public checks**, six API inference requests per surface, plus three fresh public samples matching the local instance. It did not perform a second managed near-1M request, prove each connector separately or claim kernel-native attestation; its identity scope is selected executable/configuration bytes, managed frontend record and authenticated API instance/model/context. The separate owned API05 supplies native-handle/normal-exit and near-1M proof.

The first managed verifier remains failed before HTTP, raw SHA `8dd31094...`: it assumed a loopback bind instead of the preserved existing bind, and confused the manager's stored model tag with the API's model ID. The replacement validates the original configuration and the distinct fields; it did not change the application to satisfy the check. This was a verifier preflight error, not an observed endpoint failure. The architecture-document updates and final preset/scope addendum are indexed as documentation-only receipts. Runtime/default qualification does not attest a remote Git publication action.

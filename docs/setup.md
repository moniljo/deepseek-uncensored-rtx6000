# Setup: exact Mix, custom Windows CUDA engine

**The selected V6b default passed the declared cancellation, API05 and managed local/public checks on the measured machine.** Public Q02's raw 25/26 failure remains disclosed; its predeclared corrected-primary 24/24, paired decode and native cleanup passed. Commands below are a reproducible setup sequence, not evidence that the complete sequence has already been repeated on a second computer. Run them from the recipe checkout root. Files under `.work/` are local and must stay out of Git.

## 1. Prepare the machine and developer shell

The measured machine is Windows x64 with an RTX PRO 6000 Blackwell Max-Q 96 GB/300 W, Ryzen 9 7950X3D, nominal 256 GiB RAM (255.2 GiB reported physical) and NVMe storage. Install Git, Python 3.13.15, CUDA Toolkit 13.2, and Visual Studio 2022 C++ Build Tools with MSVC **14.44**. Open an **x64 Developer PowerShell for VS 2022** using that toolset. `cl.exe` must report compiler 19.44, and CUDA must report 13.2. The native builder checks these versions and targets `120a`.

Keep the checkout path short. Avoid quotes, percent signs, exclamation marks, or newlines in paths passed through the generated Windows build batch files. `checkout_source.py` enables Git long paths in its own repositories; it does not change global Git or Windows policy.

```powershell
py -3.13 -m venv .venv
$Py = (Resolve-Path .venv\Scripts\python.exe).Path
& $Py --version
& $Py -m pip install -r requirements.txt
$CMake = (Resolve-Path .venv\Scripts\cmake.exe).Path
$Ninja = (Resolve-Path .venv\Scripts\ninja.exe).Path
$Cuda = $env:CUDA_PATH
if (-not $Cuda) { throw 'Set CUDA_PATH to the installed CUDA 13.2 toolkit.' }
& "$Cuda\bin\nvcc.exe" --version
cl.exe /?
```

The six direct Python pins are NumPy, Jinja2, requests, psutil, CMake, and Ninja; the exact versions are in `requirements.txt`. An explicit `$Py` prevents accidentally using another Python installation for packing, runtime, or the benchmark.

Before downloading, allow **354,355,210,720 bytes** for originals and **33,443,020,800 bytes** for sidecars, plus temporary packing, source, build and log space. Check available physical RAM and Windows commit/limit before loading. A large page file can support commit but cannot substitute for physical RAM needed by retained page-locked weights. Do not automatically set a page file from this document.

## 2. Prepare and verify the pinned sources

```powershell
& $Py scripts\checkout_source.py
& $Py scripts\checkout_source.py --prepare
& $Py scripts\checkout_source.py --verify
```

The first command prints a plan. Preparation fetches the exact Strata and llama.cpp commits in `pins/source-locks.json`, applies the declared LF overlays, and writes `.work/source/PREPARED-SOURCE.json`. Verification checks that identity. Optional local Git object sources must contain the same pinned public commits; they do not authorize a different branch.

Preserve the repository's `.gitattributes` rule, `* -text`. Some tested benchmark Python/JSON files and routing-profile metadata use CRLF, while the 60 native overlays use LF. Raw SHA-256 pins depend on those exact bytes. Do not apply a repository-wide LF conversion, formatter, or Git `text`/`eol` override. The attribute disables automatic conversion; it does not convert CRLF files to LF. Publication verifies the actual index blobs and committed export, rather than assuming working-tree hashes describe what Git will publish. This packaging fix changes no tested benchmark/pin/native bytes and needs no native rebuild.

Partial or incompatible existing checkouts are retained. Inspect the error and use a new `--work-dir` if necessary; pass the matching `-WorkDir` to the native builder. Do not substitute a current upstream release for the locked base.

## 3. Build native code and the lossless codec

```powershell
# Plan only.
.\scripts\build_native.ps1 -PythonExe $Py -CMakeExe $CMake -NinjaExe $Ninja -CudaRoot $Cuda

# Fresh full native source/GGML build. No model inference.
.\scripts\build_native.ps1 -Run -PythonExe $Py -CMakeExe $CMake -NinjaExe $Ninja -CudaRoot $Cuda -Parallel 3

# This script builds immediately; it has no -Run switch.
.\scripts\build_codec.ps1
& $Py scripts\prepare_model.py --codec-test
```

Expected outputs include `.work/build-native/strata-dsv41.exe`, its `BUILD.json`, and `.work/native/lossless_q6_codec.dll`. The native receipt must say the fresh compile/link succeeded and the source checkout was verified before and after. It must not use old private objects or libraries. The codec sample must report that the compiled codec matches the independent NumPy implementation.

The completed development build passed 230 CPU checks and the codec checked 1,048,576 values. These results do not replace the fresh build receipt on this computer. Preserve compiler errors and choose a fresh build directory for a different attempt.

## 4. Download and verify the exact original files

```powershell
& $Py scripts\download_models.py
& $Py scripts\download_models.py --download
& $Py scripts\download_models.py --verify
```

The first command prints sizes and destinations. Downloads use the repositories, revisions, complete-file sizes, and SHA-256 pins in `pins/model-downloads.json`; resumable `.part` files are verified before publication. If repository terms require a Hugging Face token, set `HF_TOKEN` locally through your secret-management method. Do not put it in a command argument, checked-in file, or shared transcript.

Keep all original GGUF shards and the pinned DSpark GGUF. The compatibility pack references those files. Passing a different quantization or a smaller drafter is not this recipe.

## 5. Build the compatibility/model pack

```powershell
& $Py scripts\prepare_model.py
& $Py scripts\prepare_model.py --run
```

The default source is `.work/data/models/Mix-256G`; the drafter is under `.work/data/models/dspark`. The final pack is `.work/data/packs/strata-mix256g-1m`. Original GGUFs remain unchanged. The compatibility converter adds sidecars whose decoded values must match the source values exactly, with per-file checksums and a complete conversion audit. It is not a smaller lossy quantization.

If an already-audited compatibility pack is supplied through `--q6-pack`, the wrapper still checks the complete originals, sidecar sizes/SHA-256, prior full-audit count, and model-directory binding. The development recipe attempt used this verified reuse path and generated 9,026,811 bytes of new metadata; it did **not** rerun all 40,768,634,880 conversion-value checks. A new machine without matching audited sidecars must do the full conversion and keep its receipt.

The initial 61,440-byte routing profile comes from `bench/data/initial-routing-profile.json`. It is placement metadata, not weights, KV state, or hidden benchmark answers. The local server will receive its own mutable copy. Existing final packs are not overwritten; use a new `--pack` and matching configuration argument for another attempt.

## 6. Build and run the synthetic GPU fixture separately

Stop your own other inference workloads before the GPU step. Run a single GPU owner at a time; these scripts do not authorize terminating another application.

```powershell
# Compile the fixture against this fresh native build; no model payload is needed.
& $Py scripts\test_gpu.py --build --native-build .work\build-native\BUILD.json --out .work\gpu-test-01

# Separate explicit CUDA execution.
& $Py scripts\test_gpu.py --run --native-build .work\build-native\BUILD.json --out .work\gpu-test-01
```

Require `RUN.json` passed, native exit 0, **133 numeric cases and 96 guards**. The actual direct-copy, registration, stream/event and expert math tests span the eight real role-format families. Development also checked all 132 inherited numeric hashes against the old fixture. `--compare-numeric` can enforce that comparison when a pinned earlier numeric result is available.

The fixture's epoch/pool adapters are synthetic. A passing fixture does not attest loaded cache workers, full cancellation recovery, model quality, or 1M throughput. Keep its output separate from those loaded gates. Existing output directories are intentionally not recycled.

## 7. Prepare the separate Python/API runtime

```powershell
& $Py scripts\prepare_runtime.py
& $Py scripts\prepare_runtime.py --prepare
& $Py .work\runtime\serve\server.py --help
```

The runtime starts from a Git archive of the pinned upstream commit and receives four declared runtime overlays. It does not modify the native build source tree. Use a new `--output` for a new attempt. A successful `--help` validates import/argument startup only; it does not load the model or prove API inference.

## 8. Create a private 1M configuration

```powershell
& $Py scripts\configure.py --cuda-root $Cuda
& $Py scripts\configure.py --cuda-root $Cuda --write
```

The first command prints a plan. `--write` creates `.work/local/server.json`, a fresh local profile, and `.work/local/connection.txt`. It generates a local key, or uses `STRATA_API_KEY` from the environment. **Both configuration and connection file contain a credential and must remain private.** Use a new `--local-dir` instead of overwriting an existing configuration or learned profile.

The selected template preserves:

- total context 1,048,576; one GPU; CPU threads 16;
- DSpark on, block 3, confidence 0.65; expert mmap off, SSD cache 0;
- prefill chunk 65,280 and repeat minimum 32,768;
- `DS4_GPU_MISSES=1`, `DS4_GPU_MISS_DIRECT=1`;
- `DS4_PREFILL_REGISTER_RAM_MB=196608`, a 192 GiB host-registration cap;
- original six-expert routing, exact weights, and the selected prefill/cache features.

There is no overlap flag in the selected configuration. Preserve the template's existing reasoning-budget value; reducing it is not a speed test of this engine. The startup reserve and actual registration floors are different controls. Recovery's extra 512 MiB pool reserve does not disable either admission guard.

## 9. Start the authenticated local server

```powershell
& $Py scripts\serve.py
& $Py scripts\serve.py --run
```

The first command is a plan. The second starts the configured API in this terminal; Ctrl+C stops it. Startup can take minutes because it loads and copies the model. Keep it on `127.0.0.1` as generated. Use the addresses, model name, and credential in the private connection file for your client. Do not paste the credential into a public issue or expose a tunnel as part of reproducing the local result.

Wait for actual readiness and check the engine log for copied-RAM and registration admission. A parsed argument or an open TCP port is not proof that the model, direct path, or 1M workspace is ready. A direct registration refusal is a real resource limitation; do not lower the floor to hide it.

## 10. Loaded benchmark and cancellation gates

The portable adapter under development uses `bench/prepare_inputs.py` and `bench/run_native.py`. Its source/input parity and actual public Q02 execution are now checked. The interface below runs a standalone measurement; it makes no paired speed-improvement claim without a compatible control report.

```powershell
$Reference = '.work\data\models\Mix-256G\DeepSeek-V4.1-Flash-1017-abliterated-Mix-256G-00001-of-00007.gguf'
& $Py bench\prepare_inputs.py --prepare --strata-source .work\runtime --tokenizer-dir .work\data\packs\strata-mix256g-1m\tokenizer --reference-gguf $Reference --output .work\bench-inputs
& $Py bench\run_native.py --run --config .work\local\server.json --build-receipt .work\build-native\BUILD.json --inputs .work\bench-inputs --reference-gguf $Reference --strata-source .work\runtime --output .work\bench\run-01
```

Do not run the native benchmark concurrently with the server. The benchmark owns its child and cleanup and needs a private fresh profile. It prepares inputs from the licensed public-source snapshot, verifies tokenizer/reference rendering, and measures two fresh 999,764-token seeds in one process. It retains raw original-13 grades and the predeclared corrected primary set; report both honestly. The ID-interval decode rate is `sum(emitted IDs - 1) / sum(generation seconds)`. Report TTFT separately and use the same weights/inputs/settings for the control.

Loaded recovery qualification must additionally cover repeated setup/active-prefill cancellation, the first graded short response after cancellation, direct-versus-bounce counters, unchanged token/checkpoint/sequence state, physical registration retirement, normal native exit, and owned cleanup. The five-mode private V6b campaign and the selected portable active-prefill rerun passed. API05 and managed verification passed as separately scoped gates. Synthetic numeric success is not a substitute.

## 11. Retain evidence and keep the upload small

```powershell
& $Py scripts\check_package.py
```

The package checker inspects the distributable source/data, excluding local work and virtual environments. It rejects weight/binary/log/private-config extensions and scans text, including the bounded compressed public benchmark corpus. Keep build/model/benchmark receipts locally. Do not add `.work`, DLLs, GGUFs, sidecars, `.env`, server configurations, private paths, or full raw prompts/logs to Git. Final seal/staging verification belongs to the release procedure, after the new loaded gates pass.

## API and shutdown checks

The portable API checker defaults to dry. CPU parser/redaction tests do not make network or model calls:

```powershell
& $Py scripts\check_api.py --self-test
& $Py scripts\check_api.py
# Explicit checks against your already-ready local server:
& $Py scripts\check_api.py --run --config .work\local\server.json --output .work\api-check-01.json
```

It requires authenticated models/metrics, missing-key refusals, loaded native 1M context and idle state, then natural arithmetic, complete SSE stop and `[DONE]`, both forced synthetic tool dialects (no execution), active-content disconnect, idle recovery and a natural followup. `--base` permits an explicitly selected HTTPS origin; inherited proxies and redirects are disabled. Credentials stay internal to the private configuration. The exclusive JSON receipt stores hashes, counts, usage, timing and validation, not plaintext prompts or response bodies. Stream metadata is recorded before assertions, and failures retain a bounded key-redacted reason. The [closed API05 test](../evidence/api05-qualified.json) passed these gates with the long-only preset below.

For this recipe's prescribed near-1M API qualification, use the public synthetic archive/question and the explicit long-only thinking preset:

```powershell
& $Py -c "from pathlib import Path; import runpy; m=runpy.run_path('bench/prepare_inputs.py'); Path('.work/api-1m-prompt.txt').write_text(m['long_prompt_text']('audit-sum'),encoding='utf-8',newline='')"
& $Py scripts\check_api.py --run --config .work\local\server.json --long-prompt .work\api-1m-prompt.txt --expected-long-answer 828639 --long-thinking-preset high-budget256 --deadline 1800 --output .work\api-long-check-01.json
```

The long request is FIRST inference after read-only readiness/authentication checks. It requires actual near-1M usage with zero cached tokens, exact integer content, natural stop and terminal SSE. The preset sends `reasoning_effort="high"` (DeepSeek effort level 75), `reasoning_budget_tokens=256` and `max_tokens=1024`. The effort level is distinct from the token budget. The prompt and expected answer are unchanged. This is a declared client request mode, not a configuration or model change; the installation's global 32,768-token reasoning budget stays intact. Natural arithmetic, small SSE and tool checks keep their original nonthinking settings, including the unchanged bounded disconnect request.

The checker's default long preset remains `none` with a 256-token total cap for compatibility. In attempt 04 that mode returned the correct arithmetic as an expression and failed the exact-integer contract. The failure is retained; it is not passed by accepting extra text. API05 with the declared preset passed: 999,797 uncached prompt tokens, 64 completion tokens, exact bare integer, natural stop and one `[DONE]`. `--expected-long-prompt-tokens N` can pin an exact API count; otherwise actual near-1M uncached usage is required. API rendering can differ from native frozen token IDs. Heartbeats and reasoning frames never count as emitted tokens. Use public synthetic text, not private conversations.

The runtime overlay sets module-level `ENGINE_QUIT_WAIT_S` to a 60-second QUIT wait, followed by unchanged 5/20-second terminate/kill waits. There is no JSON or native CLI setting for it. Allow normal large-RAM teardown to finish. A qualification run that needs escalation fails graceful-shutdown admission. The private decode close took 16.682 seconds; a later public attempt recorded a normal 20.432-second native wait. These traces do not prove the cause of the older untraced failure. The longer wait supplies teardown headroom.

Interpreter scope: fresh full-source build/packing/recipe setup and the API-observer campaign use 3.13.15. Historical private Stop and paired benchmark harnesses use 3.12.12. Keep each paired benchmark on the same interpreter and preserve its actual version in the receipt. Do not describe all earlier runtime results as Python 3.13 results.

## Reproduce the 230 CPU policy checks

There is no CMake target for this standalone test. In MSVC 14.44 x64 Developer PowerShell after source preparation, keep all output under ignored `.work`. `/UNDEBUG` keeps assertions active. These commands do not load a model or use CUDA:

```powershell
New-Item -ItemType Directory -Force .work\cpu-policy | Out-Null
cl.exe /nologo /std:c++17 /EHsc /O2 /UNDEBUG /I .work\source\strata\src\dsv41 .work\source\strata\tests\dsv41\gpu_miss_recovery_policy.cpp /Fo.work\cpu-policy\gpu_miss_recovery_policy.obj /Fe.work\cpu-policy\gpu_miss_recovery_policy.exe
if ($LASTEXITCODE -ne 0) { throw "CPU policy compilation failed; preserve the output." }
& .work\cpu-policy\gpu_miss_recovery_policy.exe
if ($LASTEXITCODE -ne 0) { throw "CPU policy test failed." }
```

Record the actual compiler/test return codes and check count. The retained development receipt reports 230 checks. This invocation does not substitute for CUDA numeric tests or loaded cancellation/1M gates.

Profile scope: a fresh recipe installation and its private API integration test start with the bundled 40-row profile (61,440 bytes), which DSpark can extend to 43 rows. Existing managed production keeps its own learned 43-row profile and receives a separate short API check. Never replace that learned profile with the benchmark or fresh-install initial profile. The numeric 66 label denotes a hash prefix, not 66 rows.

### Interpret the completed Q02 result

The [closed result](../evidence/qualification-v6b-q02.json) is deliberately two statements: raw report `passed=false`, status `measurement-complete-quality-failed`, 25/26 FIRST grades and command exit 1; separately, the predeclared first-eleven-plus-existing-v2 primary passed 24/24, both paired original-12 rates exceeded stock by 5%, and native phase/cleanup passed. Do not rewrite the raw report or call every test passed. Wave 2 retained the legacy attribute/dictionary contract failure; corrected-v2 passed 11/11 in both waves.

Original-12 rates were 38.4915/39.1636 versus 35.2192/35.3267 ID/s (+9.291%/+10.861%). TTFT was 329.164/330.404 versus 334.862/328.409 seconds; both metrics and both waves matter. The candidate used a fresh public-source native build and the same existing verified pack namespace as the paired control. Fresh-install API05 and managed verification subsequently passed in their separate scopes. Standalone runs on another machine have no automatic >=5% claim without an applicable paired control.

### Cancellation observer newline pitfall

The [first public cancellation attempt](../evidence/public-cancel-attempt01.json) completed its long request without sending STOP, so it failed cancellation admission. Its log reader decoded binary bytes without normalizing CRLF, unlike the private text reader. A strict readiness expression therefore never matched. The isolated fix normalizes CRLF to LF in that observer view only; CPU tests exercise the actual interrupt function with both newline forms. Preserve the underlying source and log bytes for hash checks. Do not globally rewrite pinned files or weaken the readiness expression. The [fresh actual rerun](../evidence/public-cancel-active02.json) passed Stop, the immediate FIRST code response, direct recovery, three 131K prefix reuses and normal owned cleanup. The first failure is not retroactively passed, and this selected-mode result remains separate from the subsequent API checks.

### API readiness is not inference or shutdown qualification

The [closed API attempts](../evidence/api-attempts01-03.json) show why these gates are separate. Attempts 01/02 reached native READY with 1M capacity, but their private observer had not independently held the native, so no controlled inference ran. Attempt 02 recorded Windows access denied when reopening the already-started native for a query. The observer fix receives the creator's restricted read handle before native resume and validates image, creation time and owned Job membership. It also uses that retained capability for library inspection. It does not change the model, public runtime, DACL or privileges. This private qualification instrumentation is not an extra step required to use the normal public `serve.py` and HTTP checker.

Attempt 03 passed those observer checks and normal native 0/owned cleanup, and processed an actual uncached 999,771-token API request. However, the FIRST stream failed an assertion and the old checker omitted its exact reason and parsed-answer metadata. The failure remains unresolved in that receipt. The updated checker records answer hash/length/exact-match, finish/DONE counts and numeric usage before the unchanged assertions. Attempt 04 then identified a formatting failure: a correct sum with extra expression text, while stop/DONE/usage and native cleanup passed. The declared thinking-mode API05 passed separately. Preserve every failed attempt; do not infer a complete API pass from a READY banner, HTTP 200, natural native exit or a successful numerical fixture.

The [managed default verification](../evidence/default-deployment-verified.json) later passed 15 local and 14 public checks; three public samples matched the local instance. It preserves the installation's existing bind policy, model alias/tag distinction, key, profile and tunnel configuration. It does not independently prove each connector or repeat the near-1M request on that managed process. The first verifier's pre-HTTP bind/tag assumption failure remains failed. On another installation, verify your own selected configuration and endpoints rather than copying a private deployment or assuming all bind addresses are loopback.

# Own the frontend and native process during API qualification

This optional Windows qualification starts a separate authenticated loopback
server on port 18080. Run it only after the normal server and native benchmarks
have stopped. It never stops an existing model or changes a tunnel. The generic
`check_api.py` checks HTTP behavior; this adapter additionally holds the original
native creation handle and requires actual native exit 0 and empty owned Job/V3.

Use the active Python 3.13 environment. Prepare the new native build, complete API
runtime, exact model pack and input first, then a separate fresh-profile config:

```powershell
& $Py scripts\configure.py --write --port 18080 --local-dir .work\local-api-qualify --cuda-root $env:CUDA_PATH
& $Py scripts\prepare_api_prompt.py --prepare
```

`configure.py` reuses `STRATA_API_KEY` internally if supplied; do not put a key on
the command line. Use a new config/profile directory, not a learned profile.
All execution and artifact pins are explicit:

```powershell
$cfg = '.work/local-api-qualify/server.json'
$build = '.work/build-native/BUILD.json'
$runtime = '.work/runtime/RECIPE-RUNTIME.json'
$source = 'pins/source-overlays.json'
$b = Get-Content -LiteralPath $build -Raw | ConvertFrom-Json
$sha = { param($p) (Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant() }
& $Py scripts/check_owned_api_v2.py --run `
  --config $cfg --expected-config-sha256 (& $sha $cfg) `
  --build-receipt $build --expected-build-receipt-sha256 (& $sha $build) `
  --source-manifest $source --expected-source-manifest-sha256 (& $sha $source) `
  --runtime-manifest $runtime --expected-runtime-manifest-sha256 (& $sha $runtime) `
  --expected-exe-sha256 $b.exe_sha256 `
  --expected-harness-manifest-sha256 (& $sha 'pins/owned-api-adapter-v2.json') `
  --long-prompt .work/api-first-long.txt --expected-long-answer 828639 `
  --long-thinking-preset high-budget256 --output .work/api-owned-qualification-01
```

Default without `--run` performs no process or network action. V2 requires the
four exact build recipe pins and repeat1/min32768/CPU16/noSMT/noOverlap, in
addition to complete raw current source, fresh libraries and complete runtime
proof. The original v1 source packet is preserved. No historical native result
is imported as a new result.

The FIRST input is the fixed public near-1M question with literal answer 828639;
the new run must observe its own uncached usage. Historical API05 usage 999797/64
and the updated runtime's private usage 999797/97 are separate observations, not
inherited results for another installation. Effort high/native 75 differs from
the 256-token reasoning budget. All short protocol checks remain unchanged.

Receipts and key-redacted logs stay under ignored `.work`. Do not publish raw
configurations or failure captures. The adapter proves the selected local native
identity, strict API behavior and natural cleanup; managed/public routing remains
a separate verification step.

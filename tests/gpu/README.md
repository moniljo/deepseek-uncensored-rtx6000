# Synthetic GPU regression fixture

This is the canonical-LF copy of the tested C++ source, including its actual
staging/registration helpers, controls and full-output bit comparisons. It uses
synthetic experts, all eight format families and T=1..4. No weights or API keys
are required. The inherited suite has132 numeric cases and64 lifetime guards;
the additive reserve cases are recorded separately in the JSON output.

After a fresh source build, from the repository root (MSVC14.44/CUDA13.2):

```powershell
python scripts/test_gpu.py --native-build .work/build-native/BUILD.json --out .work/gpu-test-01 --build
python scripts/test_gpu.py --out .work/gpu-test-01 --run
```

The first command freshly compiles the fixture main and links only the ten
native objects and four custom libraries from that fresh source build. CUDA
import libraries remain vendor binaries. The second command runs one bounded
synthetic process and retains numeric JSON and logs. To compare all inherited
output hashes, pass `--compare-numeric <previous-numeric.json>` to the run.
The driver is dry without explicit flags. Close other GPU workloads yourself
before running; this script never stops a model or server.

The real CUDA registration and stream tests use owned page-aligned test
domains. Worker epochs, pool restoration and47-buffer compaction in these
fixtures use clearly labelled synthetic adapters. Successful fixture output
does not replace loaded-model cancellation, repetition, quality or speed tests.

The pure actual-header recovery policy/ledger test can also be built from
public source in an x64 MSVC Developer PowerShell (no CUDA or weights):

```powershell
New-Item -ItemType Directory .work/policy-test-01
cl /std:c++20 /EHsc /O2 /MT /Z7 /Isource/strata/src/dsv41 source/strata/tests/dsv41/gpu_miss_recovery_policy.cpp /Fo:.work/policy-test-01/policy.obj /Fe:.work/policy-test-01/policy.exe
& .work/policy-test-01/policy.exe
```

V5 had162 checks; the current reserve/initial-cancellation policy version
has230 checks. These are pure policy/ledger callbacks, not a Model runtime.

Source licenses are retained under `source/`. The source manifest records
canonical-LF/raw reference hashes and the exact native model-source binding.

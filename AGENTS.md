# Reproduce the Mix-256G recipe

Read `README.md`, the source/model pins, `docs/experiments.md`, and the recorded
validation results before changing this recipe. The experiment history includes
rejected approaches; do not treat a source review or successful compile as an
inference benchmark.

## Required behavior

- Keep the exact seven OrcaRouter Mix-256G shards and separately pinned DSpark
  drafter. Verify complete-file hashes. The generated Q6 compatibility sidecars
  preserve every decoded F32 bit; they are not a smaller lossy quantization.
- Retain 1,048,576 **total** context tokens, all six routed trunk experts,
  DSpark block 3/confidence 0.65, and the recorded 16-thread preset. A nearly
  one-million-token input and a reused prefix are different measurements.
- Reconstruct the exact upstream commits and apply every locked overlay. The
  GPU-miss controls belong to this custom build. Setting them in an arbitrary
  official release does not reproduce the engine.
- Build from source with the recorded Windows/MSVC/CUDA/SM120a toolchain.
  Never import another machine's object files or libraries as a shortcut.
- Preserve the exact tested file bytes, including their newline forms. The
  `* -text` Git attribute prevents implicit conversion from breaking raw SHA
  pins. Verify staged and exported Git blobs against the package and source
  manifests; do not globally normalize files to LF or CRLF.
- Run the CPU codec/policy and CUDA fixture gates, then loaded cancellation,
  immediate short-request recovery, repeated cancellation, initial-request
  cancellation, and fresh/reused long-context checks. Keep first results,
  failed attempts, actual native Stop, and owned-process cleanup evidence.
- Preserve the original ambiguous balance question and its raw grade. The
  pre-existing dictionary-explicit version is the declared quality task for
  this recipe; disclose both. Never edit generated code or an oracle to repair
  a score, force output tokens, or combine selected retries into a passing run.
- Compute decode rate from actual emitted token-ID intervals, including
  reasoning and EOS. Report prompt processing, first-token latency, startup,
  and one-time recovery work separately. HTTP chunks are not token counts.

## Memory and operations

System RAM is a separate transfer domain, not additional VRAM. CUDA registration
consumes driver resources. Preserve the recovery headroom, exact cache-epoch
drains, physical unregister-before-maintenance, and current-slot validation.
Do not remove a failed admission guard to obtain a higher benchmark number.

Keep an existing installation until the replacement has passed its relevant
gates. Use dedicated work/config/profile directories and track owned process
identities or Windows Jobs. Never terminate every process with a matching name.
Do not compile or run competing inference while measuring decode throughput.
Do not silently change pagefile, firewall, administrator, or GPU power settings.

Private paths, API keys, profiles learned from a recipient's activity, model
payloads, binaries, build outputs and logs belong under ignored `.work/` or an
explicit external data directory. The public benchmark's declared initial
profile is separate reproducibility metadata; it is not a production profile.
Start the authenticated API on loopback. Configure remote access separately
for the recipient; never copy another installation's tunnel or credentials.

Before publication, run `scripts/check_package.py` on both the candidate files
and the Git staging area. Keep the source/evidence package within its size
budget. Publish compact outcome summaries and hashes, not raw client requests
or a whole installation directory.

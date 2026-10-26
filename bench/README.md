# Frozen native benchmark

These adapters reconstruct the public benchmark inputs and run two fresh
999,764-token prefixes in one owned Windows engine process. They do not call an
API or change a server configuration. Stop any running model server explicitly
before using `--run`; the adapter refuses a detected competing model process.

The corpus is a frozen, annotated public Python source snapshot, not an
importable Python distribution. Its deterministic gzip is about 690 KiB. The
original corpus, all thirteen full-input hashes, questions, bounded AST policy,
and eleven code vectors are unchanged. See [the corpus notice](data/CORPUS-NOTICE.md),
[oracle provenance](ORACLE-PROVENANCE.json), and
[helper provenance](SOURCE-PROVENANCE.json).

No million-token ID array or private request archive is stored in Git. Input
preparation writes one local little-endian prefix plus small suffix metadata
under the ignored `.work` directory. It verifies the corpus bytes, renderer,
reference and packed tokenizer parity, and every full input count/hash before
any model process can start.

## Prepare inputs

First follow the main recipe to check out the pinned source, prepare the Python
runtime, and construct the exact model pack. Pass the already-downloaded
**OrcaRouter Mix shard 1** as `--reference-gguf`. Only its GGUF vocabulary and
merge metadata are read. No XQ model or second set of weights is needed.
The optional original reference tokenizer JSON remains supported when already
available (SHA-256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`).

```powershell
python bench/prepare_inputs.py
python bench/prepare_inputs.py --prepare `
  --strata-source .work/runtime `
  --tokenizer-dir .work/data/packs/strata-mix256g-1m/tokenizer `
  --reference-gguf $MixShard1 `
  --output .work/bench-inputs
```

The first command is dry. The second performs CPU tokenization only. Do not run
input preparation concurrently with a timed model benchmark.

## Run two waves

Use the recipe's freshly built engine and its matching `BUILD.json`. Supply the
predeclared initial benchmark profile, SHA-256
`66b2d59fa7fa02cb6ab641846640a8e8c5fbc50906b8a1892140787e8ccf3428`.
The approved benchmark heat metadata is bundled in
`data/initial-routing-profile.json`; input preparation decodes it into the local
input directory. It contains no recipient's learned routing profile. The
adapter copies it once to its exclusive output directory. It never writes
the profile named in the input server configuration. Each run needs a new output
directory. The base configuration is read privately; API credentials are neither
printed nor saved in the benchmark report.

```powershell
python bench/run_native.py
python bench/run_native.py --run `
  --config .work/local/server.json `
  --build-receipt .work/build-native/BUILD.json `
  --reference-gguf $MixShard1 `
  --strata-source .work/runtime `
  --inputs .work/bench-inputs `
  --output .work/bench/run-01
```

The first command is dry. The actual run preserves the canonical pipe timing,
generation, independently held process identity, and Windows Job V3 cleanup
bodies. Native loading and tokenizer preparation are outside TTFT. Native rearm
work inside a submitted GEN remains inside TTFT. An unrelated graded short
request separates the waves; the second unchanged long prefix must report zero
actual reuse. All thirteen followups in each wave require natural EOS and exact
999,764-token prefix reuse. The adapter does not retry a failed result.

## Read results

The primary quality set was declared before the recovery campaign: original
first eleven tasks plus the already-frozen explicit-dictionary balance v2,
twelve per wave. The original balance question is still generated and graded.
Its raw verdict and the original thirteen-task score remain visible. No output
is transformed or regraded to make a failed legacy case pass.

The primary decode rate remains the **original twelve** tasks, excluding the
separate v2 case: `sum(actual emitted IDs - 1) / sum(last-ID time - first-ID
time)`. Reasoning, control and EOS IDs count. This differs from visible-answer
tokens/s and from the supplemental corrected-twelve rate. Report both waves,
their TTFTs, output lengths, quality scores, and cleanup result. Do not select a
faster wave or omit an original timing interval after seeing a grade.

This public adapter reproduces the performance and quality workload. It does
not certify cancellation recovery, every native registration/epoch invariant,
or deployment eligibility. Those require the separate loaded cancellation and
native lifecycle evidence. A completed source build alone is not a speed or
quality result. Generated `.work` reports and engine logs can contain local
paths; review and summarize them before publishing, rather than uploading raw
runtime directories.

For a separate normal API probe, `prepare_inputs.long_prompt_text('audit-sum')`
returns the checked public archive plus that unchanged question as user-message
text. It neither tokenizes nor sends a request. The API renders its own chat
wrapper; this is distinct from submitting the native bare-prefix seed IDs.

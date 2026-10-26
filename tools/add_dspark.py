"""CPU metadata-only DSpark addition; original Q6 trunk and payloads are read only.

Default dry. --run invokes the installed pack tool, then verifies/merges only
its mtp tensors into a separately staged pack. Never calls a GPU/engine tool.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / ".work/packing"
DATA = ROOT / ".work/data"
OLD = DATA / "packs/deepseek-orca-mix256g-q6compat"
FINAL = DATA / "packs/strata-mix256g-1m"
RAW = HERE / "raw-dspark-pack"
PINNED_SIZE = 7976914240
PINNED_SHA = "6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a"


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def stats(path):
    s = path.stat()
    return {"bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "file_id": s.st_ino}


def geometry(path):
    return dict(line.rstrip("\r\n").split(" ", 1) for line in path.read_text(encoding="utf-8").splitlines() if line)


def index(path):
    files, tensors, by_name = {}, [], {}
    for line in path.read_bytes().splitlines(keepends=True):
        words = line.decode("utf-8").split()
        if words[0] == "file":
            need(len(words) == 3 and int(words[1]) not in files, "Malformed/duplicate file record")
            files[int(words[1])] = {"path": words[2], "line": line}
        elif words[0] == "t":
            need(len(words) >= 7 and words[1] not in by_name, "Malformed/duplicate tensor record")
            tensor = {"name": words[1], "dtype": words[2], "file": int(words[3]), "offset": int(words[4]),
                      "bytes": int(words[5]), "shape": list(map(int, words[6:])), "line": line}
            by_name[tensor["name"]] = tensor
            tensors.append(tensor)
        else:
            raise RuntimeError("Unknown tensor index record")
    return files, tensors, by_name


def small_hashes(directory):
    return {str(path.relative_to(directory)).replace(os.sep, "/"): sha(path)
            for path in directory.rglob("*") if path.is_file() and path.stat().st_size < 16 << 20}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({"prepared_only": True, "old_pack": str(OLD), "new_pack": str(FINAL),
                          "raw_pack": str(RAW), "model_or_GPU_load": False, "giant_payload_copy": False}))
        return 0
    began = time.perf_counter()
    receipt_path = HERE / "dspark-pack-build.json"
    need(not receipt_path.exists() and not RAW.exists() and not FINAL.exists(), "Exclusive new raw/final pack and receipt required")
    staging = FINAL.with_name(FINAL.name + ".building-20261024")
    need(not staging.exists(), "Staging directory already exists")
    receipt = {"schema": "strata-q6compat-dspark-metadata-pack-v1", "status": "starting", "passed": False,
               "old_pack": str(OLD), "raw_pack": str(RAW), "new_pack": str(FINAL),
               "model_or_GPU_load": False, "weight_training_or_conversion": False, "giant_payload_copies": False}
    def save():
        temporary = receipt_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, receipt_path)
    save()
    try:
        download = read(HERE / "dspark-download.json")
        draft = Path(download["path"])
        need(download.get("verified") is True and download["SHA256"] == PINNED_SHA and download["bytes"] == PINNED_SIZE,
             "Download receipt does not match pinned drafter")
        need(draft.stat().st_size == PINNED_SIZE and sha(draft) == PINNED_SHA, "Pinned drafter file SHA/size mismatch")
        receipt["draft"] = {"path": str(draft), "bytes": PINNED_SIZE, "sha256": PINNED_SHA,
                            "independently_rehashed": True, "revision": download["revision"]}
        old_geo = geometry(OLD / "ds4.txt")
        model = Path(old_geo["model_dir"])
        model_before = {str(path): stats(path) for path in sorted(model.glob("*.gguf"))}
        need(len(model_before) == 7, "Expected original seven-shard model")
        old_small_before = small_hashes(OLD)
        old_payload_before = {str(path): stats(path) for path in (OLD / "q6compat").glob("*.bin")}
        need(len(old_payload_before) == 9, "Expected original nine Q6 payloads")
        frozen = HERE / "frozen-expert-profile.bin"
        frozen_sha = sha(frozen)
        tool = ROOT / "tools/pack_gguf_dspark.py"
        command = [sys.executable, "-B", str(tool), "--model", str(model), "--dspark-gguf", str(draft),
                   "--dspark-noise-token", "128799", "--dspark-markov-rank", "256", "--out", str(RAW)]
        receipt.update(status="packing-metadata", pack_tool_sha256=sha(tool), tool_arguments=command[3:],
                       original_small_file_hashes=old_small_before, original_payload_stats=old_payload_before,
                       original_gguf_stats=model_before, frozen_expert_profile_sha256=frozen_sha)
        save()
        clean_env = {name: value for name, value in os.environ.items() if name.upper() in
                     {"SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH", "COMSPEC", "USERPROFILE"}}
        with (HERE / "dspark-pack-tool.log").open("x", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, env=clean_env, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=300, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        need(result.returncode == 0, "CPU pack tool failed; original inputs untouched")
        raw_geo = geometry(RAW / "ds4.txt")
        nondraft = lambda geo: {k: v for k, v in geo.items() if not k.startswith("dspark_")}
        need(nondraft(raw_geo) == nondraft(old_geo), "Non-drafter geometry changed")
        need(raw_geo["model_dir"] == old_geo["model_dir"], "model_dir changed")
        for name in ("engram.bin",):
            need(sha(RAW / name) == sha(OLD / name), "Raw support bytes differ: " + name)
        raw_tokenizer, old_tokenizer = small_hashes(RAW / "tokenizer"), small_hashes(OLD / "tokenizer")
        need(raw_tokenizer == old_tokenizer, "Raw tokenizer differs from original")
        spec = importlib.util.spec_from_file_location("dspark_pack_metadata", tool)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gguf = module.Gguf(draft)
        need(gguf.kv.get("general.architecture") == "dflash", "Drafter architecture differs")
        blocks, dim, vocab = int(gguf.kv["dflash.block_count"]), int(old_geo["hidden_size"]), int(old_geo["vocab_size"])
        need(blocks == 3 and gguf.kv["dflash.embedding_length"] == dim, "Drafter layers/width differ")
        need(raw_geo["dspark_noise_token_id"] == "128799" and raw_geo["dspark_markov_rank"] == "256", "Noise/rank overrides differ")
        targets = list(map(int, raw_geo["dspark_target_layer_ids"].split(",")))
        need(targets == gguf.kv["dflash.target_layers"] == [37, 38, 39] and all(0 <= i < int(old_geo["num_hidden_layers"]) for i in targets), "Draft target layer metadata differs")
        need(int(raw_geo["dspark_block_size"]) == gguf.kv["dflash.block_size"] == 5, "Draft block size differs")
        need(int(raw_geo["dspark_n_routed_experts"]) == gguf.kv["dflash.expert_count"] == 128 and
             int(raw_geo["dspark_num_experts_per_tok"]) == gguf.kv["dflash.expert_used_count"] == 3, "Drafter routing geometry differs")
        old_files, old_tensors, old_names = index(OLD / "tensors.txt")
        raw_files, raw_tensors, raw_names = index(RAW / "tensors.txt")
        need(set(old_files) == set(range(16)) and not any(name.startswith("mtp.") for name in old_names), "Original file IDs/trunk differ")
        mtp = [t for t in raw_tensors if t["name"].startswith("mtp.")]
        need(mtp and {int(t["name"].split(".")[1]) for t in mtp} == set(range(blocks)), "Missing/unexpected MTP layer names")
        source_ids = {t["file"] for t in mtp}
        need(len(source_ids) == 1, "Drafter tensors span unexpected files")
        source_id = next(iter(source_ids))
        need((model / raw_files[source_id]["path"]).resolve() == draft.resolve(), "Draft file mapping differs")
        for layer in range(blocks):
            for role in ("w1", "w2", "w3"):
                names = {f"mtp.{layer}.ffn.experts.{e}.{role}.weight" for e in range(128)}
                need(names <= raw_names.keys(), "Missing drafter expert tensor role")
            need(f"mtp.{layer}.attn.wq_a.weight" in raw_names and f"mtp.{layer}.ffn_norm.weight" in raw_names, "Missing drafter dense/norm tensors")
        expected_heads = {"mtp.0.main_proj.weight": ("Q8_0", [dim, len(targets) * dim]),
                          "mtp.0.main_norm.weight": ("F32", [dim]), "mtp.2.norm.weight": ("F32", [dim]),
                          "mtp.2.confidence_head.proj.weight": ("F32", [1, dim + 256]),
                          "mtp.2.markov_head.embed.weight": ("BF16", [vocab, 256]),
                          "mtp.2.markov_head.head.weight": ("BF16", [vocab, 256])}
        for name, (dtype, shape) in expected_heads.items():
            need(name in raw_names and (raw_names[name]["dtype"], raw_names[name]["shape"]) == (dtype, shape), "Drafter head shape/type differs: " + name)
        staging.mkdir()
        shutil.copyfile(RAW / "ds4.txt", staging / "ds4.txt")
        shutil.copyfile(RAW / "engram.bin", staging / "engram.bin")
        shutil.copytree(RAW / "tokenizer", staging / "tokenizer")
        shutil.copyfile(frozen, staging / "expert-profile.bin")
        old_bytes = (OLD / "tensors.txt").read_bytes()
        newline = b"\r\n" if next(iter(old_files.values()))["line"].endswith(b"\r\n") else b"\n"
        relative = os.path.relpath(draft, model).replace(os.sep, "/")
        need(not any(c.isspace() for c in relative), "Draft relative filename has whitespace")
        draft_lines = []
        for tensor in mtp:
            fields = tensor["line"].rstrip(b"\r\n").split()
            fields[3] = b"16"
            draft_lines.append(b" ".join(fields) + newline)
        new_bytes = b"".join(old_files[i]["line"] for i in range(16)) + f"file 16 {relative}".encode() + newline + b"".join(t["line"] for t in old_tensors) + b"".join(draft_lines)
        (staging / "tensors.txt").write_bytes(new_bytes)
        files, tensors, names = index(staging / "tensors.txt")
        reconstructed_old = b"".join(line for line in new_bytes.splitlines(keepends=True)
                                      if not line.startswith(b"file 16 ") and not line.startswith(b"t mtp."))
        need(reconstructed_old == old_bytes, "An original file/tensor line changed")
        need(set(files) == set(range(17)) and len(names) == len(old_names) + len(mtp), "Final index counts differ")
        types = {name: (per, bpb) for name, per, bpb in module.TYPES.values()}
        bounds = {i: {"path": str((model / f["path"]).resolve()), "bytes": (model / f["path"]).stat().st_size, "tensors": 0, "max_end": 0} for i, f in files.items()}
        for tensor in tensors:
            need(tensor["file"] in files and tensor["offset"] >= 0 and tensor["bytes"] > 0 and all(d > 0 for d in tensor["shape"]), "Invalid tensor extent")
            info = bounds[tensor["file"]]
            end = tensor["offset"] + tensor["bytes"]
            need(end <= info["bytes"], "Tensor extends past referenced file: " + tensor["name"])
            per, bpb = types[tensor["dtype"]]
            count = math.prod(tensor["shape"])
            need(count % per == 0 and count // per * bpb == tensor["bytes"], "Tensor shape/quant byte accounting differs")
            info["tensors"] += 1
            info["max_end"] = max(info["max_end"], end)
        need(small_hashes(staging / "tokenizer") == old_tokenizer and sha(staging / "engram.bin") == sha(OLD / "engram.bin"), "Merged support bytes changed")
        need(sha(staging / "expert-profile.bin") == frozen_sha, "Expert profile differs from frozen baseline")
        need(small_hashes(OLD) == old_small_before and {str(Path(p)): stats(Path(p)) for p in old_payload_before} == old_payload_before, "Original pack changed")
        need({str(path): stats(path) for path in sorted(model.glob("*.gguf"))} == model_before, "Original GGUF stats changed")
        need(stats(draft)["bytes"] == PINNED_SIZE, "Draft size changed during build")
        payload = sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
        need(payload < 64 << 20 and not (staging / "q6compat").exists(), "Unexpected copied weight payload")
        receipt.update(status="metadata-verified-not-loaded", passed=True, original_trunk_file_records_byte_identical=True,
                       original_trunk_tensor_lines_byte_identical=True, original_index_reconstructed_sha256=sha(OLD / "tensors.txt"),
                       original_trunk_tensors=len(old_tensors), added_mtp_tensors=len(mtp), final_tensors=len(tensors),
                       new_draft_file_id=16, new_draft_relative_path=relative, referenced_file_bounds=bounds,
                       all_tensor_offset_and_quant_byte_bounds_passed=True, non_draft_geometry_identical=True,
                       geometry_changes={k: {"old": old_geo.get(k), "new": v} for k, v in raw_geo.items() if old_geo.get(k) != v},
                       tokenizer_sha256=old_tokenizer, engram_sha256=sha(staging / "engram.bin"),
                       original_pack_metadata_and_payload_stats_unchanged=True, original_GGUF_stats_unchanged=True,
                       draft_tensor_dtype_counts=dict(Counter(t["dtype"] for t in mtp)), mtp_layer_ids=list(range(blocks)),
                       expected_head_shapes_and_types_verified=True, frozen_expert_profile_copy_sha256=frozen_sha,
                       final_metadata_bytes=payload, depends_on_original_q6_payload_paths=True)
        # This small receipt is the only additional file after payload accounting.
        (staging / "dspark-merge-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        need(not FINAL.exists(), "Final pack appeared during staging; refusing replacement")
        staging.rename(FINAL)
        receipt.update(status="complete-metadata-verified-not-loaded", passed=True, seconds=time.perf_counter() - began)
        save()
        print(json.dumps({"passed": True, "pack": str(FINAL), "trunk_tensors": len(old_tensors), "mtp_tensors": len(mtp),
                          "metadata_bytes": payload, "model_or_GPU_load": False}), flush=True)
        return 0
    except BaseException as error:
        receipt.update(status="failed-originals-not-replaced", passed=False, error_type=type(error).__name__, error=str(error) if isinstance(error, RuntimeError) else "Inspect the metadata build log", seconds=time.perf_counter() - began)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())

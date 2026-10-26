"""Fresh pack-only MXFP4 -> Q6_K compatibility transform, with exhaustive value audit.

Original GGUF files are opened only for reading. This is NOT an ordinary lossy
requantizer. It accepts only exactly representable blocks and publishes atomically.
The stock engine still needs the independently tested Q6_K expert dispatch patch.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PACK_TOOL = HERE / "pack_gguf_trunk.py"
MODEL = ROOT / ".work/data/models/Mix-256G"
TARGET = ROOT / ".work/data/packs/deepseek-orca-mix256g-q6compat"
LUT = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.int16)
ROLE = {"gate": "w1", "up": "w3", "down": "w2"}
NATIVE_DLL = Path(os.environ.get("STRATA_Q6_CODEC", ROOT / ".work/native/lossless_q6_codec.dll"))
NATIVE = ctypes.CDLL(str(NATIVE_DLL)) if NATIVE_DLL.is_file() else None
if NATIVE is not None:
    NATIVE.convert_mx_q6.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p, ctypes.c_size_t]
    NATIVE.convert_mx_q6.restype = ctypes.c_int


def native_encode_audit(mx):
    out = np.empty((len(mx), 210), dtype=np.uint8)
    audited = ctypes.c_ulonglong()
    error = ctypes.create_string_buffer(256)
    status = NATIVE.convert_mx_q6(ctypes.c_void_p(mx.ctypes.data), ctypes.c_void_p(out.ctypes.data), len(mx),
                                  ctypes.byref(audited), error, len(error))
    if status:
        raise RuntimeError(error.value.decode("utf-8", "replace"))
    return out, audited.value


def encode_exact(mx):
    """N groups of eight consecutive MX32 blocks -> N Q6_K blocks."""
    if mx.dtype != np.uint8 or mx.ndim != 3 or mx.shape[1:] != (8, 17):
        raise ValueError("expected N x 8 x 17 MXFP4 bytes")
    exponent = mx[:, :, 0].astype(np.int16)
    lo = exponent.min(axis=1)
    delta = exponent - lo[:, None]
    if np.any(lo < 104) or np.any(lo > 143) or np.any(delta > 6):
        raise ValueError("MXFP4 group is outside the proven lossless Q6_K subset")
    scale = np.ldexp(np.ones(len(mx), dtype=np.float32), lo - 128).astype("<f2")
    if not np.array_equal(scale.astype(np.float32), np.ldexp(np.ones(len(mx), dtype=np.float32), lo - 128)):
        raise ValueError("F16 scale not exact")
    # MXFP4 packs values 0..15 in low nibbles and 16..31 in high nibbles.
    packed = mx[:, :, 1:]
    q = np.concatenate((LUT[packed & 15], LUT[packed >> 4]), axis=2).reshape(-1, 256)
    q = (q + 32).astype(np.uint8)
    out = np.empty((len(mx), 210), dtype=np.uint8)
    for half in range(2):
        at = half * 128
        a, b, c, d = (q[:, at + k * 32:at + (k + 1) * 32] for k in range(4))
        out[:, half * 64:half * 64 + 32] = (a & 15) | ((c & 15) << 4)
        out[:, half * 64 + 32:half * 64 + 64] = (b & 15) | ((d & 15) << 4)
        out[:, 128 + half * 32:160 + half * 32] = ((a >> 4) | ((b >> 4) << 2) |
                                                               ((c >> 4) << 4) | ((d >> 4) << 6))
    out[:, 192:208] = np.repeat(np.left_shift(1, delta), 2, axis=1).astype(np.int8).view(np.uint8)
    out[:, 208:210] = scale.view(np.uint8).reshape(-1, 2)
    return out


def decode_q6_independent(block):
    """The ggml Q6_K reference layout; does not reuse encoder intermediates."""
    raw = np.ascontiguousarray(block)
    d = np.ascontiguousarray(raw[:, 208:210]).view("<f2").reshape(-1).astype(np.float32)
    scales = raw[:, 192:208].view(np.int8)
    y = np.empty((len(raw), 256), dtype=np.float32)
    for half in range(2):
        high = raw[:, 128 + half * 32:160 + half * 32]
        for quarter in range(4):
            low = raw[:, half * 64 + (quarter & 1) * 32:half * 64 + (quarter & 1) * 32 + 32]
            q = (((low >> (4 * (quarter >> 1))) & 15) |
                 (((high >> (2 * quarter)) & 3) << 4)).astype(np.int16) - 32
            pos = half * 128 + quarter * 32
            s = np.repeat(scales[:, pos // 16:pos // 16 + 2], 16, axis=1).astype(np.float32)
            y[:, pos:pos + 32] = (d[:, None] * s) * q.astype(np.float32)
    return y


def decode_mx_reference(mx):
    """Original codebook integer times its exact E8M0 half-scale."""
    exponent = mx[:, :, 0].astype(np.int16)
    scales = np.ldexp(np.ones(exponent.shape, dtype=np.float32), exponent - 128)
    codes = mx[:, :, 1:]
    values = np.concatenate((LUT[codes & 15], LUT[codes >> 4]), axis=2).astype(np.float32)
    return (values * scales[:, :, None]).reshape(-1, 256)


def audit_bits(mx, q6):
    before = decode_mx_reference(mx)
    after = decode_q6_independent(q6)
    if not np.array_equal(before.view(np.uint32), after.view(np.uint32)):
        count = int(np.count_nonzero(before.view(np.uint32) != after.view(np.uint32)))
        raise RuntimeError(f"lossless audit failed: {count} different F32 bit patterns")
    return before.size


def codec_tests():
    rng = np.random.default_rng(20261024)
    mx = rng.integers(0, 256, size=(4096, 8, 17), dtype=np.uint8)
    bases = rng.integers(104, 137, size=(4096, 1), dtype=np.int16)
    mx[:, :, 0] = bases + rng.integers(0, 7, size=(4096, 8), dtype=np.int16)
    # Cover every nibble and exact F16 boundary, independent of random coverage.
    mx[:16, :, 0] = 104
    for i in range(16):
        mx[i, :, 1:] = i | (i << 4)
    q6 = encode_exact(mx)
    count = audit_bits(mx, q6)
    if NATIVE is not None:
        native, native_count = native_encode_audit(mx)
        assert native_count == count and np.array_equal(native, q6)
        assert audit_bits(mx, native) == count
    assert np.all(q6[:16, 208] == 1) and np.all(q6[:16, 209] == 0)
    for bad in (103, 144):
        trial = mx[:1].copy()
        trial[:, :, 0] = bad
        try:
            encode_exact(trial)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted nonrepresentable F16 exponent")
    trial = mx[:1].copy()
    trial[:, :, 0] = 116
    trial[:, 0, 0] = 123
    try:
        encode_exact(trial)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unproven scale spread")
    return {"passed": True, "audited_values": count, "all_16_codes": True,
            "f16_boundary": True, "rejects_invalid_groups": True,
            "compiled_codec_matches_independent_numpy": NATIVE is not None}


def sha_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def snapshot_sources(shards):
    return {str(g.path): {"bytes": g.path.stat().st_size, "mtime_ns": g.path.stat().st_mtime_ns} for g in shards}


def build():
    if TARGET.exists():
        raise FileExistsError("refusing to overwrite " + str(TARGET))
    spec = importlib.util.spec_from_file_location("verified_dsv41_pack", PACK_TOOL)
    pack = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pack)
    shards = [pack.Gguf(p) for p in sorted(MODEL.glob("*.gguf"))]
    if len(shards) != 7 or any(pack.short(g) for g in shards):
        raise RuntimeError("expected the seven complete model shards")
    before_stat = snapshot_sources(shards)
    roles = {}
    for g in shards:
        for name, typ, dims, off in g.tensors:
            m = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps.weight", name)
            if m:
                roles.setdefault(int(m[1]), {})[m[2]] = typ
    mixed = {n for n, r in roles.items() if 39 in r.values() and len(set(r.values())) > 1}
    if mixed != {15, 20, 23, 31, 33, 34, 37, 39}:
        raise RuntimeError("model mixed layers differ from audited input")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    stage = TARGET.with_name("." + TARGET.name + ".preparing-" + uuid.uuid4().hex[:12])
    stage.mkdir()
    payload = stage / "q6compat"
    payload.mkdir()
    started = time.monotonic()
    subprocess.run([sys.executable, "-B", str(PACK_TOOL), "--model", str(MODEL),
                    "--shards", *[g.path.name for g in shards], "--out", str(stage)], check=True)
    original_index = (stage / "tensors.txt").read_text(encoding="utf-8")
    (stage / "tensors.original.txt").write_text(original_index, encoding="utf-8", newline="\n")
    rows = original_index.splitlines()
    existing_files = [int(x.split()[1]) for x in rows if x.startswith("file ")]
    next_file = max(existing_files) + 1
    replacements, additions, reports = {}, [], []
    toy = codec_tests()
    for g in shards:
        for name, typ, dims, off in g.tensors:
            match = re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps.weight", name)
            if not match or int(match[1]) not in mixed or typ != 39:
                continue
            il, role = int(match[1]), match[2]
            values = 1
            for dim in dims:
                values *= dim
            groups = values // 256
            if values % 256 or dims[2] != 384:
                raise RuntimeError("unexpected expert tensor geometry")
            fname = f"layer-{il:02d}-{role}-q6_k.bin"
            part = payload / (fname + ".part")
            final = payload / fname
            source_hash, output_hash = hashlib.sha256(), hashlib.sha256()
            checked, done = 0, 0
            progress_at = 0
            with g.path.open("rb", buffering=0) as src, part.open("xb", buffering=0) as dst:
                src.seek(g.data_start + off)
                while done < groups:
                    ng = min(8192, groups - done)
                    raw = src.read(ng * 8 * 17)
                    if len(raw) != ng * 8 * 17:
                        raise RuntimeError("short source payload")
                    mx = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 8, 17)
                    if NATIVE is None:
                        encoded = encode_exact(mx)
                        checked += audit_bits(mx, encoded)
                    else:
                        encoded, audited_now = native_encode_audit(mx)
                        checked += audited_now
                    out = encoded.tobytes()
                    if dst.write(out) != len(out):
                        raise RuntimeError("short payload write")
                    source_hash.update(raw)
                    output_hash.update(out)
                    done += ng
                    progress = int(done * 100 / groups)
                    if progress >= progress_at:
                        print(json.dumps({"tensor": name, "percent": progress, "audited_values": checked}), flush=True)
                        progress_at = progress + 10
                os.fsync(dst.fileno())
            os.replace(part, final)
            new_bytes = groups * 210
            if final.stat().st_size != new_bytes:
                raise RuntimeError("payload size mismatch")
            fi = next_file
            next_file += 1
            reference = os.path.relpath(TARGET / "q6compat" / fname, MODEL).replace(os.sep, "/")
            if " " in reference:
                raise RuntimeError("space in pack file reference")
            additions.append(f"file {fi} {reference}")
            nrow, ncol = int(dims[1]), int(dims[0])
            per_expert = nrow * ncol // 256 * 210
            matched = 0
            for i, line in enumerate(rows):
                fields = line.split()
                if fields[:1] != ["t"]:
                    continue
                exp = re.fullmatch(rf"layers\.{il}\.ffn\.experts\.(\d+)\.{ROLE[role]}\.weight", fields[1])
                if not exp:
                    continue
                e = int(exp[1])
                if fields[2] != "MXFP4" or list(map(int, fields[6:])) != [nrow, ncol]:
                    raise RuntimeError("original pack role disagrees with source")
                replacements[i] = " ".join(["t", fields[1], "Q6_K", str(fi), str(e * per_expert),
                                             str(per_expert), str(nrow), str(ncol)])
                matched += 1
            if matched != 384:
                raise RuntimeError("did not replace exactly384 experts")
            report = {"tensor": name, "layer": il, "role": role, "source_file": str(g.path),
                      "source_offset": g.data_start + off, "source_bytes": groups * 136,
                      "source_payload_sha256": source_hash.hexdigest(), "converted_file": "q6compat/" + fname,
                      "converted_bytes": new_bytes, "converted_sha256": output_hash.hexdigest(),
                      "experts": matched, "audited_values": checked, "different_f32_bits": 0, "max_abs_error": 0.0}
            reports.append(report)
            print(json.dumps({"completed_tensor": name, "converted_bytes": new_bytes,
                              "audited_values": checked, "different_f32_bits": 0}), flush=True)
    if len(reports) != 9 or len(replacements) != 9 * 384:
        raise RuntimeError("unexpected conversion coverage")
    split = next(i for i, line in enumerate(rows) if line.startswith("t "))
    new_rows = rows[:split] + additions + [replacements.get(i, line) for i, line in enumerate(rows[split:], split)]
    new_index = "\n".join(new_rows) + "\n"
    index_part = stage / "tensors.txt.part"
    with index_part.open("w", encoding="utf-8", newline="\n") as f:
        f.write(new_index)
        f.flush()
        os.fsync(f.fileno())
    os.replace(index_part, stage / "tensors.txt")
    if before_stat != snapshot_sources(shards):
        raise RuntimeError("original source stat changed during conversion")
    unchanged = {str(p.relative_to(stage)).replace(os.sep, "/"): sha_file(p)
                 for p in stage.rglob("*") if p.is_file() and not p.is_relative_to(payload)
                 and p.name not in ("tensors.txt", "tensors.original.txt")}
    receipt = {"schema": 1, "kind": "lossless_mxfp4_to_q6_k_pack_compatibility",
               "warning": "Requires separately validated Q6_K expert admission/CUDA dispatch support; not stock engine compatible.",
               "source_model": str(MODEL), "target_pack": str(TARGET), "original_ggufs_unchanged": True,
               "source_stats": before_stat, "pack_tool": str(PACK_TOOL), "pack_tool_sha256": sha_file(PACK_TOOL),
               "codec_test": toy, "complete_value_bit_audit": True,
               "full_audit_implementation": "Independent MX and Q6 decoders in compiled codec, tested against independent NumPy codec",
               "codec_dll_sha256": sha_file(NATIVE_DLL) if NATIVE is not None else None,
               "codec_source_sha256": sha_file(HERE / "lossless_q6_codec.cpp"),
               "audited_values": sum(x["audited_values"] for x in reports), "different_f32_bits": 0,
               "max_abs_error": 0.0, "source_payload_bytes": sum(x["source_bytes"] for x in reports),
               "converted_payload_bytes": sum(x["converted_bytes"] for x in reports),
               "replaced_expert_entries": len(replacements),
               "original_index_sha256": hashlib.sha256(original_index.encode()).hexdigest(),
               "converted_index_sha256": hashlib.sha256(new_index.encode()).hexdigest(),
               "unchanged_geometry_engram_tokenizer_sha256": unchanged, "tensors": reports,
               "seconds": time.monotonic() - started}
    (stage / "q6compat-manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if TARGET.exists():
        raise FileExistsError("target appeared during conversion")
    os.replace(stage, TARGET)
    (HERE / "lossless-mix-q6-pack-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"published": str(TARGET), "audited_values": receipt["audited_values"],
                      "different_f32_bits": 0, "seconds": receipt["seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    else:
        print(json.dumps(codec_tests()))

"""Exact canonical timing, generation and owned-lifecycle blocks; see SOURCE-PROVENANCE.json."""
import ctypes
from ctypes import wintypes as W
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import threading
import time
CONTEXT = 1048576
PREFIX_N = 999764
MAXIMUM = 4096

def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def idsha(ids):
    return hashlib.sha256(struct.pack("<" + "I" * len(ids), *ids)).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_done(server, line):
    """Use this release's parser for DeepSeek's short DONE and Qwen's longer DONE."""
    holder = object.__new__(server.StrataEngine)
    server.StrataEngine._parse_done(holder, line)
    done = dict(holder.last)
    for key in ("generated", "prompt_tokens", "reused"):
        need(type(done.get(key)) is int and done[key] >= 0, "Missing/invalid native DONE accounting: " + key)
    for key in ("prompt_ms", "decode_ms"):
        need(type(done.get(key)) in (int, float) and math.isfinite(done[key]) and done[key] >= 0,
             "Missing/invalid native DONE timing: " + key)
    need(done["reused"] <= done["prompt_tokens"], "Native reused count exceeds prompt")
    return done


def replace_option(args, key, value=None):
    result = []
    i = 0
    while i < len(args):
        if args[i] == key:
            need(i + 1 < len(args), "Option has no value: " + key)
            i += 2
        elif str(args[i]).startswith(key + "="):
            i += 1
        else:
            result.append(str(args[i]))
            i += 1
    return result if value is None else result + [key, str(value)]


class DeadlineExpired(RuntimeError):
    pass


class Resources:
    """Bounded one-second sampler; file-backed RSS, physical RAM and commit stay separate."""
    def __init__(self, process, output, stage, job, native, kernel, maximum_seconds):
        self.process, self.output, self.stage = process, output, stage
        self.job, self.native, self.kernel = job, native, kernel
        self.maximum_seconds = maximum_seconds
        self.end = threading.Event()
        self.samples, self.peaks, self.error = 0, {}, None
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        import psutil
        class Performance(ctypes.Structure):
            _fields_ = [("cb", W.DWORD), *[(name, ctypes.c_size_t) for name in
                        ("CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                         "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")],
                        ("HandleCount", W.DWORD), ("ProcessCount", W.DWORD), ("ThreadCount", W.DWORD)]
        api = ctypes.WinDLL("psapi", use_last_error=True)
        api.GetPerformanceInfo.argtypes = [ctypes.POINTER(Performance), W.DWORD]
        api.GetPerformanceInfo.restype = W.BOOL
        process = psutil.Process(self.process.pid)
        created = process.create_time()
        process.cpu_percent(None)
        psutil.cpu_percent(None)
        nv = None
        try:
            import pynvml as nv
            nv.nvmlInit()
            gpu = nv.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            nv = None
        began = time.monotonic()
        try:
            with self.output.open("x", encoding="utf-8") as stream:
                while not self.end.is_set() and time.monotonic() - began <= self.maximum_seconds:
                    try:
                        need(process.create_time() == created, "Sampler PID identity changed")
                        mem, ram = process.memory_info(), psutil.virtual_memory()
                        row = {"monotonic": time.monotonic(), "stage": self.stage(), "owned_pid": process.pid,
                               "owned_rss_bytes": mem.rss, "owned_vms_bytes": mem.vms,
                               "physical_available_bytes": ram.available, "physical_total_bytes": ram.total,
                               "owned_io": process.io_counters()._asdict(),
                               "system_disk_io": psutil.disk_io_counters()._asdict(),
                               "owned_cpu_percent": process.cpu_percent(None) if self.samples else None,
                               "system_global_cpu_percent": psutil.cpu_percent(None) if self.samples else None}
                        try:
                            members = self.native.members(self.kernel, self.job.handle) if self.job.handle else []
                            row["owned_job_member_count"] = len(members)
                            row["owned_job_rss_bytes"] = sum(psutil.Process(item["pid"]).memory_info().rss for item in members)
                        except Exception:
                            row["owned_job_membership_unavailable"] = True
                        perf = Performance()
                        perf.cb = ctypes.sizeof(perf)
                        valid = bool(api.GetPerformanceInfo(ctypes.byref(perf), perf.cb))
                        row["commit_counter_valid"] = valid
                        if valid:
                            row.update(system_commit_bytes=perf.CommitTotal * perf.PageSize,
                                       system_commit_limit_bytes=perf.CommitLimit * perf.PageSize)
                        if nv:
                            memory = nv.nvmlDeviceGetMemoryInfo(gpu)
                            row.update(device_global_vram_bytes=memory.used, device_global_vram_free_bytes=memory.free,
                                       device_global_vram_total_bytes=memory.total)
                            try:
                                row.update(device_global_power_mw=nv.nvmlDeviceGetPowerUsage(gpu),
                                           device_global_gpu_utilization_percent=nv.nvmlDeviceGetUtilizationRates(gpu).gpu,
                                           device_global_temperature_c=nv.nvmlDeviceGetTemperature(gpu, nv.NVML_TEMPERATURE_GPU))
                            except Exception:
                                row["gpu_auxiliary_counters_unavailable"] = True
                        for key in ("owned_rss_bytes", "owned_job_rss_bytes", "owned_vms_bytes", "system_commit_bytes", "device_global_vram_bytes"):
                            if key in row:
                                peak = self.peaks.setdefault(row["stage"], {})
                                peak[key] = max(peak.get(key, 0), row[key])
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
                        self.samples += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        break
                    self.end.wait(1)
        except BaseException as error:
            self.error = type(error).__name__
        finally:
            if nv:
                nv.nvmlShutdown()

    def close(self):
        self.end.set()
        self.thread.join(5)
        return {"samples": self.samples, "sampler_stopped": not self.thread.is_alive(), "error_type": self.error,
                "peaks_by_stage": self.peaks, "interval_seconds": 1, "maximum_seconds": self.maximum_seconds,
                "device_and_system_counters_are_global_not_exclusive": True,
                "commit_counter_is_GetPerformanceInfo_pages_times_PageSize_not_physical_RAM": True}


class Measurement:
    """Request timing independent of engine yields: None heartbeats never count."""
    def __init__(self, engine, output, kill_owned, server, prefix_tokens=PREFIX_N):
        self.engine, self.output, self.kill_owned, self.server = engine, output, kill_owned, server
        self.prefix_tokens = prefix_tokens

    def generate(self, name, ids, maximum, deadline_seconds, reused):
        tokens, stamps = [], []
        lock, expired = threading.Lock(), threading.Event()
        input_hash = idsha(ids)
        cancel = threading.Event()
        began = time.perf_counter()
        sampling = {"temperature": 0, "seed": 12345, "presence_penalty": 0,
                    "frequency_penalty": 0, "repetition_penalty": 1,
                    "strata_checkpoint": True, "strata_prefix": {"tokens": self.prefix_tokens}}
        def expire():
            expired.set()
            with lock:
                count, first = len(tokens), stamps[0] - began if stamps else None
            receipt = {"schema": "strata-frozen-incomplete-v1", "case": name,
                       "request_deadline_seconds": deadline_seconds, "elapsed_seconds": time.perf_counter() - began,
                       "actual_input_tokens": len(ids), "actual_input_ids_sha256": input_hash,
                       "first_actual_ID_seen": count > 0, "first_actual_ID_elapsed_seconds": first,
                       "emitted_ID_count": count, "timer_fired": True, "native_final_received": False,
                       "non_target_probe": self.prefix_tokens != PREFIX_N, "requested_prefix_tokens": self.prefix_tokens,
                       "ttft_at_least_deadline": not count, "credentials_or_raw_prompt_saved": False}
            # Persist before ending the owned child so a stuck drain cannot hide
            # the observed deadline. The caller never proceeds to another case.
            safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
            try:
                path = self.output / ("INCOMPLETE-" + safe_name + ".json")
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(receipt, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            finally:
                cancel.set()
                self.kill_owned()
        timer = threading.Timer(deadline_seconds, expire)
        timer.daemon = True
        timer.start()
        progress_stop = threading.Event()
        def progress():
            while not progress_stop.wait(10):
                with lock:
                    count = len(tokens)
                print(json.dumps({"stage": name, "elapsed_s": round(time.perf_counter() - began, 1),
                                  "actual_emitted_IDs": count, "prompt_progress": self.engine.progress}), flush=True)
        observer = threading.Thread(target=progress, daemon=True)
        observer.start()
        self.engine.last = {}
        self.engine.raw_done = None
        try:
            for token in self.engine.generate(ids, maximum, sampling, cancel):
                if expired.is_set():
                    break
                if token is None:
                    continue
                need(type(token) is int and token >= 0, "Invalid actual token ID")
                with lock:
                    need(len(tokens) < maximum, "Engine exceeded output cap")
                    tokens.append(token)
                    stamps.append(time.perf_counter())
            if expired.is_set():
                raise DeadlineExpired("Request deadline exceeded; inspect the INCOMPLETE receipt")
            need(self.engine.raw_done is not None, "No native DONE for this request")
            done = parse_done(self.server, self.engine.raw_done)
            need(done["generated"] == len(tokens) and done["prompt_tokens"] == len(ids), "Actual native ID/accounting mismatch")
            need(bool(tokens), "No actual emitted IDs")
            if reused == 0:
                need(done["reused"] == 0, "Fresh seed reused cached tokens")
            natural = tokens[-1] == 1 and done["finish"] == "stop"
            need(maximum == 1 or natural, "Natural EOS required; capped or cancelled answers fail")
            seconds = stamps[-1] - stamps[0] if len(stamps) > 1 else None
            native_seconds = done["decode_ms"] / 1000
            return {"name": name, "prompt_tokens": len(ids), "input_ids_sha256": input_hash,
                    "requested_output": maximum, "token_ids": tokens, "output_ids_sha256": idsha(tokens),
                    "native_generated_tokens": done["generated"], "emitted_token_ids": len(tokens),
                    "EOS_included_in_stream_IDs": tokens[-1] == 1, "natural_eos": natural,
                    "native_prompt_ms": done["prompt_ms"], "native_decode_ms": done["decode_ms"],
                    "reused_tokens": done["reused"], "requested_prefix_tokens": self.prefix_tokens,
                    "non_target_probe": self.prefix_tokens != PREFIX_N,
                    "prompt_read": done.get("prompt_read", len(ids) - done["reused"]),
                    "prompt_read_source": "native DONE" if "prompt_read" in done else "native prompt minus native reused",
                    "generation_ID_intervals_s": (len(tokens) - 1) / seconds if seconds and seconds > 0 else None,
                    "observed_generation_ID_intervals_s": (len(tokens) - 1) / seconds if seconds and seconds > 0 else None,
                    "native_generation_ID_intervals_s": (done["generated"] - 1) / native_seconds if native_seconds > 0 and done["generated"] > 1 else None,
                    "ttft_seconds": stamps[0] - began, "generation_seconds": seconds,
                    "wall_seconds": time.perf_counter() - began, "native_done": done,
                    "None_heartbeats_counted": False}
        except BaseException:
            if expired.is_set():
                raise DeadlineExpired("Request deadline exceeded; inspect the INCOMPLETE receipt") from None
            raise
        finally:
            timer.cancel()
            timer.join(timeout=5)
            progress_stop.set()
            observer.join(timeout=1)


class OwnedNative:
    def __init__(self, a, cfg, output, server, job_class, native):
        self.output, self.server, self.native = output, server, native
        self.kernel, self.job = native.windows(), job_class()
        from owned_windows import Ownership
        self.ownership = Ownership(self.kernel, self.job)
        self.engine, self.sampler = None, None
        self.processes, self.identities = [], []
        self.job_lock = threading.Lock()
        self.stage = "startup"
        self.closed = False
        self.cfg = cfg
        args = replace_option(cfg["args"], "--dspark", a.dspark)
        # --ssd-cache budgets the expert RAM store, not conversation caching.
        # Preserve its configured value/default. "auto" omits the prefill option
        # so the engine selects its actual default (-1), never atoi("auto")=0.
        args = replace_option(args, "--prefill-chunk", None if a.prefill_chunk == "auto" else a.prefill_chunk)
        need("--max-context" in args and args[args.index("--max-context") + 1] == str(CONTEXT), "Explicit 1048576 native context required")
        need("--pack" in args, "Native DeepSeek --pack is required")
        clean = {key: value for key, value in os.environ.items() if key.upper() in
                 {"COMSPEC", "HOMEDRIVE", "HOMEPATH", "NUMBER_OF_PROCESSORS", "PATH", "PATHEXT", "PROCESSOR_IDENTIFIER", "SYSTEMROOT", "TEMP", "TMP", "USERPROFILE", "WINDIR", "CUDA_PATH"}}
        from unittest import mock
        with mock.patch.dict(os.environ, clean, clear=True):
            env = server.child_env(cfg, offline=True)
        env.update(STRATA_DS4_SLOT_KEEP="0", STRATA_DS4_SLOT_CAP="0", STRATA_DS4_SLOT_GB="0",
                   STRATA_DS4_SLOT_DIR=str(output / "disabled-slots"), STRATA_DS4_AUTO_PIN="0",
                   STRATA_DS4_REUSE="1", CUDA_DEVICE_ORDER="PCI_BUS_ID")
        self.effective_args = args
        self.effective_env = {key: value for key, value in env.items() if key.startswith(("STRATA_", "DS4_", "CUDA_"))}
        save(output / "EFFECTIVE.json", {"native_args": args, "native_environment": self.effective_env,
                                        "prefill_auto_means_flag_omitted": a.prefill_chunk == "auto",
                                        "expert_RAM_store_budget_preserved": True,
                                        "conversation_disk_slots_disabled": True,
                                        "config_sha256": sha(a.config)})
        self.kernel.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)]
        self.kernel.OpenThread.argtypes = [W.DWORD, W.BOOL, W.DWORD]
        self.kernel.OpenThread.restype = W.HANDLE
        self.kernel.ResumeThread.argtypes = [W.HANDLE]
        self.kernel.ResumeThread.restype = W.DWORD
        def spawn(what, command, **kw):
            import psutil
            need(not self.processes, "Fresh benchmark must spawn exactly one engine")
            need(Path(command[0]).resolve() == Path(cfg["exe"]).resolve(), "Unexpected executable spawn")
            kw.pop("role", None)
            kw["creationflags"] = kw.get("creationflags", 0) | 0x08000000 | 0x00000004
            process = subprocess.Popen(command, **kw)
            self.processes.append(process)
            try:
                identity = self.ownership.attach(process, cfg["exe"], role="engine")
                self.identities.append(identity)
                save(output / "OWNERSHIP.json", {"processes": self.identities, "attached_suspended": True,
                                                "independent_SYNCHRONIZE_QUERY_handle_held_before_resume": True})
                threads = psutil.Process(process.pid).threads()
                need(len(threads) == 1, "Suspended engine has unexpected threads")
                thread = self.kernel.OpenThread(0x0002, False, threads[0].id)
                need(thread, "Cannot open owned primary thread")
                try:
                    need(self.kernel.ResumeThread(thread) == 1, "Cannot resume owned engine")
                finally:
                    self.kernel.CloseHandle(thread)
                self.sampler = Resources(process, output / "RESOURCES.jsonl", lambda: self.stage,
                                         self.job, native, self.kernel,
                                         a.startup_deadline + a.seed_deadline + (0 if a.seed_only else 13 * a.case_deadline) + 120)
                return process
            except BaseException:
                process.kill()
                process.wait(timeout=30)
                raise
        class Captured(server.StrataEngine):
            def _parse_done(self, line):
                self.raw_done = line
                super()._parse_done(line)
        try:
            # Own all process creation here; suppress generic leftover-process
            # discovery so this helper can never end another installation's PID.
            with mock.patch.object(server, "popen", spawn), mock.patch.object(server, "end_leftover_engines", lambda exe: None), \
                    mock.patch.object(server, "ENGINE_READY_S", a.startup_deadline), mock.patch.object(server, "START_RETRIES", 0):
                self.engine = Captured(str(Path(cfg["exe"]).resolve()), args, cwd=cfg.get("cwd"),
                                       log=str(output / "ENGINE.log"), env=env)
            need(self.engine.max_context == CONTEXT and not self.engine.batch, "Actual solo 1048576 READY required")
            # The request's wall timer is authoritative. A long GPU prefill must
            # not be mistaken for a frozen process by the generic CPU/I/O probe.
            server.ENGINE_STALL_S = 0
            self.engine.silence_s = 0
        except BaseException:
            self.close()
            raise
        self.measurement = Measurement(self.engine, output, self.kill_owned, server, a.probe_prefix_tokens or PREFIX_N)

    def kill_owned(self):
        with self.job_lock:
            self.ownership.terminate()

    def generate(self, *args):
        self.stage = "fresh-prefill" if args[0] == "fresh-prefix-initialization" else "followups"
        return self.measurement.generate(*args)

    def close(self):
        if self.closed:
            return
        self.closed = True
        graceful_error = None
        try:
            if self.engine is not None:
                self.engine.close()
        except BaseException as error:
            graceful_error = type(error).__name__
        finally:
            proof = self.ownership.finish(timeout=30)
            popen_rows = []
            for process in self.processes:
                row = {"pid": process.pid, "returncode": process.returncode}
                try:
                    process.wait(timeout=1)
                    row["returncode"] = process.returncode
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    row["wait_error_type"] = type(error).__name__
                popen_rows.append(row)
            self.cleanup = {**proof, "recorded_processes": self.identities,
                            "popen_metadata_not_exit_proof": popen_rows,
                            "graceful_engine_close_error_type": graceful_error,
                            "resources": self.sampler.close() if self.sampler else None}
            save(self.output / "CLEANUP.json", self.cleanup)
            need(proof["passed"], "Owned cleanup unproven; inspect CLEANUP.json")

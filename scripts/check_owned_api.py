"""Private default-dry owner for one fresh FIRST-long API integration test.

Only --run starts a frontend or sends HTTP. No production configuration, service,
tunnel, model payload, or canonical ownership source is changed. The owner runs this
only after its GPU qualification slot is free.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes as C
from ctypes import wintypes as W
import hashlib
import hmac
import importlib.util
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import threading
import time
import owned_api_handoff as H

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PUB = ROOT
BASE = 'http://127.0.0.1:18080'
RUNTIME = ROOT / '.work/runtime'

def need(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def launch_mac(key, raw_config, argv):
    encoded = json.dumps(argv, ensure_ascii=False, separators=(',', ':')).encode()
    message = (b'strata-owned-launch-v1\0' + len(raw_config).to_bytes(8, 'big') + raw_config
               + len(encoded).to_bytes(8, 'big') + encoded)
    return hmac.new(bytes.fromhex(key), message, hashlib.sha256).hexdigest()


def sanitized(value, secrets_to_hide):
    if isinstance(value, dict):
        return {name: ('[REDACTED]' if name.lower() in
                      {'authorization', 'api_key', 'owner_key', 'attestation_key'}
                      else sanitized(item, secrets_to_hide)) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitized(item, secrets_to_hide) for item in value]
    if isinstance(value, str):
        for secret in secrets_to_hide:
            if secret:
                value = value.replace(secret, '[REDACTED]')
    return value


def error_detail(error, secrets_to_hide=()):
    message = sanitized(str(error), secrets_to_hide).replace('\r', ' ').replace('\n', ' ')
    detail = {'error_type': type(error).__name__, 'message': message[:2048]}
    for name in ('winerror', 'errno'):
        value = getattr(error, name, None)
        if type(value) is int:
            detail[name] = value
    return detail


class Checkpoints:
    """Private bounded diagnostics; identities come only from the caller's Job."""
    def __init__(self, path, hidden, clock=time.monotonic):
        self.stream = Path(path).open('x', encoding='utf-8')
        self.hidden, self.clock, self.started = hidden, clock, clock()
        self.count, self.previous = 0, {}

    def record(self, stage, detail=None):
        detail = sanitized(detail, self.hidden())
        encoded = json.dumps(detail, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if self.previous.get(stage) == encoded:
            return
        need(self.count < 128, 'Bounded private checkpoint count exceeded')
        self.previous[stage] = encoded
        self.count += 1
        row = {'checkpoint': stage, 'elapsed_seconds': self.clock() - self.started,
               'detail': detail}
        self.stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.stream.flush()

    def close(self):
        self.stream.close()


def configure_kernel(k):
    k.OpenThread.argtypes, k.OpenThread.restype = [W.DWORD, W.BOOL, W.DWORD], W.HANDLE
    k.ResumeThread.argtypes, k.ResumeThread.restype = [W.HANDLE], W.DWORD
    k.GetExitCodeProcess.argtypes, k.GetExitCodeProcess.restype = [W.HANDLE, C.POINTER(W.DWORD)], W.BOOL


def resume_suspended(k, process):
    import psutil
    threads = psutil.Process(process.pid).threads()
    need(len(threads) == 1, 'Suspended frontend must have one primary thread')
    thread = k.OpenThread(0x0002, False, threads[0].id)
    need(thread, 'Cannot open exact owned primary thread')
    try:
        need(k.ResumeThread(thread) == 1, 'Cannot resume exact owned frontend')
    finally:
        k.CloseHandle(thread)


def hold_native(owner, kernel, identity_module, expected, observer=None):
    """Validate the received creator capability; never reopen its native PID."""
    known = [item for item in owner.held if item['role'] == 'native']
    need(len(known) <= 1, 'More than one native capability is held')
    if not known:
        return None  # Only authenticated Client.poll may publish the native.
    item = H.validate_item(kernel, owner, known[0], expected)
    pids = owner.pids()
    need(item['identity']['pid'] in pids, 'Held native is absent from its owned Job')
    members = [{'identity': item['identity'], 'expected_native_image': True,
                'identity_source': 'retained_creator_read_capability'}]
    matches = [item['identity']['pid']]
    for pid in pids:
        if pid == item['identity']['pid']:
            continue
        try:
            ident = identity_module.identity(kernel, pid)
        except RuntimeError as error:
            members.append({'pid': pid, 'identity_query_error': error_detail(error)})
            if observer is not None:
                observer({'owned_job_members': members, 'matching_native_pids': matches})
            need(pid not in owner.pids(), 'A live additional owned Job member cannot be identified')
            continue  # A rechecked, departed optional member is not a live image.
        matched = Path(ident['image']).resolve() == expected
        members.append({'identity': ident, 'expected_native_image': matched})
        if matched:
            matches.append(pid)
    if observer is not None:
        observer({'expected_native_image': str(expected), 'owned_job_members': members,
                  'matching_native_pids': matches,
                  'previously_held_native_pids': [item['identity']['pid']]})
    need(matches == [item['identity']['pid']], 'Originally held native must remain the sole native')
    return item


def native_exit(kernel, item, milliseconds):
    need(item is not None, 'No independently held native identity')
    result = int(kernel.WaitForSingleObject(item['handle'], milliseconds))
    need(result == 0, 'Held native has not exited naturally')
    code = W.DWORD()
    need(kernel.GetExitCodeProcess(item['handle'], C.byref(code)), 'Cannot read held native exit code')
    return {'held_handle_signaled': True, 'wait_result': result, 'native_returncode': int(code.value)}


def shutdown_owned(process, owner, kernel, native_item, seconds=120, clock=time.monotonic, sleep=time.sleep):
    trace = {'frontend_stdin_EOF_requested': False, 'timeout_seconds': seconds,
             'live_owned_job_forced_termination': False, 'natural_native0': False}
    began, end = clock(), clock() + seconds
    try:
        if process is not None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
                trace['frontend_stdin_EOF_requested'] = True
            trace['frontend_returncode'] = process.wait(timeout=max(.01, end - clock()))
            need(trace['frontend_returncode'] in (0, 130), 'Frontend exit is neither normal0 nor owner-pipe130')
        trace.update(native_exit(kernel, native_item, max(0, int((end - clock()) * 1000))))
        need(trace['native_returncode'] == 0, 'Native exit must be actual0')
        while owner.active() and clock() < end:
            sleep(min(.05, max(0, end - clock())))
        trace['owned_job_active_BEFORE_finish'] = owner.active()
        need(trace['owned_job_active_BEFORE_finish'] == 0, 'Owned Job must be empty before final cleanup')
        trace['natural_native0'] = True
    except BaseException as error:
        trace['failure_type'] = type(error).__name__
        trace['failure_reason'] = str(error)[:2048]
        try:
            live = owner.active()
            trace['owned_job_active_on_failure'] = live
            if live:
                trace['live_owned_job_forced_termination'] = True
                owner.terminate()  # Never a PID/global kill; this is a failed qualification.
        except BaseException as cleanup_error:
            trace['forced_cleanup_error_type'] = type(cleanup_error).__name__
    finally:
        trace['elapsed_seconds'] = clock() - began
        # Reuse unchanged canonical finish. Its TerminateJobObject on an already
        # proven EMPTY Job is bookkeeping, not escalation of a live process.
        trace['canonical_finish_called_after_proven_empty'] = trace.get('owned_job_active_BEFORE_finish') == 0
        try:
            trace['cleanup'] = owner.finish(timeout=30)
        except BaseException as error:
            trace['cleanup_error_type'] = type(error).__name__
            # The canonical finalizer normally returns errors as data. Retain
            # failed evidence and close only this private kill-on-close Job if
            # an unexpected exception prevents that finalizer from returning.
            with contextlib.suppress(BaseException):
                owner.job.close()
    return trace


def shutdown_admitted(trace):
    cleanup = trace.get('cleanup', {})
    checks = cleanup.get('separate_process_handle_checks', [])
    return (trace.get('natural_native0') is True and trace.get('native_returncode') == 0
            and trace.get('frontend_returncode') in (0, 130)
            and trace.get('owned_job_active_BEFORE_finish') == 0
            and trace.get('live_owned_job_forced_termination') is False
            and cleanup.get('cleanup_proof_version') == 3 and cleanup.get('passed') is True
            and cleanup.get('owned_job_closed') is True and cleanup.get('owned_job_empty_proven') is True
            and cleanup.get('owned_job_active_processes_before_close') == 0
            and cleanup.get('errors') == []
            and {'frontend', 'native'} <= {x.get('role') for x in checks}
            and all(x.get('separately_held_handle_signaled') is True and x.get('wait_result') == 0
                    and x.get('pid_absence_claimed') is False for x in checks))


def loaded_cublas(kernel, owner, native_item, expected):
    """Read modules through a local duplicate of the retained creator capability."""
    H.validate_item(kernel, owner, native_item, Path(native_item['identity']['image']).resolve())
    handle = H.duplicate_local(kernel, native_item['handle'])
    need(handle, 'Cannot duplicate retained native module-read capability')
    try:
        import owned_windows
        ident = owned_windows.identity_from_handle(kernel, handle, native_item['identity']['pid'])
        need(ident == native_item['identity'], 'Native identity changed before module readback')
        inside = W.BOOL()
        need(kernel.IsProcessInJob(handle, owner.job.handle, C.byref(inside)) and inside.value,
             'Module query target is outside private Job')
        psapi = C.WinDLL('psapi.dll', use_last_error=True)
        psapi.EnumProcessModulesEx.argtypes = [W.HANDLE, C.POINTER(W.HMODULE), W.DWORD, C.POINTER(W.DWORD), W.DWORD]
        psapi.EnumProcessModulesEx.restype = W.BOOL
        psapi.GetModuleFileNameExW.argtypes = [W.HANDLE, W.HMODULE, W.LPWSTR, W.DWORD]
        psapi.GetModuleFileNameExW.restype = W.DWORD
        modules, count = (W.HMODULE * 4096)(), W.DWORD()
        need(psapi.EnumProcessModulesEx(handle, modules, C.sizeof(modules), C.byref(count), 3),
             'Cannot enumerate owned native modules')
        need(count.value <= C.sizeof(modules), 'Unexpected native module count')
        found = {}
        for module in modules[:count.value // C.sizeof(W.HMODULE)]:
            text = C.create_unicode_buffer(32768)
            need(psapi.GetModuleFileNameExW(handle, module, text, len(text)), 'Cannot read owned module path')
            name = Path(text.value).name.lower()
            if name in expected:
                actual = Path(text.value).resolve()
                need(actual == expected[name], 'Loaded cuBLAS path differs from selected public toolkit')
                found[name] = str(actual)
        need(set(found) == set(expected), 'Both selected cuBLAS DLLs must actually be loaded')
        return found
    finally:
        kernel.CloseHandle(handle)


def install_failed_long_capture(api, output, report, prompt_sha256, hidden, writer=save_new, limit=2 << 20):
    """Private diagnostics for this exact public synthetic input, same consumer.

    No event is replayed. Successful responses are discarded. Only a failed
    FIRST-long parse may save bounded response events under the private output.
    """
    need(prompt_sha256 == '6c4f575406cf9303355867354ced548d7c4789830ff6f817ae1f8de83fd17c2a',
         'Private response capture permits only the fixed public synthetic input')
    original, used = api.stream_result, False
    def checked(values, expected, **kwargs):
        nonlocal used
        if not kwargs.get('require_long') or used:
            return original(values, expected, **kwargs)
        used = True
        records, total, truncated = [], 0, False
        def observed():
            nonlocal total, truncated
            for event in values:
                if not truncated:
                    try:
                        safe = sanitized(event, hidden())
                        raw = json.dumps(safe, ensure_ascii=False, allow_nan=False).encode()
                        if total + len(raw) > limit or len(records) >= 4096:
                            truncated = True
                        else:
                            total += len(raw); records.append(safe)
                    except Exception:
                        truncated = True  # Diagnostics never replace parser validation.
                yield event
        try:
            return original(observed(), expected, **kwargs)
        except BaseException:
            payload = {'schema':'private-synthetic-FIRST-long-failure-events-v1',
                       'public_synthetic_input_sha256':prompt_sha256,
                       'response_events':records,'event_count':len(records),
                       'serialized_event_bytes':total,'byte_limit':limit,
                       'capture_truncated':truncated,'prompt_saved':False,
                       'private_only_never_publish':True,'same_original_consumer':True}
            try:
                writer(Path(output) / 'FIRST-LONG-SSE-FAILED-PRIVATE.json', payload)
                report['private_synthetic_failure_events'] = {
                    'saved':True,'event_count':len(records),'event_bytes':total,
                    'capture_truncated':truncated,'private_only_never_publish':True}
            except Exception as error:
                report['private_synthetic_failure_capture_error'] = error_detail(error, hidden())
            raise
    api.stream_result = checked


def redacted_log(pipe, destination, secrets_to_hide, result):
    """Private file only; preserve enough carry to redact keys split across reads."""
    keys = [key.encode() for key in secrets_to_hide if key]
    carry = max([len(key) for key in keys] + [1])
    pending = b''
    try:
        with destination.open('xb') as f:
            while True:
                block = pipe.read(4096)
                if not block:
                    break
                pending += block
                for key in keys:
                    pending = pending.replace(key, b'[REDACTED]')
                if len(pending) > carry:
                    f.write(pending[:-carry]); pending = pending[-carry:]
            for key in keys:
                pending = pending.replace(key, b'[REDACTED]')
            f.write(pending)
        result['reader_finished'] = True
    except BaseException as error:
        result['reader_error_type'] = type(error).__name__


def pack_metadata(pack, model_pins):
    geometry = dict(line.split(' ', 1) for line in (pack / 'ds4.txt').read_text().splitlines() if ' ' in line)
    model_dir = Path(geometry['model_dir'])
    paths = []
    for line in (pack / 'tensors.txt').read_text().splitlines():
        if line.startswith('file '):
            words = line.split()
            need(len(words) == 3, 'Unexpected model file record')
            paths.append((model_dir / words[2]).resolve())
    expected = {Path(x['file']).name: x['bytes'] for x in model_pins['files']}
    expected.update({x['converted_file']: x['converted_bytes'] for x in model_pins['compatibility_files']})
    need(len(paths) == 17 and len({x.name for x in paths}) == 17 and {x.name for x in paths} == set(expected),
         'Exact17 original/drafter/lossless file mapping required')
    for path in paths:
        need(path.is_file() and path.stat().st_size == expected[path.name], 'Prepared model payload size differs')
    merged = read(pack / 'dspark-merge-receipt.json')
    need(merged.get('passed') is True and merged.get('original_trunk_tensor_lines_byte_identical') is True
         and merged.get('all_tensor_offset_and_quant_byte_bounds_passed') is True,
         'Completed exact metadata merge receipt required')
    return {str(path): {'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns} for path in paths}


def preparation(a):
    import owned_api_admission as A
    return A.prepare(a, HERE, ROOT, need, digest, read, pack_metadata)


def run(a):
    need(os.name == 'nt' and sys.version_info[:2] == (3, 13), 'Run this integration with the selected Python3.13')
    output = a.output.resolve()
    need(output.is_relative_to(ROOT / '.work') and not output.exists(), 'Use an exclusive new private output directory')
    output.mkdir()
    report = {'schema': 'owned-portable-v027-API-integration-v1', 'passed': False, 'finished': False,
              'no_production_service_or_tunnel_changes': True, 'model_payloads_rehashed': False,
              'private_credentials_printed': False, 'single_frontend_attempt': True}
    proc = owner = job = kernel = native_item = None
    reader = None; log_result = {}; cfg = {}; pins = {}; payload_stats = {}; owner_key = ''
    trace = Checkpoints(output / 'CHECKPOINTS.jsonl', lambda: [cfg.get('api_key', ''), owner_key])
    rc = 1; began = time.monotonic()
    def checkpoint(name, detail=None):
        report['last_checkpoint'] = name
        trace.record(name, detail)
    def observed_members(value):
        report['last_owned_job_members'] = value
        checkpoint('owned_job_member_identity_snapshot', value)
    def cleanup_checkpoint(name, detail=None):
        nonlocal rc
        try:
            checkpoint(name, detail)
        except BaseException as error:
            report['checkpoint_failure'] = error_detail(error, [cfg.get('api_key', ''), owner_key])
            rc = 1
    try:
        checkpoint('preparation_entered')
        cfg, runtime, engine, pins, payload_stats, expected_dlls = preparation(a)
        checkpoint('preparation_completed', {'expected_native_image': str(engine)})
        save_new(output / 'PINS-BEFORE.json', pins)
        save_new(output / 'PACK-STATS-BEFORE.json', payload_stats)
        with socket.socket() as sock:
            sock.settimeout(1)
            need(sock.connect_ex(('127.0.0.1', 18080)) != 0, 'Port18080 is already listening; no existing server is touched')
        checkpoint('exclusive_port_preflight_passed')
        sys.path.insert(0, str(PUB / 'bench'))
        import job_support, owned_windows, identity_support
        kernel = identity_support.windows(); configure_kernel(kernel)
        job = job_support.OwnedJob(); owner = owned_windows.Ownership(kernel, job)
        checkpoint('owned_job_created')
        owner_key = secrets.token_hex(32)
        server_args = ['--engine', 'strata', '--config', str(a.config.resolve()), '--host', '127.0.0.1',
                       '--port', '18080', '--owner-fd', '0', '--shutdown-timeout', '0']
        handoff = H.Client(output / 'NATIVE-HANDOFF', owner_key, engine, owner, kernel,
                           {a.python.resolve(), Path(sys._base_executable).resolve()})
        command = [str(a.python.resolve()), str(HERE / 'owned_api_frontend.py'),
                   str(runtime)] + server_args
        allowed = {'COMSPEC', 'HOMEDRIVE', 'HOMEPATH', 'NUMBER_OF_PROCESSORS', 'PATH', 'PATHEXT',
                   'PROCESSOR_IDENTIFIER', 'SYSTEMROOT', 'TEMP', 'TMP', 'USERPROFILE', 'WINDIR', 'CUDA_PATH'}
        env = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        env.update(STRATA_OWNER_ATTEST_KEY=owner_key, STRATA_KEEP_LEFTOVERS='1',
                   STRATA_START_RETRIES='0', PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        env.update(handoff.environment())
        proc = subprocess.Popen(command, cwd=runtime, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, creationflags=0x08000000 | 0x00000004)
        try:
            frontend_ident = owner.attach(proc, a.python.resolve(), role='frontend')
            save_new(output / 'OWNERSHIP.json', {'frontend': frontend_ident, 'attached_suspended': True,
                     'held_before_resume': True, 'no_global_process_discovery_or_kill': True})
            checkpoint('frontend_attached_suspended', {'identity': frontend_ident})
            reader = threading.Thread(target=redacted_log, args=(proc.stdout, output / 'FRONTEND.log',
                                      [cfg['api_key'], owner_key], log_result), daemon=True)
            reader.start(); resume_suspended(kernel, proc)
            checkpoint('frontend_resumed')
        except BaseException:
            # Exact newly-created suspended Popen only; never a discovered PID.
            # Any such constructor failure remains a failed qualification.
            proc.kill(); proc.wait(timeout=30)
            report['constructor_failed_exact_created_child_ended'] = True
            raise
        expected_mac = launch_mac(owner_key, a.config.read_bytes(), server_args)
        checkpoint('launch_attestation_expected')
        api = load(PUB / 'scripts/check_api.py', 'owned_integration_public_API')
        import requests
        with requests.Session() as session:
            session.trust_env = False
            session.headers['Authorization'] = 'Bearer ' + cfg['api_key']
            until, polls, ready = time.monotonic() + a.startup_deadline, 0, False
            def startup_tick():
                need(time.monotonic() < until, 'Readiness deadline exceeded')
            def response_json(response):
                return api.strict_json(api.bounded_bytes(response.iter_content(chunk_size=1),
                                                        startup_tick, limit=2 << 20).decode('utf-8'))
            while time.monotonic() < until:
                need(proc.poll() is None, 'Owned frontend exited before readiness')
                native_item = handoff.poll()
                if native_item is not None:
                    native_item = hold_native(owner, kernel, identity_support, engine, observed_members)
                if native_item is not None:
                    report['native_identity'] = native_item['identity']
                    checkpoint('native_independently_held', {'identity': native_item['identity']})
                polls += 1
                try:
                    with session.get(BASE + '/health', timeout=(2, 2), allow_redirects=False, stream=True) as response:
                        if response.status_code == 200:
                            body = response_json(response)
                            if body.get('loaded') is True and body.get('max_context') == 1048576 and body.get('model') == cfg['model_name']:
                                checkpoint('loaded_health_observed', {'native_held': native_item is not None,
                                                                     'max_context': 1048576})
                                need(native_item is not None, 'Ready model has no held owned native')
                                ready = True; break
                except requests.RequestException:
                    pass
                time.sleep(.25)
            need(ready, 'Owned API did not become ready within bounded startup')
            checkpoint('owned_readiness_passed')
            with session.get(BASE + '/v1/telemetry', timeout=(2, 10), allow_redirects=False, stream=True) as response:
                need(response.status_code == 200, 'Authenticated launch telemetry failed')
                body = response_json(response)
                need(hmac.compare_digest(body.get('instance', {}).get('launch_attestation', ''), expected_mac),
                     'Ready endpoint is not this exact owned/configured launch')
            checkpoint('launch_attestation_matched')
            report['readiness'] = {'loaded': True, 'native_context': 1048576, 'polls': polls,
                                  'launch_attestation_matched': True, 'native_held_BEFORE_inference': True,
                                  'creator_read_capability_acquired_BEFORE_native_resume': True,
                                  'read_capability_rights': H.READ_RIGHTS}
        report['native_identity'] = native_item['identity']
        report['loaded_CUDA_modules'] = loaded_cublas(kernel, owner, native_item, expected_dlls)
        checkpoint('owned_native_CUDA_module_paths_verified')
        api_args = argparse.Namespace(config=a.config, output=output / 'API.json', base=BASE,
            deadline=a.api_deadline, expected_config_sha256=pins[str(a.config.resolve())],
            long_prompt=a.long_prompt, expected_long_answer=a.expected_long_answer,
            expected_long_prompt_tokens=a.expected_long_prompt_tokens,
            long_thinking_preset=a.long_thinking_preset)
        install_failed_long_capture(api, output, report, pins[str(a.long_prompt.resolve())],
                                    lambda: [cfg.get('api_key', ''), owner_key])
        checkpoint('FIRST_long_API_checks_entered')
        api_rc = api.run(api_args)  # Synchronous validation in the owning parent; no unowned checker child.
        api_result = read(output / 'API.json')
        need(api_rc == 0 and api_result.get('passed') is True and api_result.get('finished') is True
             and api_result.get('configuration_unchanged') is True and api_result.get('long_is_FIRST_inference') is True,
             'FIRST-long API/protocol validation failed')
        checkpoint('API_checks_returned', {'returncode': api_rc, 'passed': api_result.get('passed')})
        post_native = hold_native(owner, kernel, identity_support, engine, observed_members)
        need(post_native is native_item, 'API validation changed the original held native')
        report.update(API_passed=True, API_receipt_sha256=digest(output / 'API.json'),
                      original_native_still_sole_live_AFTER_API=True, native_constructor_retries=0,
                      native_PID_reopens_by_controller=0)
        checkpoint('API_and_original_native_binding_passed')
        rc = 0
    except BaseException as error:
        report.update(error_type=type(error).__name__, failure='Single owned API integration failed; no retry',
                      failure_checkpoint=report.get('last_checkpoint'),
                      actual_exception=error_detail(error, [cfg.get('api_key', ''), owner_key]))
        with contextlib.suppress(BaseException):
            trace.record('failure', {'at_checkpoint': report.get('last_checkpoint'),
                                     'exception': report['actual_exception']})
        rc = 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        if owner is not None:
            cleanup_checkpoint('owned_shutdown_entered', {'native_independently_held': native_item is not None})
            shutdown = shutdown_owned(proc, owner, kernel, native_item, 120)
            shutdown = sanitized(shutdown, [cfg.get('api_key', ''), owner_key])
            save_new(output / 'SHUTDOWN.json', shutdown)
            report['shutdown'] = shutdown
            cleanup_checkpoint('owned_shutdown_completed', {'natural_native0': shutdown.get('natural_native0'),
                       'failure_type': shutdown.get('failure_type'),
                       'failure_reason': shutdown.get('failure_reason'),
                       'live_owned_job_forced_termination': shutdown.get('live_owned_job_forced_termination')})
            if not shutdown_admitted(shutdown):
                rc = 1
        elif job is not None:
            job.close(); rc = 1
        if reader is not None:
            reader.join(timeout=5)
            report['frontend_log_reader'] = log_result
            if reader.is_alive() or not log_result.get('reader_finished'):
                rc = 1
        try:
            changed = [name for name, value in pins.items() if digest(name) != value]
            report['pinned_sources_config_exe_DLLs_unchanged'] = not changed
            report['changed_pin_count'] = len(changed)
            report['payload_stats_unchanged'] = all(Path(name).stat().st_size == value['bytes']
                and Path(name).stat().st_mtime_ns == value['mtime_ns'] for name, value in payload_stats.items())
            if changed or not report['payload_stats_unchanged']:
                rc = 1
        except BaseException as error:
            report['pin_recheck_error_type'] = type(error).__name__; rc = 1
        cleanup_checkpoint('finished', {'passed': rc == 0})
        try:
            trace.close()
        except BaseException as error:
            report['checkpoint_close_failure'] = error_detail(error, [cfg.get('api_key', ''), owner_key])
            rc = 1
        report.update(finished=True, passed=rc == 0, elapsed_seconds=time.monotonic() - began,
                      profile_learning_not_overwritten=True)
        # Never write config/API/attestation key values or response/prompt bodies.
        serialized = json.dumps(sanitized(report, [cfg.get('api_key', ''), owner_key]),
                                indent=2, allow_nan=False)
        with (output / 'REPORT.json').open('x', encoding='utf-8') as f:
            f.write(serialized + '\n')
        print(json.dumps({'finished': True, 'passed': rc == 0, 'receipt': str(output / 'REPORT.json'),
                          'keys_printed': False}))
    return rc


def self_test(runtime=None):
    class In:
        closed = False
        def close(self): self.closed = True
    class Process:
        def __init__(self, code=130, timeout=False): self.stdin=In();self.code=code;self.timeout=timeout
        def wait(self, timeout):
            if self.timeout: raise subprocess.TimeoutExpired('synthetic', timeout)
            return self.code
    class Owner:
        def __init__(self, active=0): self.n=active;self.forced=False;self.finished=False
        def active(self): return self.n
        def terminate(self): self.forced=True;self.n=0
        def finish(self, timeout):
            self.finished=True
            return {'passed':True,'cleanup_proof_version':3,'owned_job_closed':True,
                    'owned_job_empty_proven':True,'owned_job_active_processes_before_close':0,'errors':[],
                    'separate_process_handle_checks':[{'role':role,'separately_held_handle_signaled':True,
                     'wait_result':0,'pid_absence_claimed':False}for role in ('frontend','native')]}
    class Kernel:
        def __init__(self, code=0, wait=0): self.code=code;self.wait=wait
        def WaitForSingleObject(self,h,t): return self.wait
        def GetExitCodeProcess(self,h,p): p._obj.value=self.code;return True
    item={'handle':1};checks=0
    o=Owner();r=shutdown_owned(Process(),o,Kernel(),item)
    need(shutdown_admitted(r) and r['frontend_returncode']==130 and not o.forced,'Valid ownerEOF/native0 refused');checks+=1
    import copy
    for key,value in [('native_returncode',130),('frontend_returncode',1),('owned_job_active_BEFORE_finish',1),
                      ('live_owned_job_forced_termination',True)]:
        bad=copy.deepcopy(r);bad[key]=value;need(not shutdown_admitted(bad),'Bad shutdown admission accepted');checks+=1
    for key,value in [('cleanup_proof_version',2),('errors',['synthetic']),('owned_job_empty_proven',False),
                      ('separate_process_handle_checks',[])]:
        bad=copy.deepcopy(r);bad['cleanup'][key]=value;need(not shutdown_admitted(bad),'Incomplete V3 accepted');checks+=1
    o=Owner();r=shutdown_owned(Process(0),o,Kernel(),item);need(r['natural_native0'],'Frontend0 refused');checks+=1
    for process,kernel in [(Process(1),Kernel()),(Process(),Kernel(code=1)),(Process(),Kernel(wait=258)),(Process(),Kernel())]:
        o=Owner(1);r=shutdown_owned(process,o,kernel,None if process is not None and kernel.code==0 and kernel.wait==0 and process.code==130 else item,seconds=0)
        need(not r['natural_native0'] and o.forced and r['live_owned_job_forced_termination'],'Bad close accepted');checks+=1
    o=Owner(1);r=shutdown_owned(Process(timeout=True),o,Kernel(),item,seconds=0)
    need(not r['natural_native0'] and o.forced and o.finished,'Timeout did not bound owned cleanup');checks+=1
    need(launch_mac('ab'*32,b'{}',['--owner-fd','0'])!=launch_mac('ab'*32,b'{}',['--owner-fd','1']), 'Launch argv not bound');checks+=1
    need(launch_mac('ab'*32,b'{}',[])!=launch_mac('cd'*32,b'{}',[]),'Attestation key not bound');checks+=1
    class Sink:
        def __init__(self): self.data=b''
        def write(self,data): self.data+=data
        def __enter__(self): return self
        def __exit__(self,*args): pass
    class Dest:
        def __init__(self): self.sink=Sink()
        def open(self,mode): need(mode=='xb','Exclusive log required');return self.sink
    class Pipe:
        def __init__(self,chunks): self.chunks=iter(chunks)
        def read(self,n): return next(self.chunks,b'')
    key='synthetic-private-key';dest=Dest();state={}
    redacted_log(Pipe([b'prefix synthetic-',b'private-',b'key suffix']),dest,[key],state)
    need(state.get('reader_finished') and key.encode() not in dest.sink.data
         and dest.sink.data==b'prefix [REDACTED] suffix','Split-chunk secret leaked');checks+=1
    akey='a'*43;bkey='b'*64;dest=Dest();state={}
    redacted_log(Pipe([akey[:20].encode(),(akey[20:]+bkey[:31]).encode(),bkey[31:].encode()]),dest,[akey,bkey],state)
    need(akey.encode()not in dest.sink.data and bkey.encode()not in dest.sink.data
         and state.get('reader_finished'),'API/attestation key boundary leaked');checks+=1
    expected=Path('synthetic-native.exe').resolve()
    held={'role':'native','handle':1,'identity':{'pid':11,'created_filetime':7,'image':str(expected)}}
    class CapKernel(Kernel):
        def OpenProcess(self,*args): raise AssertionError('Native PID must never be reopened')
        def GetProcessId(self,h):return 11
        def GetProcessTimes(self,h,c,e,k,u):c._obj.dwLowDateTime=7;return True
        def QueryFullProcessImageNameW(self,h,f,text,n):text.value=str(expected);return True
        def IsProcessInJob(self,h,j,value):value._obj.value=True;return True
    class Known:
        job=argparse.Namespace(handle=99)
        def __init__(self,pid):self.held=[held];self.pid=pid
        def pids(self):return [self.pid]
    identity=argparse.Namespace(identity=lambda k,pid:(_ for _ in ()).throw(AssertionError('Known native PID reopen')))
    need(hold_native(Known(11),CapKernel(wait=258),identity,expected)is held,'Same retained native refused');checks+=1
    for owner,kernel in [(Known(12),CapKernel(wait=258)),(Known(11),CapKernel(wait=0))]:
        try:hold_native(owner,kernel,identity,expected)
        except RuntimeError:checks+=1
        else:raise RuntimeError('Absent/exited native accepted')
    # Compare against the actual prepared runtime's pure attestation method,
    # not two invocations of this helper's own framing.
    actual = load((runtime or RUNTIME) / 'serve/launch_identity.py', 'owned_API_actual_launch_identity_test')
    for raw, argv in [(b'{}', []), (b'{"synthetic":1}', ['--owner-fd','0']),
                      ('{"label":"caf\u00e9"}'.encode(), ['--config','synthetic.json','--port','8080'])]:
        need(launch_mac('ab'*32, raw, argv) == actual.attestation(bytes.fromhex('ab'*32), raw, argv),
             'Helper framing differs from the actual prepared runtime');checks+=1
    hidden=['synthetic-api-key','synthetic-attestation-key']
    err=error_detail(RuntimeError('detail synthetic-api-key\nsynthetic-attestation-key'), hidden)
    need(err['message']=='detail [REDACTED] [REDACTED]', 'Diagnostic exception leaked a key');checks+=1
    need(error_detail(RuntimeError('x'*3000))['message']=='x'*2048,'Exception detail is unbounded');checks+=1
    need(sanitized({'api_key':'anything','authorization':'anything','nested':['synthetic-api-key']},hidden)==
         {'api_key':'[REDACTED]','authorization':'[REDACTED]','nested':['[REDACTED]']},
         'Nested diagnostic redaction failed');checks+=1
    observed=[]
    need(hold_native(Known(11),CapKernel(wait=258),identity,expected,observed.append)is held
         and observed[-1]['matching_native_pids']==[11]
         and observed[-1]['owned_job_members'][0]['identity_source']=='retained_creator_read_capability',
         'Retained-capability identity was not recorded');checks+=1
    class Missing:
        held=[]
        def pids(self):raise AssertionError('Discovery must be authenticated handoff only')
    need(hold_native(Missing(),CapKernel(),identity,expected,observed.append)is None,
         'A native was discovered without authenticated creator handoff');checks+=1
    class Extra(Known):
        def pids(self):return [11,12]
    extra_identity=argparse.Namespace(identity=lambda k,pid:{'pid':pid,'image':str(expected)})
    try:hold_native(Extra(11),CapKernel(wait=258),extra_identity,expected,observed.append)
    except RuntimeError:checks+=1
    else:raise RuntimeError('A second native image was admitted')
    class TextSink:
        def __init__(self):self.rows=[]
        def write(self,s):self.rows.append(s)
        def flush(self):pass
    cp=object.__new__(Checkpoints);cp.stream=TextSink();cp.hidden=lambda:hidden
    cp.clock=lambda:0;cp.started=0;cp.count=0;cp.previous={}
    cp.record('synthetic',{'message':'synthetic-api-key'})
    cp.record('synthetic',{'message':'synthetic-api-key'})
    need(cp.count==1 and 'synthetic-api-key'not in ''.join(cp.stream.rows)
         and json.loads(cp.stream.rows[0])['detail']['message']=='[REDACTED]',
         'Checkpoint redaction/deduplication failed');checks+=1
    cp.count=128
    try:cp.record('new',{})
    except RuntimeError:checks+=1
    else:raise RuntimeError('Checkpoint count was not bounded')
    parser = load(PUB / 'scripts/check_api.py', 'owned_API_v4_actual_parser_test')
    def sample(answer):
        return [{'choices':[{'delta':{'content':answer}}]},
                {'choices':[{'delta':{},'finish_reason':'stop'}],
                 'usage':{'prompt_tokens':999771,'completion_tokens':11,
                          'prompt_tokens_details':{'cached_tokens':0},'total_tokens':999782}}, '[DONE]']
    prompt_hash='6c4f575406cf9303355867354ced548d7c4789830ff6f817ae1f8de83fd17c2a'
    writes=[];report={};target=argparse.Namespace(stream_result=parser.stream_result)
    install_failed_long_capture(target,Path('synthetic-private'),report,prompt_hash,lambda:[],
                                lambda path,value:writes.append(value))
    seen=[]
    def once():
        for event in sample('828639'):seen.append(event);yield event
    need(target.stream_result(once(),'828639',require_long=True)['natural_stop'] and not writes
         and len(seen)==3,'Successful capture wrote events or changed the consumer');checks+=1
    target=argparse.Namespace(stream_result=parser.stream_result);report={}
    install_failed_long_capture(target,Path('synthetic-private'),report,prompt_hash,lambda:[],
                                lambda path,value:writes.append(value))
    try:target.stream_result(iter(sample('828638')),'828639',require_long=True)
    except RuntimeError as error:
        need(str(error)=='Complete natural SSE stop, exact answer and DONE required'
             and len(writes)==1 and writes[0]['response_events'][0]['choices'][0]['delta']['content']=='828638',
             'Failed capture erased the original reason/events');checks+=1
    else:raise RuntimeError('Failed capture changed original validation')
    try:install_failed_long_capture(target,Path('synthetic-private'),{},'wrong',lambda:[])
    except RuntimeError:checks+=1
    else:raise RuntimeError('Nonpublic input was allowed into raw response capture')
    writes=[];target=argparse.Namespace(stream_result=parser.stream_result)
    install_failed_long_capture(target,Path('synthetic-private'),{},prompt_hash,lambda:[],
                                lambda path,value:writes.append(value),limit=1)
    try:target.stream_result(iter(sample('bad')),'828639',require_long=True)
    except RuntimeError:
        need(writes[0]['capture_truncated'] and writes[0]['serialized_event_bytes']<=1,
             'Failure capture exceeded its memory bound');checks+=1
    else:raise RuntimeError('Truncated capture changed parser failure')
    writes=[];target=argparse.Namespace(stream_result=parser.stream_result)
    install_failed_long_capture(target,Path('synthetic-private'),{},prompt_hash,lambda:['synthetic-secret'],
                                lambda path,value:writes.append(value))
    try:target.stream_result(iter([{'error':{'message':'synthetic-secret'}}]),'828639',require_long=True)
    except RuntimeError:
        need('synthetic-secret'not in json.dumps(writes),'Private capture leaked a key');checks+=1
    else:raise RuntimeError('Error-frame capture changed parser failure')
    return {'passed':True,'checks':checks,'process_HTTP_model_GPU_calls':0,'files_written':0}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true');p.add_argument('--self-test',action='store_true')
    for name in ('config','build-receipt','source-manifest','runtime-manifest'):
        p.add_argument('--'+name,type=Path)
        p.add_argument('--expected-'+name+'-sha256')
    p.add_argument('--expected-exe-sha256')
    p.add_argument('--expected-harness-manifest-sha256')
    p.add_argument('--python',type=Path,default=Path(sys.executable))
    p.add_argument('--runtime',type=Path,default=RUNTIME,help='Prepared API runtime for pure attestation cross-check')
    p.add_argument('--output',type=Path);p.add_argument('--long-prompt',type=Path)
    p.add_argument('--expected-long-answer',default='828639')
    p.add_argument('--expected-long-prompt-tokens',type=int)
    p.add_argument('--long-thinking-preset',choices=('high-budget256',),default='high-budget256')
    p.add_argument('--startup-deadline',type=int,default=900);p.add_argument('--api-deadline',type=int,default=1800)
    a=p.parse_args();need(not(a.run and a.self_test),'Choose one mode')
    need(1<=a.startup_deadline<=900 and 600<=a.api_deadline<=1800,'Invalid bounded deadlines')
    if a.self_test: print(json.dumps(self_test(a.runtime)));return 0
    if not a.run:
        print(json.dumps({'default_dry':True,'process_HTTP_model_GPU_calls':0,'files_written':0,
            'required':'Explicit config/BUILD/source/runtime/image/harness byte pins; fresh private66; FIRST frozen long input; exclusive GPU slot AFTER qualification gates',
            'base':BASE,'runtime_scope':'Prepared complete027 server plus no-store/QUIT60 and retained12web',
            'long_client_settings':{'reasoning_effort':'high','reasoning_budget_tokens':256,'max_tokens':1024},
            'owned_frontend_suspended_before_attach':True,'shutdown_native0_and_empty_Job_required':True}));return 0
    required=(a.config,a.build_receipt,a.source_manifest,a.runtime_manifest,a.output,a.long_prompt,
        a.expected_config_sha256,a.expected_build_receipt_sha256,a.expected_source_manifest_sha256,
        a.expected_runtime_manifest_sha256,a.expected_exe_sha256,a.expected_harness_manifest_sha256)
    need(all(required),'Run requires all explicit reviewed artifact pins and exclusive output')
    need(a.expected_long_answer=='828639' and a.long_thinking_preset=='high-budget256',
         'Only the actual successful API05 exact-answer/high256 contract is admitted')
    return run(a)


if __name__=='__main__':
    try: raise SystemExit(main())
    except (RuntimeError,ValueError) as error:
        print(json.dumps({'refused':True,'error_type':type(error).__name__,'keys_printed':False}));raise SystemExit(2)

"""Owned Windows Job proof independent of Popen handle closure/PID visibility.

Importing this module creates no handles or processes. Call attach() only on an
exact newly-created suspended Popen. The caller retains its kill-on-close Job.
"""
import ctypes as C
from ctypes import wintypes as W
from pathlib import Path
import threading
import time


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


class Accounting(C.Structure):
    _fields_ = [(name, C.c_int64) for name in ("TotalUserTime", "TotalKernelTime", "ThisPeriodTotalUserTime", "ThisPeriodTotalKernelTime")] + [
        (name, W.DWORD) for name in ("TotalPageFaultCount", "TotalProcesses", "ActiveProcesses", "TotalTerminatedProcesses")]


class Pids(C.Structure):
    _fields_ = [("assigned", W.DWORD), ("count", W.DWORD), ("pids", C.c_size_t * 4096)]


def identity_from_handle(kernel, handle, pid):
    times = [W.FILETIME() for _ in range(4)]
    need(kernel.GetProcessTimes(handle, *[C.byref(value) for value in times]), "Cannot inspect separately-held process times")
    image, length = C.create_unicode_buffer(32768), W.DWORD(32768)
    need(kernel.QueryFullProcessImageNameW(handle, 0, image, C.byref(length)), "Cannot inspect separately-held process image")
    return {"pid": pid, "created_filetime": (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime,
            "image": str(Path(image.value).resolve())}


class Ownership:
    def __init__(self, kernel, job):
        self.kernel, self.job = kernel, job
        self.lock = threading.RLock()
        self.held, self.notes = [], []
        self.termination_requested, self.finished = False, False
        self.result = None

    def active(self):
        accounting = Accounting()
        need(self.job.handle and self.kernel.QueryInformationJobObject(self.job.handle, 1, C.byref(accounting), C.sizeof(accounting), None),
             "Cannot query the owned Job active-process count")
        return int(accounting.ActiveProcesses)

    def pids(self):
        listing = Pids()
        need(self.job.handle and self.kernel.QueryInformationJobObject(self.job.handle, 3, C.byref(listing), C.sizeof(listing), None),
             "Cannot query owned Job process IDs")
        need(listing.count < 4096, "Unexpected owned Job size")
        return [int(pid) for pid in listing.pids[:listing.count]]

    def hold(self, pid, role, expected_image=None, required=True):
        # Independent SYNCHRONIZE + QUERY_LIMITED_INFORMATION handle. The engine
        # library owns/occasionally closes Popen._handle; this handle is ours.
        handle = self.kernel.OpenProcess(0x100000 | 0x1000, False, pid)
        if not handle:
            need(not required, "Cannot separately hold the suspended owned process")
            self.notes.append({"pid": pid, "note": "member exited or became unavailable before optional hold; Job-empty proof still required"})
            return None
        try:
            inside = W.BOOL()
            need(self.kernel.IsProcessInJob(handle, self.job.handle, C.byref(inside)), "Cannot inspect owned Job membership")
            if not inside.value:
                need(not required, "Exact suspended child is outside the owned Job")
                self.notes.append({"pid": pid, "note": "member snapshot raced exit/reuse; foreign process was not signaled"})
                self.kernel.CloseHandle(handle)
                return None
            identity = identity_from_handle(self.kernel, handle, pid)
            if expected_image is not None:
                need(Path(identity["image"]).resolve() == Path(expected_image).resolve(), "Owned image differs from expected executable")
            self.held.append({"handle": handle, "identity": identity, "role": role})
            return identity
        except BaseException:
            self.kernel.CloseHandle(handle)
            raise

    def attach(self, process, expected_image, role="engine"):
        with self.lock:
            need(not self.finished, "Ownership already finalized")
            self.job.attach(process)
            return self.hold(process.pid, role, expected_image, required=True)

    def terminate(self):
        """End only this Job; retain its handle for authoritative empty proof."""
        with self.lock:
            if self.finished or self.termination_requested:
                return
            self.termination_requested = True
            if self.job.handle:
                ok = bool(self.kernel.TerminateJobObject(self.job.handle, 1))
                self.notes.append({"owned_job_terminate_returned": ok})

    def finish(self, timeout=30):
        with self.lock:
            if self.finished:
                return self.result
            deadline = time.monotonic() + timeout
            errors, checks, last_active = [], [], None
            try:
                # Raw PIDs first: native.members() queried identity per PID and
                # failed when a short-lived verifier exited during enumeration.
                # Optional extra-member holds never replace Job accounting.
                known = {item["identity"]["pid"] for item in self.held}
                for pid in self.pids():
                    if pid not in known:
                        try:
                            self.hold(pid, "job-descendant", required=False)
                        except RuntimeError:
                            self.notes.append({"pid": pid, "note": "optional member hold unavailable; Job-empty proof still required"})
                self.terminate()
                while True:
                    last_active = self.active()
                    if last_active == 0 or time.monotonic() >= deadline:
                        break
                    time.sleep(min(0.05, max(0, deadline - time.monotonic())))
                if last_active != 0:
                    errors.append("OwnedJobNotEmpty")
            except BaseException as error:
                errors.append(type(error).__name__)
                self.terminate()
            finally:
                for item in self.held:
                    try:
                        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
                        result = self.kernel.WaitForSingleObject(item["handle"], remaining_ms)
                        signaled = result == 0
                        checks.append({"role": item["role"], "recorded_identity": item["identity"],
                                       "separately_held_handle_signaled": signaled, "wait_result": int(result),
                                       "pid_absence_claimed": False})
                        if not signaled:
                            errors.append("SeparateOwnedHandleNotSignaled")
                    except BaseException as error:
                        errors.append(type(error).__name__)
                    finally:
                        self.kernel.CloseHandle(item["handle"])
                self.job.close()  # last-resort kill-on-close even on proof error
                self.finished = True
            self.result = {"cleanup_proof_version": 3, "owned_job_closed": True,
                           "owned_job_active_processes_before_close": last_active,
                           "owned_job_empty_proven": last_active == 0,
                           "separate_process_handle_checks": checks, "notes": self.notes, "errors": errors,
                           "pid_absence_claimed": False,
                           "passed": last_active == 0 and not errors and all(row["separately_held_handle_signaled"] for row in checks)}
            return self.result


def offline_self_test():
    """Fake Win32 only: no actual Job/process/handle/GPU operation."""
    class Kernel:
        active_count = 1
        raced_member = False
        terminate_works = True
        wait_result = 0
        def OpenProcess(self, rights, inherit, pid):
            need(rights == 0x101000, "Missing independent SYNCHRONIZE/QUERY rights")
            return 142 if pid == 42 else 0
        def GetProcessTimes(self, handle, created, exited, kernel, user):
            created._obj.dwLowDateTime = 123
            return True
        def QueryFullProcessImageNameW(self, handle, flags, image, length):
            image.value = str(Path("mock.exe").resolve())
            return True
        def IsProcessInJob(self, handle, job, inside):
            inside._obj.value = True
            return True
        def QueryInformationJobObject(self, handle, kind, result, size, ignored):
            if kind == 1:
                result._obj.ActiveProcesses = self.active_count
            elif kind == 3:
                pids = [42, 77] if self.raced_member else [42]
                result._obj.count = len(pids)
                for index, pid in enumerate(pids):
                    result._obj.pids[index] = pid
            else:
                raise AssertionError("Unexpected Job query")
            return True
        def TerminateJobObject(self, handle, code):
            if self.terminate_works:
                self.active_count = 0
            return self.terminate_works
        def WaitForSingleObject(self, handle, timeout):
            need(handle == 142, "Popen handle used instead of separate held handle")
            return self.wait_result
        def CloseHandle(self, handle):
            return True
    class Job:
        handle = 900
        def attach(self, process):
            pass
        def close(self):
            self.handle = None
    class ClosedHandle:
        def __int__(self):
            raise AssertionError("Closed Popen handle must never be inspected")
    class Process:
        pid, returncode, _handle = 42, -1, ClosedHandle()
    for raced in (False, True):
        kernel, job = Kernel(), Job()
        kernel.raced_member = raced
        owner = Ownership(kernel, job)
        owner.attach(Process(), "mock.exe")
        result = owner.finish(0.02)
        need(result["passed"] and result["owned_job_empty_proven"] and not result["pid_absence_claimed"], "Separate handle/racy enumeration cleanup failed")
    for nonempty, failed_wait in ((True, False), (False, True)):
        kernel, job = Kernel(), Job()
        kernel.terminate_works = not nonempty
        kernel.wait_result = 0xFFFFFFFF if failed_wait else 258
        owner = Ownership(kernel, job)
        owner.attach(Process(), "mock.exe")
        need(not owner.finish(0.01)["passed"], "Ambiguous/running cleanup falsely passed")
    return ["independent handle survives closed Popen handle", "exiting member enumeration race tolerated only with Job-empty proof",
            "nonempty Job rejected", "failed independent-handle wait rejected"]

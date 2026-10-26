"""Only canonical owned-process identity/Job query blocks; no start/stop command."""
import ctypes as C
from ctypes import wintypes as W
from pathlib import Path
from native_support import need

def windows():
    k = C.WinDLL('kernel32', use_last_error=True)
    definitions = {
        'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
        'CloseHandle': ([W.HANDLE], W.BOOL),
        'GetCurrentProcess': ([], W.HANDLE),
        'GetProcessTimes': ([W.HANDLE] + [C.POINTER(W.FILETIME)] * 4, W.BOOL),
        'QueryFullProcessImageNameW': ([W.HANDLE, W.DWORD, W.LPWSTR, C.POINTER(W.DWORD)], W.BOOL),
        'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE),
        'OpenJobObjectW': ([W.DWORD, W.BOOL, W.LPCWSTR], W.HANDLE),
        'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
        'QueryInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.POINTER(W.DWORD)], W.BOOL),
        'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
        'IsProcessInJob': ([W.HANDLE, W.HANDLE, C.POINTER(W.BOOL)], W.BOOL),
        'TerminateJobObject': ([W.HANDLE, W.UINT], W.BOOL),
        'WaitForSingleObject': ([W.HANDLE, W.DWORD], W.DWORD),
        'CreateMutexW': ([C.c_void_p, W.BOOL, W.LPCWSTR], W.HANDLE),
        'ReleaseMutex': ([W.HANDLE], W.BOOL),
        'GetSystemDirectoryW': ([W.LPWSTR, W.UINT], W.UINT),
    }
    for name, (args, result) in definitions.items():
        f = getattr(k, name)
        f.argtypes, f.restype = args, result
    return k


def identity(k, pid):
    handle = k.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
    need(handle, 'Owned process is unavailable')
    try:
        times = [W.FILETIME() for _ in range(4)]
        need(k.GetProcessTimes(handle, *[C.byref(x) for x in times]), 'Cannot inspect process creation time')
        text, count = C.create_unicode_buffer(32768), W.DWORD(32768)
        need(k.QueryFullProcessImageNameW(handle, 0, text, C.byref(count)), 'Cannot inspect process image')
        return {'pid': pid, 'created_filetime': (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime,
                'image': str(Path(text.value).resolve())}
    finally:
        k.CloseHandle(handle)


class BasicLimits(C.Structure):
    _fields_ = [('process_time', C.c_int64), ('job_time', C.c_int64), ('flags', W.DWORD),
                ('min_ws', C.c_size_t), ('max_ws', C.c_size_t), ('active_limit', W.DWORD),
                ('affinity', C.c_size_t), ('priority', W.DWORD), ('scheduling', W.DWORD)]


class ExtendedLimits(C.Structure):
    _fields_ = [('basic', BasicLimits), ('io', C.c_uint64 * 6),
                ('process_memory', C.c_size_t), ('job_memory', C.c_size_t),
                ('peak_process_memory', C.c_size_t), ('peak_job_memory', C.c_size_t)]


def members(k, job):
    class IdList(C.Structure):
        _fields_ = [('assigned', W.DWORD), ('count', W.DWORD), ('pids', C.c_size_t * 4096)]
    listing = IdList()
    need(k.QueryInformationJobObject(job, 3, C.byref(listing), C.sizeof(listing), None), 'Cannot query owned Job members')
    need(listing.count < 4096, 'Unexpectedly large owned Job')
    return [identity(k, int(pid)) for pid in listing.pids[:listing.count]]

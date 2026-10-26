"""Exact retained owned Job class; no setup or model work on import."""
import ctypes
import os
from native_support import need

class OwnedJob:
    """A Windows Job whose last handle kills only attached child processes."""
    def __init__(self):
        need(os.name=='nt','This recipe is validated for native Windows only')
        from ctypes import wintypes as W
        class IO(ctypes.Structure):_fields_=[(n,ctypes.c_ulonglong)for n in('r','w','o','rb','wb','ob')]
        class Basic(ctypes.Structure):_fields_=[('p',ctypes.c_longlong),('j',ctypes.c_longlong),('flags',W.DWORD),('min',ctypes.c_size_t),('max',ctypes.c_size_t),('count',W.DWORD),('affinity',ctypes.c_size_t),('priority',W.DWORD),('sched',W.DWORD)]
        class Limits(ctypes.Structure):_fields_=[('basic',Basic),('io',IO),('pm',ctypes.c_size_t),('jm',ctypes.c_size_t),('peakp',ctypes.c_size_t),('peakj',ctypes.c_size_t)]
        self.k=ctypes.WinDLL('kernel32',use_last_error=True);self.k.CreateJobObjectW.restype=W.HANDLE;self.k.CreateJobObjectW.argtypes=[ctypes.c_void_p,W.LPCWSTR]
        self.k.SetInformationJobObject.argtypes=[W.HANDLE,ctypes.c_int,ctypes.c_void_p,W.DWORD];self.k.AssignProcessToJobObject.argtypes=[W.HANDLE,W.HANDLE];self.k.CloseHandle.argtypes=[W.HANDLE]
        self.handle=self.k.CreateJobObjectW(None,None);need(self.handle,'Cannot create owned Job')
        limits=Limits();limits.basic.flags=0x2000
        need(self.k.SetInformationJobObject(self.handle,9,ctypes.byref(limits),ctypes.sizeof(limits)),'Cannot set kill-on-close Job')
    def attach(self,proc):need(self.k.AssignProcessToJobObject(self.handle,int(proc._handle)),'Cannot attach owned child to Job')
    def close(self):
        if self.handle:self.k.CloseHandle(self.handle);self.handle=None
    def __enter__(self):return self
    def __exit__(self,*args):self.close()

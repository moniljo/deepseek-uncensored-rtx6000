"""Private launch-time native read capability. No privilege/DACL changes.

Import is inert. Only a caller's authenticated suspended native creator may
offer a handle; only the existing owned Job is admitted. Never published.
"""
import ctypes as C
from ctypes import wintypes as W
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time

READ_RIGHTS = 0x100000 | 0x1000 | 0x0400 | 0x0010
DOMAIN = b'owned-native-pre-resume-read-capability-v1\0'

def need(ok, message):
    if not ok: raise RuntimeError(message)

def configure(k):
    definitions = {
        'GetCurrentProcess': ([], W.HANDLE),
        'OpenProcess': ([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
        'CloseHandle': ([W.HANDLE],W.BOOL),
        'DuplicateHandle': ([W.HANDLE,W.HANDLE,W.HANDLE,C.POINTER(W.HANDLE),W.DWORD,W.BOOL,W.DWORD],W.BOOL),
        'GetProcessTimes': ([W.HANDLE]+[C.POINTER(W.FILETIME)]*4,W.BOOL),
        'GetProcessId': ([W.HANDLE],W.DWORD),
        'QueryFullProcessImageNameW': ([W.HANDLE,W.DWORD,W.LPWSTR,C.POINTER(W.DWORD)],W.BOOL),
        'IsProcessInJob': ([W.HANDLE,W.HANDLE,C.POINTER(W.BOOL)],W.BOOL),
        'WaitForSingleObject': ([W.HANDLE,W.DWORD],W.DWORD),
        'OpenThread': ([W.DWORD,W.BOOL,W.DWORD],W.HANDLE),
        'ResumeThread': ([W.HANDLE],W.DWORD),
    }
    for name,(args,result) in definitions.items():
        fn=getattr(k,name);fn.argtypes,fn.restype=args,result
    return k

def identity(k,handle,pid):
    need(int(k.GetProcessId(handle))==pid,'Read-capability kernel PID differs')
    times=[W.FILETIME() for _ in range(4)]
    need(k.GetProcessTimes(handle,*[C.byref(x)for x in times]),'Read-capability process time query failed')
    image=C.create_unicode_buffer(32768);length=W.DWORD(len(image))
    need(k.QueryFullProcessImageNameW(handle,0,image,C.byref(length)),'Read-capability image query failed')
    return {'pid':pid,'created_filetime':(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime,
            'image':str(Path(image.value).resolve())}

def inside(k,handle,job):
    value=W.BOOL();need(k.IsProcessInJob(handle,job,C.byref(value)),'Owned Job membership query failed')
    return bool(value.value)

def encoded(body):return json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()

def sign(key,body):return hmac.new(bytes.fromhex(key),DOMAIN+encoded(body),hashlib.sha256).hexdigest()

def write_frame(path,key,body):
    raw=encoded({'body':body,'mac':sign(key,body)})
    need(len(raw)<=65536,'Private handoff frame exceeds bound')
    temporary=path.with_suffix('.pending')
    with temporary.open('xb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    need(not path.exists(),'Private handoff frame already exists')
    os.replace(temporary,path)

def read_frame(path,key,kind,nonce):
    need(path.stat().st_size<=65536,'Private handoff frame exceeds bound')
    with path.open('rb')as stream:raw=stream.read(65537)
    need(len(raw)<=65536,'Private handoff frame exceeds bound')
    frame=json.loads(raw.decode('utf-8'));body=frame['body']
    need(hmac.compare_digest(frame['mac'],sign(key,body)),'Private handoff authentication failed')
    need(body['kind']==kind and body['nonce']==nonce,'Private handoff launch binding differs')
    return body

def duplicate_local(k,handle,rights=READ_RIGHTS):
    other=W.HANDLE();need(k.DuplicateHandle(k.GetCurrentProcess(),handle,k.GetCurrentProcess(),
        C.byref(other),rights,False,0),'Cannot duplicate retained local read capability')
    return other.value

def validate_item(k,owner,item,expected):
    need(k.WaitForSingleObject(item['handle'],0)==258,'Original held native has exited')
    need(identity(k,item['handle'],item['identity']['pid'])==item['identity'],'Original held native identity changed')
    need(Path(item['identity']['image']).resolve()==Path(expected).resolve()
         and inside(k,item['handle'],owner.job.handle),'Original native image/Job binding differs')
    return item

class Client:
    def __init__(self,folder,key,expected,owner,k,creator_images):
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False)
        self.key,self.expected,self.owner,self.k=key,Path(expected).resolve(),owner,configure(k)
        self.creator_images={Path(x).resolve()for x in creator_images}
        need(bool(self.creator_images),'Explicit owned frontend creator images required')
        self.nonce=secrets.token_hex(16);self.item=None

    def environment(self):
        return {'STRATA_PRIVATE_HANDOFF_DIR':str(self.folder),'STRATA_PRIVATE_HANDOFF_NONCE':self.nonce,
                'STRATA_PRIVATE_HANDOFF_EXE':str(self.expected)}

    def poll(self):
        if self.item is not None:return validate_item(self.k,self.owner,self.item,self.expected)
        request=self.folder/'REQUEST.json'
        if not request.exists():return None
        body=read_frame(request,self.key,'request',self.nonce)
        need(type(body['creator_pid'])is int and type(body['source_handle'])is int,'Invalid creator capability descriptor')
        creator=self.k.OpenProcess(0x100000|0x1000|0x40,False,body['creator_pid'])
        need(creator,'Cannot open the owned Python creator for handle duplication')
        native=None
        try:
            creator_identity=identity(self.k,creator,body['creator_pid'])
            need(inside(self.k,creator,self.owner.job.handle)
                 and Path(creator_identity['image']).resolve()in self.creator_images
                 and self.k.WaitForSingleObject(creator,0)==258,'Creator is not a live admitted Python in this Job')
            result=W.HANDLE()
            need(self.k.DuplicateHandle(creator,body['source_handle'],self.k.GetCurrentProcess(),
                 C.byref(result),READ_RIGHTS,False,0),'Cannot receive native creator read capability')
            native=result.value
            actual=identity(self.k,native,body['native_identity']['pid'])
            need(actual==body['native_identity'] and Path(actual['image']).resolve()==self.expected
                 and actual['created_filetime']>=creator_identity['created_filetime']
                 and inside(self.k,native,self.owner.job.handle)
                 and self.k.WaitForSingleObject(native,0)==258,'Offered native image/time/Job/live identity differs')
            need(not any(x['role']=='native'for x in self.owner.held),'A native capability is already held')
            self.item={'handle':native,'identity':actual,'role':'native','read_capability_rights':READ_RIGHTS}
            self.owner.held.append(self.item);native=None
            write_frame(self.folder/'ACK.json',self.key,{'kind':'ack','nonce':self.nonce,
                'request_sha256':hashlib.sha256(encoded(body)).hexdigest(),'native_identity':actual,
                'read_capability_rights':READ_RIGHTS})
            return self.item
        finally:
            if native:self.k.CloseHandle(native)
            self.k.CloseHandle(creator)

def gate_created(proc,folder,key,nonce,expected,timeout=60):
    """Called by the real creator after original lifeline/winjob, before resume."""
    import psutil
    try:
        k=configure(C.WinDLL('kernel32',use_last_error=True));folder=Path(folder)
        source=int(proc._handle);ident=identity(k,source,proc.pid)
        need(Path(ident['image']).resolve()==Path(expected).resolve(),'Created suspended native image differs')
        body={'kind':'request','nonce':nonce,'creator_pid':os.getpid(),'source_handle':source,'native_identity':ident}
        write_frame(folder/'REQUEST.json',key,body)
        end=time.monotonic()+timeout
        while not(folder/'ACK.json').exists():
            need(proc.poll()is None and time.monotonic()<end,'Owned native read-capability handoff timed out or child exited')
            time.sleep(.01)
        ack=read_frame(folder/'ACK.json',key,'ack',nonce)
        need(ack['request_sha256']==hashlib.sha256(encoded(body)).hexdigest()
             and ack['native_identity']==ident and ack['read_capability_rights']==READ_RIGHTS,
             'Acknowledged native read-capability identity differs')
        need(proc.poll()is None and identity(k,source,proc.pid)==ident,'Native exited during launch handoff')
        threads=psutil.Process(proc.pid).threads();need(len(threads)==1,'Suspended native must have one primary thread')
        thread=k.OpenThread(2,False,threads[0].id);need(thread,'Cannot open exact suspended native primary thread')
        try:need(k.ResumeThread(thread)==1,'Cannot resume exact acknowledged native')
        finally:k.CloseHandle(thread)
        write_frame(folder/'RESUMED.json',key,{'kind':'resumed','nonce':nonce,'native_identity':ident})
        return proc
    except BaseException:
        # Exact Popen creator capability, never a global discovered PID.
        if proc.poll()is None:proc.kill()
        proc.wait(timeout=20)
        raise

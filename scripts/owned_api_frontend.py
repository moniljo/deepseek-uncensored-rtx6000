"""Private creation-only hook around the unchanged actual API server main."""
import importlib.util
import os
from pathlib import Path
import sys
import owned_api_handoff as H

def main():
    runtime=Path(sys.argv[1]).resolve();server_args=sys.argv[2:]
    folder=os.environ.pop('STRATA_PRIVATE_HANDOFF_DIR')
    nonce=os.environ.pop('STRATA_PRIVATE_HANDOFF_NONCE')
    expected=Path(os.environ.pop('STRATA_PRIVATE_HANDOFF_EXE')).resolve()
    key=os.environ['STRATA_OWNER_ATTEST_KEY']  # Actual main still pops/uses the same launch key.
    sys.path.insert(0,str(runtime));sys.argv=[str(runtime/'serve/server.py'),*server_args]
    spec=importlib.util.spec_from_file_location('private_actual_API_server',runtime/'serve/server.py')
    server=importlib.util.module_from_spec(spec);sys.modules[spec.name]=server;spec.loader.exec_module(server)
    original=server.popen;count=0
    def popen(what,args,role='engine',**kw):
        nonlocal count
        if what!='the Strata engine' or role!='engine':return original(what,args,role=role,**kw)
        H.need(count==0 and Path(args[0]).resolve()==expected and '--serve'in args,
               'Private API allows exactly one expected native constructor')
        count+=1;kw['creationflags']=kw.get('creationflags',0)|0x00000004
        proc=original(what,args,role=role,**kw)  # Real lifeline, nested Job, logger, Popen and reaper.
        return H.gate_created(proc,folder,key,nonce,expected)
    server.popen=popen
    return server.main(server_args)

if __name__=='__main__':raise SystemExit(main())

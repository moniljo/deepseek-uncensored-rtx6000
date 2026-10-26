"""Exact phase markers and bounded stderr reader."""
import re
import native_support as B
need=B.need

COMMIT=re.compile(r"^ds4: experimental HELD workspace regrow committed47 buffers: C1024->65280 A1024->4096 AI128 retained;.*$",re.M)


READY=re.compile(r"^ds4: experimental repeat prefill rearm ready: generation(\d+) attempt(\d+), remaining(\d+) C(\d+) A(\d+) AI(\d+), epoch(\d+) issued(\d+) unchanged, pool(\d+) bytes, registration(\d+) bytes; ([\d.]+) s charged inside feed; next compute is existing chunk-major path$",re.M)


DELTA=re.compile(r"^ds4: experimental repeat prefill generation(\d+) work delta: cold (\d+) sets/(\d+) unpacks; Engram (\d+) fills/(\d+) hits; dense (\d+) fills/(\d+) hits; registration ranges(\d+) after retirement; sequence(\d+) retained$",re.M)


END=re.compile(r"^ds4: experimental chunk-major phase ended: C=(\d+), layers grown=(\d+) skipped=(\d+), .*RAM maps retained, ([\d.]+) s.*$",re.M)


def log_since(path,offset):
    with path.open('rb') as f:f.seek(offset);raw=f.read(2*1024*1024+1)
    need(len(raw)<=2*1024*1024,"Bounded setup log observation exceeded2MiB")
    # A partially written final line cannot establish the triggering event.
    text=raw.decode('utf-8',errors='replace').replace('\r\n','\n')
    return text[:text.rfind('\n')+1] if '\n' in text else ''

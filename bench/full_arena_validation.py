"""Pure startup inventory and normal retained-ledger binding; no native calls."""
import math
import re

def need(value,message):
    if not value:raise RuntimeError(message)

DOMAIN=re.compile(r'^ds4: experimental copied-RAM domains recorded: (\d+) layers/(\d+) slots, (\d+) occupied/(\d+) spares/(\d+) free holes; free bytes untrusted, no rearm or expert move performed$',re.M)
COPY=re.compile(r"^ds4: (\d+) of the CPU lane's (\d+) experts copied into RAM \(([0-9.]+) GB, (\d+) NUMA part\) in ([0-9.]+) s$",re.M)
PHASE=re.compile(r'^ds4: experimental chunk-major phase: C=(\d+), (\d+)/(\d+) CPU experts copied at startup; normal GPU pool retained; post-feed shrink1024 and monotonic growth timed before first token$',re.M)
SETUP=re.compile(r'^ds4: experimental registered-copy setup: selected (\d+)/(\d+) whole CPU arenas, published (\d+) ranges/([0-9.]+) MiB, CUDA floor ([0-9.]+) MiB; ([0-9.]+) s setup/([0-9.]+) s register calls; direct H2D first-prefill only$',re.M)

def startup_inventory(text):
    text=text.replace('\r\n','\n')
    domains,copies,phases,setups=(p.findall(text)for p in(DOMAIN,COPY,PHASE,SETUP))
    need(len(domains)==len(copies)==len(phases)==len(setups)==1,'One exact initial native inventory/copy/phase/registration record required')
    layers,slots,occupied,spares,holes=map(int,domains[0]);copied,total=map(int,copies[0][:2])
    c,phase_copied,phase_total=map(int,phases[0]);selected,arenas,published=map(int,setups[0][:3])
    mib,floor,setup_s,driver_s=map(float,setups[0][3:]);copy_gb,copy_s=float(copies[0][2]),float(copies[0][4])
    need(all(math.isfinite(v)and v>=0 for v in(mib,floor,setup_s,driver_s,copy_gb,copy_s)),
         'Invalid finite startup allocation/registration observations')
    need(layers==spares==selected==arenas==published==40 and holes==0 and
         40<=occupied<=15360 and slots==occupied+spares and
         copied==total==phase_copied==phase_total==occupied and int(copies[0][3])==1 and c==65280,
         'Full40 owned domains/all CPU copies/NH1/slot-spare inventory differs')
    need(0<mib<=196608 and floor>=3072 and setup_s+.002>=driver_s,
         'Full40 registration cap/headroom/driver timing differs')
    positions=[p.search(text).start()for p in(DOMAIN,COPY,PHASE,SETUP)]
    need(positions==sorted(positions)and'GPU_MISS_PIN 'not in text,
         'Initial native inventory/copy/phase/registration order differs')
    return {'layers':layers,'slots':slots,'occupied_CPU_experts':occupied,'reserved_spares':spares,
        'free_holes':holes,'selected_arenas':selected,'total_arenas':arenas,'published_ranges':published,
        'published_MiB_rounded':mib,'CUDA_floor_MiB':floor,'copied_CPU_experts':copied,'NUMA_parts':1,
        'native_reconstruct_slot_count_times_part_stride_equals_owned_arena_bytes':True,
        'slot_stride_proof_scope':'Pinned native reconstruct() executes before the actual domain record; no independent payload checksum',
        'exact_bytes_or_fingerprint_before_GEN_not_claimed':True}

def bind_exact(startup,anchor):
    need(anchor['ranges']==startup['published_ranges']==40 and type(anchor['bytes'])is int and
         0<anchor['bytes']<=196608<<20 and re.fullmatch('[0-9a-f]{16}',anchor['fingerprint']),
         'Actual first normal retained ledger differs from full startup selection')
    # fprintf prints the allocation aggregate with one decimal of MiB. This is
    # only a rounding consistency check, not an invented exact startup total.
    need(abs(anchor['bytes']/1048576-startup['published_MiB_rounded'])<=.050001,
         'Exact first retained bytes disagree with rounded startup allocation')
    return True

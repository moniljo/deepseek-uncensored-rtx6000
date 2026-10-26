"""Exact canonical retained-direct observer/phase admission; no extra consumer."""
import re
import time
from pathlib import Path
import native_support as B
import initial_phase as W
import gpu_miss_validation as V
import direct_pin_validation as P
need=B.need

class Observer:
    def __init__(self, owner, output, cap, overlap):
        self.owner = owner; self.output = output; self.overlap = overlap
        self.history = P.PinHistory(cap); self.rows = []

    def generate(self, measurement, name, ids, maximum, deadline, reused):
        log = self.output/'ENGINE.log'; begin = log.stat().st_size
        # Exact canonical generator invocation/timers. All observation below is
        # after its final timestamp and adds no stdout consumer or native call.
        row = measurement.generate(name, ids, maximum, deadline, reused)
        self.rows.append(row) # retain the actual first result even if a later evidence gate fails
        started = time.monotonic()
        while True:
            end = log.stat().st_size
            need(0 <= end-begin <= 64 << 20, 'Bounded native GEN log required')
            with log.open('rb') as stream: stream.seek(begin); raw = stream.read(end-begin)
            text = raw.decode('utf-8', 'replace')
            complete = V.GPU_ANSWER.search(text) and P.ANSWER.search(text)
            if self.overlap: complete = complete and self.overlap.ANSWER.search(text)
            if complete or time.monotonic()-started >= 1: break
            time.sleep(.005)
        window = self.output/(name+'-ENGINE-window.log'); window.write_bytes(raw)
        row['native_log_window'] = {'start': begin, 'end': end, 'path': str(window), 'sha256': B.sha(window)}
        self.history.feed(text)
        primary = V.gpu_answer(text, maximum != 1)
        row['native_GPU_miss_answer'] = primary
        row['native_GPU_miss_direct_answer'] = P.direct_answer(text, primary, self.history.active, maximum != 1)
        if self.overlap: row['native_GPU_miss_overlap_answer'] = self.overlap.overlap_answer(text, primary, maximum != 1)
        row['post_measurement_counter_drain_seconds'] = time.monotonic()-started
        return row

    def finish(self):
        log = self.output/'ENGINE.log'; need(log.stat().st_size <= 256 << 20, 'Bounded complete qualification log required')
        text = log.read_text(encoding='utf-8', errors='replace')
        need(len(V.GPU_ANSWER.findall(text)) == len(P.ANSWER.findall(text)) == len(self.rows) == 29,
             'Exact28full-wave GENs plus one interlude required')
        if self.overlap: need(len(self.overlap.ANSWER.findall(text)) == len(self.rows), 'All overlap final records required')
        history = P.PinHistory(self.history.cap); history.feed(text)
        need(history.active is None and [event['kind'] for event in history.events] == ['retained','retired','retained','retired'] and
             history.events[1]['reason'] == 'epoch' and history.events[-1]['reason'] == 'exit',
             'Exact initialretention/epochzero/re-retention/exitphysicalzero required')
        counters = [row['native_GPU_miss_answer'] for row in self.rows]
        need(all(a['cumulative_entries'] <= b['cumulative_entries'] and a['cumulative_bytes'] <= b['cumulative_bytes']
                 for a,b in zip(counters,counters[1:])), 'Actual cumulative GPU-miss counters regressed')
        return {'passed': True, 'initial_setup': V.gpu_setup(text), 'exit': V.gpu_exit(text, counters[-1]),
                'physical_pin_history': history.events, 'exact_completed_GENs': len(self.rows),
                'overlap_mode': self.overlap is not None, 'Stop_recovery_not_claimed': True}


def seed_telemetry(text, index, previous=None):
    if index == 1:
        result = W.seed_telemetry(text, 1)
        need(result['admission_passed'] and not P.RETIRE.search(text), 'Initial canonical cache/phase proof failed')
        keep = P.retained(text); need(keep['epoch'] == 0, 'Initial retained owner must precede any epoch')
        result['initial_retained_decode_owner'] = keep
        return result
    need(index == 2 and previous and previous.get('admission_passed'), 'Valid original firstwave required')
    phase = P.phase_evidence(text, 1, B.PREFIX_N)
    cold = re.findall(r'bounded cold F16 retired [0-9.]+ MiB: (\d+) sets/(\d+) batches/(\d+) buckets, (\d+) expert unpacks', text)
    eg = re.findall(r'first-prefill Engram weight cache retired [0-9.]+ MiB: (\d+) fills/(\d+) slab hits/(\d+) original-shape GEMMs', text)
    dense = re.findall(r'first-prefill dense projection cache retired [0-9.]+ MiB: (\d+) matrix fills/(\d+) slab hits/(\d+) original-shape GEMMs', text)
    need(len(cold) == len(eg) == len(dense) == 1, 'One actual cumulative retirement record per cache required')
    cumulative = dict(cold_sets=int(cold[0][0]), cold_unpacks=int(cold[0][3]), eg_fills=int(eg[0][0]), eg_hits=int(eg[0][1]),
                      eg_gemms=int(eg[0][2]), dense_fills=int(dense[0][0]), dense_hits=int(dense[0][1]), dense_gemms=int(dense[0][2]))
    delta = re.findall(r'repeat prefill generation1 work delta: cold (\d+) sets/(\d+) unpacks; Engram (\d+) fills/(\d+) hits; dense (\d+) fills/(\d+) hits; registration ranges(\d+) after retention; sequence(\d+) retained', text)
    need(len(delta) == 1, 'One actual repeat delta required')
    differences = dict(zip(('cold_sets','cold_unpacks','eg_fills','eg_hits','dense_fills','dense_hits'), map(int, delta[0][:6])))
    old = previous['cumulative_retirement_counters']
    need(all(v > 0 and cumulative[k]-old[k] == v for k,v in differences.items()) and
         cumulative['eg_gemms'] > old['eg_gemms'] and cumulative['dense_gemms'] > old['dense_gemms'],
         'Independent cumulative differences do not match repeated actual cache work')
    return dict(phase, cumulative_retirement_counters=cumulative, actual_generation1_work_delta=differences,
                initial_only_work_from_cumulative_counters=False)

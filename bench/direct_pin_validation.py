"""Exact noncancelled physical-pin parsers/state machine."""
import re
import gpu_miss_validation as V
need=V.need

RETAIN = re.compile(r'ds4: GPU_MISS_PIN retained: context=DECODE ranges=(\d+) bytes=(\d+) fingerprint=([0-9a-f]{16}) epoch=(\d+) issued=(\d+) streams_drained=1\r?$', re.M)


RETIRE = re.compile(r'ds4: GPU_MISS_PIN retired: reason=(epoch|exit|phase-stop|phase-abort) context=NONE ranges_before=(\d+) bytes_before=(\d+) fingerprint=([0-9a-f]{16}) ranges_after=0 bytes_after=0 epoch=(\d+) issued=(\d+) streams_drained=1\r?$', re.M)


ANSWER = re.compile(r'ds4: GPU_MISS_DIRECT answer: direct_unique=(\d+) direct_bytes=(\d+) bounce_unique=(\d+) bounce_bytes=(\d+) retained_ranges=(\d+) retained_bytes=(\d+)\r?$', re.M)


def retained(text):
    rows = RETAIN.findall(text)
    need(len(rows) == 1, 'One explicit actual retained-DECODE record required')
    n, b, fp, epoch, issued = rows[0]
    need(int(n) > 0 and int(b) > 0, 'Positive retained physical ranges required')
    return dict(ranges=int(n), bytes=int(b), fingerprint=fp, epoch=int(epoch), issued=int(issued))


def retired(text, reason):
    rows = RETIRE.findall(text)
    need(len(rows) == 1 and rows[0][0] == reason, 'One exact retirement reason/physicalzero record required')
    _, n, b, fp, epoch, issued = rows[0]
    need(int(n) > 0 and int(b) > 0, 'Retirement must identify positive prior ownership')
    return dict(reason=reason, ranges=int(n), bytes=int(b), fingerprint=fp, epoch=int(epoch), issued=int(issued))


class PinHistory:
    def __init__(self, cap):
        need(cap in (98304, 196608), 'Bounded explicitlydeclared registration cap required')
        self.cap = cap; self.active = None; self.events = []; self.previous_retired = None

    def feed(self, text):
        for line in text.splitlines():
            if 'GPU_MISS_PIN ' not in line: continue
            if RETAIN.search(line):
                event = retained(line)
                need(self.active is None and event['bytes'] <= self.cap << 20, 'Overlapping or over-cap retained owner')
                if self.previous_retired:
                    old = self.previous_retired
                    need(old['reason'] == 'epoch' and event['epoch'] == old['epoch'] and event['issued'] >= old['issued'] and
                         all(event[k] == old[k] for k in ('ranges', 'bytes', 'fingerprint')),
                         'Re-registration changed immutablephysical96/192GiB selection or epoch state')
                self.active = event
                self.events.append(dict(kind='retained', **event))
            elif RETIRE.search(line):
                reason = RETIRE.search(line).group(1)
                # Stop/abort needs its own first-result campaign, not inferred
                # success in this ordinary probe's noncancelled state machine.
                need(reason in ('epoch', 'exit'), 'Unexpected phaseStop/abort in ordinary probe')
                event = retired(line, reason)
                need(self.active is not None and all(event[k] == self.active[k] for k in ('ranges', 'bytes', 'fingerprint')) and
                     event['epoch'] >= self.active['epoch'] and event['issued'] >= self.active['issued'],
                     'Retirement did not match actual retained physical owner')
                self.active = None; self.previous_retired = event
                self.events.append(dict(kind='retired', **event))
            else:
                raise RuntimeError('Malformed/failed native retained-pin transition')


def direct_answer(text, primary, active, require_work):
    rows = ANSWER.findall(text); need(len(rows) == 1, 'One final native DIRECT answer required')
    result = dict(zip(('direct_unique', 'direct_bytes', 'bounce_unique', 'bounce_bytes', 'retained_ranges', 'retained_bytes'), map(int, rows[0])))
    need(active is not None and result['retained_ranges'] == active['ranges'] and result['retained_bytes'] == active['bytes'],
         'Answer does not match a published retained physical context')
    need(result['direct_unique'] + result['bounce_unique'] == primary['unique_blobs'] and
         result['direct_bytes'] + result['bounce_bytes'] == primary['h2d_bytes'], 'DIRECT/bounce totals do not match primary actual work')
    if require_work: need(result['direct_unique'] > 0 and result['direct_bytes'] > 0, 'Actual covered-RAM direct decode work required')
    result['scope'] = 'Answer decode delta; retained fields are live physical ownership, not deltas or requested cap'
    return result


def compact_evidence(text):
    current = retained(text)
    ends = re.findall(r'[^\r\n]*chunk-major phase ended: C=1024,[^\r\n]*', text)
    need(len(ends) == 1 and 'stopped' not in ends[0] and not RETIRE.search(text) and current['epoch'] == 0,
         'Initial successful compact phase/retained context differs')
    need(text.index('GPU_MISS_PIN retained:') < text.index('chunk-major phase ended:'), 'Retention publication order differs')
    return dict(compact_C=1024, physical_registrations_retained=True, **current)


def phase_evidence(text, generation, remaining):
    ready = re.findall(r'repeat prefill rearm ready: generation(\d+) attempt(\d+), remaining(\d+) C(\d+) A(\d+) AI(\d+), epoch(\d+) issued(\d+) unchanged, pool(\d+) bytes, registration(\d+) bytes; ([0-9.]+) s charged inside feed;', text)
    delta = re.findall(r'repeat prefill generation(\d+) work delta: cold (\d+) sets/(\d+) unpacks; Engram (\d+) fills/(\d+) hits; dense (\d+) fills/(\d+) hits; registration ranges(\d+) after retention; sequence(\d+) retained', text)
    need(len(ready) == len(delta) == 1 and 'rearm declined' not in text and 'rearm not activated' not in text, 'One actual retained-mode rearm/work record required')
    q, d = ready[0], delta[0]; old, new = retired(text, 'epoch'), retained(text)
    need(tuple(map(int, q[:3])) == (generation, generation, remaining) and tuple(map(int, q[3:6])) == (65280, 4096, 128) and
         0 < int(q[6]) == old['epoch'] == new['epoch'] and int(q[7]) == old['issued'] and int(q[8]) > 0 and int(q[9]) == new['bytes'],
         'Retired-beforeHELD to newlyregistered READY geometry/epoch/owner mismatch')
    need(all(old[k] == new[k] for k in ('ranges', 'bytes', 'fingerprint')) and int(d[0]) == generation and
         all(int(v) > 0 for v in d[1:7]) and int(d[7]) == new['ranges'] and int(d[8]) == new['issued'] >= old['issued'],
         'Retained physical owner/workdelta/sequence handoff differs')
    ends = re.findall(r'[^\r\n]*chunk-major phase ended: C=1024,[^\r\n]*', text)
    need(len(ends) == 1 and 'stopped' not in ends[0], 'Successful compact phase missing')
    order = ['GPU_MISS_PIN retired:', 'repeat prefill rearm ready:', 'GPU_MISS_PIN retained:', ' work delta:', 'chunk-major phase ended:']
    need([text.index(s) for s in order] == sorted(text.index(s) for s in order), 'Physical unregister/READY/retention/work/end order differs')
    return {'admission_passed': True, 'generation': generation, 'remaining_feed_tokens': remaining,
            'physical_registration_handoff': {'retired_to_zero_before_HELD': old, 'new_retained_decode': new},
            'registered_bytes': new['bytes'], 'maintenance_seconds_inside_GEN': float(q[10]),
            'physical_zero_after_phase_not_claimed': True}

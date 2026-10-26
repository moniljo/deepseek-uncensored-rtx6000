"""Exact native per-answer/setup/exit parsers; no private config dependencies."""
import re

GPU_ANSWER = re.compile(r'ds4: GPU_MISS answer: layer_windows=(\d+) kernel_windows=(\d+) staged_entries=(\d+) unique_blobs=(\d+) h2d_bytes=(\d+) fallbacks=(\d+); cumulative_entries=(\d+) cumulative_bytes=(\d+)\r?$', re.MULTILINE)


GPU_SETUP = re.compile(r'ds4: GPU_MISS setup: capacity(\d+) maxT(\d+) maxK(\d+) stride(\d+) device_bytes(\d+) host_bounce_bytes(\d+)\r?$', re.MULTILINE)


GPU_EXIT = re.compile(r'ds4: GPU_MISS exit: layer_windows=(\d+) kernel_windows=(\d+) staged_entries=(\d+) unique_blobs=(\d+) h2d_bytes=(\d+) fallbacks=(\d+) drained=(\d+)\r?$', re.MULTILINE)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def gpu_answer(text, require_work):
    matches = GPU_ANSWER.findall(text)
    need(len(matches) == 1, 'Exactly one native per-answer GPU-miss counter record required')
    keys = ('layer_windows', 'kernel_windows', 'staged_entries', 'unique_blobs', 'h2d_bytes',
            'fallbacks', 'cumulative_entries', 'cumulative_bytes')
    result = dict(zip(keys, map(int, matches[0])))
    need(result['kernel_windows'] == result['layer_windows'] and
         result['unique_blobs'] <= result['staged_entries'], 'Native staged-window/dedup accounting differs')
    need(result['cumulative_entries'] >= result['staged_entries'] and
         result['cumulative_bytes'] >= result['h2d_bytes'], 'Native GPU-miss cumulative counters regressed')
    if require_work:
        need(all(result[key] > 0 for key in keys[:5]) and result['fallbacks'] == 0,
             'Actual positive GPU-miss decode work without fallback required')
    result['positive_decode_work_required'] = require_work
    result['scope'] = 'Native answer delta from s_dec, excludes prefill; cumulative fields remain cumulative'
    return result


def gpu_setup(text):
    matches = GPU_SETUP.findall(text)
    need(len(matches) == 1, 'Exactly one native GPU-miss setup record required')
    values = list(map(int, matches[0]))
    need(values[:3] == [24, 4, 6] and all(v > 0 for v in values[3:]), 'Actual24/maxT4/maxK6 staging allocation required')
    return dict(zip(('capacity', 'maxT', 'maxK', 'stride', 'device_bytes', 'host_bounce_bytes'), values))


def gpu_exit(text, last_answer):
    matches = GPU_EXIT.findall(text)
    need(len(matches) == 1, 'Exactly one actual GPU-miss exit-drain record required')
    keys = ('layer_windows', 'kernel_windows', 'staged_entries', 'unique_blobs', 'h2d_bytes', 'fallbacks', 'drained')
    result = dict(zip(keys, map(int, matches[0])))
    need(result['drained'] == 1 and result['kernel_windows'] == result['layer_windows'] and
         0 < result['unique_blobs'] <= result['staged_entries'] == last_answer['cumulative_entries'] and
         result['h2d_bytes'] == last_answer['cumulative_bytes'] > 0, 'Native GPU-miss drain/cumulative accounting differs')
    return result

"""Bench-only, bounded observation of numeric native STAT lines.

The existing engine pump remains the sole stdout reader. Every original line
reaches it unchanged; this observer creates no thread, queue consumer, file,
device event, token, request, or lifecycle action. Use the same tap in both arms.
"""
from collections import deque
import copy
import math
import re
import threading
import time

FIELDS = frozenset(("tok_s", "ms_tok", "vram_hit", "cpu", "ram", "promo", "engram_ms", "disk_mb", "drafts", "skipped"))
REQUIRED = frozenset(("tok_s", "ms_tok", "vram_hit", "cpu", "ram", "promo", "engram_ms"))
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


def parse_stat(line):
    """Reject nonnumeric/unreviewed content instead of retaining arbitrary text."""
    if not isinstance(line, str) or len(line) > 1024 or not line.startswith("STAT "):
        return None
    fields = {}
    for item in line.strip().split()[1:]:
        key, separator, value = item.partition("=")
        if not separator or key not in FIELDS or key in fields or not NUMBER.fullmatch(value):
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0 or (key in ("vram_hit", "drafts") and number > 1):
            return None
        fields[key] = number
    return fields if REQUIRED <= fields.keys() else None


class StatCapture:
    def __init__(self, max_samples=256, max_completed=64):
        if type(max_samples) is not int or not 1 <= max_samples <= 4096 or type(max_completed) is not int or not 1 <= max_completed <= 256:
            raise ValueError("Invalid bounded STAT capture capacity")
        self._lock = threading.Lock()
        self._samples = deque(maxlen=max_samples)
        self._completed = deque(maxlen=max_completed)
        self._sequence = self._count = self._rejected = self._errors = 0

    def observer_error(self):
        with self._lock:
            self._errors += 1

    def observe(self, line):
        if not isinstance(line, str):
            return
        with self._lock:
            if line.startswith("STAT "):
                fields = parse_stat(line)
                if fields is None:
                    self._rejected += 1
                else:
                    self._count += 1
                    self._samples.append({"raw_STAT": line, "fields": fields, "observed_monotonic": time.monotonic()})
            elif line.startswith("DONE ") or line.startswith("ERR "):
                self._sequence += 1
                self._completed.append({"segment": self._sequence, "terminal": "DONE" if line.startswith("DONE ") else "ERR",
                                        "STAT_count": self._count, "retained": len(self._samples), "dropped": self._count - len(self._samples),
                                        "rejected_STAT_count": self._rejected, "observer_errors": self._errors,
                                        "final_STAT": copy.deepcopy(self._samples[-1]) if self._samples else None,
                                        "samples": list(self._samples)})
                self._samples.clear()
                self._count = self._rejected = self._errors = 0

    def mark(self):
        with self._lock:
            return self._sequence

    def completed_since(self, mark):
        with self._lock:
            if type(mark) is not int or mark < 0 or mark > self._sequence:
                raise ValueError("Invalid STAT segment mark")
            if self._completed and mark < self._completed[0]["segment"] - 1:
                raise ValueError("STAT segment history expired")
            return copy.deepcopy([row for row in self._completed if row["segment"] > mark])


class LineTap:
    """One next() on the original iterator, then an observation, then same line."""
    def __init__(self, stream, observe, observer_error):
        self._stream, self._observe, self._observer_error = stream, observe, observer_error

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._stream)
        try:
            self._observe(line)
        except Exception:
            try:
                self._observer_error()
            except Exception:
                pass                       # observation failure must not alter the pipe protocol
        return line

    def __getattr__(self, name):
        return getattr(self._stream, name)  # close/fileno/encoding/etc keep the original owner


def capture_class(base, *, max_samples=256, max_completed=64):
    """Patch only an isolated bench module's StrataEngine class before OwnedNative.

    mark = engine.native_stat_tap.mark() immediately before one Measurement call;
    completed_since(mark) afterwards must contain exactly one DONE segment.
    Require rejected_STAT_count/observer_errors==0 and a nonempty final_STAT.
    The returned class neither parses nor changes native T/DONE or sampling.
    """
    class StatCaptured(base):
        def __init__(self, *args, **kwargs):
            self.native_stat_tap = StatCapture(max_samples, max_completed)
            super().__init__(*args, **kwargs)

        def _pump(self):
            proc = self.proc
            original = proc.stdout
            tap = LineTap(original, lambda line: self.native_stat_tap.observe(line) if self.proc is proc else None,
                          self.native_stat_tap.observer_error)
            proc.stdout = tap
            try:
                return super()._pump()
            finally:
                if proc.stdout is tap:
                    proc.stdout = original

    return StatCaptured

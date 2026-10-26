"""Exact strict control/worker JSON parser."""
import json
from native_support import need

def strict_json(text):
    def pairs(entries):
        out = {}
        for key, value in entries:
            need(key not in out, "Duplicate control JSON key")
            out[key] = value
        return out
    def bad(_):
        raise RuntimeError("Non-finite control JSON")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=bad)
    need(isinstance(value, dict), "Control JSON must be an object")
    return value

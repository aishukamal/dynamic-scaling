"""Build-time check: the only differences between pip freeze before/after are the three added packages."""
import re
def load(p):
    out = {}
    for line in open(p):
        line = line.strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)==(.+)$", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = m.group(2)
    return out
a, b = load("/opt/freeze_before.txt"), load("/opt/freeze_after.txt")
changed = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
allowed = {"kubernetes": (None, "37.0.1"), "websocket-client": (None, "1.9.2"), "durationpy": (None, "0.11")}
unexpected = {k: v for k, v in changed.items() if allowed.get(k) != v}
print("changed:", changed)
assert not unexpected, f"unexpected package changes: {unexpected}"
print("PINS-UNCHANGED")

#!/usr/bin/env python3
"""Per-step generation latency from the orchestrator's [pipeline] DISPATCH/ARRIVE lines (kubectl --timestamps log).
For each batch_idx: first DISPATCH -> first ARRIVE (time to first completed prompt group), -> median ARRIVE, -> last ARRIVE.
Also prints per-worker arrival counts per batch (which rollout worker served how many groups)."""
import re, sys, datetime, collections, statistics as st
def ts(line):
    m = re.match(r"^(\S+?)(\.\d+)?Z\s", line)
    if not m: return None
    base = datetime.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp()
    return base + (float("0" + m.group(2)) if m.group(2) else 0.0)
disp, arr, per_worker = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(collections.Counter)
for line in open(sys.argv[1], errors="replace"):
    t = ts(line)
    if t is None: continue
    m = re.search(r"\[pipeline\] DISPATCH prompt_id=(\S+) prompt_idx=\d+ batch_idx=(\d+)", line)
    if m: disp[int(m.group(2))].append(t); continue
    m = re.search(r"\[pipeline\] ARRIVE prompt_id=(\S+) prompt_idx=\d+ batch_idx=(\d+) .*worker_id=(\S+)", line)
    if m: arr[int(m.group(2))].append(t); per_worker[int(m.group(2))][m.group(3)] += 1
print("batch | n_dispatch | n_arrive | first_dispatch->first_arrive s | ->median_arrive s | ->last_arrive s | dispatch span s | arrivals per worker")
for b in sorted(disp):
    if not arr.get(b): continue
    d0 = min(disp[b]); a = sorted(arr[b])
    print(f"{b} | {len(disp[b])} | {len(a)} | {a[0]-d0:.1f} | {st.median(a)-d0:.1f} | {a[-1]-d0:.1f} | {max(disp[b])-d0:.1f} | {dict(per_worker[b])}")

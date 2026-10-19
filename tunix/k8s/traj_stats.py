#!/usr/bin/env python3
"""Per-step trajectory statistics from a rollout worker log ([DEBUG_INFERENCE][Trajectory] lines).
Usage: traj_stats.py <rollout-worker.log> [more logs...]"""
import re, sys, collections, statistics as st
rx = re.compile(r"traj_id=(\S+), global_step=(\d+), status=(\S+), reward=([0-9.]+), steps=(\d+), total_time=([0-9.]+)s, model_time=([0-9.]+)s, env_time=([0-9.]+)s, reward_time=([0-9.]+)s.*?completion_tokens=(\d+), preemptions=(\d+)")
by = collections.defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = rx.search(line)
        if m:
            by[int(m.group(2))].append(dict(status=m.group(3), reward=float(m.group(4)), turns=int(m.group(5)), total=float(m.group(6)), model=float(m.group(7)), env=float(m.group(8)), rew=float(m.group(9)), tok=int(m.group(10)), pre=int(m.group(11))))
print("step | n | SUCCEEDED/other | turns mean | total mean / p50 / max s | model mean | env mean | reward_time mean | sum(model) s | tok mean | reward>0 | preemptions")
for s in sorted(by):
    xs = by[s]; n = len(xs)
    succ = sum(1 for x in xs if x["status"] == "SUCCEEDED")
    tot = [x["total"] for x in xs]
    print(f"{s} | {n} | {succ}/{n-succ} | {st.mean(x['turns'] for x in xs):.1f} | {st.mean(tot):.1f} / {st.median(tot):.1f} / {max(tot):.1f} | {st.mean(x['model'] for x in xs):.1f} | {st.mean(x['env'] for x in xs):.1f} | {st.mean(x['rew'] for x in xs):.1f} | {sum(x['model'] for x in xs):.0f} | {st.mean(x['tok'] for x in xs):.0f} | {sum(1 for x in xs if x['reward'] > 0)} | {sum(x['pre'] for x in xs)}")

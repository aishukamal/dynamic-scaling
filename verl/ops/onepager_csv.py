#!/usr/bin/env python3
"""onepager_csv.py <versions csv> <out csv> : the coordinator's one-pager layout (step, wall, wait, train, sync, live, held, chip_s, events,
sandbox_pending, sandbox_confounded) from a build_csv.py per-version CSV. held = 4 trainer chips + 4 x time-weighted active samplers."""
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f); w.writerow(["step", "wall", "wait", "train", "sync", "live", "held", "chip_s", "events", "sandbox_pending", "sandbox_confounded"])
    for r in rows:
        w.writerow([r["version"], r["step_s"], r["wait_s"], r["train_s"], r["sync_s"], r["active_replicas"], r["held_chips"], r["chip_s"], r["events"], r["sandbox_pending"], r["sandbox_confounded"]])
print(f"{len(rows)} rows -> {sys.argv[2]}")

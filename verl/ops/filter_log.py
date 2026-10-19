#!/usr/bin/env python3
"""filter_log.py <job_id> <out_path> : run inside the head pod; keeps only the analysis-relevant lines of `ray job logs`."""
import re, subprocess, sys
job, out = sys.argv[1], sys.argv[2]
pat = re.compile(r"step:[0-9]+ -|\[POOL\]|RAIDEN TELEMETRY \| Orchestrator|destination sampler|Traceback|ActorDied|EngineDead|init engine|KV cache size|Aborted [0-9]+ requests|RAIDEN PARITY|spawn|No available servers|Error in _run_prompt|health|swe_agent|RuntimeError|Error executing|throughput|Running: |TURN1|TPUM3 layout|SUCC|FAIL")
n = 0
with open(out, "w") as f:
    p = subprocess.Popen(["ray", "job", "logs", job], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace")
    for line in p.stdout:
        if pat.search(line):
            f.write(line); n += 1
print(f"kept {n} lines -> {out}")

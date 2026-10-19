#!/usr/bin/env python3
"""ctl.py <host> metrics|status|scale <N> : stdlib client for the tpu_pool_control HTTP endpoint (the image has no curl)."""
import json, sys, urllib.request
host, cmd = sys.argv[1], sys.argv[2]
base = f"http://{host}:8787"
if cmd == "metrics":
    print(urllib.request.urlopen(f"{base}/metrics", timeout=8).read().decode())
elif cmd == "status":
    print(urllib.request.urlopen(f"{base}/status", timeout=8).read().decode())
elif cmd == "scale":
    req = urllib.request.Request(f"{base}/scale", data=json.dumps({"standalone": int(sys.argv[3])}).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    print(urllib.request.urlopen(req, timeout=8).read().decode())

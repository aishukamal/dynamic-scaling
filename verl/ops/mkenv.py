#!/usr/bin/env python3
"""mkenv.py <name> [pool_json or '-'] [extra KEY=VAL ...] -> runtime_env.<name>.json (base env + VERL_TPU_POOL_CONTROL + extras)."""
import json, sys
base = json.load(open(__import__("os").environ.get("RT_BASE","runtime_env.base.json")))["env_vars"]
name = sys.argv[1]
pool = sys.argv[2] if len(sys.argv) > 2 else "-"
if pool != "-":
    json.loads(pool); base["VERL_TPU_POOL_CONTROL"] = pool
for kv in sys.argv[3:]:
    k, v = kv.split("=", 1); base[k] = v
json.dump({"env_vars": base}, open(f"runtime_env.{name}.json", "w"))
print(f"runtime_env.{name}.json", {k: base[k] for k in base if k.startswith("VERL_TPU")})

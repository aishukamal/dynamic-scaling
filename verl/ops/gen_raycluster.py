#!/usr/bin/env python3
"""gen_raycluster.py <name> <trainer host short> <sampler hosts short, comma> <max sampler replicas> > yaml
Renders k8s/raycluster-tpum3.yaml for another cluster: name, trainer host (head + tpu-group-0), sampler worker group pinned to the
given hosts (replicas = number of hosts given, maxReplicas as given; nodeAffinity In on the listed hosts plus any extra hosts in
EXTRA_HOSTS env for cold joins), image digest kept."""
import os, re, sys
name, th, shs, mx = sys.argv[1], sys.argv[2], sys.argv[3].split(","), int(sys.argv[4])
y = open(os.path.join(os.path.dirname(__file__), "k8s", "raycluster-tpum3.applied.yaml")).read()
full = lambda h: f"gke-tpu-dac1e83d-{h}"
y = y.replace("name: ray-tpum3\n", f"name: {name}\n", 1)
y = y.replace("kubernetes.io/hostname: gke-tpu-dac1e83d-8f4h", f"kubernetes.io/hostname: {full(th)}")
hosts = [full(h) for h in shs] + [full(h) for h in os.environ.get("EXTRA_HOSTS", "").split(",") if h]
y = re.sub(r"values: \[gke-tpu-dac1e83d-8w3s, gke-tpu-dac1e83d-t529, gke-tpu-dac1e83d-tdsr\]", "values: [" + ", ".join(hosts) + "]", y)
y = re.sub(r"(groupName: tpu-group-1\n    numOfHosts: 1\n    replicas: )\d+(\n    minReplicas: )\d+(\n    maxReplicas: )\d+",
           lambda m: f"{m.group(1)}{len(shs)}{m.group(2)}{min(1, len(shs))}{m.group(3)}{mx}", y)
y = y.replace("TPU_LOG_DIR, value: /data/logs/tpu-m3}", f"TPU_LOG_DIR, value: /data/logs/{name}}}")
sys.stdout.write(y)

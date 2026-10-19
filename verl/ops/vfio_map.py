# vfio_map.py : pid -> (vfio devices, comm, cmdline head) for every process holding /dev/vfio/* (run inside a hostPID pod)
import os, glob, json
out = {}
for fd in glob.glob('/proc/[0-9]*/fd/*'):
    try:
        t = os.readlink(fd)
    except OSError:
        continue
    if t.startswith('/dev/vfio/') and t != '/dev/vfio/vfio':
        pid = fd.split('/')[2]
        out.setdefault(pid, set()).add(t)
res = []
for pid, devs in sorted(out.items(), key=lambda kv: sorted(kv[1])):
    try:
        cmd = open(f'/proc/{pid}/cmdline', 'rb').read().replace(b'\0', b' ').decode(errors='replace')[:90]
        comm = open(f'/proc/{pid}/comm').read().strip()
    except OSError:
        cmd, comm = '?', '?'
    res.append({"pid": int(pid), "vfio": sorted(devs), "comm": comm, "cmd": cmd})
print(json.dumps(res))

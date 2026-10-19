#!/usr/bin/env python3
"""steps.csv (analyze_sg.py) -> markdown per-step table. Usage: md_table.py steps.csv"""
import csv, sys
def f(x, p=1):
    try: return f"{float(x):.{p}f}" if x not in ("", "None", None) else "-"
    except ValueError: return "-"
rows = list(csv.DictReader(open(sys.argv[1])))
print("| step | wall s | wait s | wait share | slot util peak | slot util mean | in-flight peak / cap | pending queue peak | active hosts | chips held | chip-s | sandbox Pending begin / end | sandbox pods (end) | confounded | band | gate verdict | decision |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    dec = r["decision"] if r["decision"] not in ("", "None") else "-"
    if dec != "-" and dec != "hold": dec = f"**{dec}**"
    pend = f"{r['sbx_pending_begin'] if r['sbx_pending_begin'] not in ('', 'None') else '-'} / {r['sbx_pending_end'] if r['sbx_pending_end'] not in ('', 'None') else '-'}"
    print(f"| {r['step']} | {f(r['wall'])} | {f(r['wait'])} | {f(r['share'],3)} | {f(r['util_peak'],3)} | {f(r['util_mean'],3)} | {r['inflight_peak'] or '-'} / {r['cap'] or '-'} | {r['pending_peak'] if r['pending_peak'] not in ('', 'None') else '-'} | {r['live']} | {r['held']} | {f(r['chip_s'],0)} | {pend} | {r['sbx_total_end'] if r['sbx_total_end'] not in ('', 'None') else '-'} | {r.get('sandbox_confounded','')} | {r['band'] if r['band'] not in ('', 'None') else '-'} | {r['verdict'] if r['verdict'] not in ('', 'None') else '-'} | {dec} |")

import sys, json
# usage: render_chipduty.py duty.json out.svg stack_label events_json phases_json
# duty.json rows: [step, trainer_pct, sampler_pct, n_sampler_hosts]
data=json.load(open(sys.argv[1])); out=sys.argv[2]; stack=sys.argv[3]
events=json.loads(sys.argv[4]); phases=json.loads(sys.argv[5])
steps=[r[0] for r in data]; tr=[r[1]/100 for r in data]; sa=[r[2]/100 for r in data]
blue,orange,ink,ink2,grid,shade="#2a78d6","#eb6834","#0b0b0b","#52514e","#e6e5e1","#f1f0ec"
W=1100; L,R=70,30; pw=W-L-R; T,ph=40,240; H=360
x0,x1=steps[0],steps[-1]; X=lambda s: L+(s-x0)/(x1-x0)*pw; Y=lambda v: T+ph-v*ph
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{ink}">',f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>']
for k,(a,b,lab) in enumerate(phases):
    if k%2==1: o.append(f'<rect x="{X(a)}" y="{T}" width="{X(b)-X(a)}" height="{ph}" fill="{shade}" opacity="0.7"/>')
    o.append(f'<text x="{(X(a)+X(b))/2}" y="{T-8}" text-anchor="middle" fill="{ink2}" font-size="11">{lab}</text>')
o.append(f'<text x="{L-50}" y="{T+ph/2}" transform="rotate(-90 {L-50},{T+ph/2})" text-anchor="middle" fill="{ink2}">duty cycle</text>')
for v in (0,0.25,0.5,0.75,1.0):
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="{grid}"/>'); o.append(f'<text x="{L-8}" y="{Y(v)+4}" text-anchor="end" fill="{ink2}">{int(v*100)}%</text>')
for series,col in ((sa,orange),(tr,blue)):
    o.append('<polyline points="'+" ".join(f"{X(s)},{Y(v)}" for s,v in zip(steps,series))+f'" fill="none" stroke="{col}" stroke-width="2"/>')
    o+=[f'<circle cx="{X(s)}" cy="{Y(v)}" r="3" fill="{col}" stroke="#fcfcfb" stroke-width="1.5"/>' for s,v in zip(steps,series)]
for s,lab,dy in events:
    o.append(f'<line x1="{X(s)}" x2="{X(s)}" y1="{T}" y2="{T+ph}" stroke="{ink2}" stroke-dasharray="2,3"/>')
    right=(s-x0)/(x1-x0)>0.72; xx=X(s)-3 if right else X(s)+3; anc="end" if right else "start"
    o.append(f'<text x="{xx}" y="{T+16+dy}" text-anchor="{anc}" fill="{ink}" font-size="10.5">{lab}</text>')
o+=[f'<text x="{X(s)}" y="{T+ph+14}" text-anchor="middle" fill="{ink2}">{s}</text>' for s in range(x0,x1+1)]
o.append(f'<text x="{L+pw/2}" y="{T+ph+34}" text-anchor="middle" fill="{ink2}">training step ({stack}, dynamically scaled run; x marks the end of each step). Per-step mean of the per-minute accelerator duty cycle per host.</text>')
o.append(f'<line x1="70" x2="92" y1="338" y2="338" stroke="{blue}" stroke-width="2"/><text x="98" y="342" fill="{ink2}">trainer host</text>')
o.append(f'<line x1="220" x2="242" y1="338" y2="338" stroke="{orange}" stroke-width="2"/><text x="248" y="342" fill="{ink2}">sampler hosts (mean over the hosts serving for the whole step)</text>')
o.append('</svg>'); open(out,'w').write("\n".join(o)); print("written",out)

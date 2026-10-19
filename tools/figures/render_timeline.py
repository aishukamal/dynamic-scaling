import csv, sys, json, os
# usage: render_timeline.py steps.csv out_prefix stack_label ref_json events_json phases_json [ceiling_json]
src,prefix,stack=sys.argv[1],sys.argv[2],sys.argv[3]
refs=json.loads(sys.argv[4]); events=json.loads(sys.argv[5]); phases=json.loads(sys.argv[6])
ceil=json.loads(sys.argv[7]) if len(sys.argv)>7 else None
rows=[r for r in csv.DictReader(open(src)) if int(r['step'])>=1]
f=lambda r,k: float(r[k])
steps=[int(r['step']) for r in rows]; walls=[f(r,'wall') for r in rows]
waitsh=[f(r,'wait')/f(r,'wall') for r in rows]; duty=[f(r,'train')/f(r,'wall') for r in rows]
blue,ink,ink2,grid,shade="#2a78d6","#0b0b0b","#52514e","#e6e5e1","#f1f0ec"
W=1100; L,R=70,30; pw=W-L-R
x0,x1=steps[0],steps[-1]; X=lambda s: L+(s-x0)/(x1-x0)*pw
def header(H): return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{ink}">',f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>']
def frame(T,ph,ylabel):
    o=[]
    for k,(a,b,lab) in enumerate(phases):
        if k%2==1: o.append(f'<rect x="{X(a)}" y="{T}" width="{X(b)-X(a)}" height="{ph}" fill="{shade}" opacity="0.7"/>')
        o.append(f'<text x="{(X(a)+X(b))/2}" y="{T-8}" text-anchor="middle" fill="{ink2}" font-size="11">{lab}</text>')
    o.append(f'<text x="{L-50}" y="{T+ph/2}" transform="rotate(-90 {L-50},{T+ph/2})" text-anchor="middle" fill="{ink2}">{ylabel}</text>')
    return o
def xaxis(T,ph,title):
    o=[f'<text x="{X(s)}" y="{T+ph+14}" text-anchor="middle" fill="{ink2}">{s}</text>' for s in range(x0,x1+1)]
    o.append(f'<text x="{L+pw/2}" y="{T+ph+34}" text-anchor="middle" fill="{ink2}">{title}</text>'); return o
def evlines(T,ph):
    o=[]
    for s,lab,dy in events:
        o.append(f'<line x1="{X(s)}" x2="{X(s)}" y1="{T}" y2="{T+ph}" stroke="{ink2}" stroke-dasharray="2,3"/>')
        right = (s-x0)/(x1-x0) > 0.72
        xx = X(s)-3 if right else X(s)+3; anc = "end" if right else "start"
        o.append(f'<text x="{xx}" y="{T+16+dy}" text-anchor="{anc}" fill="{ink}" font-size="10.5">{lab}</text>')
    return o
def line(Y,series,r=3.5):
    o=['<polyline points="'+" ".join(f"{X(s)},{Y(v)}" for s,v in zip(steps,series))+f'" fill="none" stroke="{blue}" stroke-width="2"/>']
    o+=[f'<circle cx="{X(s)}" cy="{Y(v)}" r="{r}" fill="{blue}" stroke="#fcfcfb" stroke-width="1.5"/>' for s,v in zip(steps,series)]
    return o
def dashed(Y,val,lab,xf=None):
    w=len(lab)*6.3+8; xx=L+pw*(xf if xf is not None else float(os.environ.get('REF_X','0.42')))
    return [f'<line x1="{L}" x2="{L+pw}" y1="{Y(val)}" y2="{Y(val)}" stroke="{ink2}" stroke-dasharray="6,4" stroke-width="1.2"/>',
            f'<rect x="{xx-4}" y="{Y(val)-17}" width="{w}" height="15" fill="#fcfcfb" opacity="0.9"/>',
            f'<text x="{xx}" y="{Y(val)-5}" fill="{ink2}" font-size="11">{lab}</text>']
# A: step time
T,ph=40,300; H=T+ph+60; ymax=float(os.environ['YMAX']) if os.environ.get('YMAX') else max(460,max(walls)*1.08); Y=lambda v: T+ph-(v/ymax)*ph
o=header(H)+frame(T,ph,"step time (s)")
step=100 if ymax<=600 else 200; v=step
while v<ymax:
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="{grid}"/>'); o.append(f'<text x="{L-8}" y="{Y(v)+4}" text-anchor="end" fill="{ink2}">{v}</text>'); v+=step
for ref in refs: o+=dashed(Y,*ref)
o+=line(Y,walls)+evlines(T,ph)+xaxis(T,ph,f"training step ({stack}, dynamically scaled run; x marks the end of each step). Solid: step time. Dashed: fixed-ratio averages.")
o.append('</svg>'); open(prefix+'-steptime.svg','w').write("\n".join(o))
# B: wait share, C: duty cycle
for suffix,series,ylab,c in (("wait",waitsh,"share of step the trainer spent waiting for samples",None),("duty",duty,"trainer duty cycle (training phase time / wall time, from phase timers)",ceil)):
    T,ph=40,240; H=T+ph+60; Y=lambda v: T+ph-v*ph
    o=header(H)+frame(T,ph,ylab)
    for v in (0,0.25,0.5,0.75,1.0):
        o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(v)}" y2="{Y(v)}" stroke="{grid}"/>'); o.append(f'<text x="{L-8}" y="{Y(v)+4}" text-anchor="end" fill="{ink2}">{int(v*100)}%</text>')
    if c: o+=dashed(Y,c[0],c[1])
    o+=line(Y,series,3)+evlines(T,ph)+xaxis(T,ph,f"training step ({stack}, dynamically scaled run; x marks the end of each step)")
    o.append('</svg>'); open(prefix+'-'+suffix+'.svg','w').write("\n".join(o))
print("written", prefix, "steptime wait duty")

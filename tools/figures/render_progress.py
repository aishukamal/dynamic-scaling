import csv, sys, json
# usage: render_progress.py dyn_steps.csv out.svg fixed_json markers_json
# fixed_json: [[label, step0_s, mean_s], ...]  markers_json: [[minutes, label], ...]
rows=[r for r in csv.DictReader(open(sys.argv[1]))]; out=sys.argv[2]
fixed=json.loads(sys.argv[3]); markers=json.loads(sys.argv[4])
walls=[float(r['wall']) for r in rows]; n=len(walls)
cum=[0.0]
for w in walls: cum.append(cum[-1]+w)
dyn=[(t/60,k) for k,t in enumerate(cum)]  # (minutes, steps completed)
fx=[(lab,[((s0+k*m)/60 if k>0 else 0.0,k) for k in range(0,n+1)]) for lab,s0,m in fixed]
# note: fixed step 0 counted as s0 for k=1
fx=[(lab,[(0.0,0)]+[((s0+(k-1)*m)/60,k) for k in range(1,n+1)]) for lab,s0,m in fixed]
W,H=1000,480; L,R,T,B=70,30,40,70; pw=W-L-R; ph=H-T-B
xmax=max(max(t for t,_ in s) for _,s in fx+[("d",dyn)])*1.04; xmax=15*((int(xmax)//15)+1)
X=lambda t: L+t/xmax*pw; Y=lambda k: T+ph-k/n*ph
blue,ink,ink2,grid="#2a78d6","#0b0b0b","#52514e","#e6e5e1"
o=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Helvetica, Arial, sans-serif" font-size="12" fill="{ink}">',f'<rect width="{W}" height="{H}" fill="#fcfcfb"/>']
for k in range(0,n+1,4):
    o.append(f'<line x1="{L}" x2="{L+pw}" y1="{Y(k)}" y2="{Y(k)}" stroke="{grid}"/>'); o.append(f'<text x="{L-8}" y="{Y(k)+4}" text-anchor="end" fill="{ink2}">{k}</text>')
t=0
while t<=xmax:
    o.append(f'<text x="{X(t)}" y="{T+ph+16}" text-anchor="middle" fill="{ink2}">{t}</text>'); t+=15
o.append(f'<text x="{L+pw/2}" y="{T+ph+38}" text-anchor="middle" fill="{ink2}">wall-clock minutes from job start</text>')
o.append(f'<text x="{L-45}" y="{T+ph/2}" transform="rotate(-90 {L-45},{T+ph/2})" text-anchor="middle" fill="{ink2}">training steps completed</text>')
for lab,s in fx:
    o.append('<polyline points="'+" ".join(f"{X(t)},{Y(k)}" for t,k in s)+f'" fill="none" stroke="{ink2}" stroke-width="1.5" stroke-dasharray="6,4"/>')
    tend=s[-1][0]; o.append(f'<text x="{X(tend)}" y="{Y(n)-8}" text-anchor="end" fill="{ink2}" font-size="11">{lab}: {round(tend)} min</text>')
o.append('<polyline points="'+" ".join(f"{X(t)},{Y(k)}" for t,k in dyn)+f'" fill="none" stroke="{blue}" stroke-width="2.5"/>')
o.append(f'<text x="{X(dyn[-1][0])}" y="{Y(n)+16}" text-anchor="end" fill="{blue}" font-size="11">dynamic: {round(dyn[-1][0])} min</text>')
for i,(m,lab) in enumerate(markers):
    o.append(f'<line x1="{X(m)}" x2="{X(m)}" y1="{T}" y2="{T+ph}" stroke="{ink2}" stroke-dasharray="2,3"/>')
    o.append(f'<text x="{X(m)+4}" y="{T+14+14*i}" fill="{ink}" font-size="10.5">{lab}</text>')
o.append('</svg>'); open(out,'w').write("\n".join(o)); print("written",out, "dyn", round(dyn[-1][0],1), [(lab,round(s[-1][0],1)) for lab,s in fx])

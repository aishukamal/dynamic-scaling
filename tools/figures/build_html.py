import re, html
import sys
SRC=sys.argv[1] if len(sys.argv)>1 else "ONE-PAGER.md"
OLD=set(l.strip() for l in open(sys.argv[2]).read().split("\n")) if len(sys.argv)>2 else None
src=open(SRC).read().split('\n'); out=[]; i=0; inlist=[]; para=[]
def chg(lines):
    return OLD is not None and any(l.strip() and l.strip() not in OLD for l in lines)
HL=' class="chg"'
def inline(t):
    t=html.escape(t, quote=False).replace("&lt;br&gt;","<br>"); t=re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t=re.sub(r'!\[(.*?)\]\((.*?)\)', r'<img src="\2" alt="\1" style="max-width:100%;border:1px solid #e6e5e1;margin:8px 0">', t)
    return re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', t)
def flush():
    global para
    if para: out.append('<p'+(HL if chg(para) else '')+'>'+inline(' '.join(para))+'</p>'); para=[]
def close(level=0):
    while len(inlist)>level: out.append('</ul>'); inlist.pop()
while i<len(src):
    line=src[i]
    if line.startswith('|'):
        flush(); close(); rows=[]
        raw=[]
        while i<len(src) and src[i].startswith('|'):
            rows.append([c.strip() for c in src[i].strip().strip('|').split('|')]); raw.append(src[i]); i+=1
        keep=[(r,l) for r,l in zip(rows,raw) if not all(re.fullmatch(r'-+',c) for c in r)]
        out.append('<table>')
        for k,(r,l) in enumerate(keep):
            tag='th' if k==0 else 'td'; out.append('<tr'+(HL if chg([l]) else '')+'>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in r)+'</tr>')
        out.append('</table>'); continue
    m=re.match(r'^(#{1,3}) (.*)',line)
    if m: flush(); close(); out.append(f'<h{len(m.group(1))}'+(HL if chg([line]) else '')+f'>{inline(m.group(2))}</h{len(m.group(1))}>'); i+=1; continue
    m=re.match(r'^(\s*)- (.*)',line)
    if m:
        flush(); level=len(m.group(1))//2+1
        while len(inlist)<level: out.append('<ul>'); inlist.append(1)
        close(level); out.append('<li'+(HL if chg([line]) else '')+'>'+inline(m.group(2))+'</li>'); i+=1; continue
    if line.strip()=='': flush(); close(); i+=1; continue
    para.append(line.strip()); i+=1
flush(); close()
css="body{font-family:Helvetica,Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 24px;color:#0b0b0b;line-height:1.45}h1{font-size:26px}h2{font-size:20px;margin-top:32px;border-bottom:1px solid #e6e5e1;padding-bottom:4px}h3{font-size:16px}table{border-collapse:collapse;margin:12px 0;font-size:14px}th,td{border:1px solid #d9d8d3;padding:6px 10px;text-align:left;vertical-align:top}th{background:#f1f0ec}ul{padding-left:22px}li{margin:4px 0}p{margin:8px 0}.chg{background:#fff3a3}tr.chg td{background:#fff3a3}"
open(SRC.replace(".md",".html"),"w").write(f'<!doctype html><html><head><meta charset="utf-8"><title>Elastic sampler pools for agentic RL on TPU</title><style>{css}</style></head><body>\n'+'\n'.join(out)+'\n</body></html>')
print("rebuilt")

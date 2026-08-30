import re, json, sys, os
SP=os.environ['SP']
SITES=["Bobbili","Gogada","Panuku","Reyavani","Kotha","Suregadi","Chinachipuru","TELUGU"]
# OCR mangles site names; match leniently
def site_of(tok):
    t=re.sub(r'[^A-Za-z]','',tok).lower()
    if not t: return None
    best=None
    for s in SITES:
        a=s.lower()
        if t==a or (len(t)>=4 and (t.startswith(a[:5]) or a.startswith(t[:5]))): best=s
    return best
ITEM=re.compile(r'^[.\s]*(\d{1,3})\s*[.,]\s*(.+?)\s*$')
recs=[]; col=None; item=None; gloss=None; last=None
for ln in open(f'{SP}/gadaba-ocr.txt',encoding='utf-8'):
    ln=ln.rstrip('\n')
    if ln.startswith('@@'):
        col=ln[3:].strip(); last=None; continue
    s=ln.strip()
    if not s: continue
    if re.fullmatch(r'\d{1,3}', s): continue        # page number
    if s.startswith('A.3'): continue
    parts=s.split()
    sm=site_of(parts[0]) if parts else None
    m=ITEM.match(s)
    if m and not sm and len(m.group(2))<40 and not m.group(2)[0].isdigit():
        item=int(m.group(1)); gloss=m.group(2).strip(); last=None; continue
    if sm:
        rest=parts[1:]; last=sm
    else:
        rest=parts
    if not rest: continue
    g=rest[0]
    grp=None
    if re.fullmatch(r'[0-9lLIO]{1,2}', g):
        grp=g.translate(str.maketrans('lLIO','1110')); rest=rest[1:]
    form=' '.join(rest)
    if not form: continue
    recs.append({"col":col,"item":item,"gloss":gloss,"site":last or sm,"group":grp,"ocr":form})
json.dump(recs,open(f'{SP}/scaffold.json','w'),ensure_ascii=False,indent=0)
print("records:",len(recs))
items=sorted({r['item'] for r in recs if r['item']})
print("items:",len(items),"range",items[0],items[-1])
missing=[i for i in range(1,max(items)+1) if i not in items]
print("missing items:",missing)
from collections import Counter
print("per-site:",Counter(r['site'] for r in recs))

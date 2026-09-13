import json,statistics,re,gzip
from pathlib import Path
ROOT=Path(__file__).parent

def lines(words,tolerance=14):
    groups=[]
    for w in sorted(words,key=lambda w:(w['y']+w['h']/2,w['x'])):
        cy=w['y']+w['h']/2
        g=next((g for g in groups[-3:] if abs(statistics.mean(t['y']+t['h']/2 for t in g)-cy)<tolerance),None)
        if g is None: groups.append([w])
        else:g.append(w)
    return [sorted(g,key=lambda w:w['x']) for g in groups]

def modal(values,step):
    counts={}
    for v in values:
        k=round(v/step);counts[k]=counts.get(k,0)+1
    best=max(counts,key=counts.get)
    return statistics.median(v for v in values if round(v/step)==best)

def parse(name,page):
    with gzip.open(ROOT/f'{name}-ocr.json.gz','rt') as f: d=json.load(f)[str(page)]
    W,H=d['width'],d['height']
    top=(.23 if name=='kudali' else .30) if page=={'kudali':105,'konkani':128}[name] else .13
    # Header positions vary: identify the main header and discard through it.
    if page not in (105,128):
        hdr=[w['y']+w['h'] for w in d['words'] if w['y']<.15*H and re.search('KUD|VOCAB|KONK|SOUTH|KANARA',w['text'],re.I)]
        if hdr:top=(max(hdr)+12)/H
    bottom=.515 if name=='kudali' and page==160 else .89
    words=[w for w in d['words'] if top*H<w['y']<bottom*H and len(re.sub(r'\W','',w['text']))]
    if name=='kudali':
        gloss=modal([w['x'] for w in words if .46*W<w['x']<.65*W],.012*W)
        sections=[(.12*W,W,gloss-.015*W)]
        headstarts=[modal([w['x'] for w in words if .15*W<w['x']<.40*W],.012*W)]
    else:
        # Each side is an independent form/gloss table. Modes locate their column starts.
        starts=[modal([w['x'] for w in words if a*W<w['x']<b*W],.012*W) for a,b in [( .10,.27),(.29,.44),(.47,.64),(.65,.82)]]
        split=starts[2]-.025*W
        sections=[(starts[0]-.06*W,split,starts[1]-.015*W),(split,.96*W,starts[3]-.015*W)]
    if name=='konkani': headstarts=[starts[0],starts[2]]
    result=[]
    for col,(lo,hi,boundary) in enumerate(sections,1):
        entries=[]
        for group in lines([w for w in words if lo<=w['x']<hi],.0065*H):
            left=[w for w in group if w['x']<boundary];right=[w for w in group if w['x']>=boundary]
            # Low-confidence ink far inside the otherwise blank head field is not a new head.
            if left and min(w['x'] for w in left)>headstarts[col-1]+.04*W and max(w['confidence'] for w in left)<50:
                left=[]
            form=' '.join(w['text'] for w in left);gloss=' '.join(w['text'] for w in right)
            if left:
                # Indented left-only lines continue a compound form.
                if not right and entries and min(w['x'] for w in left)>entries[-1]['x']+.02*W:
                    entries[-1]['left']+=' '+form;entries[-1]['raw_words']+=group
                else:
                    entries.append(dict(source=name,pdf_page=page,printed_page=page-{'konkani':8,'kudali':10}[name],col=col,ordinal=len(entries)+1,x=min(w['x'] for w in left),y=min(w['y'] for w in group),left=form,gloss=gloss,raw_words=group))
            elif entries:
                entries[-1]['gloss']+=' '+gloss;entries[-1]['raw_words']+=group
            elif right:
                entries.append(dict(source=name,pdf_page=page,printed_page=page-{'konkani':8,'kudali':10}[name],col=col,ordinal=0,x=0,y=min(w['y'] for w in group),left='',gloss=gloss,raw_words=group))
        result+=entries
    return result


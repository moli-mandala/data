"""OCR/layout library for the ESIJa (Rastorgueva & Edel'man 2000-2007) lemma layer.

Stages: text-layer lines (pdfplumber) -> outdented headword lines -> article segmentation ->
headword-line crops -> Tesseract script/Latin hOCR with character boxes -> geometric diacritic
analysis -> fusion with the ABBYY text layer -> skeleton agreement status -> Russian gloss parse.
All thresholds are fixed here so the extraction is reproducible.
"""
import re, statistics, unicodedata
from PIL import Image

# --- crop trimming ---
def band(im, thr=128):
    im=im.convert('L'); w,h=im.size; px=im.load()
    ink=[sum(1 for x in range(w) if px[x,y]<thr) for y in range(h)]
    segs=[]; y=0
    while y<h:
        if ink[y]>0:
            s=y
            while y<h and ink[y]>0: y+=1
            segs.append([s,y,sum(ink[s:y])])
        else: y+=1
    if not segs: return im
    mi=max(range(len(segs)), key=lambda i: segs[i][2])
    a,b=segs[mi][0],segs[mi][1]; H=b-a
    # merge small neighbours (diacritics) separated by small gaps
    i=mi-1
    while i>=0 and a-segs[i][1]<=max(4,H*0.12) and (segs[i][1]-segs[i][0])<0.45*H: a=segs[i][0]; i-=1
    i=mi+1
    while i<len(segs) and segs[i][0]-b<=max(4,H*0.12) and (segs[i][1]-segs[i][0])<0.45*H: b=segs[i][1]; i+=1
    H0=b-a
    return im.crop((0,max(0,a-max(6,int(0.3*H0))),w,min(h,b+max(6,int(0.35*H0)))))

# --- geometry ---
XLET=set('acemnorsuvwxz')
BASE={'ø':'g','ą':'g','ĝ':'g','ğ':'g','ġ':'g','Ğ':'g','ǫ':'g','ð':'ϑ','þ':'ϑ','ħ':'h','ň':'n','ř':'r','ż':'z','Z':'z','S':'s','I':'i','İ':'i','Ï':'i','Í':'i','Ī':'i','Y':'y','U':'u','Ù':'u','A':'a','D':'d','T':'t','K':'k','B':'b','X':'x','L':'l','Ž':'z','Ś':'s','Ĉ':'c','ĉ':'c','ć':'c','ĵ':'j','į':'i','ė':'e','é':'e'}
def strip(ch):
    return unicodedata.normalize('NFD',ch)[0]
def components(im):
    W,H=im.size; px=im.load(); seen=bytearray(W*H); comps=[]
    for y in range(H):
        for x in range(W):
            if seen[y*W+x] or px[x,y]>=128: continue
            st=[(x,y)]; seen[y*W+x]=1; pts=[]
            while st:
                a,b=st.pop(); pts.append((a,b))
                for da in (-1,0,1):
                    for db in (-1,0,1):
                        u,v=a+da,b+db
                        if 0<=u<W and 0<=v<H and not seen[v*W+u] and px[u,v]<128:
                            seen[v*W+u]=1; st.append((u,v))
            comps.append(pts)
    return comps
def analyse(path, chars):
    im=Image.open(path).convert('L'); W,H=im.size; px=im.load()
    letters=[(c,b) for c,b in chars if c.strip()]
    xl=[(c,b) for c,b in letters if strip(BASE.get(c,c)).lower() in XLET]
    if len(xl)<2:
        xl=[(c,b) for c,b in letters if strip(BASE.get(c,c)).lower() in set('abdehiklmnortuvxz')]
    if len(xl)<2: return None
    base=statistics.median(b[3] for c,b in xl)
    tops=[b[1] for c,b in xl if c in XLET] or [b[1] for c,b in xl]
    xtop=statistics.median(tops)
    # x-height band from the horizontal ink projection (robust to boxes that swallow marks)
    prof=[sum(1 for x in range(W) if px[x,y]<128) for y in range(H)]
    if prof and max(prof)>0:
        rows=[y for y,v in enumerate(prof) if v>=0.35*max(prof)]
        pb,pt=rows[-1]+1,rows[0]
        if pb-pt>=4 and abs(pb-base)>0.25*max(4,base-xtop):
            base,xtop=pb,pt
    xh=max(4,base-xtop)
    marks=[]
    for pts in components(im):
        xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
        bx0,by0,bx1,by1=min(xs),min(ys),max(xs)+1,max(ys)+1
        if len(pts)<3: continue
        if by1<=xtop+0.3*xh and (by1-by0)<=0.75*xh and by0>=xtop-1.3*xh:
            w=bx1-bx0; h=by1-by0
            tl=sum(1 for x,y in pts if x<bx0+w/3 and y<by0+h/2)
            marks.append(dict(kind='above',cx=(bx0+bx1)/2,w=w,h=h,n=len(pts),tl=tl,box=[bx0,by0,bx1,by1]))
        elif by0>=base+0.08*xh and (by1-by0)<=0.7*xh:
            marks.append(dict(kind='below',cx=(bx0+bx1)/2,w=bx1-bx0,h=by1-by0,n=len(pts),box=[bx0,by0,bx1,by1]))
    out=[]
    for c,b in letters:
        x0,y0,x1,y1=b
        info={'raw':c,'box':b,'sup': y1 < base-0.45*xh,'base':strip(BASE.get(c,c))}
        if not info['sup']:
            cw=x1-x0
            cx0=x0+int(cw*0.2); cx1=max(x1-int(cw*0.2),cx0+1)
            info['below']=sum(1 for y in range(int(base+0.12*xh),min(H,int(base+0.55*xh))) for x in range(cx0,cx1) if px[x,y]<128)
            ms=[m for m in marks if m['kind']=='above' and x0-1<=m['cx']<=x1+1]
            if ms: info['above']=max(ms,key=lambda m:m['n'])
            mb=[m for m in marks if m['kind']=='below' and x0-1<=m['cx']<=x1+1]
            if mb: info['belowmark']=max(mb,key=lambda m:m['n'])
        out.append(info)
    return dict(base=base,xh=xh,chars=out,unreliable=xh>0.5*H)

# --- fusion ---
MAC={'ä':'ā','ã':'ā','å':'ā','ă':'ā','á':'ā','â':'ā','à':'ā','ā':'ā',
     'ü':'ū','ù':'ū','ű':'ū','ŭ':'ū','ú':'ū','û':'ū','ū':'ū',
     'ï':'ī','ì':'ī','í':'ī','Ï':'ī','Í':'ī','ī':'ī','Ī':'ī','î':'ī'}
SIB={'š':'š','ś':'ś','Ś':'ś','ź':'ź','ż':'ź','ž':'ž','Ž':'ž','č':'č','Ĉ':'č','ĉ':'č','ć':'č'}
PLAIN={'ø':'g','ą':'g','ĝ':'g','ğ':'g','ġ':'g','Ğ':'g','ð':'ϑ','þ':'ϑ','ϑ':'ϑ','ħ':'h','ň':'n','ř':'r','Z':'z','S':'s','I':'i','İ':'i','Y':'y','U':'u','A':'a','D':'d','T':'t','K':'k','B':'b','X':'x','L':'l','į':'i','ė':'e','é':'e','“':'"','”':'"','‘':"'",'’':"'"}
SUPV=set('v"\'“”‘’`^')
def fuse(g):
    """g: geometry analysis -> (lemma, homonym digits from superscripts, flags)"""
    out=[]; flags=[]; sup_digits=''
    chars=list(g['chars'])
    B,XH=g['base'],g['xh']
    # drop tesseract duplicates: same letter class with heavily overlapping boxes
    ded=[]
    for c in chars:
        if ded:
            p=ded[-1]; a,b=p['box'],c['box']
            ov=min(a[2],b[2])-max(a[0],b[0])
            if ov>0.5*min(a[2]-a[0],b[2]-b[0]) and p['base']==c['base']: continue
        ded.append(c)
    chars=ded
    # superscripts: raised small glyphs (homonym numerals, the v of x/h) by geometry
    for c in chars:
        c['sup']=False
        if c['raw'].isalnum() or c['raw'] in SUPV:
            c['sup']= c['box'][3] < B-0.28*XH and (c['box'][3]-c['box'][1]) < 1.05*XH
    for k,c in enumerate(chars):
        r=c['raw']
        if c['sup']:
            if out and out[-1] in ('x','h') and (r.lower()=='v' or r in SUPV or r=='*'):
                out.append('ᵛ'); continue
            if out and out[-1]=='*' or not out or (out and out[-1] in '*0123456789'):
                sup_digits+= r if r.isdigit() else ('1' if r in "lIi|!'‘’" else '?'); continue
            if r=='*': out.append('*'); continue
            flags.append(f'sup:{r}'); continue
        if r.isdigit() and all(o in '*' for o in out):
            sup_digits+=r; continue
        b=c['base']; a=c.get('above')
        flat = a and a['w']/max(a['h'],1)>=2.5
        if r in MAC:
            if r in 'áíú' and a and not flat:
                ch=r; flags.append('accent')
            elif a and not flat and MAC[r] not in r:
                ch=strip(r).lower(); flags.append('vowel-mark-unclear')
            else: ch=MAC[r]
        elif r in SIB: ch=SIB[r]
        else: ch=PLAIN.get(r,r)
        if ch in 'au' and a and not flat:
            flags.append('vowel-mark-unclear')
        if ch in 'sz' and a and a['n']>=6:
            t=a['tl']/a['n']; ch={'s':'š','z':'ž'}[ch] if t>=0.16 else {'s':'ś','z':'ź'}[ch]; flags.append('sibilant:geom')
        if ch=='c' and a and a['n']>=6:
            ch='č'; flags.append('sibilant:geom')
        if ch in 'aiu' and flat:
            ch={'a':'ā','i':'ī','u':'ū'}[ch]; flags.append('macron:geom')
        if ch=='ī' and a and not flat:
            flags.append('macron:tess-only')
        if ch in ('j','ĵ'):
            ch='i̯' if c.get('belowmark') else 'ǰ'
        elif ch in ('y','ų','ụ'):
            ch='u̯'; flags.append('u̯:from-'+r)
        elif b in 'iur' and ch in 'iīuūr' and c.get('below',0)>40 and (c.get('belowmark') or not any(
                0<=k+d<len(chars) and chars[k+d]['base'] in 'gpjyqfϑ' for d in (-1,1))):
            ch={'i':'i̯','u':'u̯','r':'r̥','ī':'i̯','ū':'u̯'}[ch]
        if b in 'z' and ch in 'źž' and a:
            t=a['tl']/a['n']
            if (ch=='ź' and t>=0.16) or (ch=='ž' and t<0.16): flags.append('sibilant:conflict')
        if b=='s' and ch in 'śš' and a:
            t=a['tl']/a['n']
            if (ch=='ś' and t>=0.16) or (ch=='š' and t<0.16): flags.append('sibilant:conflict')
        out.append(ch)
    s=unicodedata.normalize('NFC',''.join(out))
    return s, sup_digits, flags

# --- skeletons ---
AB={'£':'c','6':'c','{':'c','б':'c','Ć':'c','é':'c','С':'c','с':'c','Ч':'c','ч':'c','É':'c','Č':'c',
    '§':'s','$':'s','5':'s','8':'s','S':'s','Š':'s','ś':'s','š':'s','Ś':'s',
    'а':'a','е':'e','о':'o','р':'p','х':'x','у':'y','к':'k','т':'t','п':'n','г':'r','и':'u','ц':'u','щ':'u','і':'i','í':'i','ï':'i','ì':'i','î':'i','T':'i','l':'l','I':'i','¡':'i','j':'j',']':'j','J':'j',
    'û':'t','ô':'t','&':'t','ϑ':'t','Ф':'t','ö':'t','θ':'t','в':'b','ь':'b','Ь':'b','д':'d','з':'z','Z':'z','ž':'z','ź':'z','2':'z','7':'z','y':'u','ų':'u','ᵛ':'v'}
def skel(s, abbyy=False):
    s=re.sub(r'^[*•#^"\'«»+†%\s]*(?:\d+\s+)?','',s)
    out=''
    for ch in s:
        if abbyy and ch in AB: ch=AB[ch]
        elif not abbyy and ch in 'ϑ': ch='t'
        elif not abbyy and ch=='ᵛ': ch='v'
        d=unicodedata.normalize('NFD',ch)[0].lower()
        if d in 'yj' and not abbyy: d={'y':'u','j':'j'}[d]
        if d.isascii() and d.isalpha(): out+=d
    return out.replace('j','i').replace('y','u')

# --- reconciliation ---
SUPTOK=re.compile(r"^[\dlI!n|]$")
SUPONE="lIJi!|'‘’"
def small_tokens(words):
    """Indices of superscript-size tokens (homonym numerals) in the ABBYY headword segment."""
    if not words: return set()
    big=max(w[2] for w in words)
    return {k for k,w in enumerate(words) if k>0 and w[2]<0.8*big and len(w[0])<=2}
def abbyy_homonym(words):
    """words: ABBYY headword-segment words [text,font,size,x0]; returns homonym digit, '?' or ''."""
    sm=small_tokens(words)
    if 1 in sm:
        t=words[1][0]
        return t if t.isdigit() else ('1' if t in SUPONE else '?')
    m=re.match(r"^\W{0,3}(\d)",words[0][0]) if words else None
    if m and len(words[0][0])>2: return m.group(1)
    return ''
def abbyy_clean(words):
    sm=small_tokens(words)
    return ' '.join(w[0] for k,w in enumerate(words) if k not in sm)
def reconcile(g, words):
    lem,supd,flags=fuse(g)
    if g.get('unreliable'): flags=flags+['geometry-unreliable']
    lem=lem.lstrip('*"\'“”«»•#^+')
    ab=' '.join(w[0] for w in words)
    hom=abbyy_homonym(words)
    abc=re.sub(r'(^|\s)[\dlI!n|]\s',' ',abbyy_clean(words))
    abs_=skel(abc,True)
    # drop superscript misreads at lemma start when ABBYY confirms a superscript numeral
    if hom or supd:
        for cut in (0,1,2):
            if skel(lem[cut:])==abs_: lem=lem[cut:]; break
        lem=re.sub(r'^[\d?]+','',lem)
    if supd and supd.strip('?') and (not hom or hom=='?'): hom=supd.strip('?')[:1]
    lem=unicodedata.normalize('NFC',lem).rstrip(',;/ ')
    first=lambda x: re.split(r'\s*[:,<>~]\s*|\s*/\s*(?=\*)|(?<=-)\s*/\s*|\s+/\s+|\s+или\s+|\s*\d\)', x.strip(), 1)[0]
    f_lem=first(lem); f_ab=skel(first(re.sub(r'^\W*[\dlI!n|]\s+','',abc).strip()),True)
    first_ok = skel(f_lem)==f_ab and len(f_ab)>=1
    status='ocr_consensus' if skel(lem)==abs_ and not any(f.startswith(('sibilant:conflict','sup:','vowel-mark-unclear','geometry-unreliable')) for f in flags) else 'ocr_disputed'
    if status!='ocr_consensus' and first_ok and not any(f.startswith(('sibilant:conflict','vowel-mark-unclear','geometry-unreliable')) for f in flags): status='ocr_consensus_first'
    return dict(lemma=lem, first=f_lem, homonym=hom, flags=flags, abbyy=ab, abbyy_clean=abc, status=status)

# --- glosses ---
QUOTE=re.compile(r"(?:(?<=\s)|^)(\d\)\s*)?['‘]([^'’]{1,160}?)['’*](?=[\s,;.:()?!—\-]|$)")
END=re.compile(r"\s—\s|\s\[|\sВосход|\sиз\s+арийск|\sот\s+и\.-е\.|\sиз\s+и\.-е\.|\s<\s|\sДальнейш|\sИсточник|\.\s+[А-Я]")
XREF=re.compile(r"^[^']{0,80}?\sсм\.?\s+(.{1,60}?)(?:[.,;]\s|[.,;]?$)")
GAP=re.compile(r"^(?:[\s,;:.\-»>→*]|\d\)|\([^()]{0,40}\)|\sили\s|\sи\s|\(?\?\)?)*$")
def contiguous_glosses(region):
    """Quoted glosses that open the paragraph and follow each other separated only by punctuation,
    sense numbers or short parentheticals; later quotes belong to comparanda."""
    out=[]; last=None
    for m in QUOTE.finditer(region):
        n,q=m.group(1),m.group(2)
        if not re.search('[а-яё]',q): continue
        gap=region[0 if last is None else last:m.start()]
        if last is None and len(gap.strip())>45: break
        if last is not None and not GAP.match(gap): break
        out.append((n or '')+q.strip()); last=m.end()
    return out
def parse(text, head):
    """text: article text; head: ABBYY headword-segment text. Returns dict."""
    if not text.startswith(head):
        return dict(kind='unanchored', ru='', region=text[:200])
    rest=text[len(head):].strip()
    if len(text)<160 and re.search(r"\sсм\.?\s", text):
        q=[x.strip() for _,x in QUOTE.findall(rest.split('см',1)[0]) if re.search('[а-яё]',x)]
        return dict(kind='cross_reference', target=rest.split('см',1)[1].strip(' .')[:60], ru='; '.join(q))
    m=re.match(r"^(?:см\.?|см)\s+(.+)", rest)
    if m and len(text)<220:
        return dict(kind='cross_reference', target=re.split(r'[.;]\s|\.$', m.group(1))[0].strip(), ru='')
    if re.match(r"^(?:см\.?)\s", rest):
        return dict(kind='cross_reference', target=rest[3:60].strip(), ru='')
    region=END.split(rest,1)[0] if not rest.startswith('—') else ''
    spans=contiguous_glosses(region)
    if spans:
        return dict(kind='gloss', ru='; '.join(spans), region=region[:200])
    m=re.match(r"^(?:\([^)]{0,60}\)\s*)?—\s*(.+?)(?:\s—\s|\sиз\s|\.\s|$)", rest)
    if m and re.search('[а-яё]', m.group(1)):
        return dict(kind='description', ru=m.group(1).strip(' ,;'), region=rest[:200])
    if re.search(r"(^|\s)см\.?\s", rest[:25]) and len(text)<220:
        return dict(kind='cross_reference', target=rest.split('см',1)[1].strip(' .')[:60], ru='')
    m=re.match(r"^([^'\[\]]{3,120}?)\s—\s", rest)
    if m and re.search('[а-яё]{3}', m.group(1)):
        return dict(kind='description', ru=m.group(1).strip(' ,;'), region=rest[:200])
    return dict(kind='none', ru='', region=rest[:200])


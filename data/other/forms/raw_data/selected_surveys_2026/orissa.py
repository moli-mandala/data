"""Source-located Orissa grid interpretation, retaining unresolved OCR explicitly."""
import json,re,unicodedata
from pathlib import Path
R=Path(__file__).resolve().parent
# Page transitions checked against the printed serial-number column. A table
# category heading is never counted as a lexical prompt.
STARTS=[1,25,50,75,100,124,149,173,197,221,246,271,296,320,343,368,393,418,443,468,493,518,543,567,592,617,642,666,688,712,737,762,785,809,833,858,883,907,932,955,980,1005,1014]
SKIP={217:{0,1},235:{0,1},239:set(),244:{0,1},247:{0,1},255:{0,2}}
def text(c,lang='eng'):
    # Tesseract TSV fields are literal, not RFC-CSV quoted fields. Use stored
    # word boxes to reject boundary rules, retaining internal punctuation.
    words=c['ocr'][lang]['words'];out=[]
    for w in words:
        t=w['text'].strip()
        if not t or re.fullmatch(r'[|{}\[\]\\]+',t):continue
        out.append(t)
    s=' '.join(out).replace('|','')
    return unicodedata.normalize('NFC',' '.join(s.split()).strip('‘’`" '))
def records():
    out=[];excluded=[]
    for pi,first,last in zip(range(217,259),STARTS,STARTS[1:]):
        page=json.loads((R/f'orissa-p{pi}-cells.json').read_text())
        bands=[page['cells'][j:j+7] for j in range(0,len(page['cells']),7)]
        skip=SKIP.get(pi,{0});valid=[row for j,row in enumerate(bands) if j not in skip]
        # Grid detection may include a footer band beneath the last table rule.
        count=last-first
        assert len(valid)>=count,(pi,len(valid),count)
        selected=valid[:count]
        for row in bands:
            if row not in selected:excluded.append({'pdf_page':pi,'row':row[0]['row'],'reason':'table/category header or footer outside the numbered lexical rows','raw':row})
        matches=0
        for item,row in zip(range(first,last),selected):
            if any(str(item) in re.findall(r'\d+',text(row[0],lang)) for lang in ['eng','script/Latin']):matches+=1
            gloss=text(row[1]);gloss=re.sub(r'^Particles and Miscellaneous\s*','',gloss,flags=re.I)
            for col,c in enumerate(row[2:]):
                out.append({**c,'item':item,'column':col,'raw_column':col+2,'gloss':gloss,'gloss_ocr':row[1]['ocr'],'number_ocr':row[0]['ocr'],'eng':text(c),'latin':text(c,'script/Latin')})
        assert matches>=count*.5,(pi,'serial-number alignment mismatch',matches,count)
    assert len(out)==5065
    assert {(r['item'],r['column']) for r in out}=={(i,c) for i in range(1,1014) for c in range(5)}
    return out,excluded
def parse(r):
    corrections=json.loads((R/'orissa-readings.json').read_text())
    key=f"{r['item']}:{r['column']}"
    s=r['eng']
    # The English model drops nasalization; accept a Latin-pass tilde only
    # where every base character and word boundary otherwise agrees exactly.
    latin_nfd=unicodedata.normalize('NFD',r['latin'])
    if '\u0303' in latin_nfd and unicodedata.normalize('NFC',latin_nfd.replace('\u0303',''))==s:
        s=r['latin']
    s=corrections.get('forms',{}).get(key,s)
    g=corrections.get('glosses',{}).get(str(r['item']),r['gloss'])
    if not s or not re.search('[A-Za-z]',s):return []
    if not g or not re.search('[A-Za-z]',g):return []
    # Badly damaged OCR remains in the audit. No headword is synthesized.
    if any(c in s for c in ['\t','\n','�']) or len(s)>100:return []
    review=['OCR: source-located unreviewed italic transcription; both independent passes retained','transcription: ambiguous printed capitalization and vowel notation preserved']
    if key in corrections.get('forms',{}):review[0]='OCR: reading collated against the source scan'
    if key in corrections.get('form_editorial',{}):review.append('transcription: '+corrections['form_editorial'][key])
    if r['eng']!=r['latin']:review.append('OCR: independent passes disagree')
    # Parenthetical qualifiers may contain commas/slashes. Split only at depth
    # zero, and after a closing annotation followed by a coordinated head.
    s=re.sub(r'\)\s+(?=[A-Za-z])', '), ',s)
    chunks=[];depth=0;start=0
    for i,c in enumerate(s):
        if c=='(':depth+=1
        elif c==')':depth=max(0,depth-1)
        elif c in ',/' and depth==0:chunks.append(s[start:i]);start=i+1
    chunks.append(s[start:]);out=[]
    for f in chunks:
        f=f.strip(' .;|‘’');tags=[];gg=g;notes=[]
        for label,tag in [('male','m'),('female','f'),('plural','pl'),('singular','sg')]:
            if '('+label+')' in gg:gg=gg.replace(' ('+label+')','').replace('('+label+')','');tags.append(tag)
        mt=re.search(r'\s*\(([^()]+)\)$',f)
        if mt and mt[1].strip() not in {'i','a'}:
            label=mt[1].strip();f=f[:mt.start()].strip()
            if label in {'m','f','sg','pl'}:tags.append(label)
            elif label in {'e','y','elder','younger'}:gg += ' ('+{'e':'elder','y':'younger'}.get(label,label)+')'
            elif label in {'FF','MF','FM','MM'}:gg={'FF':'paternal grandfather','MF':'maternal grandfather','FM':'paternal grandmother','MM':'maternal grandmother'}[label]
            elif label in {'H','NH','H/NH','H,NH','H, NH'}:gg+=' ('+{'H':'human','NH':'nonhuman'}.get(label,'human and nonhuman')+')'
            elif label.startswith(('for ','made of ','decorated ')):notes.append(label)
            else:gg+=' ('+label+')'
        if r['item']==139:tags.append('m');gg='buffalo'
        if r['item']==140:tags.append('f');gg='buffalo'
        if key=='548:0':gg='light (opposite to heavy)' if not out else 'light-eyed retina'
        if f and re.search('[A-Za-z]',f):out.append({'form':f,'gloss':gg,'tags':list(dict.fromkeys(tags)),'notes':notes,'review':review})
    return out
if __name__=='__main__':
    rows,excluded=records()
    (R/'orissa-cells.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    (R/'orissa-layout-excluded.json').write_text(json.dumps(excluded,ensure_ascii=False,indent=2)+'\n')
    print('cells',len(rows),'forms',sum(len(parse(r)) for r in rows),'excluded bands',len(excluded))

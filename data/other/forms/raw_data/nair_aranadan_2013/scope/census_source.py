"""Read-only original census; extracted text is not accepted transcription."""
from pathlib import Path
import hashlib,json,re,collections,unicodedata
import fitz
H=Path(__file__).resolve().parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
P=W/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
doc=fitz.open(P)
def sha(b):return hashlib.sha256(b).hexdigest()
def save(name,content):
    p=H/name
    if p.exists():assert p.read_bytes()==content,(name,'immutable output drift')
    else:p.write_bytes(content)
def js(name,x):save(name,(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
pages=[];hits=[];urls=[];fonts={};rare=collections.Counter();positions=[]
pattern=re.compile(r'aran|arana|eranad|usha|ush[aā]|copyright|licen[cs]e|creative commons',re.I)
for i,p in enumerate(doc):
    text=p.get_text()
    save(f'page-{i+1:02d}-raw-text.txt',text.encode())
    spans=[dict(text=s['text'],bbox=s['bbox'],font=s['font'],size=s['size']) for b in p.get_text('dict')['blocks'] if 'lines' in b for line in b['lines'] for s in line['spans']]
    match=[s for s in spans if pattern.search(s['text'])]
    if match:hits.append(dict(pdf_page=i+1,matches=match))
    for link in p.get_links():
        if link.get('uri'):urls.append(dict(pdf_page=i+1,uri=link['uri']))
    for f in p.get_fonts(full=True):
        xref=f[0]
        if xref not in fonts:
            fonts[xref]=dict(xref=xref,extension=f[1],type=f[2],basefont=f[3],resource=f[4],encoding=f[5],pages=[])
        fonts[xref]['pages'].append(i+1)
    pages.append(dict(pdf_page=i+1,characters=len(text),text_sha256=sha(text.encode()),size=list(p.rect),font_xrefs=[f[0] for f in p.get_fonts()]))
    if i+1 in (9,10,11):
        positions.append(dict(pdf_page=i+1,spans=spans))
        rare.update(c for c in text if ord(c)>127)
        output=H/f'original-page-{i+1:02d}-150dpi.png'
        if not output.exists():p.get_pixmap(dpi=150).save(output)
js('whole-pdf-census.json',dict(source=str(P.relative_to(W)),sha256=sha(P.read_bytes()),bytes=P.stat().st_size,pages=pages,metadata=doc.metadata,urls=urls,search_pattern=pattern.pattern,search_hits=hits,fonts=list(fonts.values()),chapter_nonascii=[dict(character=c,codepoint=f'U+{ord(c):04X}',name=unicodedata.name(c,'UNASSIGNED/PRIVATE USE'),count=n) for c,n in sorted(rare.items())],transcription_accepted=False))
js('chapter-positioned-text-unaccepted.json',positions)
js('render-manifest.json',[dict(path=p.name,sha256=sha(p.read_bytes()),bytes=p.stat().st_size,dpi=150,pdf_page=int(p.stem.split('-')[2])) for p in sorted(H.glob('original-page-*-150dpi.png'))])
print(json.dumps({'pages':len(pages),'search_hit_pages':[r['pdf_page'] for r in hits],'font_count':len(fonts),'chapter_symbols':len(rare)},indent=2))

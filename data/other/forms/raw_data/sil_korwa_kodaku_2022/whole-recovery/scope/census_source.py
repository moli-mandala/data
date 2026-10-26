"""Whole report native-text/style scope evidence, not accepted extraction."""
from pathlib import Path
import csv,hashlib,json,re,collections
import fitz
H=Path(__file__).resolve().parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
P=W/'tmp/pdfs/korwa-kodaku-2022/source.pdf'
doc=fitz.open(P);assert len(doc)==115
assert hashlib.sha256(P.read_bytes()).hexdigest()=='a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c'
pages=[];leads=[];spans_all=[]
for index,page in enumerate(doc):
    num=index+1;text=page.get_text();out=H/f'page-{num:03d}-native.txt'
    if out.exists():assert out.read_text()==text
    else:out.write_text(text)
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    headings=[line for line in lines if re.match(r'^(?:[ABC](?:\.\d+)?|[1-4](?:\.\d+){1,3}|Appendix|References|Table \d|Map \d)',line)]
    spans=[dict(text=s['text'],font=s['font'],bbox=s['bbox'],size=s['size']) for b in page.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    spans_all.append(dict(pdf_page=num,spans=spans))
    for b in page.get_text('blocks'):
        t=b[4]
        if num<66 or num>90:
            if any(c in t for c in '‘’“”ɐəɪɨʊɔŋɲɳʃʒʈɖ') or re.search(r'\b(?:called|means|meaning|word|term|story|transcription|language name)\b',t,re.I):
                leads.append(dict(pdf_page=num,bbox=list(b[:4]),raw_text=t,status='scope lead; unaccepted'))
    pages.append(dict(pdf_page=num,printed_page=num-10 if num>=11 else None,characters=len(text),headings=headings,
        first_lines=lines[:5],text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        dimensions=list(page.rect),fonts=[dict(xref=f[0],basefont=f[3],type=f[2],encoding=f[5]) for f in page.get_fonts()],
        phonetic_markers=len(re.findall(r'\\p\s',text)),number_markers=re.findall(r'\\n\s+(\d+(?:\.\d+)*)',text)))
(H/'whole-page-census.json').write_text(json.dumps(dict(source_path=str(P.relative_to(W)),source_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),pages=pages,accepted_transcription=False),ensure_ascii=False,indent=2)+'\n')
(H/'positioned-native-spans.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in spans_all))
(H/'outside-wordlist-scope-leads.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in leads))
summary=[]
for p in pages:summary.append(f"PDF{p['pdf_page']} printed{p['printed_page']}: "+' | '.join(p['first_lines']+p['headings'])[:900])
(H/'page-navigation.txt').write_text('\n'.join(summary)+'\n')
print(json.dumps(dict(pages=len(pages),scope_leads=len(leads),phonetic_markers=sum(p['phonetic_markers'] for p in pages))))

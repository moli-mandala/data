"""Bounded original-PDF evidence extraction; does not install lexical data."""
import csv, hashlib, json
from pathlib import Path
import fitz
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[6]
PDF=ROOT/'tmp/pdfs/turi-2024/publisher-current.pdf'
EXPECTED='d4accb5d0e011b58cf486e538cf3d77d9c88ba9b4179cf6d34826e073237e42d'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert sha(PDF)==EXPECTED
    doc=fitz.open(PDF); assert len(doc)==42
    pages=HERE/'native-pages'; pages.mkdir(exist_ok=True)
    census=[]
    for i,p in enumerate(doc):
        folio=i+261
        spans=[]
        for block in p.get_text('dict')['blocks']:
            for line in block.get('lines',[]):
                for s in line['spans']:
                    spans.append({k:s[k] for k in ('text','font','size','bbox','origin')})
        text=p.get_text('text')
        (pages/f'p{folio}.txt').write_text(text)
        (pages/f'p{folio}-spans.json').write_text(json.dumps(spans,ensure_ascii=False,indent=2)+'\n')
        census.append(dict(printed_page=folio,pdf_page=i+1,span_count=len(spans),review_status='pending_original_page_review',scope='pending_census'))
    canonical=HERE.parent.parent.parent/'20260921-peterson-turi.csv'
    rows=list(csv.reader(canonical.open()))
    assert len(rows)==246 and len({r[10] for r in rows})==246
    (HERE/'baseline-installed.json').write_text(json.dumps(dict(path=str(canonical.relative_to(ROOT)),sha256=sha(canonical),rows=246,keys=[r[10] for r in rows]),ensure_ascii=False,indent=2)+'\n')
    (HERE/'page-census.json').write_text(json.dumps(census,ensure_ascii=False,indent=2)+'\n')
    (HERE/'source-manifest.json').write_text(json.dumps(dict(source_id='peterson2024turi',pdf=str(PDF.relative_to(ROOT)),sha256=EXPECTED,pages=42,printed_pages=[261,302],status='whole_source_recovery_in_progress',extraction='native PDF spans, no OCR; not accepted lexical transcription'),indent=2)+'\n')
    print('42 original pages extracted; baseline 246 keys pinned')
if __name__=='__main__': main()

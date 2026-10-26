"""Recover Chapter 3 aligned analyses omitted by the earlier quote-only census.

Candidate evidence only. Existing canonical rows are neither overwritten nor installed.
"""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
PLAN={69:[(48,55)],70:[(10,19)],71:[(9,15),(24,34)],72:[(6,15)],73:[([5,6],13),(15,23),(25,32),(43,57),(60,72)],74:[(12,18),(20,26),(28,31),(33,36),(46,57)],75:[(4,18),(29,50),(53,64)]}
def build():
    pages={}
    for line in (P/'whole-native-pages.jsonl').open():
        r=json.loads(line)
        if r.get('printed_page') in PLAN:pages[r['printed_page']]=r
    out=[]
    for n,items in PLAN.items():
        for surface,translation in items:
            lines=surface if isinstance(surface,list) else [surface]
            def txt(i):return ''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip()
            out.append(dict(source_unit_key=f'dhakal2011:p{n}:ch3:surface:l{lines[0]}',printed_page=n,pdf_page=n+22,kind='whole-expression',form=' '.join(txt(i) for i in lines),gloss=txt(translation),surface_lines=lines,translation_lines=[translation],interlinear_evidence=[pages[n]['lines'][i-1] for i in range(max(lines)+1,translation) if txt(i)],status='Native-text candidate; original review and old-key reuse reconciliation pending'))
    assert len(out)==18
    return out
if __name__=='__main__':
    rows=build()
    (P/'chapter3-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'whole-expression source blocks; 16 prior whole-clause recoveries and 2 already installed single-token expressions')

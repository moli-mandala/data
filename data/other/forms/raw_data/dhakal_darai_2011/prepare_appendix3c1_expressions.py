"""Reviewed give-paradigm examples, preserving surface and interlinear witnesses."""
import json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent
PLAN={
371:[(8,14,[(9,11),(10,13)]),(16,22,[(17,20),(19,21)]),(24,30,[(25,27),(26,29)]),(32,38,[(33,35),(34,37)]),(40,47,[(41,44),(42,46)]),(49,55,[(50,52),(51,54)]),(58,64,[(59,61),(60,63)])],
372:[(6,12,[(7,9),(8,11)]),(14,20,[(15,17),(16,19)]),(23,30,[(24,27),(26,29)]),(32,39,[(33,36),(35,38)]),(41,48,[(42,45),(44,47)]),(51,58,[(52,55),(54,57)]),(60,67,[(61,64),(63,66)])],
373:[(5,12,[(6,9),(8,11)]),(15,21,[(16,19),(18,20)]),(23,33,[(24,30),(26,31),(28,32)]),(35,41,[(36,39),(38,40)]),(43,49,[(44,47),(46,48)]),(51,57,[(52,55),(54,56)]),(59,65,[(60,63),(62,64)])]}

def build():
    pages={}
    for s in (P/'whole-native-pages.jsonl').open():
        r=json.loads(s)
        if r.get('printed_page') in PLAN:pages[r['printed_page']]=r
    records=[]
    for n,examples in PLAN.items():
        def t(i):return unicodedata.normalize('NFC',''.join(z['decoded'] for z in pages[n]['lines'][i-1]['spans']).strip())
        for surface,trans,words in examples:
            note=''
            if (n,surface) in [(371,16),(373,23)]:note='Source surface terake differs from aligned toi-ke; both literal witnesses retained.'
            if (n,surface) in [(372,60),(373,5)]:note='Surface dehaleu- differs from interlinear de-hal-u-; neither witness is regularized.'
            for kind,f,g in [('whole-expression',surface,trans)]+[('aligned-word',a,b) for a,b in words]:
                form=t(f)
                if (n,f)==(372,16):form=form.replace('kan -səb','kan-səb')
                r=dict(source_unit_key=f'dhakal2011:p{n}:appendix3c1:{kind}:l{f}',printed_page=n,pdf_page=n+22,
                       kind=kind,form=form,gloss=t(g).strip('‘’ '),source_lines=[f,g],
                       evidence=[pages[n]['lines'][i-1] for i in [f,g]],
                       parent_expression_key=f'dhakal2011:p{n}:appendix3c1:whole-expression:l{surface}',
                       status='Owner original review passed; independent review pending',note=note)
                if (n,surface)==(373,23):
                    r['source_interpretation_uncertainty']='Printed verbal gloss includes 1SG although overt subject and full translation specify he; literal analysis retained, no coherent 1SG assertion inferred.'
                records.append(r)
    assert len(records)==64
    return records

if __name__=='__main__':
    rows=build();(P/'appendix3c1-expression-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'records: 21 whole expressions and 43 aligned words')

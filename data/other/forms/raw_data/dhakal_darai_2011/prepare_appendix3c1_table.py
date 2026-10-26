"""Physical subject/object-agreement grid, with absent cells retained as controls."""
import json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent

def build(page_number=370):
    assert page_number in (370, 374)
    page=next(r for s in (P/'whole-native-pages.jsonl').open() if (r:=json.loads(s)).get('printed_page')==page_number)
    def txt(i):return unicodedata.normalize('NFC',''.join(s['decoded'] for s in page['lines'][i-1]['spans']).strip())
    objects=[txt(i) for i in range(9 if page_number == 370 else 10,18 if page_number == 370 else 19)]
    specs=[('1SG','NPST',20,[20,21,23,25,26,28,30,31,33]),('1SG','PST',36,[36,37,39,41,42,44,46,47,49]),
           ('1PL','NPST',52,[52,53,54,55,56,58,60,61,63]),('1PL','PST',66,[66,67,68,69,70,72,74,75,77]),
           ('2SG','NPST',80,[80,81,82,83,84,86,88,89,91]),('2SG','PST',94,[94,95,96,97,98,100,102,103,105]),
           ('2SG.H','NPST',107,[107,108,109,110,111,113,115,116,118]),('2SG.H','PST',121,[121,122,123,124,125,127,129,130,132]),
           ('2PL','NPST',135,list(range(135,144))),('2PL','PST',146,list(range(146,155))),
           ('3SG','NPST',157,[157,158,160,162,163,165,167,168,170]),('3SG','PST',173,[173,174,176,178,179,181,183,184,186]),
           ('3PL','NPST',189,[189,190,192,194,195,197,199,200,202]),('3PL','PST',206,[206,207,209,211,212,214,216,217,219])]
    if page_number == 374:
        def shifted(i): return i + (1 if i < 107 else 2)
        specs = [(sub, tense, shifted(start), [shifted(i) for i in ends]) for sub,tense,start,ends in specs]
    section = "appendix3c1" if page_number == 370 else "appendix3c2"
    out=[]
    for subject,tense,start,ends in specs:
        for obj,end in zip(objects,ends):
            ls=list(range(start,end+1));raw=[txt(i) for i in ls];form=''.join(raw);absent=form=='-'
            out.append(dict(source_unit_key=f'dhakal2011:p{page_number}:{section}:l{start}',printed_page=page_number,pdf_page=page_number+22,
                            form='' if absent else form,gloss='' if absent else ('give' if page_number == 370 else 'kill'),kind='paradigm-absence-control' if absent else 'paradigm-inflection',
                            disposition='source-absence-control' if absent else 'candidate-target',
                            source_subject=subject,source_object=obj,source_tense=tense,raw_fragments=raw,
                            source_lines=ls,evidence=[page['lines'][i-1] for i in ls],
                            note=('Source excludes reflexives and states that blank slots indicate subject-only agreement; no absent form is generated.' if page_number == 370 else 'Absent source cells retained as controls; no missing form is generated.'),
                            status='Owner original-page review passed; independent review pending'))
            start=end+1
    assert len(out)==126
    return out

if __name__=='__main__':
    for number,section in [(370,'appendix3c1'),(374,'appendix3c2')]:
        rows=build(number);(P/f'{section}-table-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        print(number,len(rows),'cells;',sum(r['disposition']=='candidate-target' for r in rows),'attested inflections')

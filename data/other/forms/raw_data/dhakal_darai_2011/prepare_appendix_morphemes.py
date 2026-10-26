"""Complete owner-reviewed grammatical morpheme index; no inferred lexical glosses."""
import json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent

def build():
    pages={}
    for s in (P/'whole-native-pages.jsonl').open():
        r=json.loads(s)
        if r.get('printed_page') in (384,385):pages[r['printed_page']]=r
    rows=[]
    for n,first,last in [(384,9,99),(385,3,90)]:
        def text(i):return unicodedata.normalize('NFC',''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip())
        for i in range(first,last+1,3):
            form=text(i);lines=[i,i+1,i+2]
            rows.append(dict(source_unit_key=f'dhakal2011:p{n}:appendix4:l{i}',printed_page=n,pdf_page=n+22,
                             form=form,gloss='',kind='bound-morpheme',source_grammar=text(i+1),source_use=text(i+2),
                             source_alternates=form.split('/') if '/' in form else [form],source_lines=lines,
                             evidence=[pages[n]['lines'][l-1] for l in lines],
                             note='Source labels -kan 3SG.H while its usage text also lists third-person plural; both claims retained.' if (n,i)==(384,30) else '',
                             status='Owner original-page review passed; independent review pending'))
    assert len(rows)==61
    return rows

if __name__=='__main__':
    rows=build();(P/'appendix4-morphemes-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'physical morpheme rows; all source functions retained')

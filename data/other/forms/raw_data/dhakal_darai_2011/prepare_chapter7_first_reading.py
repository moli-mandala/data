"""Complete Chapter 7 native-text first reading; no canonical installation."""
import json
import re
import unicodedata
from pathlib import Path

P = Path(__file__).resolve().parent
SURFACES = {
207:[(5,19),(22,27),(30,35)],208:[(5,15),(17,26)],
209:[(5,[20,21]),(59,[67,68])],210:[(4,11),(25,35),(37,47),(68,72)],
211:[(4,18),(49,59),(61,68)],212:[(4,12),(33,[47,48]),(50,64),(69,85)],
213:[(4,[18,19]),(29,48),(50,72),(75,90)],
214:[(5,21),(28,41),(43,51),(53,[63,64]),(67,76),(78,88)],
215:[(5,13),(19,29),(31,41)],216:[(9,20),(22,29),(57,68),(70,81)],
217:[(5,17),(19,28),(53,62),(64,75),(78,89),(91,101)],
218:[(9,18),(20,29),(36,47),(49,59),(61,69),(71,79)],
219:[(5,15),(17,26),(28,37),(39,48),(58,[67,68])],
220:[([4,5],[29,30]),(33,[44,45]),(50,67)],
}
LISTS = {
206:[(26,27),(30,31),(32,33),(34,35),(36,37),(38,39),(40,40),(41,42),(43,43),(44,45)],
207:[(42,43),(44,45),(46,47),(48,50),(51,52),(53,54),(55,57),(58,59),(60,61),(62,63),(64,65),(66,67)],
208:[(31,33),(34,36),(37,39),(40,42),(43,45),(46,48),(49,51),(52,55),(56,58),(59,61),(62,64),(65,67),(68,70),(71,73),(74,76),(77,79)],
209:[(24,26),(27,28),(29,30),(31,32),(33,34),(35,36),(42,43),(44,46),(47,48),(49,50),(51,52),(53,54),(55,56)],
210:[(17,18),(19,22),(53,54),(55,57),(58,60),(61,62),(63,65)],
211:[(26,27),(28,30),(31,32),(33,34),(35,36),(37,38),(39,40),(41,42),(43,44),(45,46)],
212:[(19,21),(22,23),(24,26),(27,28),(29,30)],
217:[(33,34),(36,36),(38,38),(40,40),(42,43),(45,45),(47,48),(50,50)],
}
PROSE = [
(212,[65],'kucʰi','where','adv interr','Source explicitly describes ablative inflection.'),
(213,[24,25],'ete','this side/hither','adv prox','Source identifies proximal location.'),
(213,[24,25],'ote','that side/thither','adv dist','Source identifies distal location.'),
(214,[22],'icʰi','here','adv prox',''),(214,[22],'ucʰi','there','adv dist',''),
(214,[23],'-n','','suffix emph','Source identifies an emphatic marker.'),
(214,[23],'icʰin','right here','adv prox emph',''),(214,[23],'ucʰin','right there','adv dist emph',''),
(215,[14,15],'esnə/eske','like this','adv prox','Explicit co-listed manner demonstratives; relation interpretation pending.'),
(215,[15],'osnə/oske','like that','adv dist','Explicit co-listed manner demonstratives; relation interpretation pending.'),
(215,[15,16],'esnə','like this','adv prox','Proximal manner context.'),
(216,[6],'ramrəse','well','adv derived','Source explicitly contrasts derived adverb with a simple adverb.'),
(216,[6],'behaudo','daybreak','adv','Source explicitly identifies this as simple.'),
(216,[32,33],'-se','','suffix','Adverbs derived by suffixation to adjectives.'),
(216,[85],'-se','','suffix','Footnote also identifies ablative and comitative functions; distinguish these from adverb formation.'),
(217,[29,30],'-n','','suffix emph','Source says this emphatic marker attaches only to adverbs.'),
(217,[102],'-n','','suffix emph','Source example numbers appear inconsistent; preserve contrast in preceding examples.'),
]

def build():
    pages = {p['printed_page']:p for p in map(json.loads,(P/'whole-native-pages.jsonl').open()) if p['printed_page'] and 206<=p['printed_page']<=221}
    def txt(n,i):return ''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip()
    def norm(t):return unicodedata.normalize('NFC',re.sub(r'\s+',' ',t).strip())
    def evidence(n,ls):return [pages[n]['lines'][i-1] for i in ls]
    def record(n,key,kind,form,gloss,ls,**extra):
        return dict(source_unit_key=f'dhakal2011:p{n}:ch7:{key}',printed_page=n,pdf_page=n+22,kind=kind,form=norm(form),gloss=norm(gloss),source_lines=ls,evidence=evidence(n,ls),status='Native-text first reading; original-page review pending',**extra)
    rows=[]
    for n,pairs in SURFACES.items():
        for fs,gs in pairs:
            fs=fs if isinstance(fs,list) else [fs];gs=gs if isinstance(gs,list) else [gs]
            raw=' '.join(txt(n,i) for i in gs)
            ids=re.findall(r'\(([A-Za-z][A-Za-z.]+\.\d+)\)',raw)
            gloss=re.sub(r'\s*\([A-Za-z][A-Za-z.]+\.\d+\)','',raw).strip('‘’\' ')
            rows.append(record(n,f'surface:l{fs[0]}','whole-expression',' '.join(txt(n,i) for i in fs),gloss,fs+gs,surface_lines=fs,translation_lines=gs,raw_translation=raw,text_ids=ids,source_surface_evidence=evidence(n,fs),source_translation_evidence=evidence(n,gs),interlinear_evidence=pages[n]['lines'][max(fs):min(gs)-1]))
    for n,pairs in LISTS.items():
        for f,g in pairs:
            form=txt(n,f);raw=txt(n,g)
            if f==g:form=re.split('[‘\']',form,maxsplit=1)[0].strip()
            match=re.search('[‘\'](.*)',raw)
            gloss=match.group(1).strip('’\' ') if match else raw
            note=''
            if n==210 and f in (17,19):note='Author tentatively describes these emphatic adverbs as seemingly Nepali loans; no asserted donor link.'
            rows.append(record(n,f'list:l{f}','listed-head-group',form,gloss,sorted(set([f,g])),note=note,unresolved_glyphs=re.findall(r'⟦U\+[0-9A-F]+⟧',form)))
    # All sixteen physical cells retain function and deictic column, not just heads.
    for category,heads in [('manner',[59,61,64,66]),('location/direction',[70,72,74,76]),('location',[79,81,83,85]),('time',[88,90,92,94])]:
        for role,line in zip(('interrogative','proximal','distal','relative'),heads):
            rows.append(record(215,f'table:l{line}','pronominal-adverb-table',txt(215,line),txt(215,line+1).strip("' "),[line,line+1],source_category=category,source_role=role))
    # Each derivational operand and output is retained, including bound se.
    for start,derived,glossline in [(35,38,39),(40,43,44),(46,49,49),(50,53,54)]:
        raw=txt(216,start);m=re.match(r"(.*?)\s*'(.*?)'",raw)
        assert m,raw
        rows.append(record(216,f'derivation:l{start}','derivation-base',m.group(1),m.group(2),[start],derivation_group=f'p216:l{start}'))
        rows.append(record(216,f'derivation:l{start+2}','derivation-affix',txt(216,start+2),'',[start+2],derivation_group=f'p216:l{start}',source_function='adverb formation by suffixation'))
        rawhead=txt(216,derived).lstrip('= ');head=rawhead.split("'")[0].strip();rawgloss=txt(216,glossline);g=re.search("'(.*?)'",rawgloss)
        assert g,rawgloss
        rows.append(record(216,f'derivation:l{derived}','derivation-output',head,g.group(1),sorted(set([derived,glossline])),derivation_group=f'p216:l{start}'))
    seen={}
    for n,ls,form,gloss,tags,note in PROSE:
        k=(n,ls[0]);seen[k]=seen.get(k,0)+1
        rows.append(record(n,f'prose:l{ls[0]}:a{seen[k]}','prose-attestation',form,gloss,ls,tags=tags.split(),note=note))
    assert len({r['source_unit_key'] for r in rows})==len(rows)
    return rows

if __name__=='__main__':
    rows=build();(P/'chapter7-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'first-reading groups; all pages 206–221 read; original review and metadata integration pending')

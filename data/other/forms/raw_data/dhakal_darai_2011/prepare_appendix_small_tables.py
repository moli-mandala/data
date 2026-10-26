"""Reviewed appendix morphology tables with explicit roles and source attribution."""
import json,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent

def build():
    pages={}
    for s in (P/'whole-native-pages.jsonl').open():
        r=json.loads(s)
        if r.get('printed_page') in (375,376,377,382):pages[r['printed_page']]=r
    def text(n,i):return unicodedata.normalize('NFC',''.join(s['decoded'] for s in pages[n]['lines'][i-1]['spans']).strip())
    out=[]
    labels=[text(375,i) for i in range(14,20)]
    for subject,start,ends in [('1SG',21,[21,22,23,24,25,26]),('1PL',28,[28,29,31,33,35,37]),
                               ('2SG',39,[40,42,43,44,45,46]),('2PL',48,[49,51,52,53,54,55]),
                               ('3SG',57,[58,60,61,62,63,64]),('3PL',68,[69,71,72,73,74,76])]:
        for receptor,end in zip(labels,ends):
            lines=list(range(start,end+1));fragments=[text(375,i) for i in lines];raw=' '.join(x for x in fragments if x)
            control=raw=='-' or raw=='Only subject agreement'
            form='' if control else ''.join(fragments)
            out.append(dict(source_unit_key=f'dhakal2011:p375:appendix3c3:l{start}',printed_page=375,pdf_page=397,
                            form=form,gloss='',kind='source-context-control' if control else 'bound-morphology',
                            disposition='nonlexical-source-cell' if control else 'candidate-target',raw_cell=raw,
                            source_subject=subject,source_receptor=receptor,source_lines=lines,
                            evidence=[pages[375]['lines'][i-1] for i in lines],
                            attribution='Kotapish and Kotapish (1975: 140–141), reproduced by Dhakal (2011: 375)',
                            source_dialect='Not assigned from the thesis author’s Chitwan default; historical witness requires source-specific attribution.',
                            note='Literal ø and hyphen placement preserved as morphological notation, not asserted phonetic segments.',
                            status='Owner original-page review passed; independent review pending'))
            start=end+1
    assert len(out)==36
    labels=[text(376,i) for i in range(12,22)]
    for mood,rootlines,gloss,start,ends in [
        ('Counterfactual',[22,23],'go',25,[26,28,30,32,34,36,38,40,42,44]),
        ('Possibility',[45,46],'do',48,[49,51,53,55,57,59,61,63,65,67]),
        ('Inference1',[68,69],'do',71,[72,74,76,78,80,82,84,86,89,92]),
        ('Inference2',[],'',95,[96,98,100,101,102,103,105,106,108,109]),
        ('Optative',[110,111],'read',113,list(range(113,123)))]:
        if rootlines:
            out.append(dict(source_unit_key=f'dhakal2011:p376:appendix3d:root:l{rootlines[0]}',printed_page=376,pdf_page=398,
                            form=text(376,rootlines[0]),gloss=gloss,kind='paradigm-root',source_mood=mood,
                            evidence=[pages[376]['lines'][i-1] for i in rootlines],source_lines=rootlines,
                            status='Owner original-page review passed; independent review pending'))
        for label,end in zip(labels,ends):
            lines=list(range(start,end+1));fragments=[text(376,i) for i in lines];form=''.join(fragments);absent=form=='-'
            out.append(dict(source_unit_key=f'dhakal2011:p376:appendix3d:l{start}',printed_page=376,pdf_page=398,
                            form='' if absent else form,gloss='' if absent else gloss,
                            kind='source-absence-control' if absent else 'paradigm-inflection',
                            disposition='source-absence-control' if absent else 'candidate-target',
                            source_mood=mood,source_agreement=label,raw_fragments=fragments,source_lines=lines,
                            evidence=[pages[376]['lines'][i-1] for i in lines],
                            note='Inference2 has an empty root/gloss cell; no lexical meaning is supplied by analogy.' if mood=='Inference2' else '',
                            status='Owner original-page review passed; independent review pending'))
            start=end+1
    assert len(out)==90
    labels=[text(377,i) for i in range(12,22)]
    for root_line,analysis_line in [(42,43),(84,85),(126,127)]:
        out.append(dict(source_unit_key=f'dhakal2011:p377:appendix3e:root:l{root_line}',printed_page=377,pdf_page=399,
                        form=text(377,root_line),gloss='',source_grammar=text(377,analysis_line),kind='paradigm-root',
                        source_lines=[root_line,analysis_line],evidence=[pages[377]['lines'][i-1] for i in [root_line,analysis_line]],
                        status='Owner original-page review passed; independent review pending'))
    for tense,agreement,start,ends in [
        ('COP.NPST','NOM/ERG',23,[24,26,28,29,31,33,35,37,39,41]),
        ('COP.NPST','Dative/GEN',45,[46,48,50,51,53,55,57,59,61,63]),
        ('COP.NPST','NOM/ERG',65,[66,68,70,71,73,75,77,79,81,83]),
        ('COP.NPST','Dative/GEN',87,[88,90,92,93,95,97,99,101,103,105]),
        ('COP.PST','NOM/ERG',107,[108,110,112,113,115,117,119,121,123,125]),
        ('COP.PST','Dative/GEN',129,[130,132,134,135,137,139,141,143,145,147])]:
        for label,end in zip(labels,ends):
            lines=list(range(start,end+1));fragments=[text(377,i) for i in lines]
            out.append(dict(source_unit_key=f'dhakal2011:p377:appendix3e:l{start}',printed_page=377,pdf_page=399,
                            form=''.join(fragments),gloss='',kind='paradigm-inflection',disposition='candidate-target',
                            source_grammar=tense,source_agreement_type=agreement,source_person_number=label,
                            source_lines=lines,raw_fragments=fragments,evidence=[pages[377]['lines'][i-1] for i in lines],
                            note='The source prints no hyphen between rəhi and k across the line break; no hyphen supplied.' if start==140 else '',
                            status='Owner original-page review passed; independent review pending'))
            start=end+1
    assert len(out)==153
    labels=[text(382,i) for i in range(11,20)]
    for root_line,gloss_line in [(39,40),(79,80),(100,101),(142,143),(163,164)]:
        out.append(dict(source_unit_key=f'dhakal2011:p382:appendix3f:root:l{root_line}',printed_page=382,pdf_page=404,
                        form=text(382,root_line),gloss=text(382,gloss_line).strip('‘’ '),kind='paradigm-root',
                        source_lines=[root_line,gloss_line],evidence=[pages[382]['lines'][i-1] for i in [root_line,gloss_line]],
                        note='Source root prints kəkə-, while the inflected forms begin kəha-; retained without harmonization.' if root_line==142 else '',
                        status='Owner original-page review passed; independent review pending'))
    for gloss,agreement,start,ends in [
        ('want','Dative/GEN NPST',22,[23,24,26,28,30,32,34,36,38]),
        ('want','Dative/GEN PST',43,[44,45,47,49,51,53,55,57,59]),
        ('stand','Dative/GEN NPST',62,[63,64,66,68,70,72,74,76,78]),
        ('stand','Dative/GEN PST',83,[84,85,87,89,91,93,95,97,99]),
        ('ache','Dative/GEN NPST',104,[105,107,109,111,113,115,117,119,121]),
        ('ache','Dative/GEN PST',125,[126,127,129,131,133,135,137,139,141]),
        ('say','Focus PST',145,[146,148,150,152,154,156,158,160,162]),
        ('sit','Focus PST',166,[167,169,171,173,175,177,179,181,182])]:
        for label,end in zip(labels,ends):
            lines=list(range(start,end+1));fragments=[text(382,i) for i in lines]
            out.append(dict(source_unit_key=f'dhakal2011:p382:appendix3f:l{start}',printed_page=382,pdf_page=404,
                            form=''.join(fragments),gloss=gloss,kind='paradigm-inflection',disposition='candidate-target',
                            source_agreement_type=agreement,source_person_number=label,source_lines=lines,raw_fragments=fragments,
                            evidence=[pages[382]['lines'][i-1] for i in lines],
                            note='The combined source column 3PL/3PL.H is retained; no extra attestation is invented.' if label=='3PL/3PL.H' else '',
                            status='Owner original-page review passed; independent review pending'))
            start=end+1
    assert len(out)==230
    return out

if __name__=='__main__':
    rows=build();(P/'appendix-small-tables-reviewed-records.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(len(rows),'reviewed morphology/root/paradigm cells,',sum(r.get('disposition')=='candidate-target' for r in rows),'morphology units')

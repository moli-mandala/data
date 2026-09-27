"""Preview the complete Handuri inventory as it is reviewed; never install."""
import csv
import json
from pathlib import Path
import unicodedata

HERE=Path(__file__).resolve().parent
SOURCE='grierson1916handuri'
DIALECT='dialect:Hinduri:grierson1916-handuri:Handuri%20%28LSI%201916%29'
PAGE_RANGES=[(628,1,25),(630,26,52),(632,53,79),(634,80,106),(636,107,133),(638,134,160),(640,161,187),(642,188,214),(644,215,241)]

def grammar(item):
    """Only categories explicitly supplied by source prompts or paradigm contrasts."""
    tags=[]
    if item<=13: tags=['num']
    elif item<=31:
        tags=['pron']
        if item in [15,18,21,24,27,30]: tags+=['gen']
        if item in [16,19,22,25,28,31]: tags+=['poss']
        person='first-person' if item<=19 else 'second-person' if item<=25 else 'third-person'
        tags+=[person]
    elif 32<=item<=76: tags=['noun']
    elif 77<=item<=85: tags=['verb']
    elif 86<=item<=91: tags=['adv']
    elif 92<=item<=94: tags=['interr']
    elif 95<=item<=97: tags=['conj']
    elif item==100: tags=['interj']
    elif 101<=item<=118:
        tags=['noun']
        if item in [102,107,111,116]:tags+=['gen']
        elif item in [103,108,112,117]:tags+=['dat']
        elif item in [104,109,113,118]:tags+=['abl']
        if item in [105,106,107,108,109,114,115,116,117,118]:tags+=['pl']
    elif 119<=item<=131:
        tags=['multiword-expression']
        if item in [120,125]:tags+=['gen']
        elif item in [121,126]:tags+=['dat']
        elif item in [122,127]:tags+=['abl']
    elif 132<=item<=137:tags=['adj']
    elif 138<=item<=155:tags=['noun']
    elif 156<=item<=219:
        tags=['verb']
        if item in [169,176]:tags+=['inf']
        if item in [170,177,218]:tags+=['participle']
        if item in [171,178]:tags+=['conjunctive-participle']
        if 185<=item<=190:tags+=['pret']
        if item in [173,195,196,197,198,199,200,204]:tags+=['fut']
        if item in [202,203,204]:tags+=['pass']
    else:tags=['multiword-expression']
    return tags

def generate():
    cells=list(csv.DictReader((HERE/'full-transcription-staged.tsv').open(),delimiter='\t'))
    assert [int(c['item']) for c in cells]==list(range(1,242))
    out=[];audit=[]
    for c in cells:
        i=int(c['item']);page=int(c['page'])
        assert page==next(p for p,a,b in PAGE_RANGES if a<=i<=b)
        key=f'{SOURCE}:p{page}:item:{i}'
        forms=[unicodedata.normalize('NFC',s.strip()) for s in c['printed_form_review'].split(';') if s.strip()]
        assert bool(forms)==(c['decision']!='source_blank')
        keys=[]
        for j,form in enumerate(forms):
            k=key if j==0 else f'{key}:variant:{j+1}';keys.append(k)
            tags=grammar(i)+[DIALECT]
            if c['decision'] in {'pending_review','reviewed_mark_uncertain'}:tags.append('uncertain')
            notes='Provisional visual transcription; not installed. '+c['note']
            out.append(['Hinduri','',form,c['gloss'],'','',notes,f'{SOURCE}[p. {page}, standard-list item {i}, Haṇḍūrī column]','','',k,'','','',' '.join(dict.fromkeys(tags))])
        audit.append(dict(c,source_cell_key=key,scan_page=page+16,entry_keys=keys,status=c['decision'],grammar_basis='Source English prompt/paradigm; grammatical mappings provisional until full review.'))
    with (HERE/'grammar-transcription-staged.tsv').open() as f:
        grammar_cells=list(csv.DictReader(f,delimiter='\t'))
    assert len(grammar_cells)==86
    for c in grammar_cells:
        key=f"{SOURCE}:p{c['page']}:grammar:{c['unit']}"
        inventory_only='inventory only' in c['note'].lower()
        keys=[]
        forms=[unicodedata.normalize('NFC',x.strip()) for x in c['forms'].split(';') if x.strip()]
        if not inventory_only:
            for j,form in enumerate(forms):
                k=key if j==0 else f'{key}:variant:{j+1}'
                keys.append(k)
                tags=c['provisional_tags'].split()+[DIALECT]
                notes='Grammar example; source grammatical interpretation. '+c['note']
                out.append(['Hinduri','',form,c['gloss'],'','',notes,
                            f"{SOURCE}[p. {c['page']}, grammar, {c['unit']}]",'','',k,'','','',' '.join(dict.fromkeys(tags))])
        audit.append(dict(c,source_cell_key=key,entry_keys=keys,section='grammar',
                          status='inventory_only_morphological_ending' if inventory_only else c['status']))
    with (HERE/'specimen-transcription-staged.tsv').open() as f:
        specimen=list(csv.DictReader(f,delimiter='\t'))
    exact_specimen={}
    for c in specimen:
        key=f"{SOURCE}:specimen:{c['unit']}"
        form=unicodedata.normalize('NFC',c['form'])
        gloss=c['gloss']
        citation=f"{SOURCE}[p. {c['page']}, specimen 4, line {c['line']}, aligned unit {c['aligned_unit']}]"
        lookup=(form,gloss)
        previous=exact_specimen.get(lookup)
        if previous is not None:
            row=out[previous]
            row[7]+=';'+citation
            keys=[];reused=[row[10]]
        else:
            tags=[DIALECT]
            if c['status']=='first_visual_transcription':tags.append('uncertain')
            row=['Hinduri','',form,gloss,'','',
                 'Printed interlinear surface form and aligned gloss; not a reconstructed lemma. '+c['note'],
                 citation,'','',key,'','','',' '.join(tags)]
            exact_specimen[lookup]=len(out)
            out.append(row);keys=[key];reused=[]
        audit.append(dict(c,source_cell_key=key,section='specimen',entry_keys=keys,
                          reuse_entry_keys=reused,status='reused_same_source_attestation' if reused else c['status']))
    assert len({r[10] for r in out})==len(out)
    assert all(len(r)==15 and not r[13] and DIALECT in r[14] for r in out)
    return out,audit

if __name__=='__main__':
    rows,audit=generate()
    with (HERE/'full-preview.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
    with (HERE/'full-preview-audit.jsonl').open('w') as f:
        for a in audit:f.write(json.dumps(a,ensure_ascii=False)+'\n')
    print(f'{len(audit)} reconciled source cells; {len(rows)} provisional forms; no canonical writes')

"""Reproduce complete reviewed Shoracholi source-stage proposal, without installation."""
from __future__ import annotations
import csv
import json
import unicodedata
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location("shoracholi_table_grammar",Path(__file__).resolve().parent/"table_grammar.py")
_grammar=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_grammar)
explicit_table_tags=_grammar.explicit_table_tags
HERE=Path(__file__).resolve().parent
SOURCE='grierson1916shoracholi'
PAGES=[(629,1,25),(631,26,52),(633,53,79),(635,80,106),(637,107,133),(639,134,160),(641,161,187),(643,188,214),(645,215,241)]

def read(name):
    with (HERE/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))

def nfc(s):return unicodedata.normalize('NFC',s.strip())

def row(key,form,gloss,citation,tags=(),notes=''):
    return ['Shoracholi','',nfc(form),gloss,'','',notes,citation,'','',key,'','','',' '.join(dict.fromkeys(tags))]

def table_forms(c):
    forms=[nfc(x) for x in c['printed_form_review'].split(';') if x.strip()]
    item=int(c['item'])
    if item in {156,157,158,159,160,161,209,210}:
        raw=c['printed_form_review']
        pronoun,rest=raw.split(' ',1)
        pronoun=pronoun.rstrip(',')
        forms=[nfc(pronoun+' '+part.strip()) for part in rest.split(',') if part.strip()]
    return forms

def generate():
    out=[];audit=[]
    heads=read('unusual-words-reviewed.tsv')
    assert len(heads)==27
    for c in heads:
        item=int(c['item']);key=f'{SOURCE}:p602:item:{item}';keys=[]
        # The source gloss belongs to the explicit full construction, not the isolated completion verb.
        reused=[f'{SOURCE}:p602:grammar:finish-construction'] if item==8 else []
        tags=['verb'] if item in {2,3,5,15,19,26} else ['adj'] if item in {4,9,21,25} else ['noun']
        if item==27:tags+=['uncertain']
        if not reused:
            for j,form in enumerate(c['printed_form'].split(';')):
                k=key if j==0 else f'{key}:answer{j+1}';keys.append(k)
                notes='Small raised mark beside ṭ is unclear in the scan; the reviewed uṭī reading is retained.' if item==27 else ''
                out.append(row(k,form,c['gloss'],f'{SOURCE}[p.602, unusual-word item {item}]',tags,notes))
        audit.append(dict(c,section='unusual_words',source_cell_key=key,entry_keys=keys,reuse_entry_keys=reused,status='represented_by_explicit_construction' if reused else 'ingested_uncertain' if item==27 else 'ingested',reason='Standalone head is supplied only within the explicitly translated khāyŏ chhĕkṇū construction.' if reused else c['note']))
    cells=read('full-transcription-staged.tsv')
    assert [int(c['item']) for c in cells]==list(range(1,242))
    for c in cells:
        item=int(c['item']);page=int(c['page']);key=f'{SOURCE}:p{page}:item:{item}'
        assert page==next(p for p,a,b in PAGES if a<=item<=b)
        forms=table_forms(c);keys=[];tags=explicit_table_tags(item);notes=[]
        if item in {75,130}:tags+=['uncertain'];notes.append('The source diacritic is crowded or ambiguous; the reviewed literal reading is retained with typography uncertainty.')
        if item==156:notes.append('The source prints āsū sū without a separating comma; this adjacent sequence is retained under its shared subject.')
        if item in {156,157,158,159,160,161,209,210}:notes.append('The shared printed subject is expanded over explicit comma-separated predicate alternatives.')
        for j,form in enumerate(forms):
            k=key if j==0 else f'{key}:variant:{j+1}';keys.append(k)
            out.append(row(k,form,c['gloss'],f'{SOURCE}[p.{page}, standard-list item {item}, Śōrāchōlī column]',tags,' '.join(notes)))
        audit.append(dict(c,section='table',source_cell_key=key,entry_keys=keys,reuse_entry_keys=[],status='source_blank' if not forms else 'ingested_uncertain' if item in {75,130} else 'ingested',expanded_forms=forms,grammar_basis='Explicit numbered source prompt/paradigm only.'))
    grammar=read('grammar-transcription-staged.tsv');assert len(grammar)==108
    for c in grammar:
        key=f"{SOURCE}:p{c['page']}:grammar:{c['unit']}";keys=[]
        excluded='inventory only' in c['note'].lower()
        if not excluded:
            for j,form in enumerate(c['forms'].split(';')):
                k=key if j==0 else f'{key}:variant:{j+1}';keys.append(k)
                out.append(row(k,form,c['gloss'],f"{SOURCE}[p.{c['page']}, grammar, {c['unit']}]",c['tags'].split()))
        audit.append(dict(c,section='grammar',source_cell_key=key,entry_keys=keys,reuse_entry_keys=[],status='inventory_only_control_or_ending' if excluded else 'ingested'))
    specimen=read('specimen-transcription-staged.tsv');assert len(specimen)==307
    seen={}
    for c in specimen:
        key=f"{SOURCE}:specimen:{c['unit']}";form=nfc(c['printed_form_review']);gloss=c['printed_gloss'];pair=(form,gloss)
        citation=f"{SOURCE}[p.{c['page']}, specimen7, line {c['line']}, aligned word {c['word']}]"
        if c['unit']=='p608-l5-w7':citation=f'{SOURCE}[p.608, specimen7, lines5–6, split printed compound]'
        if pair in seen:
            previous=out[seen[pair]];previous[7]+=';'+citation;keys=[];reused=[previous[10]]
        else:
            seen[pair]=len(out);keys=[key];reused=[]
            out.append(row(key,form,gloss,citation))
        audit.append(dict(c,section='specimen',source_cell_key=key,entry_keys=keys,reuse_entry_keys=reused,status='reused_same_source_attestation' if reused else 'ingested'))
    assert len(audit)==683
    allkeys={r[10] for r in out};assert len(allkeys)==len(out)
    assert all(k in allkeys for a in audit for k in a['entry_keys']+a['reuse_entry_keys'])
    assert all(len(r)==15 and not r[13] for r in out)
    return out,audit

if __name__=='__main__':
    rows,audit=generate()
    with (HERE/'full-preview.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
    with (HERE/'full-preview-audit.jsonl').open('w') as f:
        for a in audit:f.write(json.dumps(a,ensure_ascii=False)+'\n')
    print(f'{len(audit)} units; {len(rows)} forms; no canonical writes')

"""Full literal Sodochi chapter proposal; no canonical writes."""
import csv,json,unicodedata
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location('sodochi_table_grammar',Path(__file__).resolve().parent/'table_grammar.py')
_grammar=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_grammar)
explicit_table_tags=_grammar.explicit_table_tags
HERE=Path(__file__).resolve().parent
SOURCE='grierson1916sodochi'
DIALECT='dialect:Kotguru:lsi1916-kotguru-sodochi:Sodochi'
def read(name):
    with (HERE/name).open() as f:return list(csv.DictReader(f,delimiter='\t'))
def nfc(s):return unicodedata.normalize('NFC',s.strip())
def row(key,form,gloss,citation,tags=(),notes='',lang='Kotguru',dialect=True):
    return [lang,'',nfc(form),gloss,'','',notes,citation,'','',key,'','','',' '.join(dict.fromkeys(([DIALECT] if dialect else [])+list(tags)))]
def generate():
    out=[];audit=[]
    for c in read('full-table-staged.tsv'):
        item=int(c['item']);page=c['page'];key=f'{SOURCE}:p{page}:item:{item}'
        forms=[nfc(s) for s in c['printed_form_review'].split(';') if s.strip()]
        notes=[];tags=explicit_table_tags(item)
        if item in range(156,162) or item==182:
            pronoun,rest=c['printed_form_review'].split(' ',1)
            forms=[pronoun+' '+x.strip() for x in rest.split(',')]
            notes.append('The shared printed subject is expanded across its explicit predicate alternatives.')
        if item in {18,82,154}:
            tags+=['uncertain']
            notes.append({18:'Source prints Mābrō, contrasting with Māhrō at item 19; literal spelling retained.',82:'Source prints Khŏṛō, au; segmentation of the following au is unclear, so the whole expression is retained.',154:'The source explicitly marks this feminine form doubtful.'}[item])
        keys=[]
        for j,form in enumerate(forms):
            k=key if j==0 else key+f':answer{j+1}';keys.append(k)
            gloss=('elder sister' if j==0 else 'younger sister') if item==50 else c['gloss']
            out.append(row(k,form,gloss,f'{SOURCE}[p.{page}, standard-list item {item}, Śōdōchī column]',tags,' '.join(notes)))
        audit.append(dict(c,section='table',source_cell_key=key,entry_keys=keys,reuse_entry_keys=[],status='ingested' if keys else 'source_blank'))
    heads=read('glossary-staged.tsv');index={}
    for c in heads:
        for f in c['forms'].split(';'):index.setdefault(nfc(f),[]).append(c)
    for c in heads:
        key=f"{SOURCE}:p{c['page']}:glossary:{c['item']}";keys=[]
        gloss=c['gloss'];notes=[];target=None
        if c['see_target']:
            matches=[x for x in index.get(nfc(c['see_target']),[]) if x['gloss']]
            assert len(matches)==1,(c,matches)
            target=matches[0];gloss=target['gloss']
            notes.append(f"The source refers to {c['see_target']}; gloss follows that explicitly cited entry.")
        os=c['source_lect_label']=='O.S.'
        lang='OuterSiraji' if os else 'Kotguru'
        notes.append('Explicitly labelled O.S. (Outer Siraji).' if os else 'Unmarked entry in the source’s mixed Sodochi/Outer Siraji glossary; the source states mutual intelligibility but does not assign this head exclusively to either lect.')
        tags=[] if os else ['uncertain']
        if gloss.startswith('to '):tags+=['verb','inf']
        for j,form in enumerate(c['forms'].split(';')):
            k=key if j==0 else key+f':answer{j+1}';keys.append(k)
            localtags=tags+(['f'] if (c['page'],c['item']) in {('649','3'),('649','14')} and j==1 else [])
            cite=f"{SOURCE}[p.{c['page']}, glossary item {c['item']}]"
            if target:cite+=f";{SOURCE}[p.{target['page']}, glossary item {target['item']}]"
            out.append(row(k,form,gloss,cite,localtags,' '.join(notes),lang,False))
        audit.append(dict(c,section='glossary',source_cell_key=key,entry_keys=keys,reuse_entry_keys=[],status='ingested_explicit_OS' if os else 'ingested_lect_uncertain',resolved_reference=target and {'page':target['page'],'item':target['item']}))
    for c in read('grammar-staged.tsv'):
        key=f"{SOURCE}:p{c['page']}:grammar:{c['unit']}";keys=[]
        disposition=c['disposition'];lang={'outer_siraji_control':'OuterSiraji','inner_siraji_control':'insir'}.get(disposition,'Kotguru')
        if disposition!='inventory_only':
            for j,form in enumerate(c['forms'].split(';')):
                k=key if j==0 else key+f':answer{j+1}';keys.append(k)
                notes=''
                if c['unit']=='noun-table-elephant-gen':notes='The source prints bāthīau(ō), although surrounding elephant forms begin h; the literal apparent printing irregularity is retained.'
                if c['unit']=='continuative-Bailey':notes='The source translates this construction as continuing to fall; retained despite its strike-stem elsewhere.'
                out.append(row(k,form,c['gloss'],f"{SOURCE}[p.{c['page']}, grammar {c['unit']}]",c['tags'].split(),notes,lang,lang=='Kotguru'))
        audit.append(dict(c,section='grammar',source_cell_key=key,entry_keys=keys,reuse_entry_keys=[],status='inventory_only_bound_or_rejected_form' if not keys else 'ingested'))
    seen={}
    for c in read('specimen-staged.tsv'):
        key=f"{SOURCE}:specimen:{c['unit']}";form=nfc(c['printed_form_review']);gloss=c['printed_gloss'];pair=(form,gloss)
        citation=f"{SOURCE}[p.{c['page']}, specimen1, line {c['line']}, aligned word {c['word']}]"
        if pair in seen:
            previous=out[seen[pair]];previous[7]+=';'+citation;keys=[];reuse=[previous[10]]
        else:
            seen[pair]=len(out);keys=[key];reuse=[];out.append(row(key,form,gloss,citation))
        audit.append(dict(c,section='specimen',source_cell_key=key,entry_keys=keys,reuse_entry_keys=reuse,status='reused_identical_form_and_gloss' if reuse else 'ingested'))
    assert len(audit)==1034
    keys={r[10] for r in out};assert len(keys)==len(out)
    assert all(k in keys for a in audit for k in a['entry_keys']+a['reuse_entry_keys'])
    return out,audit
if __name__=='__main__':
    rows,audit=generate()
    with (HERE/'full-preview.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(rows)
    with (HERE/'full-preview-audit.jsonl').open('w') as f:
        for a in audit:f.write(json.dumps(a,ensure_ascii=False)+'\n')
    print(len(audit),'units;',len(rows),'forms; proposal only')

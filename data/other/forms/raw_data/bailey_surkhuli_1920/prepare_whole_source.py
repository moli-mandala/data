"""Prepare the complete Surkhuli source stage, without canonical writes."""
from __future__ import annotations
import csv,hashlib,json,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE='bailey1920surkhuli'

def prepare():
    baseline=HERE/'legacy-before-whole-recovery.csv'
    legacy=list(csv.reader(baseline.open()))
    assert len(legacy)==267
    raw=(HERE/'whole-chapter-first-reading.jsonl').read_bytes()
    review=json.loads((HERE/'whole-chapter-own-second-review-20260926.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==review['first_reading_sha256']
    records=[json.loads(line) for line in raw.decode().splitlines()]
    records += [json.loads(line) for line in (HERE/'whole-scope-additional-attestations.jsonl').read_text().splitlines()]
    fixes={x['entry_key']:x for x in review['corrections']}
    fixes.update({x['entry_key']:x for x in json.loads((HERE/'whole-post-audit-corrections.json').read_text())})
    morphology=json.loads((HERE/'whole-morphology-decisions.json').read_text())
    aliases={'past':['pret'],'imperf':['pret','ipfv'],'perf':['perfect'],'plup':['pret','perfect'],'cond':['conditional'],'rel':['relative'],'multiword':['multiword-expression'],'cardinal':[],'comparative':['degree'],'superlative':['degree']}
    rows=[r[:] for r in legacy]
    audit=[json.loads(line) for line in (HERE/'legacy-before-whole-recovery-audit.jsonl').read_text().splitlines()]
    for record in records:
        r=json.loads(json.dumps(record));key=r['entry_key']
        if key in fixes:
            fix=fixes[key];assert fix['old_forms']==r['forms'];r['forms']=fix['forms']
            r['source_forms']=fix.get('source_forms',fix['forms']);r['second_reading_correction']=fix
        r['exported_entry_keys']=[]
        if key in morphology:
            decision=morphology[key]
            r['status']='morphology_with_context';r['morphology_decision']=decision
            for e in decision['emissions']:
                ek=key+':'+e['key_suffix']
                citation=f"{SOURCE}[p. {r['printed_page']}, {r['section']}, item {r['item']}]"
                rows.append(['surkh','',e['form'],'','','',e['note'],citation,'','',ek,'','','',' '.join(e['tags'])])
                r['exported_entry_keys'].append(ek)
            audit.append(r);continue
        if r['status']!='target':audit.append(r);continue
        note=r['note'];tags=r['tags'][:]
        for procedural in [
            'Printed split-head paradigm; only the explicitly supplied stem and ending are joined.',
            'Case and number follow the printed paradigm.',
            'Shared printed pīṭā is retained with the explicitly supplied masculine/feminine auxiliary.',
            '; no unattested full forms are generated',
            ' no unattested combinations are generated.',
            'The second i has a raised roof-shaped mark, preserved literally pending independent review.',
            'The second i has a roof-shaped mark, preserved literally pending independent review.',
        ]:note=note.replace(procedural,'')
        note=note.replace('Sentence22 is printed as a translated phrase; no omitted predicate is reconstructed.','Printed as a translated phrase without an expressed predicate.')
        note=note.replace('are preserved as separately attested alternatives with their aligned glosses;','are printed in parentheses;')
        note=note.replace('retains only the printed alternate, not reconstructed combinations.','the complete primary reading is also attested.')
        note=' '.join(note.split()).strip()

        if 'conj' in tags and 'part' in tags:
            tags=[t for t in tags if t not in {'conj','part'}]+['conjunctive-participle']
        elif 'part' in tags and 'verb' in tags:tags=['participle' if t=='part' else t for t in tags]
        if 'plup' in tags:note+=' Source labels this pluperfect.'
        if 'comparative' in tags:note+=' Source comparison construction.'
        if 'superlative' in tags:note+=' Source comparison is glossed best.'
        normalized=[]
        for tag in tags:normalized.extend(aliases.get(tag,[tag]))
        if key in {SOURCE+':149:pronouns:this-f-agent',SOURCE+':149:pronouns:that-f-agent',SOURCE+':154:sentences:21'}:
            normalized.append('uncertain');r['uncertainty'].append('transcription: small raised roof-shaped mark on i preserved provisionally as circumflex; exact mark may be damaged nasal notation')
            note=note.replace('pending independent review.','').strip()+' The small roof-shaped mark on i is retained literally without phonological interpretation.'
        if key==SOURCE+':117:introduction:8':note=note.replace('lăgno','lăgṇo')
        if r['uncertainty']:
            if 'uncertain' not in normalized:normalized.append('uncertain')
            note+=' Source uncertainty: '+'; '.join(r['uncertainty'])+'.'
        normalized=list(dict.fromkeys(normalized));r['exported_tags']=normalized;r['exported_notes']=note.strip()
        for i,form in enumerate(r['forms'],1):
            ek=key if i==1 else key+f':answer{i}'
            citation=f"{SOURCE}[p. {r['printed_page']}, {r['section']}, item {r['item']}]"
            form=unicodedata.normalize('NFC',form)
            rows.append(['surkh','',form,r['gloss'],'','',note.strip(),citation,'','',ek,key if i>1 else '','','',' '.join(normalized)])
            r['exported_entry_keys'].append(ek)
        audit.append(r)
    assert len({r[10] for r in rows})==len(rows)
    assert rows[:267]==legacy
    return rows,audit,records

if __name__=='__main__':
    rows,audit,records=prepare()
    with (HERE/'whole-proposed.csv').open('w',newline='') as f:csv.writer(f).writerows(rows)
    (HERE/'whole-proposed-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    print(json.dumps({'rows':len(rows),'old_rows':267,'supplemental_units':len(records),'audit_units':len(audit),'source_stage_only':True}))

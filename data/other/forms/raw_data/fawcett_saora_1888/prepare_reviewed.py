"""Reconcile immutable article drafts into a reviewable whole-article inventory."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def load(name):return [json.loads(x) for x in (P/name).read_text().splitlines() if x]
def build():
    first=load('article-lexical-first-reading-20260927.jsonl');changes=[]
    rows=[]
    for original in first:
        r=json.loads(json.dumps(original));key=r['source_unit_key']
        r['tags']=['noun' if t=='n' else t for t in r['tags']]
        if ':weekday:04-telugu' in key:r['forms']=['Laksmivāram','Gurravāram'];r['notes']='Printed explicit alternatives joined by or; retained only as Telugu controls.'
        if ':weekday:05-telugu' in key:r['forms']=['Sukruvāram']
        if r['scope']=='Saora' and r['section']=='weekday':
            r['notes']+=' Source says the Saora names are evidently taken from Uriya or Telugu; individual donors are not identified.'
            r['tags']+=['loanword','uncertain']
        if key.endswith(':prose:iron'):r['tags']+=['loanword']
        if key.endswith(':prose:magatha'):r['tags']+=['loanword']
        if ':numeral:12' in key or ':numeral:13' in key:r['uncertainty']=[]
        r['status']='owner whole-article reading reconciled; independent complete-output audit pending'
        if r['forms']!=original['forms'] or r['tags']!=original['tags']:
            changes.append(dict(source_unit_key=key,old_forms=original['forms'],forms=r['forms'],old_tags=original['tags'],tags=r['tags']))
        rows.append(r)
    kinship=load('kinship-reviewed-candidate-20260927.jsonl')
    for r in kinship:r['tags']+=['kinship']
    rows+=kinship+load('documentary-evidence-20260927.jsonl')
    assert len(rows)==164 and len({r['source_unit_key']for r in rows})==164
    (P/'full-reviewed-inventory.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n'for r in rows))
    (P/'owner-reconciliation-20260927.json').write_text(json.dumps(dict(status='reconciled source proposal; independent final audit pending',units=len(rows),input_hashes={n:hashlib.sha256((P/n).read_bytes()).hexdigest()for n in ['article-lexical-first-reading-20260927.jsonl','kinship-reviewed-candidate-20260927.jsonl','documentary-evidence-20260927.jsonl']},changes=changes,decisions=['Noun draft shorthand n replaced by registered noun; no neuter gender inferred.','Thursday Telugu control has two alternatives; Friday control reads Sukruvāram.','Source-attributed borrowing claims typed with uncertainty when tentative; no donor edges invented.','Numeral9 Thīmji retained after narrow original detail adjudication;12/13 literal diaeresis confirmed independently.','Subgroup labels remain documentary evidence, not invented source-language forms.']),ensure_ascii=False,indent=2)+'\n')
    print(len(rows),'units')
if __name__=='__main__':build()

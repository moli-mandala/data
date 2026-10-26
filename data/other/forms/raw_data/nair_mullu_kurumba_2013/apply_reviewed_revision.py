"""Guarded root-adjudicated metadata/relations derivative; preserves revision1."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'data/SOURCE_INGESTION_CHECKLIST.md').exists())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
BASE = HERE/'revision1'
assert sha(HERE/'physical-units-first-reading.jsonl') == '7bd8af78668aeff2453ebb99ae9fd0e8ad57212f10af6f3867164ea8e29f51c0'
if not BASE.exists():
    freeze = json.loads((HERE/'review-freeze.json').read_text())
    assert sha(HERE/'review-freeze.json') == '3a6fda61dbfa06a27de82b857193d7be65b0bf1085faac04cb7f65938d3cc818'
    for relative, value in freeze['files'].items(): assert sha(ROOT/relative)==value
    BASE.mkdir()
    archived = {}
    for relative, value in freeze['files'].items():
        source = ROOT/relative
        local = source.relative_to(HERE)
        if source.suffix == '.png':
            archived[relative] = {'retained_immutable_original': relative, 'sha256':value}
            continue
        destination = BASE/local
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        assert sha(destination)==value
        archived[relative] = {'archive':str(destination.relative_to(ROOT)), 'sha256':value}
    shutil.copyfile(HERE/'review-freeze.json', BASE/'review-freeze.json')
    test = ROOT/'data/tests/test_nair_mullu_kurumba_proposal.py'
    assert sha(test)==freeze['test_file_sha256']
    shutil.copyfile(test, BASE/'test_nair_mullu_kurumba_proposal.py')
    (BASE/'archive-map.json').write_text(json.dumps(archived,indent=2)+'\n')
assert sha(BASE/'proposed.csv') == '4e85544cd7835bdd84eb58898b0b74368efc81c04aada4b0c7d89713297d0c83'
assert sha(BASE/'review-freeze.json') == '3a6fda61dbfa06a27de82b857193d7be65b0bf1085faac04cb7f65938d3cc818'
review = HERE/'independent-review/complete-original-review-pass1.json'
assert sha(review) == 'f4783f86c9a88c2f9ff4877738b0e63760c3e0b99391b58d7edab219962b1df8'
independent = json.loads(review.read_text())
assert independent['literal_corrections']==[]
rows = list(csv.reader((BASE/'proposed.csv').open(newline='')))
keyed = {r[10]:r for r in rows}
changes = []
for decision in independent['metadata_recommendations']:
    row=keyed[decision['entry_key']]
    index={'Tags':14,'Gloss':3}[decision['field']]
    assert row[index]==decision['old']
    row[index]=decision['proposed']
    evidence=decision['source_evidence']
    if decision['entry_key']=='nair2013mullukurumba:p55:plural-examples:w1':
        evidence='Original p55 explicitly identifies -en as masculine; segmented mutt-en-maar-u includes that printed masculine exponent. Gender is source-morphology based, not inferred solely from English old men.'
    changes.append(dict(entry_key=decision['entry_key'],field=decision['field'],old=decision['old'],new=row[index],source_evidence=evidence,
        acceptance='Root approved these eight exact metadata decisions after originals55–57 review on 2026-09-27.'))
slash=json.loads((HERE/'slash-alternative-adjudication-proposal.json').read_text())
assert slash['inputs']['csv_sha256']==sha(BASE/'proposed.csv')
for decision in slash['proposed_decisions']:
    row=keyed[decision['entry_key']]
    assert row[11]==decision['old'] and row[14]==decision['old_tags']
    for index,field,value in [(11,'Variant_Of_Key',decision['proposed_value']),
        (14,'Tags',decision['proposed_tags']), (6,'Notes',decision['proposed_notes'])]:
        changes.append(dict(entry_key=row[10],field=field,old=row[index],new=value,
            source_evidence=decision['source_evidence'], acceptance='Root approved same-function source-listed morphological alternative relation, not unrestricted free interchangeability.'))
        row[index]=value
assert len(changes)==35 and len({c['entry_key'] for c in changes})==17
assert sum(bool(r[11]) for r in rows)==11 and all(not r[11] or r[11] in keyed for r in rows)
old=list(csv.reader((BASE/'proposed.csv').open(newline='')))
assert [r[10] for r in old]==[r[10] for r in rows]
actual_changes=[dict(entry_key=b[10],index=i,old=a[i],new=b[i]) for a,b in zip(old,rows) for i in range(15) if a[i]!=b[i]]
assert len(actual_changes)==35 and all(c['index'] in (3,6,11,14) for c in actual_changes)
assert all(a[2]==b[2] and a[4:6]==b[4:6] and a[7:11]==b[7:11] for a,b in zip(old,rows))
with (HERE/'proposed.csv').open('w',newline='') as stream: csv.writer(stream).writerows(rows)
audit=[json.loads(s) for s in (BASE/'proposal-audit.jsonl').read_text().splitlines()]
for unit in audit:
    relevant=[c for c in changes if c['entry_key'] in unit['entry_keys']]
    if relevant: unit['root_adjudicated_output_changes']=relevant
    for decision in unit['witness_dispositions']:
        keys=decision['entry_keys']
        if keys and decision['disposition']=='proposed_lexical_emission': decision['grammar_tags']=keyed[keys[0]][14].split()
(HERE/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit))
metadata=json.loads((BASE/'metadata-proposal.json').read_text())
for record in metadata['keyed_records']:
    row=keyed[record['witness_key']]
    record['proposed_gloss']=row[3];record['tags']=row[14].split()
(HERE/'metadata-proposal.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
graph=json.loads((BASE/'graph-proposal.json').read_text())
for decision in slash['proposed_decisions']:
    graph['proposals'].append(dict(child_key=decision['entry_key'],variant_of_key=decision['proposed_value'],
        evidence=decision['source_evidence'],relation_scope='Same listed pronominal-suffix function only; no unrestricted free interchangeability.'))
assert len(graph['proposals'])==11
(HERE/'graph-proposal.json').write_text(json.dumps(graph,ensure_ascii=False,indent=2)+'\n')
continuity=dict(status='Root-adjudicated revision2; fresh final independent output audit pending',
    inputs={'archived_csv':sha(BASE/'proposed.csv'),'independent_review':sha(review),'slash_proposal':sha(HERE/'slash-alternative-adjudication-proposal.json')},
    changes=changes, changed_rows=17, changed_fields=35, field_counts={'Gloss':1,'Notes':9,'Variant_Of_Key':9,'Tags':16},
    all_135_ordered_keys_and_literal_forms_unchanged=True, source_first_reading_unchanged=True,
    new_csv_sha256=sha(HERE/'proposed.csv'),new_audit_sha256=sha(HERE/'proposal-audit.jsonl'),canonical_changes=False)
(HERE/'review-repair-continuity.json').write_text(json.dumps(continuity,ensure_ascii=False,indent=2)+'\n')
status=json.loads((BASE/'candidate-status.json').read_text())
status['status']='Whole-source revision2 review proposal; not installed'
status['variant_proposals']=11
status['outputs']={n:sha(HERE/n) for n in ('proposed.csv','proposal-audit.jsonl','metadata-proposal.json','graph-proposal.json')}
status['pending']=['Fresh independent actual-output audit','root acceptance and guarded installation','compiled/full-build/browser gates']
(HERE/'candidate-status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps({'rows':135,'changed_rows':17,'changed_fields':35,'variant_links':11,'canonical_changes':False}))

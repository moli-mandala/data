"""Create an UNACCEPTED 15-column review candidate; never install canonical data."""
import argparse,copy,csv,hashlib,io,json
from pathlib import Path
H=Path(__file__).resolve().parent
P=H.parent
DATA=next(p for p in P.parents if (p/'tags.py').exists())
FIRST=P/'first-reading/complete-first-reading.jsonl'
FIRST_SHA='22fe3f77dd063609585a70cbe032392e143c955dc237df471a4c16dedf316136'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
def build(overlays=()):
    assert sha(FIRST)==FIRST_SHA,'First reading is immutable; reconcile through approved overlays.'
    records=copy.deepcopy(read(FIRST))
    assert len(records)==97 and sum(len(r['witnesses']) for r in records)==105
    inventory=read(P/'scope/source-unit-inventory.jsonl')
    assert [r['source_unit_key'] for r in records]==[r['source_unit_key'] for r in inventory]
    witnesses={w['witness_key']:w for r in records for w in r['witnesses']}
    assert len(witnesses)==105
    physical_keys=set(witnesses)
    units={r['source_unit_key']:r for r in records}
    excluded={};added={};variant_links={}
    original_roles={k:'target' if w['language']=='Aranadan' else 'explicit-comparison' for k,w in witnesses.items()}
    applied=[];held=[];pins={str(FIRST.relative_to(P)):FIRST_SHA}
    for overlay_path in overlays:
        overlay_path=Path(overlay_path).resolve()
        assert overlay_path.is_relative_to((P/'root-review').resolve()),'Only root-review approvals may be applied.'
        overlay=json.loads(overlay_path.read_text())
        assert overlay['approved'] is True and overlay['first_reading_sha256']==FIRST_SHA
        for name,expected in overlay.get('evidence_sha256',{}).items():
            evidence=(P/name).resolve();assert evidence.is_relative_to(P.resolve())
            assert sha(evidence)==expected,(name,'changed evidence')
        seen=set()
        for d in overlay['decisions']:
            key=d['witness_key'];field=d['field'];pair=(key,field)
            assert pair not in seen;seen.add(pair)
            assert field in {'form','gloss','language','tags','note','typed_uncertainty'}
            assert witnesses[key][field]==d['old'],(key,field,'old value mismatch')
            if d['decision']=='accepted':
                witnesses[key][field]=copy.deepcopy(d['new']);applied.append(d)
            else:
                assert d['decision']=='held';held.append(d)
        for d in overlay.get('unit_exclusions',[]):
            unit=units[d['source_unit_key']]
            assert [w['witness_key'] for w in unit['witnesses']]==d['expected_witness_keys']
            for w in unit['witnesses']:
                assert w['witness_key'] not in excluded
                excluded[w['witness_key']]=d['reason']
        for d in overlay.get('witness_exclusions',[]):
            key=d['witness_key'];assert witnesses[key]['form']==d['expected_form']
            assert key not in excluded;excluded[key]=d['reason']
        for d in overlay.get('added_witnesses',[]):
            unit=units[d['source_unit_key']]
            anchor=witnesses[d['shared_physical_witness_key']]
            assert anchor['witness_key'] in physical_keys
            assert anchor in unit['witnesses'] and anchor['form']==d['expected_physical_form']
            w=copy.deepcopy(d['witness']);key=w['witness_key']
            assert key not in witnesses and key.startswith(unit['source_unit_key']+':')
            assert set(w)==set(anchor),'Added witness must use the same explicit field schema.'
            assert w['form']==anchor['form'],'Shared printed attestation cannot invent a new spelling.'
            assert d['reason'] and d['evidence']
            witnesses[key]=w;unit['witnesses'].append(w);added[key]=d
            original_roles[key]='explicit-comparison'
        for d in overlay.get('variant_links',[]):
            child=d['child_key'];parent=d['parent_key']
            assert child!=parent and child in witnesses and parent in witnesses
            assert child not in variant_links and witnesses[child]['form']==d['expected_child_form']
            assert witnesses[parent]['form']==d['expected_parent_form']
            assert witnesses[child]['language']==witnesses[parent]['language']
            assert d['reason'] and d['evidence'];variant_links[child]=d
        pins[str(overlay_path.relative_to(P))]=sha(overlay_path)
    languages={r['ID'] for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
    rows=[];audit=[]
    for record in records:
        keys=[]
        for w in record['witnesses']:
            if w['witness_key'] in excluded:continue
            assert w['language'] in languages and w['form']
            key=w['witness_key'];tags=list(dict.fromkeys(w['tags']))
            if w['typed_uncertainty'] and 'uncertain' not in tags:tags.append('uncertain')
            physical_locator_key=added[key]['shared_physical_witness_key'] if key in added else key
            locator=f"p. {record['printed_page']}, {record['section']}, item {record['position']}, witness {physical_locator_key.rsplit(':w',1)[1]}"
            row=[w['language'],'',w['form'],w['gloss'],'','',w['note'],f'nair2013aranadan[{locator}]','','',key,variant_links.get(key,{}).get('parent_key',''),'','',' '.join(tags)]
            assert len(row)==15;rows.append(row);keys.append(key)
        audit.append(dict(source_unit_key=record['source_unit_key'],pdf_page=record['pdf_page'],printed_page=record['printed_page'],
            raw_source_page_sha256=record['raw_source_page_sha256'],source_record=record,
            review_candidate_keys=keys,physical_witness_slots=sum(w['witness_key'] in physical_keys for w in record['witnesses']),
            added_shared_attributions=[added[w['witness_key']] for w in record['witnesses'] if w['witness_key'] in added],
            exclusions={w['witness_key']:excluded[w['witness_key']] for w in record['witnesses'] if w['witness_key'] in excluded},
            witness_roles={k:original_roles[k] for k in keys},
            disposition='review-candidate-not-accepted' if keys else 'documentary/context-only',
            accepted_for_installation=False,
            pending='Independent literal/metadata and root scope approval; no graph assertion or final language/control acceptance.'))
    assert len(rows)==105+len(added)-len(excluded) and len({r[10] for r in rows})==len(rows)
    emitted={r[10] for r in rows}
    assert {k for a in audit for k in a['review_candidate_keys']}==set(witnesses)-set(excluded)==emitted
    assert sum(a['physical_witness_slots'] for a in audit)==105
    for child,d in variant_links.items():assert child in emitted and d['parent_key'] in emitted
    out=io.StringIO(newline='');csv.writer(out).writerows(rows)
    outputs={'review-candidate.csv':out.getvalue(),'review-audit.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit)}
    for name,text in outputs.items():(H/name).write_text(text)
    status=dict(status='unaccepted complete review scaffold; canonical install forbidden',first_reading_sha256=FIRST_SHA,
        audit_units=97,physical_witness_slots=105,review_rows=len(rows),target_witnesses=sum(r[0]=='Aranadan' for r in rows),explicit_comparison_witnesses=sum(r[0]!='Aranadan' for r in rows),
        documentary_units=sum(not a['review_candidate_keys'] for a in audit),additional_alternative_rows=0,shared_attribution_rows=len(added),excluded_witnesses=excluded,variant_links=variant_links,
        count_note='Two slash alternatives were already separate physical witness slots in first reading; do not expand again. Twenty comparators are review-only pending final scope approval, not accepted Aranadan forms.',
        inputs=pins,outputs={name:sha(H/name) for name in outputs},applied_decisions=applied,held_decisions=held,
        graph_fields_blank=not variant_links,only_explicit_approved_variant_links=True,native_and_phonemic_blank=True,accepted_for_installation=False,
        pending=['Independent full original reading and metadata review.','Root approved literal/metadata overlays and comparison disposition.','Root settings, source-provenance and sound-profile decisions.','Closed source graph claims only after explicit acceptance; auxiliary Usha remains background.','Actual parser/profile checks and fresh output audit before any guarded installation.'])
    (H/'review-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    return status
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--overlay',action='append',default=[])
    print(json.dumps(build(p.parse_args().overlay),ensure_ascii=False,indent=2))

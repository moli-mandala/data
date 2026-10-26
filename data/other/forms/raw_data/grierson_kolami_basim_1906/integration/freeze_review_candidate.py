"""Pin final source-local candidate and deterministic original-output review selection."""
import csv,hashlib,json,random
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
ROOT=HERE.parents[6]
OUT=HERE/'revision4'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
    with (OUT/'combined-review-candidate.csv').open() as f:rr=list(csv.reader(f))
    by={r[10]:r for r in rr};assert len(rr)==len(by)==827
    audit=rows(OUT/'combined-review-audit.jsonl')
    records={a['source_unit_key']:a for a in audit if 'actual_row' in a}
    assert set(records)==set(by)
    parser=json.loads((OUT/'actual-parser-review.json').read_text())
    assert parser['candidate_sha256']==sha(OUT/'combined-review-candidate.csv')
    metadata=json.loads((OUT/'metadata-reference-readiness.json').read_text())
    assert metadata['input_pins'][str((OUT/'combined-review-candidate.csv').relative_to(ROOT))]==sha(OUT/'combined-review-candidate.csv')
    pins={str(p.relative_to(PACKAGE)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name not in {'review-freeze.json','fresh-output-selection-20260927.json'}}
    dependencies=[PACKAGE/'root-independent-review/residual-metadata-root-adjudication.json',
      PACKAGE/'root-independent-review/specimen-metadata-root-adjudication.json',
      PACKAGE/'root-independent-review/consolidated-24-literal-acceptance.json',
      PACKAGE/'root-independent-review/sharp-grammar-35-literal-acceptance.json',
      PACKAGE/'root-independent-review/sharp-grammar-metadata-r2-verification.json',
      PACKAGE/'root-independent-review/whole-candidate-literal-accounting.json',
      PACKAGE/'specimen-conservation-review/sharp-revised-conservation-review.json',
      PACKAGE/'grammar-review/sharp-reviewed-revision3/reviewed-child-witnesses.jsonl',
      PACKAGE/'grammar-review/sharp-reviewed-revision3/grammar-tag-proposal.jsonl',
      PACKAGE/'grammar-review/sharp-reviewed-revision3/grammar-graph-proposal.jsonl',
      PACKAGE/'specimens/sharp-reviewed-revision/atomic-emission-proposal.jsonl',
      PACKAGE/'specimens/sharp-reviewed-revision/expression-emission-proposal.jsonl',
      PACKAGE/'outside-review/outside-census-review.json',
      HERE/'apply_specimen_metadata_review.py',HERE/'close_historical_commentary.py',HERE/'apply_residual_metadata.py',
      HERE/'check_actual_parser.py',HERE/'check_metadata_readiness.py',HERE/'check_revised_accounting.py',Path(__file__)]
    pins.update({str(p.relative_to(PACKAGE)):sha(p) for p in dependencies})
    previous=PACKAGE/'root-independent-review/specimen-first-audit-selection-20260927701.json'
    excluded={r['source_unit_key'] for r in json.loads(previous.read_text())['sample']}
    physical_used=set(excluded);chosen=[];randomizer=random.Random(20260927)
    def physical(a):
        r=a['source_record'];return set(r.get('physical_source_keys',[a['source_unit_key']]))
    def select(kind,page,count):
        pool=[a for a in records.values() if a['kind']==kind and (page is None or f'[p. {page}]' in by[a['source_unit_key']][7]) and not physical(a)&physical_used]
        randomizer.shuffle(pool)
        for a in pool:
            if physical(a)&physical_used:continue
            chosen.append(a);physical_used.update(physical(a));count-=1
            if not count:return
        assert count==0,(kind,page,count)
    for page in [562,563,566,567]:select('grammar',page,2)
    for page in [564,565,568,569]:select('atomic',page,2)
    # Whole expressions must not overlap selected atom/old-sample witnesses.
    select('expression',564,1);select('expression',568,1)
    select('outside',182,1);select('outside',randomizer.choice([626,627]),1)
    assert len(chosen)==20 and len({a['source_unit_key'] for a in chosen})==20
    controls=[]
    for a in records.values():
        key=a['source_unit_key'];r=a['source_record']
        if ('ï' in by[key][2] or ':present-copula:3:' in key or
            (a['kind']=='grammar' and r.get('typed_uncertainties')) or
            len(r.get('physical_source_keys',[]))==2 and a['kind']=='atomic' or
            key in {'grierson1906kolamibasim:p564:specimen75-line:4:word:5','grierson1906kolamibasim:p564:specimen75-line:4:word:6',
                    'grierson1906kolamibasim:p568:specimen76-line:3:word:3','grierson1906kolamibasim:p568:specimen76-line:2:word:6',
                    'grierson1906kolamibasim:p569:specimen76-line:2:word:5','grierson1906kolamibasim:p564:specimen75-line:2:word:3',
                    'grierson1906kolamibasim:p565:specimen75-line:2:word:3','grierson1906kolamibasim:p565:specimen75-line:4:word:1',
                    'grierson1906kolamibasim:p564:specimen75-line:10:word:4','grierson1906kolamibasim:p564:specimen75-line:16:word:3'}):
            controls.append(a)
    def item(a):return {'source_unit_key':a['source_unit_key'],'kind':a['kind'],'actual_row':by[a['source_unit_key']],
      'source_locators_and_literal_evidence':a['source_record'],'typed_uncertainties':a.get('typed_uncertainties',[]),
      'typed_metadata_uncertainties':a.get('typed_metadata_uncertainties',[]),'scholarly_source_claims':a.get('scholarly_source_claims',[])}
    grammar_repairs=[r for r in rows(PACKAGE/'grammar-native-review/full224-sharp-review-proposal.jsonl') if r['old_form']!=r['proposed_form']]
    specimen_repairs=json.loads((PACKAGE/'specimens/sharp-reviewed-revision/keyed-literal-overlay.json').read_text())
    assert len(grammar_repairs)==35 and len(specimen_repairs)==24
    repair_guards=[]
    for repair in grammar_repairs+specimen_repairs:
        key=repair['source_unit_key']
        matches=[a for a in audit if a['source_unit_key']==key and 'source_record' in a]
        assert len(matches)==1,(key,len(matches))
        repair_guards.append({'source_unit_key':key,'accepted_literal_repair':repair,'actual_emitted_row':by.get(key),
            'current_source_record':matches[0]['source_record'],'disposition':matches[0]['disposition'],'entry_keys':matches[0]['entry_keys']})
    sample={'status':'Selected before independent original-output comparison; not acceptance','seed':20260927,
      'candidate_sha256':sha(OUT/'combined-review-candidate.csv'),'audit_sha256':sha(OUT/'combined-review-audit.jsonl'),
      'excluded_prior_sample_sha256':sha(previous),'excluded_prior_source_keys':sorted(excluded),'sample':[item(a) for a in chosen],
      'strata':{'grammar':8,'atomic':8,'whole_expression':2,'outside':2},'targeted_controls':[item(a) for a in controls],
      'literal_repair_regression_controls':repair_guards,'literal_repair_counts':{'grammar':35,'specimen_fields':24,'total':59},
      'crops':'Use retained native originals/manifests and exact source_record page/line/word or grammar paragraph labels. Existing isolated control crops are retained in sharp correction ledgers; no bulk render required.'}
    (OUT/'fresh-output-selection-20260927.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    pins[str((OUT/'fresh-output-selection-20260927.json').relative_to(PACKAGE))]=sha(OUT/'fresh-output-selection-20260927.json')
    manifest={'status':'Frozen final source-local review candidate; independent final-output audit and canonical installation pending',
      'files':pins,'counts':{'new_rows':827,'reused_existing_rows':3,'audit_records':1570,'structural_parents':169,'grammar_children':224,'physical_atoms':561,'atomic_emissions':560,'whole_expressions':51},
      'canonical_changes':False,'deferred':['Fresh independent final actual-output original audit','Exact source-stage installer root approval/canonical checks','DB/CLDF/full tests/browser/publication'],
      'reuse_policy':'Three existing witnesses retain their old keys/citations. New chapter scope mappings are audit-only, not compiled alias or new citation proof.'}
    (OUT/'review-freeze.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'candidate':sha(OUT/'combined-review-candidate.csv'),'audit':sha(OUT/'combined-review-audit.jsonl'),'freeze':sha(OUT/'review-freeze.json'),'sample':sha(OUT/'fresh-output-selection-20260927.json'),'selected':len(chosen),'targeted_controls':len(controls),'literal_repair_guards':len(repair_guards)}))
if __name__=='__main__':main()

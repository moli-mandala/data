"""Root approval of source-local Ahirwati grammar review; not installation."""
import hashlib
import json
from collections import Counter
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=Path(__file__).resolve().parents[7]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]
first=HERE/'child-witnesses-first-reading.jsonl'
reviewed=HERE/'child-witnesses-reviewed.jsonl'
review=BASE/'independent-grammar-review/independent-full-grammar-review.json'
assert sha(first)=='96ccc5261f3ddca49a5fe96b08675f800d64173c7d590a7eda4048882cb6f6d7'
assert sha(reviewed)=='5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7'
assert sha(review)=='bd3bb95c88dfbe71317838ac598897de5a947c8d74cad88643a3fc3e98c1b9a3'
a,b=rows(first),rows(reviewed)
assert len(a)==95 and len(b)==98
bykey={r['witness_key']:r for r in b};assert len(bykey)==98
assert [r['witness_key'] for r in b[:95]]==[r['witness_key'] for r in a]
expected={
 ('lsi9.2:p50:nominal-ending-comparison-continuation:w3','source_form'):('ē','ā'),
 ('lsi9.2:p51:qualified-oblique-relative-interrogative:w2','source_form'):('kaun','kauṇ'),
 ('lsi9.2:p50:nominal-ending-comparison-continuation:w4','language_label'):('Bagri','Bangaru'),
 ('lsi9.2:p50:nominal-ending-comparison-continuation:w5','language_label'):('Bagri','Bangaru'),
}
continuity=json.loads((HERE/'reviewed-derivative-continuity.json').read_text())
context={(r['key'],r['field']):(r['old'],r['new']) for r in continuity['changes'] if r['field']=='source_quote'}
assert len(context)==11
changes=Counter();seen=set()
for old in a:
    key=old['witness_key'];new=bykey[key]
    for field in old.keys()|new.keys():
        if old.get(field)==new.get(field):continue
        changes[field]+=1
        signature=(key,field)
        if field in ('source_form','language_label'):
            assert signature in expected
            assert (old[field],new[field])==expected[signature]
            seen.add(signature)
        elif field=='source_quote':assert (old[field],new[field])==context[signature]
        else:assert field in {'stage','literal_review_resolution','typed_uncertainties','historical_typed_uncertainties'},signature
    if old['typed_uncertainties']!=new['typed_uncertainties']:
        assert new['historical_typed_uncertainties']==old['typed_uncertainties']
        assert new['typed_uncertainties']==[] and new['literal_review_resolution']
assert seen==set(expected)
assert sum(r['disposition']=='unattested_source_hypothesis' for r in b)==4
added=b[95:]
assert [(r['language_label'],r['source_form']) for r in added]==[('Mewati','ō'),('Mewati','ā'),('Mewati','kō')]
for r in added:
    assert r['physical_span_shared'] is True and len(r['physical_source_keys'])==1
    original=bykey[r['physical_source_keys'][0]]
    assert r['source_form']==original['source_form'] and r['parent_key']==original['parent_key']
    assert r['source_analysis']==original['source_analysis'] and r['source_gloss']==original['source_gloss']
    assert r['disposition']=='eligible_comparator'
assets={
 'tmp/jambu-next-source/ahirwati-native-grammar-20260927/peer-p50-nominal-ending-native.png':'07d6d3ccec441bc206a391106ca91addc4ce9671e84129df2e36c4f662d6a70d',
 'tmp/jambu-next-source/ahirwati-native-grammar-20260927/p51-first-reading-hypothesis-detail.png':'daa47fae2c74d960550063ff57ca8f3299ecb1b197dd2867da3d58439ab60c6e',
}
for p,pin in assets.items():assert sha(ROOT/p)==pin
assert len(rows(HERE/'parent-coverage-reviewed.jsonl'))==51
target=HERE/'root-reviewed-grammar-acceptance.json';assert not target.exists()
report={'reviewer':'/root','status':'accepted_source_local_four_keyed_corrections_and_three_shared_language_interpretations',
        'input_sha256':{str(p.relative_to(BASE)):sha(p) for p in [first,reviewed,review,HERE/'reviewed-derivative-continuity.json',HERE/'parent-coverage-reviewed.jsonl']},
        'reviewed_interpretations':98,'preserved_first_reading_keys':95,'covered_parents':51,'field_changes':dict(changes),
        'accepted_field_corrections':[{'key':k,'field':f,'old':v[0],'new':v[1]} for (k,f),v in expected.items()],
        'shared_language_interpretations':[r['witness_key'] for r in added], 'native_evidence_sha256':assets,
        'counts_by_disposition':dict(Counter(r['disposition'] for r in b)),
        'limits':['Same printed morphological witnesses interpreted under source-explicit Mewati agreement; not three new printed tokens.',
                  'No inherited/borrowing edges inferred from agreement.', 'Whole-source rich emission/settings/profile/registry/refs/graph/final audit pending.',
                  'Four unattested obliques remain audit-only.', 'No canonical writes or full-source completion.']}
target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'path':str(target),'sha256':sha(target),'interpretations':98,'parents':51,'corrections':4,'shared_interpretations':3}))

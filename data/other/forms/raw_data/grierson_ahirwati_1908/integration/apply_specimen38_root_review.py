"""Root-approved lexical metadata and source-anchored expression derivative only."""
import copy,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
SECTION=PACKAGE/'specimen38'
OUT=HERE/'specimen38-reviewed-root-r3'
RECEIPT=SECTION/'independent-metadata-review/root-metadata-boundary-acceptance-r3.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def records(p):return [json.loads(s) for s in p.read_text().splitlines()]
def save(p,value):
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in value)
    if p.exists():assert p.read_text()==data,('Immutable derivative differs',str(p))
    else:p.write_text(data)
def main():
    assert sha(RECEIPT)=='f95fc51bb6c057bd4a9af28369fde0865995e03d2b6d4f1b1be0ef1156301889'
    accepted=json.loads(RECEIPT.read_text());assert accepted['status']=='pass'
    for name,pin in accepted['input_hashes'].items():assert sha(SECTION/name)==pin,name
    atomic_path=SECTION/'native-review/atomic-emissions-proposal.jsonl'
    expression_path=SECTION/'native-review/whole-utterances-proposal.jsonl'
    atoms=records(atomic_path);expressions=records(expression_path)
    decisions={d['source_unit_key']:d for d in accepted['decisions']}
    assert len(atoms)==132 and len(expressions)==23 and len(decisions)==115
    full_path=SECTION/'independent-metadata-review/residual-revision3/all-132-keyed-metadata-decisions.jsonl'
    assert sha(full_path)=='7aac51a57fdeced03a5f8e1f3a43b8fd4b67dc42b24d5360a225f372fd18e5b0'
    full={d['source_unit_key']:d for d in records(full_path)};assert len(full)==132
    reviewed=[]
    for source in atoms:
        a=copy.deepcopy(source);key=a['source_unit_key'];d=full[key]
        assert (a['emission_form'],a['emission_gloss'],a['tags'])==(d['exact_form_guard'],d['exact_gloss_guard'],d['old_tags'])
        if key in decisions:
            authoritative=decisions[key]
            assert authoritative['proposed_tags']==d['proposed_tags'],key
            assert authoritative['old_tags']==d['old_tags'],key
        else:assert d['proposed_tags']==d['old_tags'],key
        a['historical_owner_metadata']={'tags':a['tags'][:],'stage':'Original owner emission proposal retained'}
        a['tags']=d['proposed_tags'][:]
        a['emission_notes']='; '.join(d['proposed_source_notes'])
        a['typed_metadata_uncertainties']=copy.deepcopy(d['additional_scope_holds'])
        a['root_metadata_review']={'receipt_path':str(RECEIPT.relative_to(PACKAGE)),'receipt_sha256':sha(RECEIPT),
          'exact_keyed_decision':d,'status':'Accepted source-local metadata; combined output audit/installation pending'}
        assert all(a[k]==v for k,v in source.items() if k!='tags')
        assert not a['emission_native'] and not a['phonemic']
        reviewed.append(a)
    assert sum(a['tags']!=b['tags'] for a,b in zip(atoms,reviewed))==115
    boundary_path=SECTION/'independent-metadata-review/all-23-keyed-boundary-decisions.jsonl'
    boundaries={r['historical_whole_key']:r for r in records(boundary_path)};assert len(boundaries)==23
    whole=[]
    for source in expressions:
        e=copy.deepcopy(source);b=boundaries[e['key']]
        assert (e['emission_form'],e['emission_gloss'],e['physical_source_keys'])==(b['exact_form_guard'],b['exact_gloss_guard'],b['physical_source_keys'])
        assert b['proposed_stable_key']==e['first_word_key']+':expression'
        e['historical_ordinal_key']=e['key'];e['key']=b['proposed_stable_key']
        e['root_boundary_review']={'receipt_path':str(RECEIPT.relative_to(PACKAGE)),'receipt_sha256':sha(RECEIPT),'exact_keyed_decision':b}
        assert all(e[k]==v for k,v in source.items() if k!='key')
        assert not e['emission_native'] and not e['phonemic']
        whole.append(e)
    assert len({r['key'] for r in whole})==23
    OUT.mkdir(exist_ok=True);save(OUT/'atomic-emissions-reviewed.jsonl',reviewed);save(OUT/'expression-emissions-reviewed.jsonl',whole)
    status={'status':'Root-approved No38 source-local derivative; combined acceptance/installation pending',
      'inputs':{str(p.relative_to(PACKAGE)):sha(p) for p in [RECEIPT,atomic_path,expression_path,full_path,boundary_path]},
      'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='status.json'},
      'counts':{'atoms':132,'tag_fields_changed':115,'expressions':23,'anchored_key_changes':23,'literal_fields_changed':0},
      'canonical_changes':False,'native_policy':'All narrow atom/expression Native fields blank; broad Persian spans remain documentary provenance.'}
    p=OUT/'status.json';data=json.dumps(status,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text()==data
    else:p.write_text(data)
    print(json.dumps(status,ensure_ascii=False))
if __name__=='__main__':main()

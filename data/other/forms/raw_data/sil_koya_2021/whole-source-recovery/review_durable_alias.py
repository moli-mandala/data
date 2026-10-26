"""Read-only targeted identity census and in-memory alias regression."""
import csv
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
OUT=HERE/'durable-alias-review-20260927'
DATA=HERE.parents[5]
OLD='silkoya1985:utn:i198:v2'
TARGET='silkoya1985:utn:i198:v1'
JAG='silkoya1985:jag:i198:v1'
KEYS={OLD,TARGET,JAG}
IDS={'f_clrv55fanq2my','f_7mlyijnudplki','f_vtiprk6vhpmco'}

def select(path,predicate):
    result=[];count=0
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            count+=1
            if predicate(row):result.append(row)
    return {'path':str(path.relative_to(DATA)),'rows_scanned':count,'selected':result,'file_bytes':path.stat().st_size}

def main():
    OUT.mkdir(exist_ok=True)
    evidence={
        'registry':select(DATA/'data/form-identities.csv',lambda r:r.get('Source_Key') in KEYS or r['Form_ID'] in IDS),
        'source_keys':select(DATA/'cldf/form-source-keys.csv',lambda r:r['Source_Key'] in KEYS),
        'forms':select(DATA/'cldf/forms.csv',lambda r:r['ID'] in IDS),
        'aliases':select(DATA/'cldf/form-id-aliases.csv',lambda r:r['Legacy_ID'] in {'135-1213','135-1214','135-591','147-1214'} or r['Legacy_ID'] in IDS),
    }
    registry=evidence['registry']['selected']
    assert not [r for r in registry if r.get('Source_Key')==OLD]
    target=next(r for r in registry if r['Source_Key']==TARGET)
    jag=next(r for r in registry if r['Source_Key']==JAG)
    assert target['Form_ID']=='f_clrv55fanq2my'
    assert jag['Form_ID']=='f_7mlyijnudplki'
    assert 'item 198, Utnoor Gondi' in jag['Source'] and 'Jaganathapuram Koya' in jag['Source']
    alias_path=OUT/'proposed-source-key-alias.csv'
    with alias_path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['Retired_Source_Key','Target_Source_Key','Reason'])
        writer.writerow([OLD,TARGET,'Original PDF101 item198 is one continuous answer va:val; old slash split erroneous. No separate v2 durable registry identity exists in the reviewed baseline.'])
    sys.path.insert(0,str(DATA))
    from assign_form_ids import assign_ids
    from source_key_aliases import apply_source_key_aliases
    with (HERE/'literal-integration/proposed-wordlist.csv').open(newline='') as handle:
        raw={r[10]:r for r in csv.reader(handle) if r[10] in {TARGET,JAG}}
    forms=[];source_keys={}
    for index,key in enumerate([TARGET,JAG]):
        r=raw[key];stage=f'fixture-{index}'
        forms.append({'ID':stage,'Language_ID':r[0],'Form':r[2],'Original':r[2],'Gloss':r[3],'Source':r[7],'Status':'unlinked','Redirect':''})
        source_keys[stage]=key
    mapping,next_registry=assign_ids(forms,registry,source_keys)
    assert mapping['fixture-0']==target['Form_ID']
    assert mapping['fixture-1']==jag['Form_ID']
    aliases={'old-utn-v1-url':target['Form_ID'],'old-shared-jag-url':jag['Form_ID']}
    before=aliases.copy()
    replacements=apply_source_key_aliases(aliases,registry,next_registry,set(mapping.values()),[alias_path])
    assert replacements==0 and aliases==before
    # Demonstrate the existing route if a genuine retired keyed ID existed.
    synthetic_previous=registry+[{'Source_Key':OLD,'Form_ID':'f_test_retired'}]
    synthetic_aliases={'historic-url':'f_test_retired'}
    assert apply_source_key_aliases(synthetic_aliases,synthetic_previous,next_registry,set(mapping.values()),[alias_path])==1
    assert synthetic_aliases=={'historic-url':target['Form_ID'],'f_test_retired':target['Form_ID']}
    try:
        apply_source_key_aliases({},synthetic_previous,next_registry,set(mapping.values())|{'f_test_retired'},[alias_path])
    except ValueError as error:assert 'active ID' in str(error)
    else:raise AssertionError('must reject redirecting an active ID')
    evidence['result']={'status':'passed_readonly_identity_investigation','target_durable_id':target['Form_ID'],'old_v2_registry_matches':0,'shared_historical_attestation_id':jag['Form_ID'],'shared_id_must_remain_with':JAG,'actual_source_key_alias_replacements':replacements,'fixture_checks':['corrected/reordered v1 retains its durable ID','JAG retains shared historic ID','absent v2 key creates no fabricated ID redirect','real retired keyed ID would redirect including historic URL','active ID redirect rejected'],'missing_route':'There is no separate v2 persistent ID to redirect in this baseline. A key-alias CSV cannot and must not redirect the active JAG node. Source-local reconciliation accounts the old key; the real v1 public ID survives by immutable key.','legacy_position_warning':'135-1214 currently aliases unrelated active Tshangla f_vtiprk6vhpmco; do not infer a Koya identity from source row number.','canonical_writes':False,'build_run':False}
    evidence['implementation_sha256']={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest() for name in ['assign_form_ids.py','source_key_aliases.py']}
    (OUT/'evidence-and-fixture-result.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(evidence['result'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()

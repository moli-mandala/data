"""Prepare an explicitly disabled three-target source-spelling addendum."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
ROOT=DATA.parent
KEY='grierson1906kui:p566:outside-comparand:06'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    csv_path=DATA/'data/other/forms/20260927-grierson-kui.csv'
    audit_path=PACKAGE/'installed-audit.jsonl'
    baseline=csv_path.read_bytes();records=baseline.splitlines(keepends=True)
    rows=list(csv.reader(io.StringIO(baseline.decode(),newline='')))
    assert len(records)==len(rows)==1832 and all(len(r)==15 for r in rows)
    position=next(i for i,r in enumerate(rows) if r[10]==KEY)
    assert rows[position][0]=='Kolami' and rows[position][2]=='and'
    assert records[position].startswith(b'Kolami,,and,he,,,') and KEY.encode() in records[position]
    changed=list(records);changed[position]=changed[position].replace(b'Kolami,,and,he,,,',b'Kolami,,amd,he,,,',1)
    proposed=b''.join(changed)
    newrows=list(csv.reader(io.StringIO(proposed.decode(),newline='')))
    assert [r[10] for r in newrows]==[r[10] for r in rows]
    assert sum(a!=b for a,b in zip(records,changed))==1
    for i,(before,after) in enumerate(zip(rows,newrows)):
        assert before==after if i!=position else all(a==b for j,(a,b) in enumerate(zip(before,after)) if j!=2)
    stage=HERE/'stage';stage.mkdir(exist_ok=True)
    (stage/'corrected.csv').write_bytes(proposed)
    old_audit=audit_path.read_bytes().splitlines(keepends=True)
    audit=[json.loads(line) for line in old_audit]
    assert len(audit)==1859
    matches=[i for i,r in enumerate(audit) if r['source_unit_key']==KEY]
    assert len(matches)==1
    audit_index=matches[0];record=copy.deepcopy(audit[audit_index])
    assert record['source_record']['form']=='and' and record['entry_keys']==[KEY]
    record['historical_source_record']=copy.deepcopy(record['source_record'])
    record['source_record']['form']='amd'
    record['accepted_literal_addendum']={
        'field':'source_record.form','old':'and','accepted':'amd',
        'approval_reference':'source-printed-amd-addendum/root-acceptance.json',
        'evidence':'Isolated original p566 Kōlāmi amd: m has two shoulders; hypothetical avnd remains plain n.',
        'stable_entry_key':KEY,'all_other_lexical_fields_unchanged':True,
        'same_occurrence_new_chapter_witness':'grierson1906kolamibasim:p566:pronunciation-n-nd-d:1:independent:w10'}
    new_audit=list(old_audit)
    new_audit[audit_index]=(json.dumps(record,ensure_ascii=False)+'\n').encode()
    (stage/'accepted-audit.jsonl').write_bytes(b''.join(new_audit))
    assert sum(a!=b for a,b in zip(old_audit,new_audit))==1
    # Preserve the former root-pinned entrypoint byte-for-byte; the proposed
    # wrapper takes over regeneration after the new approved correction.
    historical=HERE/'historical-import_source.py'
    if historical.exists():assert sha(historical)==sha(PACKAGE/'import_source.py')
    else:historical.write_bytes((PACKAGE/'import_source.py').read_bytes())
    wrapper='''"""Guarded reviewed Kui source-stage importer; no database generation."""
import importlib.util
from pathlib import Path
import argparse

def main(install=False):
    path=Path(__file__).resolve().parent/'source-printed-amd-addendum/apply_correction.py'
    spec=importlib.util.spec_from_file_location('kui_printed_amd_correction',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.main(install=install)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true')
    main(parser.parse_args().install)
'''
    (stage/'import_source.py').write_text(wrapper)
    targets={
        'csv':dict(target=str(csv_path.relative_to(DATA)),input='stage/corrected.csv'),
        'audit':dict(target=str(audit_path.relative_to(DATA)),input='stage/accepted-audit.jsonl'),
        'importer':dict(target=str((PACKAGE/'import_source.py').relative_to(DATA)),input='stage/import_source.py')}
    for asset in targets.values():
        asset['baseline_sha256']=sha(DATA/asset['target']);asset['output_sha256']=sha(HERE/asset['input'])
    unchanged_paths=[DATA/'data/other/forms/20260927-grierson-kui.yaml',DATA/'conversion/grierson-kui-1906.txt',
        DATA/'cldf/sources.bib',DATA/'cldf/languages.csv',DATA/'cldf/dialects.csv']
    inputs=['prepare_plan.py','apply_correction.py','test_transaction.py','check_staged_parser.py','historical-import_source.py',*[a['input'] for a in targets.values()]]
    evidence=ROOT/'data/data/other/forms/raw_data/grierson_kolami_basim_1906/grammar-review/existing-kui-key-spelling-correction-proposal.json'
    plan=dict(installation_approved=False,result='disabled; root plan review required',
        rows=1832,audit_units=1859,entry_key=KEY,targets=targets,files={name:sha(HERE/name) for name in inputs},
        unchanged_at_initial_transaction={str(p.relative_to(DATA)):sha(p) for p in unchanged_paths},
        durable_identity_sha256=sha(DATA/'data/form-identities.csv'),
        root_historical_approval_sha256=sha(PACKAGE/'root-installation-approval.json'),
        canonical_field_changes=[dict(row=position+1,column=2,old='and',new='amd')],
        unchanged_other_rows=1831,all_keys_and_order_unchanged=True,
        source_evidence={'path':str(evidence.relative_to(ROOT)),'sha256':sha(evidence)},
        raw_first_readings_and_historical_candidate_unchanged=True,
        existing_entrypoint_archival_mapping={'import_source.py':'source-printed-amd-addendum/historical-import_source.py'},
        required_test_gates=['Exact1831rawrowbytepreservation','Actualparser1832NFC/NFDkey/Originalclosure',
            'Isolated interruptedwrite/recovery/rollback/output-drift/no-opfixtures'],
        forbidden=['DB/fullbuild','Unrelatedsettings/profile/Bib/registrychanges','Sourcekeyreplacement','Duplicate new chapter emission'])
    (HERE/'root-acceptance-draft.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))

if __name__=='__main__':main()

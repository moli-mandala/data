"""Read-only canonical baseline capture and disabled six-target asset bundle.

This does not accept an audit or activate installation. Refreshing the bundle after
root final audit acceptance is explicit; shared canonical files are never written.
"""
import csv,hashlib,io,json,sys
from pathlib import Path
from pybtex.database import parse_string
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
FINAL=PACKAGE/'integration/revision6'
STEM='20260927-grierson-kolami-basim'
TARGETS={'csv':f'data/other/forms/{STEM}.csv','yaml':f'data/other/forms/{STEM}.yaml',
 'audit':'data/other/forms/raw_data/grierson_kolami_basim_1906/installed-audit.jsonl',
 'profile':'conversion/grierson-kolami-basim-1906.txt','bib':'cldf/sources.bib','dialects':'cldf/dialects.csv'}
def sha(p):
    if not p.exists():return None
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assert not (HERE/'root-acceptance.json').exists(),'Never refresh activated assets'
    freeze=json.loads((FINAL/'review-freeze.json').read_text())
    for name,pin in freeze['files'].items():assert sha(PACKAGE/name)==pin,name
    baselines={name:sha(DATA/name) for name in TARGETS.values()}
    assert all(baselines[TARGETS[role]] is None for role in ['csv','yaml','audit','profile'])
    prior=(DATA/TARGETS['bib']).read_bytes();extra=(HERE/'staged-references.bib').read_bytes()
    merged=prior.rstrip()+b'\n\n'+extra
    before=parse_string(prior.decode(),'bibtex').entries;after=parse_string(merged.decode(),'bibtex').entries;added=parse_string(extra.decode(),'bibtex').entries
    assert not set(before)&set(added) and set(after)==set(before)|set(added)
    assert all(after[k]==v for k,v in before.items()) and len(added)==2
    (HERE/'baseline-bibliography.bib').write_bytes(prior);(HERE/'planned-bibliography.bib').write_bytes(merged)
    prior=(DATA/TARGETS['dialects']).read_bytes();added_bytes=(HERE/'staged-dialects.csv').read_bytes()
    (HERE/'baseline-dialects.csv').write_bytes(prior)
    original_reader=csv.DictReader(io.StringIO(prior.decode()));original=list(original_reader)
    added_reader=csv.DictReader(io.StringIO(added_bytes.decode()));added_dialects=list(added_reader)
    assert original_reader.fieldnames==added_reader.fieldnames and len(added_dialects)==2
    assert not {r['ID'] for r in original}&{r['ID'] for r in added_dialects}
    payload=io.StringIO(newline='');writer=csv.DictWriter(payload,fieldnames=original_reader.fieldnames);writer.writerows(added_dialects)
    merged=prior+(b'' if prior.endswith(b'\n') else b'\n')+payload.getvalue().encode()
    checked=list(csv.DictReader(io.StringIO(merged.decode())));assert checked==original+added_dialects
    (HERE/'planned-dialects.csv').write_bytes(merged)
    audit_ready=(HERE/'accepted-audit.jsonl').exists()
    audit_input='installation/accepted-audit.jsonl' if audit_ready else 'integration/revision6/combined-review-audit.jsonl'
    input_assets={'csv':'integration/revision6/combined-review-candidate.csv','yaml':f'installation/{STEM}.yaml',
      'audit':audit_input,'profile':'integration/revision6/grierson-kolami-basim-1906.txt','bib':'installation/planned-bibliography.bib','dialects':'installation/planned-dialects.csv'}
    files={}
    def include_frozen_dependencies(manifest):
        for name,pin in manifest['files'].items():
            assert sha(PACKAGE/name)==pin,('Frozen dependency drift',name)
            assert name not in files or files[name]==pin,('Conflicting historical pin',name)
            if name in files:continue
            files[name]=pin
            if name.endswith('/review-freeze.json'):
                include_frozen_dependencies(json.loads((PACKAGE/name).read_text()))
    include_frozen_dependencies(freeze)
    files['integration/revision6/review-freeze.json']=sha(FINAL/'review-freeze.json')
    for p in HERE.iterdir():
        if p.is_file() and p.name not in {'root-acceptance-draft.json','root-acceptance.json','installation-transaction.json','installation-receipt.json'}:
            files[str(p.relative_to(PACKAGE))]=sha(p)
    files[audit_input]=sha(PACKAGE/audit_input)
    if audit_ready:
        proof=PACKAGE/'root-independent-review/fresh-final-output-original-audit.json'
        assert sha(proof)=='2bc4a5c8be3ec9d127d04211ac6729f5eb2451bcba132bb7c61c46b56c737864'
        files[str(proof.relative_to(PACKAGE))]=sha(proof)
        addendum=PACKAGE/'root-independent-review/scholarly-notes-r6-acceptance.json'
        assert sha(addendum)=='6c7b7ee4ee605a2800080bf37ca801f967c01c4f5a6fade8d2134e894d722547'
        accepted=json.loads(addendum.read_text())
        assert accepted['candidate_sha256']==sha(FINAL/'combined-review-candidate.csv')
        assert accepted['audit_sha256']==sha(FINAL/'combined-review-audit.jsonl')
        assert accepted['historical_original_output_acceptance_sha256']==sha(proof)
        files[str(addendum.relative_to(PACKAGE))]=sha(addendum)
    root_account=json.loads((PACKAGE/'root-independent-review/whole-candidate-literal-accounting.json').read_text())
    reused=[]
    for witness in root_account['canonical_reuses']:
        with (DATA.parent/witness['canonical_path']).open() as stream:
            matches=[r for r in csv.reader(stream) if r[10]==witness['existing_key']]
        assert len(matches)==1
        reused.append({'canonical_path':str(Path(witness['canonical_path']).relative_to('data')),'entry_key':witness['existing_key'],'canonical_row':matches[0]})
    reuse_hashes={str(Path(p).relative_to('data')):pin for p,pin in root_account['canonical_input_sha256'].items()}
    assert all(sha(DATA/path)==pin for path,pin in reuse_hashes.items())
    images={}
    for base in ['specimen-conservation-review/sharper-selected-originals-manifest.json','grammar-native-review/selected-original-manifest.json']:
        # Both manifests are pinned inputs; retained originals are also read-only
        # workspace evidence and receive direct pins below.
        p=PACKAGE/base;files[base]=sha(p)
    for pattern in ['tmp/jambu-next-source/kolami-specimen-originals/alternate-leaf*-printed*.jpg','tmp/jambu-next-source/kolami-grammar-originals/alternate-leaf*-printed*.jpg']:
        for path in DATA.parent.glob(pattern):images[str(path.relative_to(DATA.parent))]=sha(path)
    images['tmp/pdfs/LSI-V4.djvu']=sha(DATA.parent/'tmp/pdfs/LSI-V4.djvu')
    draft={'installation_approved':False,'result':'pending_root_activation','shared_baseline_captured':True,'audit_acceptance_ready':audit_ready,
      'rows':827,'audit_units':1570,'files':files,'assets':{role:{'target':TARGETS[role],'input':name,'output_sha256':sha(PACKAGE/name)} for role,name in input_assets.items()},
      'canonical_baselines':baselines,'durable_identity_sha256':sha(DATA/'data/form-identities.csv'),
      'original_inputs':images,'canonical_reuse_inputs':reuse_hashes,'canonical_reuse_rows':reused,'reused_existing_keys':[r['entry_key'] for r in reused],
      'counts':{'prior_bibliography_entries':len(before),'added_bibliography_entries':2,'prior_dialect_rows':len(original),'added_dialect_rows':2},
      'pending':['Root final source-output acceptance and accepted audit derivative' if not audit_ready else 'Exact root activation','Real complete staged asset dry-run validation','Canonical actual-parser/sourceMeta/repeat checks','Full compiled/browser gates deferred']}
    (HERE/'root-acceptance-draft.json').write_text(json.dumps(draft,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'Disabled exact bundle captured; no canonical writes','draft_sha256':sha(HERE/'root-acceptance-draft.json'),'counts':draft['counts'],'audit_acceptance_ready':audit_ready,'files':len(files),'targets':len(TARGETS)}))
if __name__=='__main__':main()

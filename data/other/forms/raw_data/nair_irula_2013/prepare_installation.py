"""Prepare disabled four-target plan; never write canonical assets or approval."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import yaml
from pybtex.database import parse_string
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
OUT=HERE/'installation'
OUT.mkdir(exist_ok=True)
def file_hash(path):
    if not path.exists():return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((HERE/'review-manifest-r2.json').read_text())
assert all(file_hash(HERE/n)==v for n,v in manifest['sha256'].items())
review_path=HERE/'root-original-review-final-r2-20260927425.json'
review=json.loads(review_path.read_text())
assert review['material_errors']==0 and review['sample_count']==20
assert review['csv_sha256']==file_hash(HERE/'proposed.csv')
assert review['audit_sha256']==file_hash(HERE/'proposal-audit.jsonl')
assert file_hash(review_path)=='2995c6f1ecf096305e49e58cd8f2dc501d1e528a1cae045dcb0a27e835f563f1'
config=yaml.safe_load((HERE/'20260927-nair-irula.yaml').read_text())
config['sources']['nair2013irula']['reference']['etymology_provenance']='source'
config['defaults']['importer']={'commands':[['data/other/forms/raw_data/nair_irula_2013/installation/install_source.py','--install']], 'note':'Guarded exact source-stage install; repeat verifies own assets and bibliography memberships. No database build.'}
(OUT/'20260927-nair-irula.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
bib=(HERE/'proposed-reference.bib').read_text()
old='Source-local first reading, proposal and full audit in data/other/forms/raw_data/nair_irula_2013; same pinned 58-page PDF as separately cited Nair chapters, not a separate edition'
new='Canonical data/other/forms/20260927-nair-irula.csv; accepted source audit data/other/forms/raw_data/nair_irula_2013/installed-audit.jsonl; source-stage receipt installation/installation-receipt.json and frozen revision2 within that package. All 261 physical units accounted: 256 target readings, 16 audit-only units, 12 local variant pointers and four source-supported derived links. Same pinned 58-page PDF as separately cited Nair chapters, not a separate edition'
assert bib.count(old)==1
bib=bib.replace(old,new)
primary, auxiliary=bib.split('@misc',1)
assert primary.count('etymology_provenance = {none}')==1
bib=primary.replace('etymology_provenance = {none}','etymology_provenance = {source}')+'@misc'+auxiliary
original_refs=parse_string((HERE/'proposed-reference.bib').read_text(),'bibtex').entries
refs=parse_string(bib,'bibtex').entries
assert original_refs.keys()==refs.keys()
for key in refs:
    before=dict(original_refs[key].fields);after=dict(refs[key].fields)
    if key=='nair2013irula':
        before.pop('provenance');after.pop('provenance')
        assert before.pop('etymology_provenance')=='none' and after.pop('etymology_provenance')=='source'
    assert before==after and original_refs[key].persons==refs[key].persons
(OUT/'staged-references.bib').write_text(bib)
audit=[]
for a in map(json.loads,(HERE/'proposal-audit.jsonl').read_text().splitlines()):
    audit.append({**a,'historical_disposition':a['disposition'],'disposition':'accepted-target' if a['proposed_entry_keys'] else 'audit-only','accepted_for_installation':True,'acceptance_semantics':'The audit decision is accepted; audit-only units are not lexical rows. Frozen proposal history and raw source evidence remain unchanged.','review_evidence':{'path':review_path.name,'sha256':file_hash(review_path)}})
assert len(audit)==261 and sum(not a['proposed_entry_keys'] for a in audit)==16
(OUT/'accepted-audit-proposal.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
template=DATA/'data/other/forms/raw_data/nair_mannan_2013/installation/install_source.py'
engine=template.read_text().replace('Mannan','Irula').replace('mannan','irula').replace("== 154","== 256").replace("'rows': 154","'rows': 256").replace("'audit_units': 140","'audit_units': 261")
start=engine.index('def validate_assets(');end=engine.index('\ndef shared_membership()',start)
validator='''def validate_assets(approval, planned):
    rows = list(csv.reader(io.StringIO(planned['csv'].decode())))
    assert planned['csv'] == (PACKAGE / 'proposed.csv').read_bytes()
    assert len(rows) == approval['rows'] == 256
    assert all(len(r) == 15 and r[0] == 'Irula' and r[2] and r[10] for r in rows)
    keys = {r[10] for r in rows}
    assert len(keys) == 256
    assert sum(bool(r[11]) for r in rows) == 12 and sum(bool(r[13]) for r in rows) == 4
    for r in rows:
        assert not any(r[i] for i in (1, 4, 5, 8, 12))
        assert not r[11] or r[11] in keys
        assert not r[13] or all(k in keys for k in r[13].split('|'))
        assert unicodedata.is_normalized('NFC', r[2])
        assert all(unicodedata.category(c) != 'Co' and c != '\\ufffd' for c in r[2])
        assert r[7].startswith('nair2013irula[p. ')
    audit = list(map(json.loads, planned['audit'].decode().splitlines()))
    historical = list(map(json.loads, (PACKAGE / 'proposal-audit.jsonl').read_text().splitlines()))
    assert len(audit) == len({a['unit_key'] for a in audit}) == approval['audit_units'] == 261
    assert {k for a in audit for k in a['proposed_entry_keys']} == keys
    assert sum(not a['proposed_entry_keys'] for a in audit) == 16
    for a, old in zip(audit, historical):
        assert a['accepted_for_installation'] is True
        assert a['historical_disposition'] == old['disposition']
        assert a['disposition'] == ('accepted-target' if old['proposed_entry_keys'] else 'audit-only')
        assert all(a[k] == v for k, v in old.items() if k != 'disposition')
        assert a['review_evidence']['sha256'] == approval['final_review']['sha256']
    settings = yaml.safe_load(planned['yaml'])
    frozen = yaml.safe_load((PACKAGE / '20260927-nair-irula.yaml').read_text())
    expected = dict(settings['defaults']); importer = expected.pop('importer')
    frozen['sources']['nair2013irula']['reference']['etymology_provenance'] = 'source'
    assert expected == frozen['defaults'] and settings['sources'] == frozen['sources']
    assert settings['file'] == STEM + '.csv'
    assert importer['commands'] == [['data/other/forms/raw_data/nair_irula_2013/installation/install_source.py', '--install']]
    refs = parse_string((PACKAGE / 'installation/staged-references.bib').read_text(), 'bibtex').entries
    assert set(settings['sources']) == set(refs) == {'nair2013irula','zvelebil1982irulapart3','perialwar1978irula'}
    for key, source in settings['sources'].items():
        assert source['reference'] == {'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'source' if key == 'nair2013irula' else 'none'}
    assert set(SourceMeta([PACKAGE / 'installation/20260927-nair-irula.yaml']).sources) == set(refs)
    live_refs = parse_string(planned['bib'].decode(), 'bibtex').entries
    assert all(live_refs[k] == v for k, v in refs.items())
    _, languages = registry((DATA / 'cldf/languages.csv').read_bytes())
    assert languages['Irula'] == approval['existing_irula_identity']
    assert languages['Irula']['Glottocode'] == 'irul1243'

'''
engine=engine[:start]+validator+engine[end:]
(OUT/'install_source.py').write_text(engine)
live_bib=DATA/'cldf/sources.bib';old_entries=parse_string(live_bib.read_text(),'bibtex').entries
assert not set(old_entries)&set(refs),'source bibliography collision'
merged=live_bib.read_bytes()+b'\n'+bib.encode()
merged_entries=parse_string(merged.decode(),'bibtex').entries
assert all(merged_entries[k]==v for k,v in old_entries.items())
(OUT/'merged-sources.bib').write_bytes(merged)
spec=importlib.util.spec_from_file_location('irula_plan_engine',OUT/'install_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
names=list(manifest['sha256'])+['review-manifest-r2.json',review_path.name,'installation/install_source.py','installation/20260927-nair-irula.yaml','installation/staged-references.bib','installation/accepted-audit-proposal.jsonl','installation/merged-sources.bib','prepare_installation.py']
assets={role:{'target':target,'input':{'csv':'proposed.csv','yaml':'installation/20260927-nair-irula.yaml','audit':'installation/accepted-audit-proposal.jsonl','bib':'installation/merged-sources.bib'}[role]} for role,target in m.TARGETS.items()}
for asset in assets.values():asset['output_sha256']=file_hash(HERE/asset['input'])
languages={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
plan={'installation_approved':False,'result':'disabled pending root review','rows':256,'audit_units':261,'source_pdf_sha256':manifest['source_pdf_sha256'],'durable_identity_sha256':file_hash(DATA/'data/form-identities.csv'),'existing_irula_identity':languages['Irula'],'final_review':{'path':review_path.name,'sha256':file_hash(review_path)},'files':{n:file_hash(HERE/n) for n in names},'assets':assets,'canonical_baselines':{target:file_hash(DATA/target) for target in m.TARGETS.values()},'reused_transaction_engine':{'path':str(template.relative_to(DATA)),'sha256':file_hash(template)},'deferred_gates':['full CLDF/DB build','compiled identity and graph checks','full suite','browser QA']}
assert all(plan['canonical_baselines'][m.TARGETS[r]] is None for r in ['csv','yaml','audit'])
(OUT/'root-acceptance-draft.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(m.main(check_draft=True),indent=2))

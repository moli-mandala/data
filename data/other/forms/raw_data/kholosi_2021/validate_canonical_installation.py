"""Focused canonical source validation, without generating compiled tables."""
import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path
from pybtex.database import parse_file
P=Path(__file__).resolve().parent;D=P.parents[4]
os.chdir(D);sys.path.insert(0,str(D))
import make_cldf
import source_meta
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
path=D/'data/other/forms/20220913-kholosi.csv'
rows=list(csv.reader(path.open()));before=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
assert path.read_bytes()==(P/'source-stage-parent-delimiter-addendum/whole-proposed.csv').read_bytes()
assert (D/'conversion/kholosi-source.txt').read_bytes()==(P/'whole-proposal-r3/whole-profile-proposed.txt').read_bytes()
assert (path.with_suffix('.yaml')).read_bytes()==(P/'source-stage-importer-routing-addendum/whole-metadata-proposed.yaml').read_bytes()
errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(path),errors,name=path.stem)
assert not errors.getvalue(),errors.getvalue()
assert len(parsed)==len(rows)==1135==stats['converted']
for p,r in zip(parsed,rows):
 assert (p.entry_key,p.param,p.old_form,p.ipa,p.notes,p.source,p.etymology,p.variant_of_key,p.borrowed_from_key,p.derivation_parent_keys)==tuple(r[i] for i in (10,1,2,5,6,7,9,11,12,13))
 assert set(r[14].split())<=set(p.tags.split())
keys={r[10] for r in rows};assert len(keys)==1135
assert [r[10] for r in rows[:198]]==[r[10] for r in before]
assert all(not r[11] or r[11] in keys for r in rows)
assert all(k in keys and k!=r[10] for r in rows for k in [k for k in r[13].split('|') if k])
assert not {t for r in rows for t in r[14].split()}-(GRAMMATICAL_TAGS|GENDER_TAGS)
langs={r['ID']:r for r in csv.DictReader((D/'cldf/languages.csv').open())}
assert {r[0] for r in rows}=={'Khol'}
assert [langs['Khol'][k] for k in ('Name','Glottocode','Latitude','Longitude','Clade')]==['Kholosi','khol1241','27.12','54.30','Sindhic']
assert not any(t.startswith('dialect:') for r in rows for t in r[14].split())
meta=source_meta.load()
assert meta.source_owner['kholosi']==path.stem
assert meta.flag('kholosi','identity','dedupe_by_entry_key') is True
assert meta.flag('kholosi','forms','split_alternates') is False
assert meta.flag('kholosi','reference','etymology_provenance')=='mixed'
assert meta.flag('kholosi','reference','ocr') is False
assert source_meta.importer_commands(meta,path.stem)==[['data/other/forms/raw_data/kholosi_2021/install_reviewed_source.py','--install']]
entry=parse_file(str(D/'cldf/sources.bib')).entries['kholosi']
assert entry.fields['doi']=='10.5281/zenodo.7901554'
assert entry.fields['ocr']=='No' and entry.fields['etymology_provenance']=='mixed'
assert all(re.fullmatch(r'kholosi(?:\[[^\[\];]+\])?',c) for r in rows for c in r[7].split(';'))
legacykeys={r[10] for r in before}
identities=[r for r in csv.DictReader((D/'data/form-identities.csv').open()) if r['Source_Key'] in legacykeys]
assert {r['Source_Key'] for r in identities}==legacykeys
report=dict(status='passed canonical source-stage checks; compiled/app gates deferred',rows=1135,
 legacy_keys_preserved=198,new_keys=937,legacy_identity_records=identities,
 canonical_sha256={str(p.relative_to(D)):sha(p) for p in (path,path.with_suffix('.yaml'),D/'conversion/kholosi-source.txt')},
 registered_language=langs['Khol'],registered_source_owner=meta.source_owner['kholosi'],
 importer_registered=True,parser_errors=0,exact_parser_literal_and_graph_fields=True,
 borrowing_tag_rows=sum('loanword' in r[14].split() for r in rows),uncertain_rows=sum('uncertain' in r[14].split() for r in rows),
 blank_gloss_rows=sum(not r[3] for r in rows),ancestry_linked_rows=sum(bool(r[1]) for r in rows),
 variant_rows=sum(bool(r[11]) for r in rows),derivation_parent_links=sum(len([k for k in r[13].split('|') if k]) for r in rows),
 source_type_addenda=['dictionary','website/API snapshot','etymological/comparative'],
 bibliography_key='kholosi',reference_provenance='mixed',ocr=False,
 deferred=['Full CLDF/database build','Compiled durable identity/graph reconciliation','Full suite','Browser QA'])
(P/'canonical-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='legacy_identity_records'},ensure_ascii=False))

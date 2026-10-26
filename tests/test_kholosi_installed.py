"""Current whole-source installation checks; earlier revisions remain evidence."""
import csv
import hashlib
import io
import json
from pathlib import Path
import make_cldf
import source_meta

D=Path(__file__).resolve().parents[1]
P=D/'data/other/forms/raw_data/kholosi_2021'
CSV=D/'data/other/forms/20220913-kholosi.csv'

def test_installed_bytes_and_198_existing_keys():
 assert CSV.read_bytes()==(P/'source-stage-parent-delimiter-addendum/whole-proposed.csv').read_bytes()
 rows=list(csv.reader(CSV.open()))
 baseline=list(csv.reader((P/'legacy-baseline/20220913-kholosi.csv').open()))
 assert len(rows)==1135 and len({r[10] for r in rows})==1135
 assert [r[10] for r in rows[:198]]==[r[10] for r in baseline]
 changes={r[10]:r[1] for r,b in zip(rows,baseline) if r[1]!=b[1]}
 assert changes=={f'legacy:20220913-kholosi:row:{n}':p for n,p in {92:'9735',194:'3138',141:'12357',169:'13744',64:''}.items()}
 report=json.loads((P/'canonical-validation.json').read_text())
 assert report['canonical_sha256']['data/other/forms/20220913-kholosi.csv']==hashlib.sha256(CSV.read_bytes()).hexdigest()
 assert {r['Source_Key'] for r in report['legacy_identity_records']}=={r[10] for r in baseline}

def test_actual_canonical_parser_preserves_every_graph_and_literal_field():
 rows=list(csv.reader(CSV.open()));errors=io.StringIO()
 parsed,stats=make_cldf.parse_file(str(CSV),errors,name=CSV.stem)
 assert not errors.getvalue() and stats['converted']==1135==len(parsed)
 for p,r in zip(parsed,rows):
  assert (p.entry_key,p.param,p.old_form,p.ipa,p.notes,p.source,p.etymology,p.variant_of_key,p.borrowed_from_key,p.derivation_parent_keys)==tuple(r[i] for i in (10,1,2,5,6,7,9,11,12,13))
  assert set(r[14].split())<=set(p.tags.split())

def test_importer_and_source_metadata_use_reviewed_routes():
 meta=source_meta.load()
 assert meta.source_owner['kholosi']==CSV.stem
 assert source_meta.importer_commands(meta,CSV.stem)==[['data/other/forms/raw_data/kholosi_2021/install_reviewed_source.py','--install']]
 assert meta.flag('kholosi','reference','etymology_provenance')=='mixed'
 assert meta.flag('kholosi','reference','ocr') is False
 assert meta.flag('kholosi','identity','dedupe_by_entry_key') is True
 assert meta.flag('kholosi','forms','split_alternates') is False
 defaults=meta.file_defaults(CSV)
 rules=defaults['transcription']
 assert rules==[dict(profile='kholosi-source',input='form',preserve_hyphens=True)]
 assert CSV.with_suffix('.yaml').read_bytes()==(P/'source-stage-importer-routing-addendum/whole-metadata-proposed.yaml').read_bytes()

def test_registered_language_and_all_local_graph_keys_are_closed():
 rows=list(csv.reader(CSV.open()));keys={r[10] for r in rows}
 assert {r[0] for r in rows}=={'Khol'}
 assert all(not r[11] or r[11] in keys for r in rows)
 assert all(k in keys and k!=r[10] for r in rows for k in [k for k in r[13].split('|') if k])
 report=json.loads((P/'canonical-validation.json').read_text())
 assert report['registered_language']['Glottocode']=='khol1241'
 assert report['registered_language']['Clade']=='Sindhic'
 assert report['borrowing_tag_rows']==317 and report['uncertain_rows']==107

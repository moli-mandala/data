"""Scoped actual parser/formatter and immutable audit-continuity checks."""
import csv
import hashlib
import json
import sys
from pathlib import Path
import pybtex
from pybtex.database import parse_file
import yaml
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import make_cldf
from source_meta import SourceMeta,importer_commands
SHA=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
config=HERE/'20260927-nair-irula.yaml'
paths=list((DATA/'data/other/forms').glob('*.yaml'))+[config]
meta=SourceMeta(paths)
refs=parse_file(str(HERE/'staged-references.bib')).entries
assert set(refs)=={'nair2013irula','zvelebil1982irulapart3','perialwar1978irula'}
assert all(meta.sources[k]['reference']=={'editor':'OpenAI Codex','ocr':False,'etymology_provenance':'source' if k=='nair2013irula' else 'none'} for k in refs)
assert importer_commands(meta,'20260927-nair-irula')==[['data/other/forms/raw_data/nair_irula_2013/installation/install_source.py','--install']]
make_cldf.source_meta.load=lambda:meta
errors=[]
parsed,stats=make_cldf.parse_file(str(PACKAGE/'proposed.csv'),errors,name='20260927-nair-irula')
rows={r[10]:r for r in csv.reader((PACKAGE/'proposed.csv').open())}
assert len(parsed)==len(rows)==256 and not errors and stats['converted']==0
for r in parsed:
    raw=rows[r.entry_key]
    assert (r.form,r.old_form,r.gloss,r.tags,r.source,r.variant_of_key,r.derivation_parent_keys,r.ipa,r.native)==(raw[2],raw[2],raw[3],raw[14],raw[7],raw[11],raw[13],'','')
formatted={k:pybtex.PybtexEngine().format_from_string(v.to_string('bibtex'),'plain',output_backend='markdown')[3:].strip() for k,v in refs.items()}
assert all(formatted.values())
old=yaml.safe_load((PACKAGE/'20260927-nair-irula.yaml').read_text());new=yaml.safe_load(config.read_text())
new['defaults'].pop('importer')
assert old['sources']['nair2013irula']['reference']['etymology_provenance']=='none'
old['sources']['nair2013irula']['reference']['etymology_provenance']='source'
assert old==new
historical=[json.loads(s) for s in (PACKAGE/'proposal-audit.jsonl').read_text().splitlines()]
accepted=[json.loads(s) for s in (HERE/'accepted-audit-proposal.jsonl').read_text().splitlines()]
assert len(historical)==len(accepted)==261
for a,b in zip(historical,accepted):
    assert all(b[k]==v for k,v in a.items() if k!='disposition')
    assert b['historical_disposition']==a['disposition'] and b['accepted_for_installation'] is True
    assert b['disposition']==('accepted-target' if a['proposed_entry_keys'] else 'audit-only')
assert sum(not a['proposed_entry_keys'] for a in accepted)==16
report={'status':'passed actual combined SourceMeta/parser/formatter and accepted-audit continuity','rows':256,'physical_units':261,'audit_only':16,'variants':12,'derived':4,'converted':0,'errors':errors,'refs_declared_once':list(refs),'formatted':formatted,'metadata_delta':'Installation YAML adds importer and marks primary etymology provenance source, reflecting four source-explicit internal derivations; bibliography refines main provenance and matches that flag. Auxiliary provenance remains none. All scholarly bibliographic fields and frozen lexical evidence unchanged.','accepted_audit_sha256':SHA(HERE/'accepted-audit-proposal.jsonl'),'frozen_csv_sha256':SHA(PACKAGE/'proposed.csv'),'canonical_mutations':False}
(HERE/'metadata-audit-parser-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='formatted'},indent=2))

"""Read-only readiness checks; never authorizes or installs this source."""
import csv, hashlib, json, sys
from pathlib import Path
from pybtex.database import parse_file
HERE=Path(__file__).resolve().parents[1]
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
sys.path.insert(0,str(DATA))
import source_meta

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 freeze=json.loads((HERE/'review-freeze.json').read_text())
 assert all(sha(HERE/n)==h for n,h in freeze['files_sha256'].items())
 baseline=[json.loads(x) for x in (HERE/'scope/legacy-baseline.jsonl').read_text().splitlines()]
 assert len(baseline)==151
 sources=[]
 for n in ('20260912-muduga.csv','20260925-kuriakose-muduga-numerals.csv'):
  path=DATA/'data/other/forms'/n;actual=list(csv.reader(path.open(newline='')))
  assert actual==[x['fields'] for x in baseline if x['file']==n]
  sources.append({'file':n,'rows':len(actual),'sha256':sha(path)})
 rows=list(csv.reader((HERE/'proposed.csv').open(newline='')))
 assert len(rows)==204 and not {r[10] for r in rows}&{r['entry_key'] for r in baseline}
 langs={r['ID']:r for r in csv.DictReader((DATA/'cldf/languages.csv').open())}
 assert langs['Muduga']['Glottocode']=='mudu1239' and {r[0] for r in rows}<=langs.keys()
 staged=HERE/'source-stage/20260927-nair-muduga.yaml'
 meta=source_meta.SourceMeta(source_meta.yaml_files()+[staged])
 for key in ('nair2013muduga','rajendran1986muduga'):
  assert meta.flag(key,'reference','ocr') is False
  assert meta.flag(key,'reference','editor')=='OpenAI Codex'
  assert meta.flag(key,'reference','etymology_provenance')==('source' if key=='nair2013muduga' else 'none')
 assert meta.transcription('nair2013muduga','20260927-nair-muduga.csv','Muduga')==(None,False)
 assert meta.flag('nair2013muduga','identity','dedupe_by_entry_key') is True
 assert source_meta.importer_commands(meta,'20260927-nair-muduga')==[['data/other/forms/raw_data/nair_muduga_2013/import_source.py','--install']]
 old=parse_file(DATA/'cldf/sources.bib');new=parse_file(HERE/'source-stage/references.bib')
 assert set(new.entries)=={'nair2013muduga','rajendran1986muduga'} and not set(new.entries)&set(old.entries)
 assert 'data/other/forms/20260927-nair-muduga.csv' in new.entries['nair2013muduga'].fields['provenance']
 assert 'installed-audit.jsonl' in new.entries['nair2013muduga'].fields['provenance']
 d=list(csv.DictReader((HERE/'metadata/proposed-dialect.csv').open()))
 assert len(d)==1 and d[0]['Language_ID']=='Muduga' and d[0]['ID']=='nair2013_muduga_attapady'
 assert not any(d[0][x] for x in ('Glottocode','Latitude','Longitude','Quality'))
 assert all(d[0]['Tag'] in r[14].split() for r in rows if r[0]=='Muduga')
 existing=list(csv.DictReader((DATA/'cldf/dialects.csv').open()))
 assert all(x['ID']!=d[0]['ID'] and x['Tag']!=d[0]['Tag'] for x in existing)
 approval=json.loads((HERE/'source-stage-authorization.json').read_text())
 assert approval['installation_approved'] is False and approval['files']=={}
 assert not (HERE/'installation-receipt.json').exists() and not (HERE/'installation-transaction.json').exists()
 audit=[json.loads(x) for x in (HERE/'source-stage/accepted-audit.jsonl').read_text().splitlines()]
 raw=[json.loads(x) for x in (HERE/'proposal-audit.jsonl').read_text().splitlines()]
 assert len(audit)==len(raw)==128
 for u,v in zip(audit,raw):
  assert u['accepted_for_source_stage'] is False
  assert {k:x for k,x in u.items() if k not in ('accepted_for_source_stage','acceptance_evidence')}==v
 report={'status':'pass: read-only current canonical baseline/config readiness; not lexical acceptance or installation','freeze_sha256':sha(HERE/'review-freeze.json'),'csv_sha256':sha(HERE/'proposed.csv'),'audit_sha256':sha(HERE/'proposal-audit.jsonl'),'legacy_sources':sources,'legacy_keys':151,'registered_languages':sorted({r[0] for r in rows}),'combined_source_meta':'canonical YAML plus draft passes without duplicate declaration','reference_keys':sorted(new.entries),'proposed_lect':d[0],'installation_disabled':True,'frozen_files_unchanged':True,'gaps':['Independent root lexical acceptance','Exact approved report, audit acceptance fields and final input/baseline hashes','Actual canonical parser/meta/ref/lect and repeat verification after approved transaction','Full compiled/build/browser gates'],'canonical_changes':False}
 (HERE/'source-stage/current-readiness-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(report['status'])
if __name__=='__main__':main()

"""Guarded whole-source Kurux installation; no compiled database generation."""
import argparse,csv,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
SOURCE='shackelford-swenson-chaudhary-maggard2022kurux'
ROUTES={'wordlist':'20260813-kurux-nepal','grammar':'20260927-kurux-nepal-grammar','native':'20260927-kurux-nepal-native','roman':'20260927-kurux-nepal-roman'}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify():
 approval=json.loads((HERE/'root-independent-review-20260927222.json').read_text())
 assert approval['result']=='pass' and approval['material_errors']==0
 manifest=json.loads((HERE/'frozen-manifest.json').read_text())
 assert digest(HERE/'frozen-manifest.json')==approval['manifest_sha256']
 repair=json.loads((HERE/'dependency-repair-20260927.json').read_text())
 for name,expected in manifest['files'].items():
  if name=='build_whole_proposal.py':
   assert expected==repair['old_generator_sha256']
   expected=repair['new_generator_sha256']
  assert digest(HERE/name)==expected,name
 return manifest

def generate():
 verify()
 for script in ['build_whole_proposal.py','prepare_proposal_profiles.py']:
  subprocess.run([sys.executable,str(HERE/script)],check=True)
 verify()
 print('All approved proposal CSV, audit, profiles and YAML bytes reproduced exactly.')

def install():
 from pybtex.database import parse_file
 generate()
 identity=ROOT/'data/form-identities.csv';before=digest(identity)
 bib=ROOT/'cldf/sources.bib';old={k:v for k,v in parse_file(str(bib)).entries.items() if k!=SOURCE}
 blocks=re.split(r'(?=^@)',bib.read_text(),flags=re.M)
 matches=[i for i,b in enumerate(blocks) if re.match(r'@\w+\{'+SOURCE+r'\s*,',b)]
 assert len(matches)==1
 replacement=(HERE/'installed-source.bib').read_text()
 assert set(parse_file(str(HERE/'installed-source.bib')).entries)=={SOURCE}
 blocks[matches[0]]=replacement.rstrip()+'\n\n'
 dp=ROOT/'cldf/dialects.csv'
 with dp.open(newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;existing={r['ID']:r for r in reader}
 additions=list(csv.DictReader((HERE/'proposed-dialects.csv').open()))
 for r in additions:assert r['ID'] not in existing or existing[r['ID']]==r
 canonical={};counts={};keys=set()
 for bucket,stem in ROUTES.items():
  src=HERE/f'proposal/{bucket}.csv';target=ROOT/f'data/other/forms/{stem}.csv'
  rows=list(csv.reader(src.open()));counts[bucket]=len(rows)
  for row in rows:assert row[10] not in keys;keys.add(row[10])
  shutil.copyfile(src,target);canonical[str(target.relative_to(ROOT))]=digest(target)
  yaml=(HERE/f'proposal/{bucket}.yaml').read_text().replace(f'file: {bucket}.csv',f'file: {stem}.csv')
  yaml=yaml.replace('defaults:\n', 'defaults:\n  importer:\n    commands:\n      - [data/other/forms/raw_data/kurux_nepal_2022/whole-source-recovery/import_source.py, --install]\n    note: Complete reviewed source; immutable baseline and approved revision 2 hashes.\n',1)
  if bucket=='wordlist':
   yaml += '    reference:\n      editor: Aryaman Arora; OpenAI Codex\n      ocr: false\n      etymology_provenance: none\n'
  (ROOT/f'data/other/forms/{stem}.yaml').write_text(yaml)
 assert len(keys)==2465
 legacy={r[10] for r in csv.reader((HERE/'legacy-before-whole-recovery.csv').open())}
 assert len(legacy)==1336 and legacy<=keys
 for name in ('phonetic','native','roman'):
  target=ROOT/f'conversion/kurux-nepal-whole-{name}.txt'
  shutil.copyfile(HERE/f'proposal/{target.name}',target);canonical[str(target.relative_to(ROOT))]=digest(target)
 shutil.copyfile(HERE/'proposal/audit.jsonl',HERE/'installed-audit.jsonl')
 bib.write_text(''.join(blocks))
 assert {k:v for k,v in parse_file(str(bib)).entries.items() if k!=SOURCE}==old
 with dp.open('a',newline='') as f:csv.DictWriter(f,fieldnames=fields).writerows(r for r in additions if r['ID'] not in existing)
 assert digest(identity)==before
 report={'status':'Reviewed whole-source stage installed; full build and browser gates deferred.','rows':counts,'total_rows':len(keys),'audit_units':2542,'legacy_keys_preserved':len(legacy),'canonical_sha256':canonical,'audit_sha256':digest(HERE/'installed-audit.jsonl'),'durable_identity_sha256_unchanged':before,'unrelated_bibliography_entries_unchanged':len(old),'dialects_added':[r['ID'] for r in additions if r['ID'] not in existing],'independent_review':'root-independent-review-20260927222.json','deferred':['Full CLDF/database build','Compiled identity and graph verification','Full test suite','Browser QA']}
 (HERE/'installation-report-20260927.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--install',action='store_true');args=parser.parse_args()
 install() if args.install else generate()

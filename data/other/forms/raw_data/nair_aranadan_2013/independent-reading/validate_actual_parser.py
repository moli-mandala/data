from pathlib import Path
import csv,io,json,hashlib,sys,shutil,unicodedata as ud
D=Path(__file__).resolve().parent;P=D.parent;ROOT=D.parents[5];sys.path.insert(0,str(ROOT))
import source_meta,make_cldf
from tags import GRAMMATICAL_TAGS,GENDER_TAGS
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
I=P/'integration';paths=[I/'review-candidate.csv',I/'review-audit.jsonl',I/'review-status.json',P/'root-review/proposed-settings.yaml'];pins={str(p.relative_to(P)):sha(p) for p in paths}
rows=list(csv.reader(paths[0].open()));assert len(rows)==102,len(rows);assert all(len(r)==15 for r in rows);keys={r[10] for r in rows};assert len(keys)==102
languages={r['ID'] for r in csv.DictReader((ROOT/'cldf/languages.csv').open())};allowed=GRAMMATICAL_TAGS|GENDER_TAGS
for r in rows:
 assert r[0] in languages
 assert set(r[14].split())<=allowed,(r[10],set(r[14].split())-allowed)
 assert not any(ud.category(c)=='Co' or c=='\ufffd' for v in r for c in v)
 assert r[7].split('[',1)[0]=='nair2013aranadan'
 for i in (11,12,13):assert all(k in keys for k in r[i].split('|') if k)
assert sum(bool(r[11]) for r in rows)==2
F=D/'parser-fixture/data/other/forms';F.mkdir(parents=True,exist_ok=True);stem='20260927-nair-aranadan';yp=F/(stem+'.yaml');shutil.copyfile(paths[3],yp)
source_meta.yaml_files=lambda:[yp];source_meta.load.cache_clear()
fields=('lang','param','old_form','gloss','native','ipa','notes','source','cognateset','etymology','entry_key','variant_of_key','borrowed_from_key','derivation_parent_keys','tags');checks={}
for normalization in ('NFC','NFD'):
 inputs=[[ud.normalize(normalization,v) if i==2 else v for i,v in enumerate(r)] for r in rows];cp=F/(stem+'.csv')
 with cp.open('w',newline='') as f:csv.writer(f).writerows(inputs)
 errors=io.StringIO();parsed,stats=make_cldf.parse_file(str(cp),errors,name=stem)
 assert not errors.getvalue(),errors.getvalue();assert len(parsed)==102
 assert stats=={'converted':0,'for_conversion':0},stats
 for a,e in zip(parsed,inputs):
  assert [getattr(a,f) for f in fields]==e,(a.entry_key,'source-field drift')
  assert ud.normalize('NFC',a.form)==ud.normalize('NFC',e[2]),(a.entry_key,a.form,e[2])
 forms={ud.normalize('NFC',a.form) for a in parsed}
 assert 'ñãã' in forms and any('n\u0332' in f for f in forms) and any('l\u0332' in f for f in forms)
 assert any('aa' in f for f in forms) and any(f.startswith('-') or f.startswith('–') for f in forms)
 checks[normalization]={'rows':102,'all15fields_exact':True,'display_exact_normalized_equivalence':True,'errors':0,'ordered_keys':True}
assert pins=={str(p.relative_to(P)):sha(p) for p in paths},'Input changed during validation'
report={'status':'passed isolated actual parser; no canonical writes','pins':pins,'parser_sha256':sha(ROOT/'make_cldf.py'),'checks':checks,'variant_links_closed':2,'languages':sorted({r[0] for r in rows}),'no_pua_or_replacement':True,'source_mark_controls':['n/l combining lowline','ñãã three tilde marks','doubled vowels','source hyphen/dash boundaries'],'canonical_changes':False}
(D/'actual-parser-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(report['status'])

import csv, gzip, hashlib, importlib.util, json, random, re, sys, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT.parent/'tmp/berger-audit-20260914'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('berger_audit_importer',ROOT/'data/other/forms/raw_data/berger_cleanup.py')
b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
def read_csv(p): return list(csv.reader(p.open()))
def read_dict(p): return list(csv.DictReader(p.open()))
auto=read_csv(b.AUTO_OUTPUT); gold=read_csv(b.GOLD_OUTPUT); installed=auto+gold; bykey={r[10]:r for r in installed}
with gzip.open(b.AUDIT_OUTPUT,'rt') as f: audit=list(csv.DictReader(f))
assert b.IDENTITY_MAP.exists() and b.LEGACY_INDEX.exists()
tracked=[b.AUTO_OUTPUT,b.GOLD_OUTPUT,b.EDITORIAL,b.IDENTITY_MAP,b.AUDIT_OUTPUT,b.SAMPLE_OUTPUT,b.MANIFEST_OUTPUT,b.LEGACY_INDEX,ROOT/'conversion/berger.txt']
hashes={str(p.relative_to(ROOT)):b.sha256_path(p) for p in tracked}
print('Reconstructing cached source in memory only.',flush=True)
pages=b.load_pages(b.CACHE_DIR);units=b.reconstruct_units(pages);entries=b.parse_entries(units,b.legacy.load_valid_ids(ROOT/'data/cdial/params.csv'))
b.apply_identity_map(entries);b.apply_editorial(entries);regen_gold=b.align_gold(entries)
regen=b.import_rows(entries);preserved=b.catalog_preservation_rows(regen,regen_gold);regen+=preserved
regenby={r[10]:r for r in regen+regen_gold}
missing=sorted(bykey.keys()-regenby.keys());added=sorted(regenby.keys()-bykey.keys())
changed=[{'key':k,'columns':[i for i in range(15) if bykey[k][i]!=regenby[k][i]],'before':bykey[k],'after':regenby[k]} for k in sorted(bykey.keys()&regenby.keys()) if bykey[k]!=regenby[k]]
missing_audit=[a for a in audit if a['Status']!='excluded' and a['Installed_Key'] not in bykey]
auditkeys={k for a in audit for k in a['Emitted_Keys'].split('|') if k}
uncovered=[r for r in installed if r[10] not in auditkeys and r[10] not in {a['Installed_Key'] for a in audit if a['Status']=='gold'}]
maprows=read_dict(b.IDENTITY_MAP)
keydrift=[{'audit':a,'matches':[r for r in installed if r[0]==('Werch' if 'dialect:Yasin' in a['Tags'].split() else 'Bur') and r[2]==a['Final_Form'] and r[3]==a['English_Gloss']]} for a in missing_audit]
links=[(r[10],kind,k) for r in installed for col,kind in [(11,'variant'),(12,'borrowed'),(13,'derived')] for k in r[col].split('|') if k]
badlinks=[x for x in links if x[2] not in bykey]
cycles=[];parents={child:parent for child,kind,parent in links if kind=='variant'}
for key in parents:
 seen=set();cur=key
 while cur in parents:
  if cur in seen: cycles.append(key);break
  seen.add(cur);cur=parents[cur]
catalog=read_dict(ROOT/'data/burushaski_cognates.csv')
needed={k for a in catalog for k in a['Evidence_Keys'].split('|') if k.startswith(('berger-','berger:'))}
editorial=b.load_editorial(); audit_bykey={a['Installed_Key']:a for a in audit}
rowedits=[{'key':r[10],'installed_gloss':r[3],'editorial_gloss':editorial[r[10]]['English_Gloss']} for r in installed if r[10] in editorial and r[3]!=b.postprocess_translation(editorial[r[10]]['English_Gloss'])]
# Source-side labels are screening counts, not proof of the form's grammatical scope.
class_candidates=[a for a in audit if a['Status']!='excluded' and re.search(r'(?<!\w)(?:h|hm|hf|x|y)(?!\w)',a['Raw_OCR'][:180])]
plural_candidates=[a for a in audit if a['Status']!='excluded' and re.search(r'\b(?:pl\.|D\.pl\.)',a['Raw_OCR'][:350],re.I)]
summary={
'installed':{'auto':len(auto),'gold':len(gold),'total':len(installed),'languages':dict(Counter(r[0] for r in installed)),'widths':dict(Counter(len(r) for r in installed)),'duplicate_keys':len(installed)-len(bykey),'blank_glosses':sum(not r[3] for r in installed),'grammar_notes':sum('grammar' in r[6].lower() for r in installed),'nonempty_notes':sum(bool(r[6]) for r in installed),'tags':dict(Counter(t for r in installed for t in r[14].split())),'class_tagged_rows':sum('Burushaski-class-' in r[14] for r in installed),'nfc_failures':sum(any(not unicodedata.is_normalized('NFC',x) for x in r) for r in installed)},
'audit':{'rows':len(audit),'status':dict(Counter(a['Status'] for a in audit)),'review':dict(Counter(reason for a in audit for reason in a['Review'].split(';'))),'emitted_keys_missing_current':sorted(auditkeys-bykey.keys()),'installed_not_covered':len(uncovered),'class_label_candidates':len(class_candidates),'plural_label_candidates':len(plural_candidates)},
'regeneration':{'units':len(units),'entries':len(entries),'auto_rows':len(regen),'gold_rows':len(regen_gold),'preserved_rows':len(preserved),'missing_keys':missing,'added_keys':added,'changed_shared_rows':len(changed),'changed_columns':dict(Counter(i for x in changed for i in x['columns'])),'entry_review':dict(Counter(reason for e in entries for reason in e.review))},
'graph':{'relation_counts':dict(Counter(x[1] for x in links)),'missing_endpoints':badlinks,'variant_cycles':cycles,'catalog_evidence_keys':len(needed),'missing_catalog_evidence_keys':sorted(needed-bykey.keys())},
'editorial_gloss_differences':len(rowedits),'input_sha256':hashes}
for p in tracked: assert b.sha256_path(p)==hashes[str(p.relative_to(ROOT))]
# Fresh unreviewed installed primary articles, distinct physical units.
base=[];seen=set()
for a in audit:
 if a['Status']=='installed' and a['Stable_Key'] not in seen and a['Installed_Key'] in bykey:
  seen.add(a['Stable_Key']);base.append(a)
samples=random.Random(20260914).sample(sorted(base,key=lambda a:a['Stable_Key']),20)
(OUT/'sample.json').write_text(json.dumps([{'audit':a,'installed':bykey[a['Installed_Key']]} for a in samples],ensure_ascii=False,indent=2)+'\n')
(OUT/'details.json').write_text(json.dumps(dict(key_drift=keydrift,regeneration_changes=changed,uncovered=uncovered,editorial_differences=rowedits,class_candidates=class_candidates,plural_candidates=plural_candidates),ensure_ascii=False,indent=2)+'\n')
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(OUT/'regenerated-auto.csv').write_text('')
with (OUT/'regenerated-auto.csv').open('w',newline='') as f: csv.writer(f).writerows(regen)
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

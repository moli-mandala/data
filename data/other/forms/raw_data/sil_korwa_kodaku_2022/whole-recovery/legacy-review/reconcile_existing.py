from pathlib import Path
import csv,json,hashlib,collections
D=Path(__file__).resolve().parent;P=D.parents[1];ROOT=D.parents[6]
C=ROOT/'data/other/forms/20260828-sil-korwa-kodaku.csv';A=ROOT/'data/other/forms/raw_data/20260828-sil-korwa-kodaku-audit.csv';M=P/'manual_review.tsv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.reader(C.open()));audit=list(csv.DictReader(A.open()));assert len(rows)==4458;keyed={r[10]:r for r in rows};assert len(keyed)==len(rows)
keys=[k for r in audit for k in r['Entry_Keys'].split('|') if k];assert len(keys)==len(set(keys))==4458 and set(keys)==set(keyed)
errors=[]
for a in audit:
 for k in a['Entry_Keys'].split('|'):
  if not k:continue
  r=keyed[k]
  if (r[0],r[3],r[7])!=(a['Language_ID'],a['Gloss'],a['Citation']):errors.append(k)
  assert r[2]==r[5] # Existing source phonetic layer, preserve during recovery.
assert not errors
controls={}
for code in 'ABJTUVk':
 aa=[r for r in audit if r['Site_Code']==code]
 controls[code]={'site':aa[0]['Site'],'audit_records':len(aa),'prompts':len({r['Item'] for r in aa}),'nonblank_records':sum(bool(r['Manual_Form']) for r in aa),'blank_items':[r['Item'] for r in aa if not r['Manual_Form']],'statuses':dict(collections.Counter(r['Status'] for r in aa))}
assert controls['J']['audit_records']==211 and controls['J']['nonblank_records']==209
snapshot=[{'entry_key':r[10],'row':r,'row_sha256':hashlib.sha256(json.dumps(r,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()} for r in rows]
(D/'canonical-row-baseline.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in snapshot))
(D/'comparison-exclusions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit if r['Site_Code'] in controls))
report={'status':'read-only complete installed-key and exclusion reconciliation; not original transcription acceptance','pins':{str(p.relative_to(ROOT)):sha(p) for p in (C,A,M,P/'import_korwa_kodaku.py',P/'unresolved_source_codes.tsv')},'rows':4458,'unique_keys':4458,'languages':dict(collections.Counter(r[0] for r in rows)),'audit_records':len(audit),'audit_statuses':dict(collections.Counter(r['Status'] for r in audit)),'all_installed_keys_mapped_exactly_once':True,'source_language_gloss_citation_equal':True,'original_equals_existing_phonemic':True,'controls':controls,'multioutput_audit_records':[r for r in audit if '|' in r['Entry_Keys']],'undefined_source_codes':list(csv.DictReader((P/'unresolved_source_codes.tsv').open(),delimiter='\t')),'source_stage_gaps':['Importer and bibliography cover AppendixB.5 only; full115pagecensus owned separately.','Seven comparison lists excluded on scope grounds, not poor transcription. BirjiaJ211records=210prompts,209nonblankrecords,blank23/24.','Four complete interlinear narratives and printed target questions are not represented by existing AppendixB.5keys.','Existing tags are site/dialect only; source-explicit grammar in paired prompts requires complete review rather than assuming baseline metadata complete.','Existing per-row note manually verified is process metadata; separate scholarly similarity-group information from procedural claims during recovery, preserving historicalaudit.','Unknown codesu/n remain unresolved attribution; do not assign by similarity.','No Birjia registry registration or acceptednewtranscription performed.'],'canonical_writes':False,'baseline_file':'canonical-row-baseline.jsonl','exclusion_file':'comparison-exclusions.jsonl'}
(D/'reconciliation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('4458keys and allseven comparisonlists reconciled')

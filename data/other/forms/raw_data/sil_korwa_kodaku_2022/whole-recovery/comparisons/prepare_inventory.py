from pathlib import Path
import csv,json,hashlib,collections,unicodedata
D=Path(__file__).resolve().parent;P=D.parents[1];ROOT=D.parents[6]
A=ROOT/'data/other/forms/raw_data/20260828-sil-korwa-kodaku-audit.csv';C=ROOT/'data/other/forms/20260828-sil-korwa-kodaku.csv';M=P/'manual_review.tsv'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
baseline=json.loads((P/'whole-recovery/legacy-review/reconciliation.json').read_text());assert sha(C)==baseline['pins'][str(C.relative_to(ROOT))]
manual=list(csv.DictReader(M.open(),delimiter='\t'));audit=list(csv.DictReader(A.open()));selected=[];occurrences=[];counters=collections.Counter()
for n,a in enumerate(audit,1):
 code=a['Site_Code']
 if code not in 'ABJTUVk':continue
 item=int(a['Item']);counters[(code,item)]+=1;ordinal=counters[(code,item)]
 key=f'behera2022korwakodaku:comparison:item{item:03d}:site{code}:response{ordinal}'
 matches=[m for m in manual if int(m['Item'])==item and code in m['Site_Codes'] and unicodedata.normalize('NFC',m['Manual_Form'])==a['Manual_Form']] if a['Manual_Form'] else []
 assert bool(matches)==bool(a['Manual_Form']),(key,a)
 physical=[]
 for m in matches:
  pk=f"behera2022korwakodaku:p{m['Printed_Page']}:wordlist-line{m['Line']}:site{code}"
  physical.append(pk);occurrences.append({'physical_key':pk,'response_key':key,'source_record':m})
 selected.append({'source_unit_key':key,'historical_audit_row_one_based':n,'historical_audit_record':a,'source_code':code,'source_label':a['Site'],'item':item,'form':a['Manual_Form'],'gloss':a['Gloss'],'physical_keys':physical,'printed_pages':a['Printed_Page'],'status':'inherited-first-reading pending complete original review' if matches else 'source-blank pending original confirmation','language_id':None,'identity_policy':'No new base/dialect assignment; independent identity review required, Tanmai mixed attribution unresolved.','reviewed_against_original':False})
assert len(selected)==1628 and sum(bool(r['form']) for r in selected)==1611
assert len({r['physical_key'] for r in occurrences})==len(occurrences)
for filename,rr in [('comparison-first-inventory.jsonl',selected),('comparison-physical-occurrences.jsonl',occurrences)]:
 (D/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rr))
report={'status':'complete structured inventory; original review pending, no accepted transcriptions','pins':{str(p.relative_to(ROOT)):sha(p) for p in (A,C,M)},'responses':1628,'nonblank_responses':1611,'blank_cells':17,'conceptual_cells':1470,'attested_conceptual_cells':1453,'physical_site_occurrences':len(occurrences),'pages':dict(collections.Counter(r['source_record']['PDF_Page'] for r in occurrences)),'all_codes':list('ABJTUVk'),'legacy_rows_keys_preserved':4458,'canonical_changes':False}
(D/'inventory-status.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

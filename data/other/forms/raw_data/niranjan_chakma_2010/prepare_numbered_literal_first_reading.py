import json,re,hashlib
from pathlib import Path
p=Path('data/data/other/forms/raw_data/niranjan_chakma_2010'); groups={}; count=0
for f in sorted((p/'numbered-literal-first-readings').glob('p*.txt')):
 current=None
 for n,line in enumerate(f.read_text().splitlines(),1):
  if line.startswith('@'):
   g,page,col=line[1:].split('|'); current={'pdf_page':int(page),'printed_page':int(page)-2,'column':col,'literal_lines':[],'manual_file':str(f.relative_to(p)),'manual_line_start':n+1}; groups.setdefault(int(g),[]).append(current)
  else:current['literal_lines'].append(line);count+=1
raw={r['printed_group']:r for r in map(json.loads,(p/'numbered-group-raw-inventory.jsonl').read_text().splitlines())}
assert set(groups)==set(range(1,102))
rows=[]
for g,segs in sorted(groups.items()):
 r=raw[g]; assert [(s['pdf_page'],s['column']) for s in segs]==list(dict.fromkeys((f['pdf_page'],f['column']) for f in r['raw_fragments'])),(g,segs)
 rows.append({'source_unit_key':r['source_unit_key'],'printed_group':g,'literal_segments':segs,'raw_ocr_fragment_count':len(r['raw_fragments']),'source_definition_and_labels':'retained literally in ordered lines; no automatic language abbreviation expansion','status':'complete original-page first reading; fine glyph adjudication and independent second reading pending','accepted_for_installation':False})
(p/'numbered-literal-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
print(len(rows),'groups',sum(len(s) for s in groups.values()),'segments',count,'literal lines',sum(len(r['raw_fragments']) for r in raw.values()),'OCR fragments')

from pathlib import Path
import json,hashlib
p=Path('data/data/other/forms/raw_data/niranjan_chakma_2010')
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
notes=json.loads((p/'numbered-literal-first-readings/reading-notes.json').read_text())
rows=[json.loads(x) for x in (p/'numbered-literal-first-reading.jsonl').read_text().splitlines()]
for r in rows:
 g=r['printed_group'];r['source_reading_notes']=notes['group_notes'].get(str(g),[])
 r['source_head_present']=g!=48
 r['source_claim_scope']='explicit origin and cited-work discussion; relation typing pending' if g in (65,87) else 'attributed place-name/spiritual/quoted-work prose, no standalone printed head' if g==48 else 'cross-language comparison; no ancestry inferred'
 r['line_witnesses']=[]
 for s in r['literal_segments']:
  for i,line in enumerate(s['literal_lines'],1):
   r['line_witnesses'].append({'key':f"{r['source_unit_key']}:pdf{s['pdf_page']}:{s['column']}:line{i}",'pdf_page':s['pdf_page'],'printed_page':s['printed_page'],'column':s['column'],'line_in_group_segment':i,'literal':line})
 if g in (54,79,87):r['source_reading_notes'].append('FONT-CONTROL CAUTION: provisional হ্ must NOT be accepted from a single lower diagonal. Ordinary হ has such a stroke in this font; distinct extra hasanta not yet established. Preserve this first-reading spelling as unaccepted evidence only.')
(p/'numbered-literal-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
report={'status':'complete first reading of all101 groups; independent full second reading and structured extraction required before acceptance','counts':{'groups':101,'original_pages':11,'page_column_segments':115,'continuations':14,'literal_lines':620,'raw_ocr_fragments_accounted':621},'source_pdf_sha256':'f022a2a448445cdc9840891478bd5864e2ded1ae98985332b51f0938c075842d','no_omitted_groups':list(range(1,102)),'page_column_layout':'Verified each group ordered page/column sequence against immutable raw allocation; all14 continuations match.','ocr_segmentation_exception':{'group':42,'raw_fragments':6,'original_lines':5,'evidence':'Last two overlapping OCR fragments correspond to one printed Santali line; original6x crop PDF83 fractions .52,.30,.95,.48.'},'editorial_policy':notes['policy'],'exception_review':{'48':'Complete prose matches owner; no printed head, preserve both place names and named-work claim separately.','65':'All components, explicit development, quotation and work label retained. Head lower mark provisional hasanta rather than long-u; exact ড/ড় unresolved. No accepted correction to owner draft.','87':'Original confirms লম্পক origin versus লম্বক cited-work term. Final h-hasanta withdrawn pending font control.'},'pending':['Independent full original second reading, including all compact clusters and listed typed uncertainties.','Parse all literal labels, definitions, explicit source origins, cited works and witnesses with their exact local scopes.','h-hasanta typography class requires separate ordinary-h controls; prior proposals are not accepted.','No canonical rows, profiles, language IDs, IPA or graph edges installed.'],'files':{str(f.relative_to(p)):h(f) for f in sorted((p/'numbered-literal-first-readings').glob('*'))},'raw_inputs':{n:h(p/n) for n in ['numbered-group-raw-inventory.jsonl','numbered-column-raw-evidence.jsonl','numbered-exception-first-reading.json']},'generator_sha256':h(p/'prepare_numbered_literal_first_reading.py'),'output_sha256':h(p/'numbered-literal-first-reading.jsonl')}
(p/'numbered-literal-first-reading-freeze-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
font={'status':'confidence withdrawn pending original font-control adjudication; historical proposals retained unchanged','reason':'Ordinary printed হ itself has a lower diagonal. A single such stroke does not demonstrate additional U+09CD. Norton control PDF99 row17 হাগা prompted this class reopening.','affected_proposals':[{'report':'independent-kokborok-full-review-20260927.json','pdf_page':91,'row':21,'field':'Chakma','proposal':'ভাহ্‌ঝ'},{'report':'independent-kokborok-full-review-20260927.json','pdf_page':91,'row':24,'field':'Chakma','proposal':'মালাহ্'},{'report':'independent-further-full-review-20260927.json','pdf_page':101,'row':2,'field':'both witnesses','proposal':'গাহ্‌ক'},{'report':'independent-further-full-review-20260927.json','pdf_page':101,'row':10,'field':'Chakma','proposal':'থেধাহ্'},{'report':'independent-further-full-review-20260927.json','pdf_page':102,'row':8,'field':'Chakma','proposal':'গোহ্‌র'},{'report':'numbered-literal-first-reading.jsonl','group':54,'proposal':'পাহ্'},{'report':'numbered-literal-first-reading.jsonl','group':79,'proposal':'যুহ্'},{'report':'numbered-literal-first-reading.jsonl','group':87,'proposal':'লামাহ্'}],'policy':'Do not apply these h-class corrections as confident glyph recovery. Retain immutable reports as superseded confidence evidence; adjudicate against ordinary হ typeform controls. Non-h findings need their own glyph evidence, not wholesale withdrawal or automatic approval.','pinned_historical_reports':{n:h(p/n) for n in ['independent-kokborok-full-review-20260927.json','independent-further-full-review-20260927.json']}}
(p/'independent-hasanta-h-font-control-reopening-20260927.json').write_text(json.dumps(font,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report['counts']));print(report['output_sha256'])

"""Independent physical-marker parser and whole-expression grouping proposal."""
from pathlib import Path
from collections import Counter
import fitz,json,re,hashlib
H=Path(__file__).resolve().parent;P=H.parents[1];W=next(p for p in H.parents if (p/'data/tags.py').exists());pdf=W/'tmp/pdfs/korwa-kodaku-2022/source.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest()=='a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c'
doc=fitz.open(pdf);events=[];active=None;section=None;number=None;question=None
for page_number in range(93,109):
 lines=doc[page_number-1].get_text(sort=True).splitlines()
 for line_number,line in enumerate(lines,1):
  t=line.strip()
  if line_number<=3 and t==str(page_number-10):continue
  m=re.match(r'^C\.([2-5])(?:\s|$)',t)
  if m:section='C.'+m[1];number=question=None;active=None;continue
  if not section:continue
  q=re.match(r'^QUESTION\s+(\d+)',t)
  m=re.match(r'^\\([npgɡfqa])(?:\s+(.*))?$',t)
  if q or m:
   marker='question-label' if q else ('g' if m[1]=='ɡ' else m[1])
   if q:question=q[1]
   elif marker=='n':number=(m[2] or '').strip();question=None
   active=dict(physical_key=f'behera2022korwakodaku:pdf{page_number}:line{line_number}',section=section,marker=marker,raw_marker='QUESTION' if q else m[1],printed_number=number,question_number=question,pdf_page=page_number,printed_page=page_number-10,source_lines=[])
   events.append(active)
  if active is not None and t:active['source_lines'].append(dict(pdf_page=page_number,line=line_number,raw=line))
for e in events:
 e['raw_text']='\n'.join(l['raw'] for l in e['source_lines'])
 text=re.sub(r'^\\[npgɡfqa]\s*','',e['raw_text'].strip(),count=1)
 e['content']=' '.join(text.split())
root=json.loads((P/'whole-recovery/interlinear/tagged-blocks-v4.json').read_text())
assert len(events)==637 and len(root)==637
assert [(e['physical_key'],e['marker'],e['raw_text']) for e in events]==[(e['physical_key'],e['marker'],e['raw_text']) for e in root]
rows=[];account={};languages={'C.2':'kw','C.3':'Kodaku','C.4':'Sadri','C.5':'Sadri'}
for s in languages:
 numbered={}
 for e in events:
  if e['section']==s and e['question_number'] is None and e['marker'] in ('p','g','f'):
   group=e['printed_number'].split('.')[0] if s=='C.5' else e['printed_number']
   numbered.setdefault(group,[]).append(e)
 for number,es in numbered.items():
  ps=[e for e in es if e['marker']=='p'];fs=[e for e in es if e['marker']=='f'];gs=[e for e in es if e['marker']=='g']
  assert len(fs)<=1
  key=f'behera2022korwakodaku:{s}:expression:{number}'
  row=dict(source_unit_key=key,section=s,printed_number=number,role='whole-narrative-expression',language=languages[s],form=' '.join(e['content'] for e in ps),gloss=fs[0]['content'] if fs else '',raw_interlinear_glosses=[e['content'] for e in gs],physical_p_keys=[e['physical_key'] for e in ps],physical_g_keys=[e['physical_key'] for e in gs],physical_f_keys=[e['physical_key'] for e in fs],printed_subnumbers=list(dict.fromkeys(e['printed_number'] for e in ps)),source_pages=sorted({l['pdf_page'] for e in es for l in e['source_lines']}),tags=['multiword-expression','sentential'],notes=[],alignment_status='No automatic word-level alignment; raw p/g blocks preserved for independent alignment review.',accepted_for_installation=False)
  if not fs:row['notes'].append('Source supplies no separately marked free translation for this numbered expression; interlinear gloss retained without inventing a translation.')
  if not gs:row['notes'].append('Source supplies no marked interlinear gloss block for this numbered expression.')
  if s=='C.4' and number=='20':
   # Visible bare g on printed p92 is source gloss, not phonetic continuation.
   assert row['form']=='ar oh sɐmajĩ atʃʰa piʈai bʰi pɐre g and that time good beating also given'
   row['form']='ar oh sɐmajĩ atʃʰa piʈai bʰi pɐre'
   row['raw_interlinear_glosses']=['and that time good beating also given']
   row['editorial_marker_recovery']={'pdf_page':102,'printed_page':92,'raw_marker':'g','source_line':14,'raw_event_key':ps[0]['physical_key'],'raw_event_unchanged':True}
   row['notes']=['The printed interlinear gloss has a bare g marker without a backslash; its English text is retained separately from the phonetic form.']
  if len(ps)>1:row['notes'].append('Whole translation scopes the joined printed phonetic blocks; all physical blocks and numbered subdivisions remain separately accounted.')
  rows.append(row)
  for p in ps:assert p['physical_key'] not in account;account[p['physical_key']]=key
 # Only C5 has source-language question/answer forms; other q/a are English documentary evidence.
 if s=='C.5':
  for q in map(str,range(1,11)):
   es=[e for e in events if e['section']==s and e['question_number']==q]
   ps=[e for e in es if e['marker']=='p'];qs=[e for e in es if e['marker']=='q'];answers=[e for e in es if e['marker']=='a'];assert len(ps)==len(qs)==len(answers)==1
   p=ps[0];key=f'behera2022korwakodaku:{s}:question:{q}'
   rows.append(dict(source_unit_key=key,section=s,question_number=q,role='whole-target-question',language='Sadri',form=p['content'],gloss=qs[0]['content'],physical_p_keys=[p['physical_key']],physical_q_keys=[qs[0]['physical_key']],source_pages=sorted({l['pdf_page'] for e in [p,qs[0]] for l in e['source_lines']}),tags=['multiword-expression','sentential','interr'],accepted_for_installation=False))
   assert p['physical_key'] not in account;account[p['physical_key']]=key
   a=answers[0];m=re.fullmatch(r'(.*?)\\([^\\]+)\\\s*',a['content']);assert m,(q,a)
   form=m[2].strip();key=f'behera2022korwakodaku:{s}:answer:{q}'
   rows.append(dict(source_unit_key=key,section=s,question_number=q,role='source-target-answer',language='Sadri',form=form,gloss=m[1].strip(),physical_answer_key=a['physical_key'],source_pages=sorted({l['pdf_page'] for l in a['source_lines']}),tags=['multiword-expression'] if ' ' in form else [],notes=['Literal target answer is enclosed by source backslashes; English answer preserved independently.'],accepted_for_installation=False))
assert len(account)==126 and set(account)=={e['physical_key'] for e in events if e['marker']=='p'}
assert len(rows)==117
(H/'independent-physical-events.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
(H/'whole-expression-proposal.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
status=dict(status='Grouping proposal; original visual review pending',source_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),root_scaffold_sha256=hashlib.sha256((P/'whole-recovery/interlinear/tagged-blocks-v4.json').read_bytes()).hexdigest(),physical_events=637,physical_p_blocks=126,all_p_blocks_accounted=True,whole_narrative_expressions=97,whole_target_questions=10,target_answers=10,proposal_rows=117,by_section_role={s:dict(Counter(r['role'] for r in rows if r['section']==s)) for s in languages},script_g_blocks=[e['physical_key'] for e in events if e['raw_marker']=='ɡ'],physical_p_to_expression=account,word_alignment='Not inferred. A later explicit alignment pass must resolve per-token gloss scope before child attestations are accepted.')
(H/'grouping-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print({k:v for k,v in status.items() if k!='physical_p_to_expression'})

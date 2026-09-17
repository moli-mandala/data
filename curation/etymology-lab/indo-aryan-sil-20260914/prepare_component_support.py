from pathlib import Path
P=Path(__file__).resolve().parent
s=(P/'loan_fourth_save.py').read_text().replace('loan-fourth','groundnut')
s=s.replace("needed=targets|{x['parent'] for x in acc}","needed=targets|{p for x in acc for p in x.get('components',[x['parent']])}")
a=s.index(" assert not selected[x['parent']]['Redirect']");b=s.index('\nvalidate_assignments(forms,rows)',a)
s=s[:a]+''' for parent in x.get('components',[x['parent']]):
  assert not selected[parent]['Redirect'],parent
  assert selected[parent]['Status']!='unlinked' or parent in targets,parent
rows=[]
for x in acc:
 for pos,parent in enumerate(x.get('components',[x['parent']]),1):
  rows.append(dict(Form_ID=x['record']['ID'],Etymon_ID=parent,Kind=x.get('kind','reflex'),Rank='1',Status='accepted',Source=x['citation'],Notes=x['evidence']+' Joint SIL review 2026-09-14.',Pos=str(pos) if x.get('components') else ''))
'''+s[b:]
s=s.replace("parentForm=selected[parent]['Form']","parentForm=' + '.join(selected[p]['Form'] for p in ys[0].get('components',[parent]))")
(P/'groundnut_save.py').write_text(s)
# Decisions stay one per lexical record. Only explicit component lists add rows.
p=P/'render_review.py';s=p.read_text();s=s.replace("examined=accepted|held","row_count=lambda x: len(x.get('components',[])) or 1\nsurvey_rows=sum(map(row_count,acc))\nadditional_rows=sum(map(row_count,allacc))-survey_rows\nexamined=accepted|held")
s=s.replace("nrows=sum(len(x['assignments']) for x in d['proposals']);lid=", "nrows=sum(len(x['assignments']) for x in d['proposals']);nrecords=len({fid for q in d['proposals'] for fid in q['formIds']});lid=")
s=s.replace('{nrows:,} assignment rows and records.', '{nrows:,} assignment rows on {nrecords:,} records.')
s=s.replace("'component':'Component: '","'component':'Compound components: '")
s=s.replace('Multiword, slash-separated and unmatched responses are outside these saved proposals;', 'Responses not listed in the exact manifest are outside these saved proposals;')
s=s.replace('savedAssignments=len(acc)', 'savedAssignments=survey_rows').replace("savedParentNodes=len({x['parent'] for x in acc})", "savedParentNodes=len({parent for x in acc for parent in x.get('components',[x['parent']])})")
s=s.replace('additionalDictionaryAssignments=len(allacc)-len(acc)','additionalDictionaryAssignments=additional_rows')
s=s.replace('{len(acc):,} rank-1 assignments','{survey_rows:,} rank-1 assignments').replace('{len(acc):,} new survey assignment rows (plus {len(allacc)-len(acc)} dictionary rows)', '{survey_rows:,} new survey assignment rows on {len(accepted):,} records (plus {additional_rows} dictionary rows)')
p.write_text(s)
p=P/'verify_joint_checkpoint.py';s=p.read_text();s=s.replace("assert status['savedAffectedRecords']+status['additionalDictionaryAssignments']==len(rows)","assert status['savedAssignments']+status['additionalDictionaryAssignments']==len(rows)\nexcluded=set(read('scope-correction.json')['excludedIds'])\nassert status['savedAffectedRecords']==len({r['Form_ID'] for r in rows if r['Form_ID'] not in excluded})")
p.write_text(s)

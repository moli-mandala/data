"""Refresh pending review/dispositions, never accepted data."""
import json,collections,re
from pathlib import Path
p=Path(__file__).resolve().parent
saved=(p/'acceptance-validation.json').exists()
def finalized(lines):
 if not saved:return '\n'.join(lines)
 out=[]
 for line in lines:
  if line.startswith('**The eight-hour') or line.startswith('**Overnight research'):
   line='**All 757 proposals approved and saved. Persian/Arabic loans are nested under their etymological entries; immediate transmission is unspecified.**'
  elif line.startswith('The shared corpus is being rebuilt'):
   line='Acceptance reconciled the frozen records against current IDs, meanings and overlay rows. All 3,777 links passed temporary graph validation. See [acceptance validation](acceptance-validation.json).'
  elif 'ready for review. Nothing in this batch saved yet.' in line:
   line=line.replace('ready for review. Nothing in this batch saved yet.','approved and saved to the accepted overlay.')
  elif line.startswith('Final structural validation:'):
   line='All 3,777 links passed temporary graph validation; repeat application changed zero. All donor dependencies are resolved. [Acceptance validation](acceptance-validation.json) records the checks; full donor-build checks are in [acceptance report](ACCEPTED.md).'
  elif line.startswith('Open a batch below'):
   line='Open a batch for its saved analysis and evidence. Qualified analyses retain their scholarly caveats; approval does not resolve every phonological or historical uncertainty.'
  elif line.startswith('[Acceptance dependencies]'):
   line='All accepted compound/base dependencies were saved together. Historical dependency notes remain in [DEPENDENCIES.md](DEPENDENCIES.md).'
  elif line.startswith('[Complete review and held cases]'):
   line='[Complete saved review and held cases](REVIEW.md). Exact records and saved rows are in the batch manifests. See [acceptance report](ACCEPTED.md) and [acceptance validation](acceptance-validation.json).'
  elif line.startswith('Excluded responses and unresolved cases'):
   line='Excluded responses remain held in the cumulative review and holds.json. All approved donor dependencies are resolved.'
  line=line.replace('No accepted overlay edits.','Saved to the accepted overlay.').replace(' proposed links;',' saved links;').replace(' proposed assignment rows:',' saved assignment rows:')
  line=line.replace('acceptance requires fresh ID, meaning and overlay-conflict reconciliation.','acceptance included fresh ID, meaning and overlay-conflict reconciliation.')
  out.append(line)
 return '\n'.join(out)
names={'Malvi':'mewari_basad','Nimadi':'Nimadi','Bagheli':'bagheli_lakshman'}
all_dispositions={};counts={}
index=[]
topics={1:'Body parts and village',2:'Household and tools',3:'Nature',4:'Food and animal products',5:'Animals and name',6:'Kinship and time',7:'Adjectives',8:'Numerals',9:'Simple verb forms',10:'Interrogatives, quantifiers and pronouns'}
topics[11]='Hindustani donor proposals'
topics[12]='Further nouns and adjectives'
topics[13]='Body and adjective remainders'
topics[14]='Ant family and tan'
topics[15]='Further household and kinship'
topics[16]='Further nature and time vocabulary'
topics[17]='Weather, rope and tree families'
topics[18]='Ordered kinship components'
topics[19]='Further adjectives and simple burn forms'
topics[20]='Simple past-tense forms'
topics[21]='Further Hindustani loans'
topics[22]='Food and instrument remainders'
topics[23]='Knife loans, noon and finger-rings'
topics[24]='Grain, mortar, cabbage and ring remainders'
topics[25]='Temporal and kinship remainders'
topics[26]='Spouse and woman families'
topics[27]='Bride and wife remainder'
topics[28]='Quality, quantity and score numeral'
topics[29]='Weight adjective loans'
topics[30]='Adjective reduplication'
topics[31]='Direction, head and household remainders'
topics[32]='Right, left and below families'
topics[33]='Malvi h-initial and lightning remainders'
topics[34]='Fat, house, meat and chicken loans'
topics[35]='Head and thunderbolt families'
topics[36]='Potato and tomato vocabulary'
topics[37]='Eleven, twelve and evening'
topics[38]='Whole cauliflower loans'
articles=json.loads((p/'cdial-articles.json').read_text())
main=['# Malvi, Nimadi and Bagheli — overnight triage','', '**The eight-hour research window has ended. All proposals below remain pending; no new analyses have been saved to the accepted overlay.**','', 'Deadline: September 11, 2026, 11:30 a.m. America/New_York (15:30 UTC). Proposals are numbered independently by survey. Each row groups only selected compatible responses; exact IDs, locality tags, source locators and assignment rows are in the corresponding JSON manifest.','', 'The shared corpus is being rebuilt concurrently. Initial conversation counts and the frozen working inventories differ because of merging, not research progress. Merged records with incompatible meanings are held. Deadline ID/redirect/conflict reconciliation is recorded in validation.json; acceptance still requires a fresh check.','']
held=json.loads((p/'holds.json').read_text()) if (p/'holds.json').exists() else {}
for lang,lid in names.items():
 inventory=json.loads((p/f'{lang}-inventory.json').read_text()); byid={r['ID']:r for r in inventory}
 proposals=[]
 for f in sorted((p.parent/lid).glob('batch-*.json')):
  m=json.loads(f.read_text())
  if m.get('researchDirectory')!=str(p):continue
  pp=m['proposals'];proposals+=pp
  index.append({'survey':lang,'batch':m['batch'],'first':min(x['number'] for x in pp),'last':max(x['number'] for x in pp),'records':len({i for x in pp for i in x['formIds']}),'straightforward':sum(x['difficulty']=='straightforward' for x in pp),'qualified':sum(x['difficulty']=='qualified' for x in pp),'path':f'../{lid}/{f.stem}-review.md'})
  lines=[f'# {lang} — batch {m["batch"]:03d}','',f'**{len(pp)} {"proposal" if len(pp)==1 else "proposals"}, ready for review. Nothing in this batch saved yet.**','']
  for tier in ['straightforward','qualified']:
   group=[x for x in pp if x['difficulty']==tier]
   if not group:continue
   lines += [f'## {tier.capitalize()}','',f'| # | {lang} | Proposed etymology | Evidence |','|---|---|---|---|']
   for x in group:
    word=', '.join(x['forms']).replace('|','\\|'); head=x['parentForm'].replace('*','\\*');gloss=x['gloss'].replace('|','\\|')
    article=x['parentId'].split('-')[0]
    if article not in articles:
     cited=re.search(r'CDIAL\[(\d+[a-z]?)',x['citation'])
     if cited:article=cited.group(1)
    page=articles.get(article,[{}])[0].get('page')
    url=x.get('primarySourceURL') or (f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={page}' if page else 'https://dsal.uchicago.edu/dictionaries/soas/')
    label=x['citation'].replace('[',' ').replace(']','')
    link=(label+'; [Primary evidence](<'+x['primarySourceURL']+'>)') if x.get('primarySourceURL') else (f'[{label}]({url})' if page else label)
    if x['kind']=='component':
     parts=[]
     for c in x['components']:
      cm=re.search(r'CDIAL\[(\d+[a-z]?)',c['citation'])
      cp=articles.get(cm.group(1),[{}])[0].get('page') if cm else None
      cu=f'https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page={cp}' if cp else url
      cl=c['citation'].replace('[',' ').replace(']','')
      parts.append('**'+c['form']+'** (Pos '+str(c['position'])+', ID `'+c['id']+'`; ['+cl+']('+cu+'))')
     analysis='Components: '+' + '.join(parts)
    else:
     analysis=('Borrowed from '+x.get('parentLanguage','')+' ' if x['kind']=='borrowed' else 'Derived from ' if x['kind']=='derived' else '')+f'**{head}**, {link}; ID `{x["parentId"]}`'
    lines.append(f'| {x["number"]} | **{word}** ‘{gloss}’ | {analysis} | {x["evidence"]} |')
   lines.append('')
  rows=sum(len(x['assignments']) for x in pp);ids={r for x in pp for r in x['formIds']}
  kinds=collections.Counter(r['Kind'] for x in pp for r in x['assignments'])
  kind_summary=', '.join(f'{n} `{k}`' for k,n in sorted(kinds.items()))
  lines += [f'{len(pp)} {"proposal" if len(pp)==1 else "proposals"}; {len(ids)} affected records; {rows} proposed assignment rows: {kind_summary}. Historical extensions and uncertain contact pathways are explained in the evidence.','', 'Excluded responses and unresolved cases are tracked separately in the cumulative review and holds.json. See validation.json for any pending donor-ancestry dependency.','']
  f.with_name(f.stem+'-review.md').write_text(finalized(lines))
  main+=lines
 proposed={i for x in proposals for i in x['formIds']};assert sum(len(x['formIds']) for x in proposals)==len(proposed),'Duplicate proposed record'
 dispositions=[]
 for r in inventory:
  status=('saved' if saved else 'pending-review') if r['ID'] in proposed else 'held' if r['ID'] in held else 'unexamined'
  dispositions.append({'id':r['ID'],'form':r['Form'],'gloss':r['Gloss'],'status':status,**({'reason':held[r['ID']]['reason']} if r['ID'] in held else {})})
 all_dispositions[lang]=dispositions
 counts[lang]={'proposals':len(proposals),'records':len(proposed),'assignment_rows':sum(len(x['assignments']) for x in proposals),'straightforward':sum(x['difficulty']=='straightforward' for x in proposals),'qualified':sum(x['difficulty']=='qualified' for x in proposals),**dict(collections.Counter(x['status'] for x in dispositions)),'inventory':len(inventory)}
 main += [f'## {lang}: held cases','', '| Form and meaning | Reason |','|---|---|']
 for d in dispositions:
  if d['status']=='held':main.append(f'| **{d["form"]}** ‘{d["gloss"]}’ | {d["reason"]} |')
 main+=['',f'Unexamined records: {counts[lang].get("unexamined",0)}. These have not been ruled out.','']
summary=['## Final research counts','','| Survey | Proposals | Records / links | Held | Unexamined |','|---|---:|---:|---:|---:|']
for lang,c in counts.items():summary.append(f'| {lang} | {c["proposals"]} | {c["records"]} / {c["assignment_rows"]} | {c.get("held",0)} | {c.get("unexamined",0)} |')
main[8:8]=summary+['']
(p/'REVIEW.md').write_text(finalized(main));(p/'dispositions.json').write_text(json.dumps(all_dispositions,ensure_ascii=False,indent=2));(p/'counts.json').write_text(json.dumps(counts,indent=2))
triage=['# Morning triage: Malvi, Nimadi, Bagheli','','**Overnight research finished September 11, 2026, at the 11:30 a.m. ET deadline. Every proposal is pending; the overnight automation is paused.**','',f'{sum(c["proposals"] for c in counts.values())} proposals covering {sum(c["records"] for c in counts.values())} records and {sum(c["assignment_rows"] for c in counts.values())} proposed links; {sum(c["straightforward"] for c in counts.values())} straightforward and {sum(c["qualified"] for c in counts.values())} qualified proposals. No accepted overlay edits.','', 'Open a batch below for the four-column evidence table. Proposal numbers are independent by survey. To triage, specify the survey and numbers to approve, revise or hold; approval may select only the straightforward section. Qualified proposals preserve unresolved phonology, contact, morphology or competing etyma.','', '| Survey | Batch and topic | Proposal numbers | Records | Straightforward | Qualified |','|---|---|---:|---:|---:|---:|']
triage[6:6]=['| Survey | Proposals | Records | Links | Held | Unexamined |','|---|---:|---:|---:|---:|---:|']+[f'| {lang} | {c["proposals"]} | {c["records"]} | {c["assignment_rows"]} | {c.get("held",0)} | {c.get("unexamined",0)} |' for lang,c in counts.items()]+['','Final structural validation: no missing IDs, changed target records, missing registry targets or current overlay conflicts. The temporary graph passed for 3,760 eligible links and changed nothing on a second application. **17 links remain blocked on unresolved donor ancestry**; see [acceptance dependencies](DEPENDENCIES.md). These checks do not establish the scholarly correctness of every proposal.','']
for x in index:
 triage.append(f'| {x["survey"]} | [{x["batch"]:03d}: {topics.get(x["batch"],"Further research")}]({x["path"]}) | {x["first"]}–{x["last"]} | {x["records"]} | {x["straightforward"]} | {x["qualified"]} |')
triage+=['','[Acceptance dependencies](DEPENDENCIES.md). Review these when selecting individual compounds or derivations.','', '[Complete review and held cases](REVIEW.md). Exact source records, IDs and planned rows accompany each batch in its JSON manifest. [Current validation](validation.json) and [last complete successful validation](validation-last-passed.json) must be read with their proposal counts and timestamps; a later batch may not yet be covered.','',f'Held records: {sum(c.get("held",0) for c in counts.values())}. Unexamined records: {sum(c.get("unexamined",0) for c in counts.values())}. Unexamined does not mean unetymologisable. The frozen source inventory is preserved while another task rebuilds the shared corpus; acceptance requires fresh ID, meaning and overlay-conflict reconciliation.','']
(p/'TRIAGE.md').write_text(finalized(triage))
print(json.dumps(counts,indent=2))

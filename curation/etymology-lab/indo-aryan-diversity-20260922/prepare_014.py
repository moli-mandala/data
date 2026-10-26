import ast,csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
rs=json.loads((P/'bajjika-records.json').read_text())
tree=ast.parse((P/'prepare_013.py').read_text()); groups=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='groups' for t in n.targets))
forms=[['ā̃kʰ'],['kān'],['nāk'],['kes'],['muh','mũh'],['jibʰ'],['hātʰ','hā̃tʰ'],['dā̃t'],['pet'],['gʰar'],['guh'],['gām','gā̃o']]
evidence={0:'Turner gives Bihari/Maithili ā̃khi and Bengali ā̃kh(i) eye under akṣi, through akkhi. The source nasal long vowel and kh match this family.',4:'Turner explicitly gives Maithili and Bhojpuri mũh mouth/face beside Prakrit muha. The source mu(h) forms continue regular loss of k and preserve both elicited senses.',5:'Turner gives Bhojpuri jībh and Bengali jib alongside Prakrit jibbhā under jihvā section 1. The source final bh matches that series.',6:'Turner gives Bihari/Maithili/Bhojpuri hāth hand/forearm and Awadhi hā̃th. Both source vowel variants fit hasta through hattha.',7:'Turner explicitly gives Maithili/Bhojpuri dā̃t tooth. Loss of the nasal stop with vowel nasalisation matches the source; the plural elicitation gloss is preserved.',8:'Turner section 1 gives Maithili peṭ(h) and Bhojpuri peṭ belly under *peṭṭa, distinct from poṭṭa and *peḍḍuka. The survey writes plain t; this is preserved without a transcription repair.',10:'Turner section 1 gives Bhojpuri gū̆h and Hindi gūh excrement beside Prakrit gūha. The source guh fits this series; the euphemistic medān response is excluded.',11:'Turner explicitly gives Maithili gām and gā̃o village under grāma. Both the m-retaining and nasal-vowel variants are represented in this source.'}
groups=[(forms[i],g[1],g[2],g[3],evidence.get(i,g[4])) for i,g in enumerate(groups)]
groups += [(['pāni'],['water'],'8082','pānīya','Turner gives Maithili/Bhojpuri pānī and neighbouring Bengali/Nepali pāni water under pānīya. The source final short i is preserved.'),(['bāl'],['hair'],'11572','vāla','Turner gives Hindi/Nepali bāl and Bhojpuri bār hair under vāla. The b onset and long vowel support this hair family; the alternative source kes records are linked separately.')]
assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
a=[]
for fs,gs,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in fs and r['Gloss'] in gs:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' Published spelling and Bajjika locality tags are preserved.'))
aud={}
for line in open('data/other/forms/raw_data/ia_dravidian_2026/20260911-bajjika-audit.jsonl'):
 row=json.loads(line); ins=row.get('installed',[])
 if len(ins)>7:aud[ins[7]]=row
for x in a:
 z=aud[x['record']['Source']]; assert z['installed'][3]==x['record']['Gloss'];assert z['status']=='unlinked'
held=[dict(record=r,reason='Alternative lexical response requires its own immediate donor or semantic analysis; excluded from inherited mouth/excrement families.') for r in rs if r['Form'] in ['ceharā','medān'] and r['Gloss'] in ['face','feces'] and r['ID'] not in assigned]
(P/'pass014-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
(P/'pass014-source-audit.json').write_text(json.dumps({x['record']['ID']:aud[x['record']['Source']] for x in a},ensure_ascii=False,indent=2)+'\n')
lines=['# Bajjika: first diversity batch','',f'{len(a)} records / {len(a)} reflex rows, fourteen etyma. Autonomous review under the standing authorization. No rebuild.','','| # | Bajjika | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(fs,gs,parent,label,ev) in enumerate(groups,1):
 label=label.replace('*',r'\*');n=sum(x['parent']==parent for x in a);lines.append(f'| {i} | **{" / ".join(fs)}** ‘{" / ".join(gs)}’ ({n}) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} alternative-response records held. Source audit locators verified for every selected record.','', 'Validation and save status: pending.']
(P/'pass014-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')

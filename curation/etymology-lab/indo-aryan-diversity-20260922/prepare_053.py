import csv,json,ast
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'chalise2014majhi[' in r['Source'] and r['ID'] not in linked]
tree=ast.parse((P/'prepare_052.py').read_text());base=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='groups' for t in n.targets))
forms={'one':['ek'],'two':['d̪ui'],'three':['t̪in'],'four':['tsār'],'five':['pā̃ts'],'six':['tsʰa'],'seven':['sāt̪'],'eight':['ātʰ'],'nine':['naũ'],'ten':['das']}
evidence={
'one':'Turner2462 section2 places Nepali/Maithili ek under the MIA ekka replacement, rather than vowel-only descendants of section1 eka. Majhi ek retains k and fits this branch; the entry’s alternative explanations of gemination remain open.',
'two':'Turner6648 gives Nepali/Maithili/Bhojpuri dui two alongside older duvi/duve forms. Majhi d̪ui has the same diphthong, with explicit dental articulation retained.',
'three':'Turner5994 section3 gives Nepali tin and Maithili tini through MIA tiṇṇi. The Majhi nasal-final t̪in selects trīṇi rather than trayas/trāyas; the source dental sign is preserved.',
'four':'Turner4655 section2 gives Nepali cār and eastern cāri through MIA *cayāri. Source tsār is the corresponding long-vowel affricate form; retain the published ts sequence rather than replacing it with dictionary c. The caur branch is not selected.',
'five':'Turner7655 gives Nepali/Maithili pā̃c five. Majhi pā̃ts preserves the matching nasal long vowel and affricate family, in the source’s ts spelling; no phonetic normalization is made.',
'six':'Turner12803 section3 explicitly gives Nepali cha and the eastern cha/chao family under *kṣaṭ/*kṣvaṭ. Majhi tsʰa is the source aspirated affricate form, not the ṣ- series of the other branches.',
'seven':'Turner13139 gives Nepali/Maithili/Bhojpuri sāt through MIA satta. Majhi sāt̪ has the same vowel and explicit dental final, retained in the assignment.',
'eight':'Turner941 gives regional āṭh eight through MIA aṭṭha. Majhi ātʰ has the corresponding long vowel and aspirated final stop; retain its published unmarked t rather than inserting a retroflex diacritic.',
'nine':'Turner6984 gives Nepali/Maithili nau and neighboring nasalized naũ nine. Majhi naũ belongs to this numeral family, with nasalization retained; it is distinct from the new adjective.',
'ten':'Turner6227 gives Nepali/Bihari/Maithili das through MIA dasa. All five Majhi das ten attestations fit this regional numeral without added aspiration or sibilant changes.'}
a=[];lines=['# Majhi: numerals one to ten','','Fifty proposed reflex records across ten numeral families; standing autonomous authorization.','','| # | Majhi forms and meaning | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,old,parent,label,old_ev) in enumerate(base,1):
 fs=forms[g];ev=evidence[g];hits=[r for r in rs if r['Form'] in fs and r['Gloss']==g];assert len(hits)==5,(g,len(hits))
 for r in hits:a.append(dict(record={k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']},parent=parent,kind='reflex',citation=r['Source']+';CDIAL['+parent+' and addenda]',evidence=ev+' Published-notation uncertainty and all locality identities remain unchanged; no guessed transcription is introduced.'))
 lines.append(f'| {i} | **{" / ".join(fs)}** ‘{g}’ (5 records) | **{label}**, CDIAL **{parent}** | {ev} |')
aud={};full=[]
for line in open('data/other/forms/raw_data/selected_surveys_2026/majhi-audit.jsonl'):
 row=json.loads(line)
 for rd in row.get('readings',[]):
  ins=rd.get('row',[])
  if len(ins)>7:aud.setdefault(ins[7],[]).extend(rd.get('review',[]))
  if len(ins)>7 and ins[7] in {x['record']['Source'] for x in a}:full.append(row)
for x in a:
 notes=aud[x['record']['Source']];assert notes and all(n=='transcription: published notation retained; ambiguous symbols are not assigned guessed phonemic values' for n in notes),(x['record'],notes)
(P/'pass053-source-audit.json').write_text(json.dumps(dict(recordsChecked=len(a),retainedPublishedNotation=True,rows=full),ensure_ascii=False,indent=2)+'\n')
(P/'pass053-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n')
lines+=['','50 records and50 reflex rows; no additional holds in this numeral set. All source notation cautions and five locality attestations remain separate.','Research: homepage majhi-search002–003. Full dictionary prose/addenda already read in danuwar-primary006–007 are reused for these exact ten shortlisted families; source-specific evidence and audit are independently checked.','Validation: pending.'];(P/'pass053-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted')

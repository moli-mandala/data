import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'domari-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
(['ǧib'],['language; tongue'],'5228','jihvā','Turner section 1 gives Domari ǰiba and related ǰib tongue forms. Herin explicitly includes ǧib among inherited elements in his discussion of complex verbs; only the bare noun is included.'),
(['pēṭ'],['belly'],'8376','*peṭṭa','Herin explicitly derives pēṭ from Indo-Aryan pēṭṭa belly. Turner section 1 gives Domari pēṭ. Herin says the underdot here marks velarisation, not retroflexion; the source notation is preserved.'),
(['vāl'],['hair'],'11572','vāla','Turner gives the vāla hair family, including Romani wal/bal and regional vāl. Herin presents vāl as an example of the Aleppo approximant; the source vowel and consonant are unchanged.'),
(['pānī'],['water'],'8082','pānīya','Turner explicitly gives Domari pāni water. The bare source noun matches this inherited family; case-marked pāny-ə-m is not silently treated as the same complete form.'),
(['dānd'],['tooth'],'6152','danta','Turner gives the danta > dand tooth series in Romani and neighboring languages. Herin glosses dānd tooth in his existential example; the source word is an independent noun, not the accompanying mouth phrase.'),
(['qālã́'],['black'],'3083','kāla','Herin section 1.2 explicitly identifies qālã black as inherited despite uvular q. Turner gives Domari kāla black/dark. The uvular is retained, not normalised to k.'),
(['xaśt','xāśt'],['hand'],'14024','hasta','Herin explicitly compares Aleppo xāšt with Palestinian Domari xast and identifies št as an Aleppo peculiarity. Turner hasta supplies the hand family, including retained-st forms. Both source vowel-length spellings are retained.'),
(['yēk','yēka'],['one'],'2462-2','*ekka','Turner section 2 gives the retained-k *ekka family, including Romani yek and Domari εk. Herin gives yēka/yōka with a short form yē; the source yēk/yēka forms retain k and belong here.'),
(['trən'],['three'],'5994-3','trīṇi','Turner section 3 gives Romani trin and the trīṇi family with retained n. Herin lists trən among the atomic numerals; it is distinct from the n-less trayaḥ series.'),
(['śtār'],['four'],'4655','catvāraḥ','Turner section 1 gives Romani štar four, distinct from the cār and cōr subsections. Herin lists štār as the atomic numeral; the source matches the retained cluster and r.'),
(['panǧ'],['five'],'7655','pañca','Turner gives Romani panǰ/panč five, with retained nasal and affricate. Herin lists panǧ as the atomic numeral, matching the voiced-affricate variant.'),
(['dazz'],['ten'],'6227','daśa','Herin section 2.7 explicitly calls dazz ten Indic while identifying the neighboring Kurdish and Arabic numeral material. Turner daśa includes Romani deš; the Zargari/Romani spelling is not substituted for Aleppo dazz.')]
a=[]
for forms,glosses,parent,label,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form'] in forms and r['Gloss'] in glosses:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev))
held=[dict(record=r,reason='Herin section 2.7 explicitly identifies this numeral as Kurdish-derived (nine only possibly). Resolve a supported immediate Kurdish donor node before saving a borrowing link.') for r in rs if r['Form'] in ['śēś','ḣavt','ḣaśt','na'] and r['Gloss'] in ['six','seven','eight','nine'] and r['ID'] not in assigned]
(P/'pass012-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Aleppo Domari: first diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; twelve etyma. No rebuild.','','| # | Aleppo Domari | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(forms,glosses,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*');lines.append(f'| {i} | **{" / ".join(forms)}** ‘{" / ".join(glosses)}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} numeral records held for Kurdish donor resolution (nine only possibly a loan). Herin’s cached publisher HTML was read for sections 1.2 and 2.7 and the inherited complex-verb discussion; extracted prose is in domari-primary-prose.txt. No inflected phrases or complex verbs are included.','', 'Validation and save status: pending.']
(P/'pass012-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')

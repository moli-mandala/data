import json,csv
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records=json.loads((P/'varli-records.json').read_text());assigned={r['Form_ID'] for r in read_assignments()};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('cow',['gāy','gāi'],'4147-3','gāvī','Turner section 3 gives Pk gāī, Gujarati gāy and Marathi gāī cow. The retained final i/y distinguishes gāvī from the gāvā series in section 2.'),
('horse',['ghōṛo','ghoṛa'],'4516','ghoṭa','Turner gives Pk ghoḍa, Gujarati ghoṛɔ and Marathi ghoḍā horse. Intervocalic retroflex voicing/flapping and the inflectional final vowel fit the regional family. Turner regards its deeper origin as probably non-Aryan; no new modern loan route is asserted.'),
('horn',['siŋ'],'12583','śṛṅga','Pk siṃga and Gujarati sĩg/Marathi śī̃g support this horn family, alongside n/g-cluster reduction to a velar nasal in Nepali/Bengali siṅ. Both Varli varieties explicitly have siŋ.'),
('house',['gɔr','ghɔr'],'4428','ghara','Turner lists Gujarati/Marathi ghar and regional gar/ghɔr outcomes. The accepted parent is ghara, not a direct leap to gṛha: Turner explicitly discusses phonetic difficulty in the deeper derivation. Davar deaspiration is preserved.'),
('maternal uncle',['mamo','mama'],'10055','māma','Turner gives Pk māma mother’s brother, Gujarati māmɔ and Marathi māmā. He identifies the family as a nursery word; the link does not claim that similar nursery words in other families prove cognacy.'),
('wood',['lākɔḍ'],'10875-2','*lakkuṭa','Turner section 2 gives Pk lakkuḍa and Marathi lākaḍ/lākuḍ wood, retaining k unlike the lakuṭa/laüḍa series of section 1.'),
('tall',['uco','ũca'],'1634','ucca','Turner gives unnasalised uco and nasalised Gujarati ū̃cũ/Marathi ũc tall, high. His proposed influence from udañc on nasalisation is explicitly doubtful, so it is retained as uncertainty rather than a second component.'),
('wind',['vāro'],'11497','vātara','Turner section 1 gives Konkani vāro and Marathi vārā wind. This adds the Davar final-o counterpart to the Dungar vārā records already saved.')]
a=[]
for gloss,forms,parent,label,ev in groups:
 for r in records:
  if r['Form'] in forms and r['Gloss']==gloss and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev))
held=[dict(record=r,reason='Compare manuṣya 9828 and mānuṣa 10049. The regional Marathi comparator has initial long ā, whereas this source writes manus. Need stronger source-specific evidence before selecting a parent.') for r in records if r['Form']=='manus' and r['Gloss']=='person' and r['ID'] not in assigned]
(P/'pass005-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Davar and Dungar Varli: second diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; eight etyma. No rebuild.','','| # | Language forms and meanings | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(gloss,forms,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*')
 lines.append(f'| {i} | **{" / ".join(forms)}** ‘{gloss}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['','manus person is held over the manuṣya/mānuṣa distinction. Existing source spellings, IDs and Davar/Dungar language labels are preserved.','', 'Validation and save status: pending.']
(P/'pass005-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')

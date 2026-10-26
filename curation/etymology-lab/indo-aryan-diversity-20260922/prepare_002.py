import json,csv
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
records=json.loads((P/'darai-records.json').read_text())
assigned={r['Form_ID'] for r in read_assignments()}
assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('bagʰ','tiger','12193','vyāghra','Turner section 1 gives Pk vaggha and Nepali/Maithili bāgh tiger, matching initial vy > b and gh retained. The distinct *viyāghra section is not selected.'),
('bʰok','hunger','9286','bubhukṣā','Turner gives the reduced Pk bhukkhā family and Nepali bhok(h), Assamese bhok. His question about the exact reduced intermediate *bhukṣā remains; this is a family assignment, not an assertion resolving that reconstruction.'),
('kaṭʰi','wood','3120','kāṣṭha','The entry includes kāṣṭhikā and Sindhi/Hindi kāṭhī wood, with kṣṭ > ṭṭh > ṭh. The final i continues the feminine/extended form covered in this entry; no separate section node exists. This is not a new claim of synchronic derivation.'),
('ʣanike','to know','5193','jānāti','Turner gives Nepali jānnu and Maithili jānab. Dhakal p. 75 explicitly glosses -ike INF in cəl-a-ike and segments dzan-ne know-COND, confirming dzan as the verb stem; the existing profile renders dz as ʣ.'),
('ãkʰi','eye','43','akṣi','Pk akkhi and Bihari/Maithili ā̃khi eye support kṣ > kkh > kh, nasalisation and retained final i. No extra component is present.'),
('bãsi','flute','11180','vaṃśī','Compare Pk vaṃsī and Hindi bãsī flute, with v > b, nasalisation and s for ś. The unextended form fits the head, not the separate -l- extended bā̃sulī series.'),
('beŋgo','frog','12159-2','*viyaṅga','Turner puts Nepali beṅ, Oriya beṅga and Maithili beṅ frog specifically under section 2 *viyaṅga, distinct from section 1 skin marks. The source retains a final nominal o.'),
('goḍ','leg','4272','*goḍḍa','Compare Pk goḍḍa/gōḍa foot and Maithili goṛ leg. The retroflex stop and leg sense identify this family; no unsupported intermediate borrowing route is imposed.')]
a=[]
for form,gloss,parent,label,ev in groups:
 for r in records:
  if r['Form']==form and r['Gloss']==gloss and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];dhakal2011darai[pp. 42, 75]',evidence=ev+' Darai vowel length is noncontrastive (Dhakal p. 42); source spellings are preserved.'))
held=[]
for r in records:
 if r['ID'] in assigned:continue
 if (r['Form'],r['Gloss'])==('til','sesame'):held.append(dict(record=r,reason='5827 tila versus 5827-3 *tilla: Darai noncontrastive vowel length cannot distinguish these neighboring historical series.'))
 if (r['Form'],r['Gloss'])==('bʰar','load'):held.append(dict(record=r,reason='9393 bhara versus 9459 bhāra: both have Maithili load comparanda, and Darai noncontrastive length does not distinguish them.'))
(P/'pass002-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n')
lines=['# Darai: second diversity batch','',f'Autonomously reviewed under the standing authorization. {len(a)} proposed records and reflex rows; eight etyma. No rebuild.','','| # | Darai | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(form,gloss,parent,label,ev) in enumerate(groups,1):
 n=sum(x['parent']==parent for x in a);label=label.replace('*',r'\*')
 lines.append(f'| {i} | **{form}** ‘{gloss}’ ({n} records) | **{label}**, CDIAL **{parent}** | {ev} |')
lines+=['',f'{len(held)} records held: sesame and load have competing historical stems obscured by noncontrastive length. No source homonyms or multiword responses included.','', 'Validation and save status: pending.']
(P/'pass002-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')

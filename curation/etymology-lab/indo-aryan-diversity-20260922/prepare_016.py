import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');short=json.loads((P/'konkani-reviewed-shortlist.json').read_text());assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('eighteen','946','aṣṭādaśa','Turner explicitly gives Marathi aṭhrā eighteen after Prakrit aṭṭhārasa/aṭṭhāra. The source əṭhra matches that contracted numeral.'),
('eighty','911','aśīti','Turner section 2 gives Marathi aiśī̃/ẽśī eighty, expressly noting phonetic irregularity and epenthesis. The source ə̃yśi matches this series; the existing unsplit 911 node is used, with that qualification retained.'),
('courtyard','118','aṅgana','Turner gives Konkani āṅgaṇa/āṅgoṇ courtyard beside Prakrit aṁgaṇa. This is the courtyard entry, distinct from aṅga body.'),
('body','114','aṅga','Turner explicitly gives Konkani āṅga body under aṅga. The source aŋgə matches the velar-nasal cluster; the separate unreviewed nasal-vowel response is not included.'),
('mango','1268','āmra','Turner gives Konkani ā̃bo and Marathi ā̃bā mango alongside MIA amba. Source ambə continues the mb family; its transcription and final vowel are preserved.'),
('today','242','adya','Turner explicitly gives Konkani āj/āji today from adya through ajja. Source a:ǰi is the same adverb; the existing profile represents ǰ as j.'),
('eight','941','aṣṭā','Turner gives MIA aṭṭha, Marathi āṭh and deaspirated Bengali āṭ/Sinhala aṭa. Source a:ṭə continues this eight family with no aspiration marked.'),
('louse','10512','yūkā','Turner explicitly groups Marathi ū, plural uvā, and Konkani ū̃vva under the y-loss branch of yūkā. Source u:və fits this branch; the suggested taboo motivation in Turner is not asserted as established.'),
('tamarind','4792','ciñcā','Turner section 1 gives Marathi cĩc tamarind from MIA ciṁcā. Source čĩ:cə matches this family, distinct from ciñciṇī (section 2). The remote substrate discussion does not establish a recent Dravidian donor for this form.'),
('calf','11243','vatsarūpa','Turner explicitly gives Konkani vāsrũ young calf and Marathi vāsrū̃ under vatsarūpa. The r-bearing stem selects this compound etymon rather than bare vatsa.'),
('deer','13982','hariṇa','Turner section 2 gives Marathi haraṇ deer. Source hərəṇə matches that animal sense; the existing unsplit 13982 node also contains the adjective yellow, so section 2 is explicitly intended.'),
('hand','14024','hasta','Turner explicitly gives Konkani hātu hand through MIA hattha. The image-reviewed source ha:tu agrees directly.'),
('to laugh','14021','hasati','Turner gives Marathi hasṇẽ/hãsṇẽ and MIA hasaï laughs. Source ha:sə is intransitive to laugh, selecting hasati rather than causative hāsayati.')]
a=[]
for gloss,parent,label,ev in groups:
 for x in short:
  r=x['record']
  if r['Gloss']==gloss and r['ID'] not in assigned:a.append(dict(record=r,parent=parent,kind='reflex',citation=f'CDIAL[{parent}];'+r['Source'],evidence=ev+' Existing source audit explicitly marks this reading image-reviewed. Installed transcription and dialect tags are preserved.'))
reasons={'half':'Source ərdə retains rd where Turner gives Konkani ādo; learned or contact-mediated ardha requires immediate-donor analysis.','blanket':'Source kambəḷi could reflect regional transmission; distinguish inherited kambala-family vocabulary from a Kannada-mediated loan before selecting a relation.','mustard':'Turner explicitly corrects MIA sāsava from sarṣapa 13281 to sārṣapa 13371a. The source sa:səmə additionally has m; resolve its immediate history before linking.'}
held=[dict(record=x['record'],reason=reasons[x['record']['Gloss']]) for x in short if x['record']['Gloss'] in reasons and x['record']['ID'] not in assigned]
(P/'pass016-decisions.json').write_text(json.dumps(dict(accepted=a,held=held),ensure_ascii=False,indent=2)+'\n');(P/'pass016-source-audit.json').write_text(json.dumps([x for x in short if x['record']['ID'] in {z['record']['ID'] for z in a}],ensure_ascii=False,indent=2)+'\n')
lines=['# South Kanara Konkani: first diversity batch','',f'{len(a)} records / {len(a)} reflex links. Every selected reading has an image-reviewed source audit. No rebuild.','','| # | Konkani | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(g,pa,la,ev) in enumerate(groups,1):
 r=next(x['record'] for x in short if x['record']['Gloss']==g);lines.append(f'| {i} | **{r["Form"]}** ‘{g}’ | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['',f'{len(held)} held records:']+[f'- **{x["record"]["Form"]}** ‘{x["record"]["Gloss"]}’: {x["reason"]}' for x in held]+['','Validation and save status: pending.']
(P/'pass016-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'accepted',len(held),'held')

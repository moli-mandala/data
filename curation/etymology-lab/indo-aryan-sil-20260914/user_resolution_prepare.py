import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv')) if r['Status']=='accepted'}
qs=[dict(parent='13519-2',citation='CDIAL[13519.2]',evidence='CDIAL 13519 explicitly leaves many suvarṇa/sauvarṇa gold continuations indistinguishable. The user selected sauvarṇa for these ambiguous cases on 2026-09-14/15; this is an editorial branch choice, not a claim that the local forms prove that derivation.'),dict(parent='587',citation='CDIAL[587]',evidence='Chatterji 1926 Part II, pp. 832–833 derives Gujarati ā “this” from ayam via aya/āa, contrasting the Rajasthani eta paradigm. The user explicitly chose the Gujarati analysis for the Bhil-language survey forms. Their ayam link is provisional about local inheritance versus Gujarati/other Indo-Aryan transmission.')]
a=json.loads((P/'eleventh-decisions.json').read_text());acc=[]
for x in a['held']:
 r=x['record']
 if r['Gloss']!='gold' or r['ID'] in current:continue
 assert 'suvarṇa' in x['reason'] and 'sauvarṇa' in x['reason']
 acc.append(dict(record=r,family=0,parent=qs[0]['parent'],citation=qs[0]['citation'],evidence=qs[0]['evidence']+' The original competing-root hold remains preserved in eleventh-decisions.json; uncertain intra-IA borrowing is retained.'))
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())}
for fid in ['f_b24vhtsq6trag','f_cirlwby2um4z4','f_mdkwky6gitydq']:
 assert fid not in current
 acc.append(dict(record=raw[fid],family=1,parent='587',citation='CDIAL[587]',evidence=qs[1]['evidence']+' The primary grammar pages and prior withdrawal are preserved in chatterji-pronoun-primary.json and proximal-correction.json; the source spelling and nasalization are retained.'))
(P/'user-resolution-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'user-resolution-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[],userInstruction='do saurvarNa. for “this” i’d match gujarati for bhil langs'),ensure_ascii=False,indent=1));(P/'user_resolution_save.py').write_text((P/'sixth_save.py').read_text().replace('sixth-save-decisions.json','user-resolution-decisions.json').replace('sixth','user-resolution'))
print('accepted',len(acc))

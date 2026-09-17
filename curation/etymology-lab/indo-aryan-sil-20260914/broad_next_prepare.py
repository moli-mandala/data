import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[]
def add(parent,gloss,words,ev,kind='reflex'):
 rules.append(dict(parent=parent,citation='CDIAL['+parent+']',glosses=gloss.split('|'),words=words.split('|'),evidence=ev,kind=kind))
add('3677','ash','kʰarani','CDIAL 3677 *kṣāradhānikā explicitly gives Nepali kharāni ashes. These Danuwar, Majhi and Bote forms match the entire documented word, including -rani; inheritance versus regional Nepali transmission remains open.')
add('11493','wind','bataś|bataʃ','CDIAL 11493 *vātatrāsa explicitly gives Bengali bātās and neighboring batās/batāsa wind. The Bengali and Hajong final ś/ʃ spellings fit the eastern sibilant realization of that whole word; the survey transcription is retained.')
add('9353','broken','bʰaŋa|bhaŋŋa|bhaŋa','CDIAL 9353 bhaṅga gives Apabhraṃśa adjectival broken, Bengali bhāṅā to be broken and Gujarati bhā̃gũ broken to pieces. The eastern bhaŋa forms match the simple result form; auxiliary-bearing and additional suffixed responses are not included.')
add('11012','wife|husband','lāḍo|lāḍu|lāḍā|lāḍi|ladi|laɖi|laɖə|ləɽo|lāḍa|laḍa|laḍi|ḷāḍo','CDIAL 11012 *lāḍa gives Kashmiri lāra husband/lörī wife, Kangri lāṛī wife and Western Pahari lāṛā/laṛɔ bridegroom and lāṛī/laṛi bride. The western lāḍ-/laḍ- forms preserve the source etymon’s retroflex stop beside the primary flapped forms. Dental d spellings and initial ḷ are retained as survey variants, without asserting a particular sound law. Gender and the exact marital gloss are preserved; intra-Indo-Aryan transmission remains open.')
add('1577','rainbow','indrụ̄̃|indrụ|indrọ̃|īnān','CDIAL 1577 indradhanuṣ explicitly gives Kalasha indr (oblique indrūna) and Torwali inhān rainbow. The Kalasha indru variants fit the contracted stem, while Torwali īnān has lost the h of inhān; the source vowel and nasalization marks remain intact.')
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,q in enumerate(rules):
  if r['Form'] in q['words'] and r['Gloss'].lower() in q['glosses']:
   acc.append(dict(record=r,parent=q['parent'],family=i,kind=q['kind'],citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is retained.'));break
assert len({x['record']['ID'] for x in acc})==len(acc)
(P/'broad-next-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1)+'\n')
(P/'broad-next-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1)+'\n')
(P/'broad_next_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','broad-next'))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))

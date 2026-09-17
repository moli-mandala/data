import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass107';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='6261',citation='CDIAL[6261]',evidence='CDIAL 6261 *dādda explicitly gives Assamese/Bengali dādā elder brother and Nepali dājyu elder brother. Hajong dada and Nepali-area daju older-brother responses fit this kinship family, retaining source vowel length and loss of the palatal glide. Nursery formation and local Indo-Aryan transmission remain open; no single remote origin is asserted.'),dict(parent='12918',citation='CDIAL[12918]',evidence='CDIAL 12918 saṃdhyā gives Prakrit saṃjhā, Hindi sā̃j(h), Gujarati sā̃j and regional sañjh/sā̃jh evening. The selected sanj/sānj/sañja/sanjhā and hā̃j/hāndz responses fit this family with affricate, aspiration, nasal and initial h/s notation retained. The broader survey evening/afternoon sense remains a qualification; no local transmission route is settled.'),dict(parent='11813',citation='CDIAL[11813]',evidence='CDIAL 11813 *vibhāna gives Assamese/Bengali bihān morning, Nepali biyāna and Kumaoni byān. Eastern bia̯n/bhiyan and Tharu bihan fit this contracted morning family with source glide, aspiration and vowel notation retained. Turner also proposes an alternative MIA vibhāyana formation; the comparative-family link does not choose the deeper formation or settle local Indo-Aryan transmission.'),dict(parent='13290',citation='CDIAL[13290]',evidence='CDIAL 13290 *savāra explicitly gives Gujarati savār morning/dawn and savārũ early. Western Bhil havāre fits this family with initial s weakening to h and the final adverbial vowel retained. It is distinguished from *savēla; local Indo-Aryan transmission remains open.'),dict(parent='13291',citation='CDIAL[13291]',evidence='CDIAL 13291 *savēla gives Awan savēlē in the morning and Punjabi savere/savere-type adverb beside savelā/saverā early. Gojri savere/səbeḷe fit this e-vowel family, retaining the l/r correspondence, source retroflex lateral and v/b difference. This is distinct from Gujarati savār under 13290; local Indo-Aryan transmission remains open.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if (f=='dada' and g=='elder brother (gen)' and r['Language_ID']=='Hajong') or (f=='daju' and g=='older brother'):i=0
 if f in {'sañja','sanjā','sānj','sā̃j','sānjā','sanjhā','hā̃j','hāndz','sənǰhæ'} and g.startswith('evening'):i=1
 if (f in {'bia̯n','bʰiyan'} and g=='morning') or (f=='bihan' and g=='morning (after dawn)'):i=2
 if f=='havāre' and g=='morning':i=3
 if f in {'savere','səbeḷe'} and g=='morning':i=4
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass107-more-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass107_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))

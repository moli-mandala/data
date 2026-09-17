import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass99';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='6849',citation='CDIAL[6849]',evidence='CDIAL 6849 dhūma explicitly gives Awan dhūā̃ and Punjabi dhūā̃ smoke, as well as nasal dhū̃. The Pothwari tṳã/tṳ̃ã and Gojri tū̃o responses fit this family with initial devoicing, source breathy-vowel notation and nasalisation retained. Final a/o variation and local Indo-Aryan transmission remain open.'),
dict(parent='6886',citation='CDIAL[6886]',evidence='CDIAL 6886 *dhauvati explicitly gives Pothwari tō- with low rising tone, beside Lahnda dhovaṇ and Punjabi dhoṇā wash. Survey to̤ matches that regional form closely, with breathy-vowel notation retained; the isolated tṳ retains its rounded vowel-quality difference as a qualification. The specific *dhauvati entry is selected rather than dhāvati 6803; local Indo-Aryan transmission remains open.'),
dict(parent='10666',citation='CDIAL[10666]',evidence='CDIAL 10666 *rahati gives Lahnda rahaṇ, Punjabi rahiṇā and Western Pahari rēhṇā remain, with the addendum explicitly including remain/stop/live. Pothwari re̤ to live is assigned to this stay/remain family with the h/breathy-vowel correspondence and contracted vowel retained. The analysis reads live in the residence/continuance sense, not the separate jīv- be alive family; exact survey context and local Indo-Aryan transmission remain qualifications.'),
dict(parent='3208',citation='CDIAL[3208]',evidence='CDIAL 3208 kukkuṭa gives Lahnda kukkuṛ and Punjabi kukkaṛ cock, feminine chicken/hen forms, and Kashmiri kokur-type vowels. Pothwari kukoṛ and Gojri kokəṛi/ququṛī fit this poultry family with source vowel, q/k and feminine-ending notation retained. The dictionary calls the family onomatopoeic; the link does not settle its deeper history or local Indo-Aryan transmission.')]
acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];l=r['Language_ID'];i=None
 if f in {'tṳã','tṳ̃ã','tū̃o'} and g=='smoke' and l in {'poth','Goj'}:i=0
 if f in {'to̤','tṳ'} and g=='wash' and l=='poth':i=1
 if f=='re̤' and g=='to live' and l=='poth':i=2
 if f in {'kukoṛ','kokəṛi','ququṛī'} and g=='chicken' and l in {'poth','Goj'}:i=3
 if l in {'poth','awan','Goj'} and 'burn' in g and f in {'bal','balnā','bālo','bāḷ','bʰāl'}:
  held.append(dict(record=r,families=[],reason='CDIAL 6654 *dvalati explicitly describes intransitive burning and gives Punjabi balṇā. The survey asks to burn wood, apparently transitively; the response needs verification for a causative, an ambitransitive local use, or an elicitation mismatch. No bare-root assignment is saved solely from the spelling. The multi-gloss bāḷ record additionally mixes hair and burning and needs source-boundary review.',passNumber=99))
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass99-more-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass99_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),len(held),Counter(x['parent'] for x in acc))

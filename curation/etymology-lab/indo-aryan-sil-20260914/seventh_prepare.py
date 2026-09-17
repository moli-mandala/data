import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'sixth_prepare.py').read_text();exec(src.split('qs=[]')[0])
qs=[]
def add(p,g,w,e):qs.append(dict(parent=p,gloss=g,words=w.split('|'),evidence=e,citation='CDIAL['+p.replace('-','.',1)+']'))
add('9092','flower|egg','phul|phula|phuli|phuḷ|ful|fula|phullo|phullu','CDIAL 9092 phulla gives Prakrit phulla “flower”, Hindi phūl, Gujarati/Marathi phūl and Nepali phul “flower, egg”. The lateral variants identify this family. Any intra-Indo-Aryan transmission remains open under the user’s policy; the link does not prove uninterrupted inheritance.')
add('6943','river|small river|river (small)','nadi|nadī|nodi|nadiya|nadiyā|nədi|naddi|nandi|nendi|nondi|nond|nedi|nai|naī|naye|nei','CDIAL 6943 nadī gives Prakrit ṇaī, Punjabi naī, Bengali naï and Western Pahari nei. It separately discusses early nasalization *nandī with Pashai nandī, Gawri nēndi and Torwali ned. Retained-d forms may reflect intra-IA borrowing or learned restoration; the supported family link leaves that transmission unresolved.')
add('9871','die', 'mar|mara|mare|maro|mari|mariyo|maryo|marla|marlo|marli|marab|marna|marṇā|marnu|marṇu|marṇo|marvu|mariba|maribā|maral|mora|morla|mor|muro|muri','CDIAL 9871 marate compares Pali marati, Prakrit maraï, widespread mar- verbs, Bhojpuri maral and Hindi marnā. Turner leaves derivation from Vedic mar- versus remodeling from other verbal forms open. These simple stem/inflection forms are linked to this family; auxiliary-bearing and negative constructions are excluded.')
add('6507-2','see','dekh|dekha|dekho|dekhi|dekhna|dekhnu|dekhṇo|dekhṇu|dekhab|dekhal|dekhla|dekhlo|dekhli|dekhle|dekhat|dekhis|ḍekh|ḍekha|ḍekho|ḍekhlo','CDIAL 6507.2 *dēkṣati explicitly gives Ashokan dekhati, Prakrit dekkhaï, Bhojpuri dēkhal, Hindi dekhnā and Sindhi ḍekhaṇu. The e-vowel and kh select branch 2. The article describes geographical spread; intra-IA transmission is left open. Compound verbs and contracted forms without diagnostic kh are excluded.')
add('6624','run','daur|dauro|daura|dauri|daurab|daurna|dauṛ|dauṛo|dauṛa|dauṛi|dauṛab|dauṛna|doṛ|doṛo|doṛa|doṛi|doṛṇā|doṛṇo|doṛnu|dor|doro|dɔr|dɔḍa|dauḍo|doḍo','CDIAL 6624 dravati explicitly lists the -ḍ- extension in Punjabi dauṛṇā, Hindi dauṛnā, Marwari doṛṇo and Marathi dauḍṇẽ, with Bengali, Nepali and Maithili borrowing noted. These simple run-forms identify that extension; local intra-IA transmission remains unresolved under the user’s policy.')
(P/'seventh-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
# Explicit prompt aliases, preserving full multi-sense records for separate analysis.
aliases={'die':{'die','to die','die (man)','he died','he died.','(he) died','(the man) died',"don't die!, he died",'don’t die/he died','he dies; he died','don’t die!, he died','die! / (he) died'},'see':{'see','to see','see!','he sees/he saw','he sees; he saw','watch/see','watch / see','you see! /','see! / to see','see! / (he) saw'},'run':{'run','run!','to run','run!, he ran','run!; he ran','run/he ran','(you) run!','(you) run','you run!','run! / (he) ran','run! / to run'}}
ws=[{norm(w) for w in q['words']} for q in qs];cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=norm(r['Form']);g=r['Gloss'].lower().strip()
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i,q in enumerate(qs) if w in ws[i] and (g in aliases.get(q['gloss'],set(q['gloss'].split('|'))) or (q['parent']=='9092' and g=='flower; egg'))]
 if ii:cs.append(dict(record=r,families=ii))
# Restore complete immutable record content needed by save validation.
byid={r['ID']:r for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in {x['record']['ID'] for x in cs}}
for x in cs:x['record']=byid[x['record']['ID']]
(P/'seventh-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
for i,q in enumerate(qs):
 rs=[x['record'] for x in cs if i in x['families']];print(i,q['parent'],len(rs));print('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs})))

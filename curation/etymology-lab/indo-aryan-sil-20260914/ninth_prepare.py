import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'sixth_prepare.py').read_text();exec(src.split('qs=[]')[0])
qs=[]
def add(p,g,w,e,loc=None):qs.append(dict(parent=p,gloss=g,words=w.split('|'),evidence=e,citation='CDIAL['+(loc or p.replace('-','.',1))+']'))
add('1388','potato','alu|aḷu|allū|ālu|ālū','CDIAL 1388.1 ālu explicitly compares Hindi/Punjabi ālū, Nepali ālu and Oriya āḷū for potato, transferred from older edible-root senses. It marks some Bihari forms as Hindi loans; intra-IA transmission remains unresolved, while this family link is supported. Pumpkin/gourd forms and ālukī/arwī are excluded.','1388.1')
add('9209-2','father','baba|babu|bab|babo|babb|babba|bavā|bawa','CDIAL 9209.2 *bābba, a nursery-word family, explicitly includes Bengali/Oriya bābā, Nepali/Kumaoni bābu, Shina bābu and Western Pahari bābo/babb. This selects the voiced-b branch separately from *bāppa; nursery-word convergence and regional borrowing remain possible.')
add('6261','older brother|elder brother|father','dada|dado|daddā|dad','CDIAL 6261 *dādda compares Bengali, Assamese, Hindi and Marathi dādā “elder brother”, Kalasha dāda “father” and other older-relative senses. This is a nursery kinship family; the link does not assert that every regional use was inherited without contact.')
add('12335','body','sarir|śarir|śarira|sarira|śorir|sorir','CDIAL 12335 śarīra “body” gives Pali/Prakrit sarīra and Western Pahari sarīr. Conservative sibilants may reflect learned or intra-IA transmission; this provisional family link leaves that route open rather than claiming regular inherited sound change.')
add('6557','body','deh|deha|dehi|dih|diha|de','CDIAL 6557 deha “body” gives Pali/Prakrit dēha, Kashmiri dih, Hindi deh/dehī, Assamese dehā, Oriya diha and Middle Bengali de. These explicit body comparanda support the family; intra-IA transmission is not resolved.')
add('1008','sky','akas|akaś|akāśa|akasa','CDIAL 1008 ākāśa gives Pali/Prakrit ākāsa “sky” and discusses Dardic outcomes separately. Retained-k sky forms identify this lexical family, though learned restoration or intra-IA transmission may account for conservatism; the provisional link does not assert uninterrupted inheritance.')
add('1577','rainbow','indradhanus|indradhanush|indradhanuś|indradhanuṣ|indradhan|indreni|indraini|indrani|indruṇ|indran|idran|indr','CDIAL 1577 indradhanuṣ is itself the historical rainbow compound. It gives Nepali indreni, Kalasha indr and Bshk. idrān; learned full forms retain the historical compound. The link is to that complete compound, with possible learned or cross-IA transmission left open.')
add('6225','thread|rope|string','dor|dora|dori|doro|ḍor|ḍora|ḍori|ḍoro|ḍuri|duri','CDIAL 6225 davara/dōra/ḍōra gives Prakrit dōra/ḍōra, Punjabi ḍorā, Nepali ḍoro “thread” and ḍori “rope”, Bhojpuri ḍorā and Gujarati/Marathi dor/dorī. Both dental and retroflex initials are explicitly represented; their local transmission remains open.')
add('3244','axe','kuraḍ|kuraḍi|kurhaḍ|kurhaḍi|kurāḍ|kurāḍi|kuvaṛi|kuvaḍi|kural|kurali|kuṛal|kuṛali|kuṛari|kurāṛhi|kurāṛha|kuhāṛi|kuhāṛa|kulhāṛi|kulhāṛa|kurhāṛi|kurhaṛ','CDIAL 3244 kuṭhāra gives Prakrit kuhāḍa, Gujarati kuvāṛī, Bengali kuṛāl, Oriya kuṛāla/kurāṛhi, Hindi kulhāṛī and Marathi kurhāḍ. Metathesis is explicit; deeper Dravidian origin is probable rather than settled. Intra-IA transmission does not prevent this family link.')
add('10539','blood','rat|ratt|rath|rāt|rakta|rokto|rokot|ragat|rakat','CDIAL 10539 rakta explicitly means blood and compares Kashmiri rath, Shina rat, Punjabi ratt and related Dardic forms. Conservative rakt- and epenthetic ragat/rokot forms may involve learned or regional transmission; that route remains open. The link asserts the blood family, not uninterrupted inheritance.')
add('9216','child|boy','balak|bāḷak|baḷāk|balakh','CDIAL 9216 includes bālaka, Pali bālaka “boy”, Western Pahari bālak and Nepali bālakha. Turner explicitly leaves the k-extension versus borrowing from Sanskrit bālaka open; the link uses the existing entry containing that extension and preserves the uncertainty.','9216, -kk- extension or Sanskrit bālaka')
add('9331','rice|cooked rice|boiled rice','bhat|bhāt|bhatta|bhatto|bhata|batt|bat|bata|bot','CDIAL 9331 bhakta gives Pali/Prakrit bhatta “food, rice”, widespread bhāt “boiled rice”, Gawri bat and Bshk. batt. Possible intra-IA borrowing is retained; raw/paddy-specific senses and other bāt homonyms are not included.')
add('11072-2','cloth|clothes','luga|lugga|nuga|lugā','CDIAL 11072.2 *lugga explicitly gives Nepali lugā, Oriya lugā/nugā, Bihari lūgā/luggā and Hindi lugā “cloth”. The deeper connection to “broken/defective” formations is doubtful; this link stops at the exact *lugga branch.')
add('5731','palm','tali|taḷi|tal|til|tirī','CDIAL 5731.1 tala includes palm and its feminine forms: Punjabi talī, Shina talī, Sindhi tirī and Bshk. til. Doubled-l tallī belongs to the separate *talla branch and is excluded.','5731.1, feminine palm forms')
(P/'ninth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
ws=[{norm(w) for w in q['words']} for q in qs];cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=norm(r['Form']);gs={g.strip().lower() for g in r['Gloss'].split(';')}
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i,q in enumerate(qs) if w in ws[i] and gs<=set(q['gloss'].split('|'))]
 if ii:cs.append(dict(record=r,families=ii))
ids={x['record']['ID'] for x in cs};byid={r['ID']:r for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in ids}
for x in cs:x['record']=byid[x['record']['ID']]
(P/'ninth-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
for i,q in enumerate(qs):
 rs=[x['record'] for x in cs if i in x['families']];print(i,q['parent'],len(rs));print('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs})))

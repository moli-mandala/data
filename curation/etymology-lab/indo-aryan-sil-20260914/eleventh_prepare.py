import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'sixth_prepare.py').read_text();exec(src.split('qs=[]')[0])
qs=[]
def add(p,g,w,e,loc=None):qs.append(dict(parent=p,gloss=g,words=w.split('|'),evidence=e,citation='CDIAL['+(loc or p.replace('-','.',1))+']'))
add('11813','morning','bihan|bihana|bihane|bihaṇ|biana|biyan|bian|byan|byani|vihān|vihan|vihāṇ','CDIAL 11813 *vibhāna compares Bengali/Bihari bihān, Hindi bihān/bihāne, Nepali biyāna and Kumaoni byān. Turner also allows derivation from vibhāyana, which remains an unresolved deeper alternative within the reviewed morning family.')
add('9402','husband','bhatar|bhatara|bhataru|bhatār','CDIAL 9402 bhartṛ gives Pali bhattā/bhattāraṃ and Assamese, Bihari, Hindi bhatār plus Sindhi bhatāru “husband”. These dental-t forms select the husband family; unrelated Brahman-title or bearer senses are excluded.')
add('9188','broom','buhari|bohari|bahari|buhar|buhara|bahar|buharu','CDIAL 9188 bahukāra/bahukarī gives Prakrit bōhārī, Punjabi buhārī/bahārī and Sindhi buhārī “broom”. The link uses the historical family containing these feminine broom forms; reduced unrelated broom stems are not assumed to belong.')
add('13355','all|whole|whole/all|complete','sara|saro|sare|sari|sarā|saru','CDIAL 13355 sāra explicitly gives Lahnda/Punjabi sārā, Marwari sāro, Gujarati sārũ and Marathi sārā “all, whole”. The article argues for development from “best part”, while recording the competing older account; regional borrowing remains possible.')
add('13276','all|whole|whole/all','sab|sabh|sabha|saba|sabu|sabo|sabbe|sabba|sabbi|sappe|sau|sahu|habh|habha|habba|habb|habbe','CDIAL 13276 sarva gives Pali sabba, Prakrit savva, Hindi/Nepali sab, Punjabi sabh, Oriya sabu and Lahnda habh/habbā. The article explicitly discusses irregular pronoun developments and borrowing in some languages; the supported all/whole family is linked without resolving local transmission.')
add('4424','many','ghana|ghaṇa|ghano|ghaṇo|gana|gaṇa|gano|gaṇo|ghane|ghaṇe','CDIAL 4424.1 ghana explicitly gives Sindhi ghaṇo “many”, Old Gujarati ghaṇauṁ “much” and regional dense/big/many developments. The article separates uncertain Dardic *ghāna/*ghaṇḍa forms; those require a separate branch decision.','4424.1')
add('4564','good','canga|caŋga|cango|caŋgo|canga|cangho|caŋgho|cangla|caŋgla','CDIAL 4564 caṅga gives Punjabi caṅgā, Western Pahari caṅgo and Marathi cā̃glā “good”. Explicit Hindi-to-Nepali/Marathi borrowing and the -l- extension are retained in the article; the provisional family link leaves local transmission open.')
add('6065','broken','tuṭa|tuṭo|tuṭi|tuṭal|tuṭla|tuṭlo|ṭuṭa|ṭuṭo|ṭuṭi|ṭuṭal|ṭuṭla|ṭuṭlo|ṭuṭli|ṭuṭṭa|ṭuṭṭo|ṭuṭṭal|ṭuṭṭla','CDIAL 6065 truṭyati gives Prakrit tuṭṭa “broken”, Bhojpuri ṭūṭal and Hindi tūṭā/ṭūṭā. These simple resultative forms identify that broken-verb family; compounds with additional light verbs are excluded.')
add('5362','tree|bush','jhaṛ|jhaḍ|jhar|jhaṛa|jhaḍa|jhaṛi|jhaḍi','CDIAL 5362.1 jhāṭa compares Prakrit jhāḍa, Gujarati jhāṛ and Marathi jhāḍ for tree/bush. It records a proposed Munda origin without settling it. Retroflex-ṭ and nasalized broom branches remain separate.','5362.1')
add('10648','rope','rassi|rasi|rassa|rasa|raso|ras|rasri|rasari','CDIAL 10648 raśmi “rope” gives Prakrit rassi/rāsi and Punjabi rassī, explicitly transmitted to Hindi and several other IA languages. Under the user’s policy this family is linked while the survey lect’s immediate transmission remains unresolved; no uninterrupted inheritance is asserted.')
add('10286','dust|mud|earth|soil|clay','miṭi|miṭṭi|maṭi|maṭṭi|mati|matti|maṭo|miṭṭī|māṭī','CDIAL 10286 mṛttikā gives Pali mattikā, Prakrit maṭṭī/mittiā, Punjabi/Hindi miṭṭī and eastern māṭi “earth, clay”; its Kachchi addendum explicitly includes “dust”. Clay/earth used for mud is the same material family, with local transmission left open.')
add('997','mother','ai|āi|ayi|āyi|āī|aī|ayi','CDIAL 997 *āī is explicitly a probable nursery word, with Gujarati/Marathi/Assamese āi “mother”. The article warns that Dardic forms may instead continue āryikā; those are held for the exact-parent decision.')
add('9980','buffalo','manj|maŋj|mañj|manjh|mañjh|majjh|majh|maj|manji|manjhi|majhi','CDIAL 9980.1 *mahyā compares Sindhi mañjh, Lahnda mañjh/majjh and Punjabi majjh “buffalo”. Turner records an alternative derivation through mahiṃśī > mahiñjhī; the link retains that uncertainty and selects the buffalo family rather than the long-vowel adjective *māhya.','9980.1')
add('13519','gold','sona|sono|son|sun|suna|suno|suṇ|soṇ|soṇa','CDIAL 13519 explicitly says most modern gold forms cannot distinguish suvarṇa from sauvarṇa and lists them jointly. The corpus has separate persistent parent nodes; the correct subsection is unresolved for these records.')
add('6914-2','nail|fingernail','nakh|nākʰ|nak|nokh|nakhā','CDIAL 6914.2 *nakkha explicitly lists Bshk. nakh and Torwali nōkh. Retained-k forms elsewhere may be conservative/learned nakha or this strengthened branch; local exact-parent evidence is required.')
(P/'eleventh-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
ws=[{norm(w) for w in q['words']} for q in qs];cs=[]
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 w=norm(r['Form']);gs={g.strip().lower() for g in r['Gloss'].split(';')}
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i,q in enumerate(qs) if w in ws[i] and gs<=set(q['gloss'].split('|'))]
 if ii:cs.append(dict(record=r,families=ii))
ids={x['record']['ID'] for x in cs};byid={r['ID']:r for r in json.loads((P/'inventory.json').read_text()) if r['ID'] in ids}
for x in cs:x['record']=byid[x['record']['ID']]
(P/'eleventh-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
for i,q in enumerate(qs):
 rs=[x['record'] for x in cs if i in x['families']];print(i,q['parent'],len(rs));print('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs})))

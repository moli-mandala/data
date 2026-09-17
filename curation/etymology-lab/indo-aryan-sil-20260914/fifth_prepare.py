"""Fifth pass: pronouns, local Dardic comparanda and simple verbal forms."""
import csv,json,re,unicodedata,collections
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[2]
src=(P/'prepare.py').read_text();exec('def norm'+src.split('def norm',1)[1].split('families=json.loads')[0]);base_norm=norm
def norm(s):return base_norm(s.replace('ɡ','g').replace('ǰ','j').replace('č','c').replace('ˈ','').replace('ˌ',''))
qs=[]
def add(parent,gloss,words,ev,loc=None):qs.append(dict(parent=parent,gloss=gloss,words=words.split('|'),evidence=ev,citation='CDIAL['+(loc or parent.replace('-','.',1))+']'))
add('3865','eat|eat!|he ate|to eat|eat!, he ate|Eat! / he ate','khana|khāna|khāṇā|khāṇo|khāṇu|khānu|khābo|khāba|khāb|khāibā|khāiba|khāeb|khāab|khava|khāvu|khāva','CDIAL 3865.1 khādati gives Prakrit khāaï, Hindi khānā, Punjabi khāṇā, Nepali khānu, Gujarati khāvũ and eastern khāibā/khāeb. Only simple verb/infinitive forms are selected here; past stems belonging to khādita and khā-le completive constructions require separate analysis.','3865.1')
add('8209','drink|drink!|he drank|to drink|(you) drink!|drink!; he drank','pi|pi-|pīo|pio|pīnā|pina|pīṇā|pīṇo|pīṇu|piunu|piyunu|piva|pivu|pīvaṇ|piab|piiba','CDIAL 8209 pibati lists Middle Indo-Aryan pi(v)aï, Hindi pīnā, Punjabi pīṇā, Nepali piunu and Khowar/Kalasha/Palula pi-/pī-. Final hyphens on dictionary stems are notation, not morphology. Forms with possible le-auxiliaries, opaque tense endings or causatives are excluded.')
add('6327','older sister|elder sister|sister (older)','didi|dīdī|didiya','CDIAL 6327 *diddā explicitly lists Kumaoni didī, Nepali didi, Bengali/Oriya didi and Hindi dīdī “elder sister”. The nursery/kinship formation is identified without claiming a secure deeper root; extended -iyā and transmission outside the cited area require review.')
add('986','we|we (inclusive)|we (exclusive)|we (inclusive/exclusive)','ham|hami|hamī|āmī|ami|āmhī|ame|asī|asĩ|asa|asā|ispa|ispā|ābi|abi|amo|amõ|mo','CDIAL 986 asmad explains the first-person plural through Middle Indo-Aryan amhē/asmē and instrumental replacement, with Hindi ham, Gujarati ame, Marathi āmhī, Khowar ispā, Kalasha ābi, Torwali mō and Lahnda asā̃. The selected simple pronouns exclude added sab/log and dual-number constructions.')
add('992','I|I (1 person singular)|I (1st person singular)','ā|a|ai|awa|āvā|ya|yā|hu|hũ|hau|haũ','CDIAL 992 aham specifically lists Kalasha ā, Khowar awa, Gawarbati ā, Gawri ya and Torwali ai/ā; western hũ/haũ forms are also given. These are selected only in the named or directly supported groups; oblique-derived m-initial first-person pronouns are not assigned to aham.')
add('972','that|those','o|u|ū|vo|võ|vo|vɔ|vah|wah|vu|vū|va|vā|bo|bɔ','CDIAL 972 asau and its amu- oblique stem are the basis of the remote demonstratives, with Nepali u, Hindi wah/us and Old Marwari vo. This identifies the distant-deictic family for the selected plains forms. Number endings, bare vowel homophony and local proximal/distal contrasts remain relevant; Dardic matches are not generalized from these comparanda.')
add('2530','this|these','yo|yɔ|ye|yi|yī|yah|i|ī|e|ẽ','CDIAL 2530 eṣa/eta gives Nepali yo, eastern i/e, Hindi yah and Punjabi e/eh. The selected proximal demonstratives fit this historical paradigm; the survey’s singular/plural use is retained, and compounds or relative-pronoun j-forms are not collapsed into it.')
add('10511','you|you (pl)|you (plural)|you (plural informal)|you (formal singular/plural)','tum|tūm|tam|təm|timi|tumi|tus|tūs|tusa|tusā|tusã|tussã|tusːã|thā|tha|twa','CDIAL 10511 yuṣmad describes the t-initial paradigm remodeled with tvam: Hindi tum, Nepali timi, Gujarati tame and local Gawri tha, Torwali twa, Indus Kohistani/Chilisso/Palula tus. Lahnda/Punjabi tusā̃ is explicitly in the paradigm. This does not make a short tV singular form automatically a plural reflex.')
add('10104','month|moon','mas|mās|māsa|māh|mah|maha|māhu|māu','CDIAL 10104 māsa explicitly gives Khowar mas “moon, month”, Torwali mah “month”, Lahnda māh and widespread eastern mās “month”. The semantic distinction in each survey is preserved; this is not the homophonous flesh word, and bare mā/mõ contractions are excluded without a local paradigm.')
add('5301-2','moon|month','yūn|yun|yũ|yū̃|yusūn|yūsūn|yusun|yuṇ','CDIAL 5301.2 *yōtsnā specifically lists Gawri yūsun, Torwali yūn and Indus Kohistani yũ “moon, month”, with Palula and Shina yūn. This selects the y-initial branch, whose possible dissimilation from *dyōtsnā remains uncertain, rather than the j-initial jyotsnā branch.')
add('13734','woman|wife','īs|is|istri|iśtri|ištri','CDIAL 13734 strī explicitly lists Khowar istri and Gawri īs/is “woman, wife”. Those named local comparanda support the selected forms; conservative istri outside this area may reflect learned/contact transmission and is held.')
add('1921','water','ū|u|ūk|uk|ūg|ug|ūx|ux|vī|vi|wī|wi|voy|woy','CDIAL 1921.1 udaka explicitly lists Kalasha ūk with oblique ūguna, Khowar uγ, Gawri/Torwali ū and Indus Kohistani/Gowro/Palula wī, Chilisso woy. These local forms select udaka, not ap merely because the word means water.','1921.1')
add('13574-4','sun','sī|si|sīr|sir|swīr|swir','CDIAL 13574.4 sūrī, although its Sanskrit head gloss names the sun’s wife, explicitly assigns Gawri sīr, Torwali sī and Indus Kohistani swīr “sun” to this feminine branch. This is distinct from sūriya in section 3; the exact parent was checked against the full article rather than its short gloss.','13574.4')
(P/'fifth-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1))
linked={r['Child_ID'] for r in csv.DictReader((ROOT/'cldf/edges.csv').open()) if r['Rank']=='1'};linked.update(r['Form_ID'] for r in csv.DictReader((ROOT/'data/etymology-assignments.csv').open()) if r['Status']=='accepted' and r['Rank']=='1')
held={x['record']['ID'] for f in ['decisions.json','second-decisions.json','third-decisions.json','fourth-decisions.json'] for x in json.loads((P/f).read_text())['held']};excluded=linked|held
ws=[{norm(w) for w in q['words']} for q in qs];ss=[set(g.lower() for g in q['gloss'].split('|')) for q in qs];cs=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] in excluded:continue
 w=norm(r['Form']);gl={g.strip().lower() for g in r['Gloss'].split(';')}
 if re.search(r'[,;/ ()]',w):continue
 ii=[i for i in range(len(qs)) if w in ws[i] and gl<=ss[i]]
 if ii:cs.append(dict(record=r,families=ii))
(P/'fifth-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1))
with (P/'fifth-review.txt').open('w') as f:
 for i,q in enumerate(qs):
  rs=[x['record'] for x in cs if i in x['families']];f.write(f"\n{i}. {q['parent']} {q['gloss']} ({len(rs)})\n");f.write('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs}))+'\n')
print('Candidates',len(cs))

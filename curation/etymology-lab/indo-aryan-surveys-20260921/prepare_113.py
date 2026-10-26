import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[]
for r in rs:
 if r['ID'] in done:continue
 f,g,l=r['Form'],r['Gloss'],r['Language_ID'];x=None
 if g=='mango' and f in ['ā̃p','ãp','āmbū̃','āmbū','āmm','āmi'] and l in ['Majhi','DewasDoneDanuwar','Dang','Bhili','Rathawi','PalyaBareli','kaithal']:
  x=('1268','CDIAL[1268]','Turner āmra explicitly includes Nepali ā̃p, Gujarati/Marathi ā̃b/ā̃bā, Hindi ām and Sindhi āmī mango. These Terai, western and Haryana survey forms match the corresponding mango family, including the documented nasal-final ā̃p outcome. Compound answers and the distinct kairī family are excluded.')
 if g=='buffalo' and f in ['bʰæsīyā','bʰæ̃sanīyā','bʰõysa']:
  x=('9964','CDIAL[9964]','Turner mahiṣa explicitly gives Nepali bhaĩsi, Hindi bhaĩsī/bhaĩsā and Oriya bhaĩs/bhaĩsā buffalo. The Tharu extended bhaisi-yā/bhaisan-iyā forms and Halbi bhõysa preserve that diagnostic bhais stem. This is a family-level comparison allowing local nominal extensions; their suffixes are not reconstructed as separate Sanskrit roots.')
 if g=='village' and f in ['gāũ','gav','gāme','gãvo','gãve̯o','gṛā̃','gṛā̃(n)','ˈgṛā̃','gəṛā̃','gəṛō̃']:
  x=('4368','CDIAL[4368 and addenda]','Turner grāma explicitly gives Nepali gāũ, western Pahari grā̃/graũ, Gujarati gām and Marathi gāv village. These survey forms fit the corresponding regional village stems and inflectional vowels. Anomalous gʰoṛā̃ and velar-final gaŋ are excluded rather than assumed to have regular outcomes.')
 if g=='village' and f in ['gā̃vḍɔ','gamḍā','gamḍo','gãvḍo']:
  x=('4368','CDIAL[4368,extension -ṭa]','Turner grāma explicitly includes the extended family grāmaṭikā, Prakrit gāmaḍa village and Gujarati gāmṛũ small village. The western survey gamḍ-/gāvḍ- village forms are compared with this documented extension, not with grāmārdha neighbourhood solely because of similar consonants.')
 if x:a.append(dict(record=r,parent=x[0],kind='reflex',citation=x[1],evidence=x[2]))
(P/'pass113-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=1)+'\n');print('accepted',len(a))

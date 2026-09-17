from research_helpers import Batch
b=Batch(8);a=b.add
specs=[
('one','2462-2',{'Malvi':'ek|ik','Nimadi':'ek','Bagheli':'ek'},'CDIAL 2462.2 specifically assigns Hindi/Gujarati/Marathi ek and related forms to *ēkka. Turner leaves emphatic doubling versus an early Middle Indo-Aryan Sanskrit loan unresolved; the immediate *ēkka link preserves that uncertainty.'),
('two','6648',{'Malvi':'do|du','Nimadi':'dui|duy|do','Bagheli':'ḍo|ḍu|ḍui|ḍuy|ḍuyi'},'CDIAL 6648 compares Prakrit dō/duvē, Hindi and Marwari do, and Bihari/Awadhi dui. Both contracted do and dui-type forms belong to the same numeral family.'),
('three','5994-3',{'Malvi':'tin','Nimadi':'tin|tiṇ','Bagheli':'ṭin|ṭiney|ṭiṇ'},'CDIAL 5994.3 trīṇi gives Prakrit tiṇṇi, Hindi/Marwari tīn and Awadhi tīnⁱ. This selects the neuter-form branch, not the masculine trayaḥ head.'),
('four','4655-2',{'Malvi':'car|cār','Nimadi':'cār','Bagheli':'car|carey|cari'},'CDIAL 4655.2 catvāri gives Apabhramsha cāri, Hindi/Gujarati cār, Old Marwari cyāri and Awadhi cāri. Turner explicitly notes unusual contraction or analogical influence in the numeral; branch 2 is the matching source form.'),
('five','7655',{'Malvi':'pāñc|pānc|pā̃c','Nimadi':'pāñc|pā̃c|pānc','Bagheli':'pẽc|paṇcey|pac|pā̃ch'},'CDIAL 7655 compares Prakrit paṁca and Hindi/Marwari/Gujarati/Awadhi pā̃c. The family is supported by the labial onset and palatal stop with nasal-consonant/nasal-vowel variation.'),
('six','12803-3',{'Malvi':'che|ce|cho','Nimadi':'chau|cau|che|chāu','Bagheli':'ce|cey|cə|che|cheu'},'CDIAL 12803.3 specifically groups Pali/Prakrit cha, Punjabi che, Maithili chao and Marwari cha under *kṣaṭ or *kṣvaṭ. Preserve both reconstructions: the c/ch forms belong to this third branch, not root ṣaṣ or the *ṣuvaṭ branch.'),
('seven','13139',{'Malvi':'sat|sāt','Nimadi':'sat','Bagheli':'saṭ|saṭey|saṭe'},'CDIAL 13139 gives Pali/Prakrit satta and Hindi/Marwari/Gujarati/Awadhi sāt. Assimilation of pt and simplification of the resulting geminate support the numeral.'),
('eight','941',{'Malvi':'āṭ|āṭh','Nimadi':'āṭh','Bagheli':'aṭh|aṭey'},'CDIAL 941 gives Pali/Prakrit aṭṭha, Hindi/Marwari/Gujarati āṭh and Bengali āṭ. These establish the retroflex stop and the attested aspirated/unaspirated outcomes.'),
('nine','6984',{'Malvi':'no|nau','Nimadi':'nau|no|nāu','Bagheli':'neu|nev|nuey'},'CDIAL 6984 nava ‘nine’ compares Hindi/Awadhi nau, Old Marwari nova and Gujarati/Marathi nav. This is the numeral entry, distinct from nava ‘new’. Contracted and v-bearing variants belong to the same family.'),
('ten','6227',{'Malvi':'das|daś|dəs','Nimadi':'das|dās','Bagheli':'ḍes|ḍesey'},'CDIAL 6227 compares Prakrit dasa/daha and Hindi/Marwari/Gujarati/Awadhi das, with ś retained in some other regions. The numeral is daśa, not the derived daśaka or unrelated daṁśa.'),
('twenty','11616',{'Malvi':'bis|vis|viś','Nimadi':'bis','Bagheli':'bis'},'CDIAL 11616 gives Prakrit vīsaṁ/vīsā, Hindi/Bhojpuri/Awadhi bīs, Old Marwari bīsa and Gujarati/Marathi vīs. Both b- and v- initials are directly represented in the regional numeral.'),
('one hundred','12278',{'Nimadi':'so|sau|sāu','Bagheli':'sev|sau|so'},'CDIAL 12278 compares Prakrit saya/saa, Hindi/Old Marwari sau and Gujarati sɔ. Loss of intervocalic t with subsequent vowel contraction supports the hundred family.')]
for gloss,parent,langs,ev in specs:
 for l,w in langs.items():
  qualified=gloss in ['one','six'] or (l=='Bagheli' and gloss not in ['one','twenty'])
  extra=' Bagheli’s source-specific vowel, retroflex-stop or final-ey variants are retained; their exact local development remains to be checked.' if l=='Bagheli' and gloss not in ['one','twenty'] else ''
  a(l,gloss,w,parent,ev+extra,tier='qualified' if qualified else 'straightforward')
a('Malvi','four','śar','4655-2','Compare Hindi cār and Assamese sāri in CDIAL 4655.2. The source ś- may reflect local palatal sibilantisation, but this needs a Malvi correspondence check; the Assamese comparison establishes possibility, not a donor.',tier='qualified')
a('Malvi','five','pãs','7655','CDIAL 7655 compares the usual central pā̃c and s-bearing Assamese pā̃s. Malvi pãs fits the family provisionally, but the local affricate-to-sibilant development remains to be verified.',tier='qualified')
a('Malvi','seven','hat','13139','The vowel and final t fit the sāt family in CDIAL 13139, but initial h requires a local s-to-h correspondence. Do not use the dictionary’s Sinhalese h-form as proof of direct transmission.',tier='qualified')
a('Malvi','one hundred','ho','12278','Compare Gujarati sɔ and Old Marwari sau in CDIAL 12278. The h-onset could match the same local s-to-h pattern suggested by hat ‘seven’, but requires independent dialectal confirmation.',tier='qualified')
for pp in b.proposals.values():
 for x in pp:
  if x['parentId']=='12803-3':x['parentForm']='*kṣaṭ / *kṣvaṭ';x['alternatives']=[{'parentForm':'*kṣvaṭ','reason':'Explicit alternative reconstruction in CDIAL 12803.3; shared existing node.'}]
b.save()

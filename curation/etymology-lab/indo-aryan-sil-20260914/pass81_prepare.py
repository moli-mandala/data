import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass81';assert not (P/(stem+'-decisions.json')).exists();done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
spec={
'1577':'CDIAL 1577 indradhanuṣ explicitly means rainbow. Dhundari īndradanūs retains the full compound and is linked as learned Sanskrit vocabulary, with deaspiration and vowel length retained and intra-Indo-Aryan transmission open.',
'114':'CDIAL 114 aṅga explicitly means limb/body and gives regional aṅg. Bundeli ang body fits this noun without adding an unsupported suffix.',
'10434':'CDIAL 10434 yavanāla explicitly gives Hindi jundrī millet. Survey jundari millet (husked) fits this millet noun with an inserted vowel and the processing qualification preserved.',
'5239':'CDIAL 5239 jīva gives regional jiu/jiw life and explicit mind/heart senses in its addenda. Dang jiw heart is compared with that vital-spirit noun, preserving the elicited heart sense rather than claiming it is always anatomical.',
'2963':'CDIAL 2963 kavāṭa gives Hindi kiwāṛ door and regional stop-bearing kavāḍ forms. Buksa kiwaḍ fits the same whole door noun with stop/flap and vowel details retained.',
'6459':'CDIAL 6459 *duvāra gives Nepali duwār and regional duār door. Dang dawar fits that expanded door stem with the unstressed vowel retained.',
'5523':'CDIAL 5523 *ḍag section 1 explicitly gives Oriya ḍagara road and regional ḍagar. Dang ḍagara path matches the retroflex branch, not dental dag in section 3.',
'9882':'CDIAL 9882 markaṭa explicitly gives Khowar mukuḷ monkey. Survey mūkūl directly matches that whole word, with lateral quality and vowel length preserved.',
'5070-2':'CDIAL 5070.2 *chokhara explicitly gives Awan chòr boy, beside Lahnda chohur. Saraiki cor fits this regional contracted boy stem, with initial aspiration/tone notation preserved rather than imposed.',
'13676':'CDIAL 13676 stabdha section 2 documents the cold sense and the influenced thaṇḍha/ṭhaṇḍha forms, including Hindi/Punjabi ṭhaṇḍā and Lahnda ṭhaḍā. Awan ṭhanḍā and Saraiki thaḍa fit that cold family; their dental/retroflex details and the deeper contact influence noted by Turner remain qualified.',
'1340-2':'CDIAL 1340.2 *ālla explicitly gives Bashkarik āl wet and neighboring ālo. Bashkarik ālū selects this contracted lateral wet family with the final vowel retained.',
'13139':'CDIAL 13139 sapta explicitly gives Bashkarik sat and Torwali sāt seven. Survey sāʔt/sātʔ fits this numeral, with glottal articulation preserved rather than interpreted as an extra morpheme.',
'1388':'CDIAL 1388 ālu explicitly gives Bihari aluā potato, noting Hindi transmission. Danuwar alua matches that expanded potato noun; the regional route remains open.',
'10636-2':'CDIAL 10636 explicitly gives Nepali rāmro good and Western Pahari rāmṛā under its -ḍ extension. Jambu promotes this to 10636-2 *ramyaḍ-. Dotyali rāmaro fits that extension with an inserted vowel retained.'}
reason={
'8056':'Buksa paj leg requires checking whether final j represents a glide or an affricate. The pāda article gives pāy/pāi forms but does not establish this local transcription equivalence.',
'2668':'Bashkarik kāṇṭ/kãṇṭ thorn can relate to kaṇṭa or kaṇṭaka; the primary article separates those branches but lacks this exact Bashkarik form. Establish the suffix history before selecting the bare head.',
'941':'Torwali/Chiliss āth eight has a dental t in the source while the regional aṣṭa comparanda are retroflex. Check the survey transcription rather than silently flattening this distinction.',
'12732':'Goj nando/nanḍo small adds a stop absent from the nāno/nannā comparanda under ślakṣṇa. Investigate a local nasal-stop development or another smallness stem.',
'10188':'Danuwar muṭuk heart has final k absent from the cited muṭu form. Establish the extension or full local noun before attaching it to the bare muṭṭa family.',
'11515':'The Danuwar bankar/baŋar monkey forms have velar material absent from the bānar/bāndar vānara series. Compare the maṅkaḍa/markaṭa family and possible crossing or source issues.',
'4008':'Dotyali gāiyo resembles the gayo past of go under gata, but its survey gloss is bare go. Check whether this is a past form or a directive with additional morphology before choosing a participial ancestry.'}
rules=[dict(parent=p,citation=('CDIAL[10636, -ḍ extension]' if p=='10636-2' else 'CDIAL['+p.replace('-','.')+']'),evidence=e) for p,e in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];p=ps[0]
 if r['ID'] in done or len(ps)>1:continue
 if p=='5070':p='5070-2'
 if p in reason:held.append(dict(record=r,families=[],reason=reason[p],passNumber=81))
 elif p in ix:
  q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind=('borrowed' if p=='1577' else 'reflex'),citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass81_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})

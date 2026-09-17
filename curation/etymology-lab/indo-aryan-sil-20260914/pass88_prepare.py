import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass88';assert not (P/(stem+'-decisions.json')).exists()
spec={
'9238-2':('CDIAL[9238.2]','CDIAL 9238.2 *bēṭṭa explicitly gives Bengali beṭā son/beṭi daughter, Maithili beṭā/beṭī, Nepali beṭo/beṭi, Gujarati/Marwari beṭo/beṭī and Western Pahari beṭṭɔ/beṭi. The selected beta/beti pairs fit this specific child branch rather than the defective head. Survey dental t, including explicit dental diacritics, is preserved as a local transcription/phonological qualification; the link does not silently change it to ṭ or resolve regional borrowing.'),
'4701-2':('CDIAL[4701, -ḍa extension]','CDIAL 4701 places Bengali cāmṛā, Bihari camṛā, Hindi camṛā and the regional camṛī skin forms in its -ḍa extension, canonical 4701-2 *carmaḍa-. The selected camra/camri forms preserve that extended noun, with r/ṛ articulation and vowel length retained as local details; transmission within Indo-Aryan remains open.'),
'4661':('CDIAL[4661.2]','CDIAL 4661.2 candra moon explicitly gives Lahnda cann/can and Punjabi cann. Pothwari caṇ fits that contracted regional moon family, preserving the survey retroflex nasal notation and leaving the contact route open.'),
'5239':('CDIAL[5239]','CDIAL 5239 jīva gives Western Pahari jiu mind/heart in its addenda and Jaunsari jiū mind/person, alongside Oriya ji heart. Jaunsari jiw heart fits the vital-spirit/heart noun; the elicited gloss is preserved without claiming a uniformly anatomical meaning.'),
'6770':('CDIAL[6770.1]','CDIAL 6770.1 *dhāgga gives regional dhāgā thread, explicitly Jaunsari dhāgā in the addenda. Jaunsari daga and Mewari dāga fit the voiced thread family with absent aspiration retained as a local qualification. The deeper *dhārga/*dharga revision in the addenda and internal Indo-Aryan transmission remain open.'),
'1577':('CDIAL[1577]','CDIAL 1577 indradhanuṣ means rainbow. Mewari indradanuś preserves the full learned compound, with deaspiration and final sibilant articulation retained. It is linked as learned Sanskrit vocabulary with intermediate Indo-Aryan transmission open.'),
'13676':('CDIAL[13676.2]','CDIAL 13676 section 2 explicitly gives Hindi/Punjabi ṭhaṇḍā cold and discusses influence on the thaṇḍha/ṭhaṇḍha series. Pothwari ṭʰanḍa fits this cold family, with the source nasal notation retained; the deeper crossing and local transmission are not independently resolved.')}
rules=[dict(parent=p,citation=c,evidence=e) for p,(c,e) in spec.items()];ix={r['parent']:i for i,r in enumerate(rules)};inventory={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};acc=[]
for x in json.loads((P/'pass88-expansion-candidates.json').read_text()):
 if len(x['parents'])!=1 or x['parents'][0] not in ix:continue
 p=x['parents'][0];r=inventory[x['record']['ID']];q=rules[ix[p]];acc.append(dict(record=r,parent=p,family=ix[p],kind='borrowed' if p=='1577' else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass88_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc)})

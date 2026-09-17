import json,csv
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'2665':'CDIAL 2665 explicitly gives Punjabi/Lahnda kaṇak wheat. The survey label wheat (husked) is a preparation qualifier and does not change the identified grain word.',
'4445':'CDIAL 4445 gharma explicitly gives ghām heat of the sun and sunshine. These ghām/gham hot (weather) responses identify ambient heat; their source elicitation wording is retained.',
'10437':'CDIAL 10437 yavākāra explicitly gives Hindi juwār/joār sorghum and regional juār millet. The source millet (husked) label is retained as a preparation qualifier.',
'12135':'CDIAL 12135 vesavāra explicitly gives Nepali besār turmeric. The survey tumeric spelling is the same gloss; an altered sibilant in beśar needs a separate local check.',
'5090-4':'CDIAL 5090.4 *jūḍa explicitly gives Maithili jūr and Oriya juṛa cold. Kochila jurə cold (water) matches this u-vowel branch; the water qualifier does not change the adjective.',
'11250':'CDIAL 11250 vadhū explicitly gives Maithili bahu wife and Hindi bahū bride/wife. The survey breathy-vowel bahṳ keeps the same segmental form and family; voice quality is preserved.',
'9237':'CDIAL 9237 biḍāla section 1 explicitly gives Bengali/Oriya bilāi cat. Survey bilaⁱ records the final vowel as superscript; it fits this branch without source normalization.',
'10990-2':'CDIAL 10990.2 *raśuna explicitly gives Bengali rasun garlic and regional r-initial variants. Hajong rosun fits this branch with vowel colouring retained.',
'10991':'CDIAL 10991 *laṣṭi explicitly gives laṭṭhi/laṭṭhī stick. The unreleased first stop in laṭ̚ṭʰi is a phonetic detail of the same cluster, not a different root.',
'28':'CDIAL 28 akṣata explicitly gives Gujarati ākhũ and Konkani ākho whole. Dungra Bhili āːkho retains redundant length notation but matches that whole/unbroken family.',
'9250':'CDIAL 9250 bīja explicitly gives Bihari bīyā and regional biya seed. Survey biʲa uses a superscript glide in this corresponding seed form.',
'10984':'CDIAL 10984 *lavaṇḍa explicitly gives Kumaoni laũṛo/lauṛo boy. Kathoriya lãᵘra has the documented nasalized diphthong and flapped/dental r realization preserved in the source.'}
allx={}
for file in ['phonetic-third-expansion-candidates.json','semantic-third-expansion-candidates.json']:
 for x in json.loads((P/file).read_text()):
  if len(x['parents'])==1 and x['parents'][0] in E:allx[x['record']['ID']]=x
acc=[];held=[];rules=[];ix={}
for x in allx.values():
 k=x['parents'][0];r=x['record'];ev=E[k]
 if k not in ix:ix[k]=len(rules);rules.append(dict(parent=k,citation='CDIAL['+k.replace('-','.',1)+']',evidence=ev))
 i=ix[k]
 if r['Form']=='beśar':held.append(dict(record=r,families=[i],reason='Jaunsari beśar turmeric requires confirming s/ś variation locally; the primary besār comparison alone does not establish it.',passNumber=54));continue
 acc.append(dict(record=r,parent=k,family=i,citation=rules[i]['citation'],evidence=ev+' Intra-Indo-Aryan transmission remains open.'))
(P/'expansion-third-rules.json').write_text(json.dumps(rules,ensure_ascii=False,indent=1));(P/'expansion-third-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'expansion_third_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','expansion-third'))
(P/'expansion_third_decide.py').write_text(Path(__file__).read_text());print(len(acc),len(held))

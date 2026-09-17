import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass97';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[
dict(parent='14021',citation='CDIAL[14021]',evidence='CDIAL 14021 hásati explicitly gives Kumaoni hãsṇo, Maithili hasab/hãsab, Bhojpuri hãsal and Hindi hasnā. The bare has/hãs laugh responses fit this verb, with nasalisation variation explicitly documented in the family. No additional negative, auxiliary or suffix-bearing material is included in these targets. Local Indo-Aryan transmission remains open.'),
dict(parent='2854',citation='CDIAL[2854]',evidence='CDIAL 2854 kártati gives Prakrit kaṭṭai, Nepali kāṭnu, Bengali kāṭā and Hindi kāṭnā cut; Turner explicitly distinguishes the retroflex cutting family from the dental spinning family. The selected bare kaṭ and Bengali kaʈa fit this cutting verb. Source vowel length and ordinary Bengali verbal ending are retained; local Indo-Aryan transmission remains open.'),
dict(parent='12370',citation='CDIAL[12370]',evidence='CDIAL 12370 śāka potherb, vegetable gives Prakrit sāga/sāya and Pali sāka. The survey sag/sāg vegetable responses retain the voiced stop of the sāga branch; the isolated sãg nasalisation is preserved as a regional qualification. This is a comparative vegetable-family link, with local Indo-Aryan transmission unresolved.'),
dict(parent='1341',citation='CDIAL[1341]',evidence='CDIAL 1341 ārdraka explicitly gives Nepali aduwā dried ginger, alongside Kumaoni ādo, Bengali/Assamese ādā and Gujarati ādũ ginger. The selected aduva/əduwa responses fit the Nepali-type extended form; survey ginger does not specify dried versus fresh. The initial reduced vowel and v/w notation are retained, with local Indo-Aryan transmission unresolved.'),
dict(parent='12459',citation='CDIAL[12459, sense 1]',evidence='CDIAL 12459 śilā sense 1 gives Assamese xil stone/hailstone and Bengali sil flat grinding stone/hail. Eastern hil/sil stone is assigned to this family, with sibilant weakening to h explicitly retained as the regional sound qualification. This uses the eastern xil/sil comparanda, not Maldivian hila in the separate *śillā addendum. The general stone gloss and local Indo-Aryan transmission remain open.'),
dict(parent='2814',citation='CDIAL[2814]',evidence='CDIAL 2814 karóti gives Prakrit karai, Bengali karā, Assamese kariba and Oriya karibā do/make. The selected Hajong kor/kora/kore/kura and Bengali kɔɾa fit this verb with ordinary final-vowel variation and eastern rounded vowels retained. These are simple verbal responses, not analysed as causatives or multipart constructions. Local Indo-Aryan transmission remains unresolved.')]
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=None
 if f in {'has','hãs'} and g=='laugh':i=0
 if f=='kaṭ' and g=='cut' or f=='kaʈa' and g=='to cut (something)' and r['Language_ID']=='B':i=1
 if f in {'sag','sāg','sãg'} and g=='vegetable':i=2
 if f in {'aduva','əduwa'} and g=='ginger':i=3
 if f in {'hil','sil'} and g=='stone' and r['Language_ID'] in {'Hajong','Bishnupriya'}:i=4
 if f in {'kor','kora','kore','kura','kɔɾa'} and g=='to do/make' and r['Language_ID'] in {'Hajong','B'}:i=5
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a.update(json.loads((P/'pass97-kar-primary-articles.json').read_text()));f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/'pass97_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
from collections import Counter
print(len(acc),Counter(x['parent'] for x in acc))

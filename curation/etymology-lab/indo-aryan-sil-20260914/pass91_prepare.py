import json,csv
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass91';assert not (P/(stem+'-decisions.json')).exists()
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())}
rules=[dict(parent='11225',citation='CDIAL[11225];CDIAL[9661]',evidence='The older-brother expression has big/elder + brother. CDIAL 11225 vaḍra explicitly gives baṛā/baḍā/vaḍḍā big and elder usage; CDIAL 9661 gives bhrā/bharā and bhāi/bhāū brother. Ordered component-family edges preserve the complete phrase, including source labial, aspiration, rhotic and vowel notation. The vaḍra addenda discuss extraction from evaḍa-type size words rather than a simple vṛddha derivation; the component link preserves that uncertainty and does not assert a single ancient compound.'),dict(parent='11225',citation='CDIAL[11225];CDIAL[9349]',evidence='The older-sister expression has big/elder + sister. CDIAL 11225 supplies baṛā/baḍā/vaḍḍā with elder usage and feminine agreement; CDIAL 9349 gives bhēṇ/bhaiṇ, bahin/bahan and related regional sister forms. Ordered component-family edges preserve the complete phrase and source p/bh, nasal, breathy and vowel notation. The debated deeper origin of vaḍra and local Indo-Aryan transmission remain open; no single ancient compound is asserted.')]
bs={'badɔbʰāī','baḍābʰāī','baḍā bʰāī','baḍobʰāū','baḍo bɦāi','bəḍa pra','baṛa bāi','baḍːa bʰai','baḍa bʰayi','baḍobhāi','baṛa pra̤','vaḍːa pra̤','vaḍa pra̤'}
ss={'baḍībahaṇ','baṛībahaṇ','baḍī bahɛn','baḍībahin','baḍī bahīn','baḍī bahin','baḍībɛn','baḍi bahani','baḍi bahin','baḍi bayeṇ','bəḍːi pǣṇ','baṛi beiṇ','baṛi peṇ','baḍːi bahaṇ','baḍi behen','baḍi bahaṇ','baṛi pe̤n','baṛi pe̤ṇ','vaḍi pe̤ṇ'}
acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 f=r['Form'];g=r['Gloss'];i=0 if f in bs and g=='older brother' else 1 if f in ss and g=='older sister' else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],components=['11225','9661' if i==0 else '9349'],family=i,kind='component',citation=q['citation'],evidence=q['evidence']+' Exact response: '+f+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass91_save.py').write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem))
print({'records':len(acc),'rows':len(acc)*2})

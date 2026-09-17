from research_helpers import Batch,P
import json
b=Batch(25)
def a(l,g,w,p,e,t='qualified',senses=None):
 rs=[r for r in b.inv[l] if r['ID'] not in b.used and r['Form'] in w.split('|') and (set(r['Gloss'].split('; '))<=set(senses) if senses else r['Gloss']==g)]
 if rs:b.add(l,g,'|'.join(dict.fromkeys(r['Form'] for r in rs)),p,e,tier=t,source_glosses=list(dict.fromkeys(r['Gloss'] for r in rs)))
for l,w in [('Malvi','kal|kāl|kale'),('Nimadi','kal|kāl'),('Bagheli','kal|kāl')]:
 a(l,'yesterday; tomorrow',w,'3104-2','CDIAL 3104.2 kalya gives Prakrit kallaṃ/kalhiṃ and Hindi kal for yesterday/tomorrow, with long-vowel forms in Awadhi, Bengali, Gujarati and Marathi. The two temporal senses are explicitly compatible; source records retain their distinct elicited meanings. The specific short-vowel Sanskrit branch is selected, while local vowel length and final-e need review.',senses=['yesterday','tomorrow'])
a('Malvi','father','kākā','2998','CDIAL 2998 *kākka is a senior-male-relative family, with Kashmiri kākas ‘one’s own father’ and Hindi/Gujarati kākā ‘father’s brother’. Turner proposes a Dravidian source; local kinship extension and possible contact mean this is a qualified family assignment, not a secure deep inherited reconstruction.')
a('Malvi','mother','āi','997','CDIAL 997 *āī gives Gujarati āi and Marathi āī ‘mother’, with comparable kinship uses across Indo-Aryan. Turner calls it probably a nursery word and distinguishes a possible āryikā derivation for Dardic forms; no automatic āryikā ancestry is assigned here.')
for l,w in [('Malvi','dadā|dādo'),('Nimadi','dādo')]:
 a(l,'father; older brother',w,'6261','CDIAL 6261 *dādda covers father and senior relatives: Pashai/Kalasha father, Hindi father’s father/elder brother, Gujarati dādɔ grandfather. The survey’s father/elder-brother senses are compatible with this variable kinship family, while expressive renewal and regional semantic differences remain qualified.',senses=['father','older brother'])
for l,w in [('Malvi','jiji|jijā'),('Nimadi','jiji'),('Bagheli','jiji|jidyi|jiyyi')]:
 a(l,'mother; older sister',w,'5232','CDIAL 5232 *jījja is explicitly a nursery family for breast, mother and relatives; Gujarati jijī means mother and Hindi/Marathi jijī elder sister, while Sindhi jījā/jījī is affectionate address for mother/aunt. These justify the compatible source senses, but local endings, palatal variants and contact or expressive renewal remain qualified.',senses=['mother','older sister'])
b.save()
h=json.loads((P/'holds.json').read_text());reopened=[]
for xs in b.proposals.values():
 for x in xs:
  for fid in x['formIds']:
   if fid in h:reopened.append({'id':fid,'previousHold':h.pop(fid),'resolution':x['evidence'],'proposal':x['number']})
(P/'batch-025-reopened-holds.json').write_text(json.dumps(reopened,ensure_ascii=False,indent=2))
(P/'holds.json').write_text(json.dumps(h,ensure_ascii=False,indent=2));print('Reopened holds',len(reopened))

import json
from pathlib import Path
P=Path(__file__).resolve().parent
rs=json.loads((P/'remaining-initial.json').read_text())
done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())}
forms={'ṭʰaṇḍo','ṭʰaṇḍa','taṇḍʰā','tʰanḍa','tanḍā','ṭʰaṇḍā','ṭʰaṇdā','ṭʰaṇḍe','tʰaṇḍo','ṭʰanḍo','ṭʰando','tʰando','tʰanḍo','tʰanḍʰ','taṇḍo','tʰanḍā','ṭʰanda','tʰanda','tʰandʰā','tʰānḍā','tʰənda','thanḍa','tānḍɔ','ṭhanḍo','thaṇḍa','tɔṇḍa','tʰãḍo','ṭhaṇḍo','thaṇḍā','ṭhaṇḍa','tʰaṇḍu','thanḍo','t̪ʰəṇḍə','ˈt̪ʰəṇḍə','thāṇḍā','thāṇḍū','thaṇḍo'}
a=[];h=[]
for r in rs:
 if r['ID'] in done:continue
 if r['Gloss'] in ['cold','cold (water)','cold (weather)'] and r['Form'] in forms:
  loan=r['Language_ID'] in {'awan','poth','jaun','kul','dog'}
  ev=('Turner explicitly marks Hindi ṭhaṇḍā as the source of Punjabi and several West Pahari cold forms. The existing Hindi comparandum is therefore used as the immediate donor for this northwestern survey form, with local transmission provisional where the exact survey variety is not named. Ordinary final agreement is retained, and the -ha extended Kullu response is excluded.' if loan else 'Turner places the nasal ṭhaṇḍ-/thaṇḍ- cold family under stabdha, with explicit Nepali, Bengali, Oriya, Bhojpuri, Awadhi, Gujarati and Marathi comparanda and proposed Dravidian influence on the nasal formation. This is a qualified comparative-family link, not a claim of unmodified sound-law descent or exclusion of regional contact; ordinary dental/retroflex transcription and final agreement are preserved.')
  a.append(dict(record=r,parent='f_7s6lwotgtunly' if loan else '13676',kind='borrowed' if loan else 'reflex',citation='CDIAL[13676,2]',evidence=ev))
 if r['Gloss'] in ['when','when?'] and r['Form'] in {'kab','kabe','kabɛ','kabʰe','kabʰɛ','kaba','kabaī  ','kabaū'}:
  h.append(dict(record=r,reason='Turner 2528 explicitly treats kab/kabe as interrogative remodeling of the ēvam ēva series, whereas Platts s.v. kab proposes kiyat with analogy from jab/tab. The competing historical analyses are preserved for review; no automatic parent assignment from the search suggestion.'))
(P/'pass020-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n')
(P/'pass020-interrogative-evidence.md').write_text('Turner 2528 full article and addenda are in primary-040.json. Platts s.v. kab was read at https://www.rekhta.org/urdudictionary?keyword=kab (2026-09-21); he tentatively proposes kiyat with analogy from jab and tab, competing with Turner’s ēvam ēva remodeling account. The relevant forms remain held. Search links to 1605 for kahā̃ were not expanded: the full ihá article does not establish that interrogative formation.\n')
print(len(a),'accepted',len(h),'held')

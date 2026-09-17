import json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass85';assert not (P/(stem+'-decisions.json')).exists();done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
rules=[
 dict(parent='6459',citation='CDIAL[6459]',evidence='CDIAL 6459 *duvāra explicitly gives Kumauni dwār and Western Pahari dwār in its addenda, as well as Nepali duwār. Dotyali dvār/dvər therefore selects that expanded door family, preserving the local reduced vowel; the separate dvāra branch mainly supplies dār/bār forms.'),
 dict(parent='9209',citation='CDIAL[9209.1]',evidence='CDIAL 9209.1 *bāppa gives Nepali/Maithili bāp, Oriya bāpa and regional bāpū father. Danuwar bapo preserves the p-bearing nursery-word branch, with its final vowel retained, rather than the separate bābba branch.'),
 dict(parent='6770',citation='CDIAL[6770.1]',evidence='CDIAL 6770.1 *dhāgga gives Lahnda dhāggā and Awan dhāgā thread, beside Punjabi dhāggā. Saraiki dāgā fits the voiced regional thread family with aspiration loss retained as a qualification; it is not assigned to the voiceless tāga/trāgga branch. The addenda revise the deeper reconstruction toward *dhārga/*dharga, and local transmission remains open.'),
 dict(parent='6590',citation='CDIAL[6590]',evidence='CDIAL 6590 dōṣā explicitly includes Kalasha Rumbur and Khowar doṣ yesterday. The survey doś forms match those language-specific whole-word comparisons; the source ś versus printed ṣ articulation remains an explicit transcription qualification. The link does not impose an unverified sibilant merger or settle areal transmission.'),
 dict(parent='8339',citation='CDIAL[8339.1]',evidence='CDIAL 8339.1 pūrṇa filled/full supports the full learned Dhundari pūrṇa whole form, with its rṇ cluster retained. This corrects the earlier pūra search candidate and is saved as learned Sanskrit vocabulary; any intermediate Indo-Aryan transmission is open.')]
acc=[];held=[]
for x in json.loads((P/'near-expansion-candidates.json').read_text()):
 r=x['record'];ps=x['parents'];i=None;reason=None
 if r['ID'] in done:continue
 if set(ps)=={'6459','6663'}:
  if r['Language_ID']=='Dotyali':i=0
  else:reason='Goj dhar door has aspiration absent from the cited duvāra/dvāra door comparisons. Establish its local development or source reading before selecting a branch.'
 elif set(ps)=={'9209','9209-2'}:
  if r['Form']=='bapo':i=1
  else:reason='Sunha bau father lacks the medial consonant distinguishing bāppa from bābba. Compare a local bā/bāu nursery stem and contraction evidence before choosing the p or b branch.'
 elif set(ps)=={'6010','6770'}:
  if r['Language_ID']=='srk':i=2
  else:reason='Awan thāgā thread is voiceless aspirated, while trāgga gives tāgā and dhāgga gives dhāgā. Regional tone/aspiration evidence is needed to disambiguate these close thread families.'
 elif ps==['6590']:i=3
 elif set(ps)=={'6983','7025-2'}:reason='Bagheli naba new resembles both nava and the nābo/nɔbo forms that CDIAL discusses under navya. The full entries keep this competition; a short spelling match alone cannot select a historical branch.'
 elif set(ps)=={'1351','574'}:reason='Buksa aia mother has no nasal supporting ambā and lacks a specific local comparator in āryikā. Investigate the separate āī family instead of selecting either near-list parent.'
 elif ps==['4435']:reason='Buksa gharwala husband is a full ghar + wala formation. A link of the whole response to ghara would drop the second component; find supported component or whole-word donor evidence.'
 if reason:held.append(dict(record=r,families=[],reason=reason,passNumber=85))
 if i is not None:
  q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact survey response: '+r['Form']+'.'))
# Revisit the specifically rejected pūra suggestion with the now-read pūrṇa article.
for r in json.loads((P/'inventory.json').read_text()):
 if r['Language_ID']=='dhundari_badagaon' and r['Form']=='pūrṇa' and r['Gloss']=='whole':
  q=rules[4];acc.append(dict(record=r,parent=q['parent'],family=4,kind='borrowed',citation=q['citation'],evidence=q['evidence']))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass85_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem))
print({'accepted':len(acc),'held':len(held)})

import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};inv=json.loads((P/'inventory.json').read_text())
assert not (P/'pass218-decisions.json').exists()
held=[dict(record=r,families=[],reason='Awank/Gojri muc/mūc many has no verified etymological parent after homepage searches for muc/mucc and much. General web searches did not establish a primary lexical analysis, and the Bailey dictionary PDF could not be inspected through the web reader because it exceeded the reader size limit. No English much connection is inferred. Mixed muc / zīādā also needs a supported immediate donor for its second response. This is an unresolved lexical family, not merely unresolved cross-IA transmission.',passNumber=218) for r in inv if r['ID'] in remaining and r['Gloss']=='many' and r['Form'] in {'muc','mūc','muc / zīādā'}]
for suffix,obj in [('rules',[]),('decisions',dict(accepted=[],held=held))]:(P/('pass218-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
print('218 held',len(held))
rules=[dict(parent='11225',citation='CDIAL[11225]',evidence='Full CDIAL vaḍra gives Punjabi/Hindi baṛā big and explicitly records the related Kotgarhi bɔṛɔ much/very in its addenda. Gojri baṛa many fits this size-to-quantity usage with source vowel length retained. The article prefers historical extraction from evaḍa-type size expressions rather than a simple vṛddha derivation; that qualification and local IA transmission remain open.'),dict(parent='9750',citation='CDIAL[9750];CDIAL[10023]',evidence='Gojri mato/matā many lacks direct support from shortlisted matta intoxicated/proud or mātrā measure. The full articles do not give the relevant regional quantity adjective, and surface similarity alone does not choose either family.'),dict(parent='12006',citation='CDIAL[12006];CDIAL[11987.2]',evidence='Bhatri bistaṛ many could be associated with vistāra expansion/extent or vistara spreading/diffuseness, but the full entries do not directly document this many usage or settle the stem distinction. Verify a regional dictionary analysis; the homonymous Persian bedding family is not evidence for this sense.')]
acc=[];held=[]
for r in inv:
 if r['ID'] not in remaining or r['Gloss']!='many':continue
 if r['Form']=='baṛa' and r['Language_ID']=='Goj':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],family=0,kind='reflex',citation=q['citation'],evidence=q['evidence']))
 elif r['Form'] in {'mato','matā','bistaṛ'}:
  i=2 if r['Form']=='bistaṛ' else 1;held.append(dict(record=r,families=[i],reason=rules[i]['evidence'],passNumber=220))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held))]:(P/('pass220-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/'pass220_save.py').write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth','pass220'))
print('220 accepted',len(acc),'held',len(held))

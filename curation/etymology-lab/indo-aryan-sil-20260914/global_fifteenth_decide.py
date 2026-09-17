import json,csv,re
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914')
E={
'6559-3':'CDIAL 6559.3 *dehila explicitly gives Nepali dailo door/threshold. Majhi and Bote dailo select this branch rather than the dehalī or dehula branches.',
'7':'CDIAL 7 aṃsiya explicitly gives Nepali hã̄siyo, Bengali/Hindi hã̄siyā and Bihari hãsuā sickle, as well as Kumaoni ã̄sī scythe. The selected forms preserve these y-bearing, u-bearing and h-less variants; uncertain regional transmission remains open.',
'2817':'CDIAL 2817 karkaṭa explicitly gives Nepali kā̃kro and related kā̃kri cucumber. These kãkro/kãkra forms fit that named plant family; the article qualifies its deeper relation to karkaṭa crab and karkāru.',
'1728':'CDIAL 1728 utkuṇa explicitly gives Bengali ukun and Oriya ukuṇi louse, alongside the nasal uṅkuṇa variant. Majhi ukuni/ũkuni matches this louse family with source nasalization retained.',
'1348':'CDIAL 1348 āryaka explicitly gives Bhojpuri ājā grandfather, ājī grandmother and Oriya ajā maternal grandfather. These aja/aji/aje forms fit the documented kinship family; source relationship specificity is retained.',
'5539-8':'CDIAL 5539.8 *devva explicitly gives Nepali debre left. These dental-initial Majhi/Bote forms select section 8 rather than retroflex *ḍevva in section 5.',
'4999':'CDIAL 4999 chardi explicitly gives Nepali chād vomit. Majhi/Bote cʰad matches the noun in that article.',
'13734':'CDIAL 13734 strī means woman/wife. The unchanged Bengali consonant cluster identifies the learned Sanskrit word; Kalasha istrīźā is separately identified in the article as a modern compound with jāyā and needs both parts represented.',
'10383':'CDIAL 10383 mriyate explicitly gives Khowar bri- die in the addendum and brik in the main entry. These bri-/brī- stems match that direct comparison; the hyphen is stem notation.'}
done=set()
for f in json.loads((P/'pass-ledger.json').read_text())['decisionFiles']:
 d=json.loads((P/f).read_text());done.update(x['record']['ID'] for k in ['accepted','held'] for x in d[k])
cs={}
for x in json.loads((P/'global-exact-candidates.json').read_text()):
 k='/'.join(x['parents']);r=x['record']
 if k in E and r['ID'] not in done:cs[r['ID']]=(r,k)
# Explicit whole-form extensions of these already discovered primary families.
extras={
'7':({'sickle'},{'hãsu','hãsuva','hasu','ā̃si','hə̃suva','həsuva','hãse','hãssyə','hãssiya','həssiyə'}),
'2817':({'cucumber'},{'kãkra'}),
'1348':({'grandfather','grandmother'},{'ajai','aje'})}
for r in csv.DictReader((P/'unresearched-records.csv').open()):
 for k,(gs,ws) in extras.items():
  if r['Gloss'] in gs and r['Form'] in ws:cs[r['ID']]=(r,k)
acc=[];held=[];qs=[];ix={}
for r,k in cs.values():
 ev=E[k];reason=None
 if k=='13734' and r['Language_ID']=='Kal':reason='The full primary entry analyzes istrīźā as strī + jāyā. Resolve the existing nodes for both components; a single strī parent would omit part of the word.'
 if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':reason=reason or 'Verify the uncertain lexical reading first.'
 if k not in ix:ix[k]=len(qs);qs.append(dict(parent=k,citation='CDIAL['+k.replace('-','.',1)+']',evidence=ev))
 i=ix[k]
 if reason:held.append(dict(record=r,families=[i],reason=reason,passNumber=60));continue
 acc.append(dict(record=r,parent=k,family=i,kind='borrowed' if k=='13734' else 'reflex',citation=qs[i]['citation'],evidence=ev+' Exact survey form '+r['Form']+' is preserved; intra-Indo-Aryan transmission remains open.'))
for fn,obj in [('global-fifteenth-decisions.json',dict(accepted=acc,held=held)),('global-fifteenth-rules.json',qs)]: (P/fn).write_text(json.dumps(obj,ensure_ascii=False,indent=1))
(P/'global_fifteenth_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','global-fifteenth'));(P/'global_fifteenth_decide.py').write_text(Path(__file__).read_text());print(len(acc),len(held))

import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass260';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='7733',components=['7733','10560'],citation='CDIAL[7733];CDIAL[10560]',evidence='Hajong pataroŋ green is analysed provisionally as pata leaf + roŋ colour, with ordered component links. The same survey independently gives Hajong pata leaf at p.42 (exact comparison records preserved in pass260-local-comparanda.json). Full CDIAL pattra supplies Bengali pātā leaf; full raṅga supplies dye/colour and eastern raṅgā/rāṅā colour forms. This is a transparent modern leaf-colour analysis, not an inherited Sanskrit compound. The second member’s local vowel/history and cross-IA transmission remain qualified; the source spelling and green gloss remain intact.'),dict(parent='7733',citation='CDIAL[7733]',evidence='Full pattra explicitly gives Punjabi pattā/pattrā, Awankari pattar, Bshk paλ, Gawri phaṭa, Shina Kohistani păṭhṷ and Kotgarhi pāc leaf; it also documents pattrikā and Middle Indo-Aryan pattī/pattikā leaf. These support the selected simple and ordinary local leaf variants. Source aspiration, dental/retroflex notation, vowels, rhotics, affricates and ordinary endings remain intact, including complete same-family slash responses. Exact local phonetic/ending histories and cross-IA transmission remain qualified; nasal pānṭ- and other competing leaf families are excluded.')]
fs={'pʌʈːa','pʌʈːija','pʌʈija','faṭa','vaṭā','paṭor','patarā','paṭaṛ','paṭar','pator','paṭor / patar','patar / pato','pāṭʰ','pət ̚taⁱ','potor','potṛo','poˈtʃə','potʃʰ','pʰotʃʰ','potʃ','pəṭiya','pəṭhiyə','pəṭhiya'}
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 i=0 if r['Language_ID']=='Hajong' and r['Form']=='pataroŋ' and r['Gloss']=='green' else 1 if r['Gloss']=='leaf' and (r['Form'] in fs or r['Form']=='pālo' and r['Language_ID']=='Bshk') else None
 if i is not None:
  q=rules[i];x=dict(record=r,parent=q['parent'],family=i,kind='component' if i==0 else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.')
  if i==0:x['components']=q['components']
  acc.append(x)
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare260.py').read_text());print('accepted',len(acc),'rows',sum(len(x.get('components',[x['parent']])) for x in acc))

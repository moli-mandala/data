import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent;stem='pass117';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='f_dvwxidw6762ww',citation='strand[s.v. silʹêni];CDIAL[13444.1]',evidence='Strand’s original cached Palula entry explicitly gives silʹêni needle, morphological analysis sil-ʹân-i, and OIA sīvyati sews (T. 13444.1). The survey selenī is linked as a same-language variant of that existing Palula noun, preserving vowel and stress/length transcription differences. The noun’s derivational account comes from Strand, not from an unattested noun quotation in CDIAL. Primary: https://nuristan.info/IndoAryan/Indus/Atsaret/AtsaretLanguage/Lexicon/alph-s.html'),dict(parent='13551-2',citation='CDIAL[13551.2]',evidence='CDIAL 13551.2 *sūñcī explicitly gives Khowar šunǰ needle. Survey śūnź fits this nasal branch with fricative/affricate notation retained; it is not inferred from the unrelated English syringe loan label in another database entry. Local transmission remains unresolved.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='needle':continue
 i=0 if r['Language_ID']=='Phal' and r['Form']=='selenī' else 1 if r['Language_ID']=='Kho' and r['Form']=='śūnź' else None
 if i is None:continue
 q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='variant' if i==0 else 'reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
f=P/(stem+'-primary-articles.json');a=json.loads(f.read_text());a['13551']=json.loads((P/'pass116-primary-articles.json').read_text())['13551'];f.write_text(json.dumps(a,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));print(len(acc))

import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass270';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='9051',components=['9051','9092'],citation='CDIAL[9051];CDIAL[9092]',evidence='Full phala and phulla entries explicitly document Nepali/Hindi phal fruit and phul flower. The same Kochila survey independently records phal phul fruit at p.33 (Siraha), supporting the Bara pəlepʰul response as the corresponding fruit-plus-flower expression. Ordered component-family links preserve the collective fruit gloss and full source spelling. The first member’s reduced vowel and intervening e are retained without inventing an affix or ancient compound; exact local formation and cross-IA transmission remain qualified.')]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[];held=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining or r['Gloss']!='fruit':continue
 if r['Form']=='pəlepʰul':
  q=rules[0];acc.append(dict(record=r,parent=q['parent'],components=q['components'],family=0,kind='component',citation=q['citation'],evidence=q['evidence']))
 elif r['Form']=='phalphɔl':held.append(dict(record=r,families=[0],passNumber=270,reason='Full phala documents Niya phalophala fruits of all kinds, while the local survey also has phal-phul fruit-plus-flower expressions. Kathoriya phalphɔl could represent a repeated fruit form or a flower second component with shifted vowel. Need local morphological/lexical evidence to choose the second parent rather than infer it from the collective fruit gloss.'))
 elif r['Form']=='phəl phulwar':held.append(dict(record=r,families=[0],passNumber=270,reason='Initial phəl fruit and phul- flower are recognizable, but the complete phulwar member needs its ending or compound analysis. Simple phala+phulla links would leave an unexplained part of this response. Held for direct local lexical evidence.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=held)),('primary-articles',json.loads((P/'pass267-primary-articles.json').read_text()))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'groundnut_save.py').read_text().replace('groundnut',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare270.py').read_text());print('accepted',len(acc),'held',len(held))

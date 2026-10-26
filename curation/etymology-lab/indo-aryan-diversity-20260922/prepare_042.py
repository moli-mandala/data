import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922')
linked={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'}
linked.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
rs=[r for r in csv.DictReader(open('cldf/forms.csv')) if 'dhakal2011darai[' in r['Source'] and r['ID'] not in linked]
rec=lambda r:{k:r[k] for k in ['ID','Language_ID','Form','Gloss','Source','Tags']}
a=[];h=[]
for r in rs:
 if r['Form'] in ['ʦər','ʦərike'] and 'graze' in r['Gloss']:
  parent='4686';ev='Turner4686 carati gives Prakrit carai, Nepali carnu and Maithili/Awadhi carab to graze. Dhakal p44 explicitly labels cər graze.IMP and p54 gives cərike to graze; p75 identifies -ike as INF. These two inflections continue the unextended intransitive stem; the longer caralə record is excluded pending causative analysis.'
 elif r['Form']=='kʰəsike' and r['Gloss']=='to fall down':
  parent='3856';ev='Turner3856 *khasati gives Prakrit khasai, Nepali khasnu, Maithili khasab and Hindi khasna fall/slip. Dhakal p52 contrasts kəsike tighten with kʰəsike fall down, verifying initial aspiration and the unextended khas- stem. The infinitive -ike is documented in p75. No -kk-, -ḍ- or -r- extension is present.'
 else:
  reasons={'ʦaralə':'Source p43 labels caralə graze-PST, but p75 productive causative -a and vowel harmony cər > car-a suggest a causative rather than ordinary stem inflection; verify the specific example before linking.','uḍʰike':'Turner1697-2 fits the fly family but source p53 explicitly contrasts and exemplifies aspirated ḍʰ; its historical aspiration is not explained by the shortlisted entry.','bapʰaike':'Turner9223 documents regional bāph steam but verbal formation requires distinguishing inherited bāṣpāyate from Darai denominal/causative formation. No direct noun-to-verb reflex assigned.'}
  if r['Form'] in reasons:h.append(dict(record=rec(r),reason=reasons[r['Form']]))
  continue
 a.append(dict(record=rec(r),parent=parent,kind='reflex',citation=r['Source']+';dhakal2011darai[p. 75];CDIAL['+parent+' and addenda]',evidence=ev))
assert len(a)==3,len(a)
(P/'pass042-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=2)+'\n')
source=Path('data/other/forms/raw_data/dhakal_darai_2011/decoded')
(P/'pass042-source-evidence.json').write_text(json.dumps({str(n):(source/f'p{n:03}.txt').read_text() for n in [43,44,52,53,54,75]},ensure_ascii=False,indent=2)+'\n')
print(len(a),'accepted',len(h),'held')

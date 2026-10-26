import json
from pathlib import Path
P=Path(__file__).resolve().parent;rs=json.loads((P/'remaining-initial.json').read_text());done={x['Form_ID'] for f in P.glob('pass*-saved-assignments.json') for x in json.loads(f.read_text())};a=[];h=[]
for r in rs:
 if r['ID'] in done:continue
 f,g,l=r['Form'],r['Gloss'],r['Language_ID'];x=None;reason=None
 if l=='Kal' and g=='red' and f in ['lāẓīā','lāʦ̣īā','laʦ̣īa']:
  x=dict(parent='11003',kind='reflex',citation='CDIAL[11003,1];trail-cooper1999[printed p. 184, PDF p. 209]',evidence='Turner lākṣika, section 1, explicitly gives Kalasha lac̣hīa red. Trail and Cooper confirm Lac̣hía and the voiced variant LaJía. Their printed reference suffix .2 conflicts with Turner’s full article: section 2 is the separate *lākṣaka head, so the exact first-branch public node is retained.')
 if l=='Kal' and g=='gold' and f in ['sụ̄̃ā','sụ̄ɛ̣̃','sụ̄̃ā̃']:
  x=dict(parent='13519',kind='reflex',citation='CDIAL[13519,1 or 2];trail-cooper1999[printed p. 277, PDF p. 302]',evidence='Trail and Cooper explicitly derive retroflex-nasal suạ̃ gold from suvarṇa; Turner includes the older Kalasha sū̃ṛa/sū̃ŕä forms. Turner groups suvarṇa and sauvarṇa because their reflexes often cannot be distinguished. This uses that combined public head and preserves the survey’s retroflex vowel notation.')
 if l in ['Kal','Kho'] and g=='feces' and f in ['riʦ̣','rīʦ̣','reʦ̣','īric']:
  x=dict(parent='10731',kind='reflex',citation='CDIAL[10731]'+(';trail-cooper1999[printed p. 259, PDF p. 284]' if l=='Kal' else ';bashir2023[s.v. ric̣]'),evidence='Turner explicitly gives Kalasha ric̣ excrement and Khowar ric̣(h) fresh dung of humans and animals under the combined *rikṣa or *rikṣya head. Trail and Cooper confirm the Urtsun variant iric̣; Bashir includes human feces. The link retains the unresolved reconstruction alternative already present in this public head.')
 if l=='Kho' and g=='dry' and f in ['ʦ̣ūʦ̣hū','ʦ̣ūʦ̣ū']:
  x=dict(parent='12508',kind='reflex',citation='CDIAL[12508];bashir2023[s.v. c̣uc̣hú]',evidence='Turner *śukṣa explicitly gives Khowar c̣uc̣hú/c̣huc̣u dried, and Bashir gives c̣uc̣hú dry with the same etymological reference. The survey’s differing placement/presence of aspiration is retained; no separate śuṣṭa adjective is substituted.')
 if l=='Kal' and g=='dry' and f=='āṣūṣal':
  x=dict(parent='f_7npoxdh57yiug',kind='borrowed',citation='CDIAL[12508];trail-cooper1999[printed p. 70, PDF p. 95];bashir2023[s.v. c̣uc̣hú]',evidence='Trail and Cooper explicitly list Urtsun aṣúṣal as a variant of c̣uc̣hu dry and identify the *śukṣa family as borrowed through Khowar. The existing Bashir Khowar c̣uc̣hú entry is the immediate donor. This preserves the documented borrowing route and the dialect variant instead of asserting direct Kalasha inheritance.')
 if l=='Kal' and g in ['moon','month'] and f in ['mastrūk','māstrūk']:reason='Turner 10104 recognizes māsa in mastruk but tentatively suggests an additional roka component with two question marks. Trail and Cooper cite māsa without resolving the remainder. Hold the potentially compound whole word until its full formation is established.'
 if l in ['Kal','Kho'] and g=='arm' and f in ['bāzā','bāzū']:reason='Turner 9229 explicitly treats Kalasha baza and Khowar bazu as borrowed from Nuristani or Iranian, leaving the immediate route unresolved. Do not use a direct bāhu reflex edge or assume a Persian donor from resemblance alone.'
 if l=='Kal' and g=='egg' and f=='āyūkūn':reason='Trail and Cooper explicitly identify ayukun as borrowed from Khowar. The matching Bashir Khowar donor f_xlofqmd7h2uis currently has Status unlinked, which the shared save helper disallows as a parent; no new donor/source ingestion is authorized in this no-rebuild pass.'
 if x:a.append(dict(record=r,**x))
 if reason:h.append(dict(record=r,reason=reason))
(P/'pass114-decisions.json').write_text(json.dumps(dict(accepted=a,held=h),ensure_ascii=False,indent=1)+'\n');print('accepted',len(a),'held',len(h))
p=P/'lexical-reference-links.json';links=json.loads(p.read_text())
for x in a:
 for c in x['citation'].split(';'):
  if c.startswith('bashir2023'):links[c]='https://escholarship.org/uc/item/955239w9'
p.write_text(json.dumps(links,ensure_ascii=False,indent=1)+'\n')

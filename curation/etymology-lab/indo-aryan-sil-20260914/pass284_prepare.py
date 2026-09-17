import csv,json
from pathlib import Path
P=Path('/Users/aryamanarora/Documents/Code/jambu-all/data/curation/etymology-lab/indo-aryan-sil-20260914');stem='pass284';assert not (P/(stem+'-decisions.json')).exists()
rules=[dict(parent='1111',citation='CDIAL[1111]',evidence='Full āṇḍa explicitly includes Kalasha ṓṇḍrak/hā̃ṭrak, Lahnda āṇḍṛā and Awankari ā̃ṭṛā, Torwali āṇ, Bshk ã, Maiya āṛa, regional aṇḍā and Gujarati ĩḍũ. These support the selected simple, reduced, r-extended and front-vowel egg responses. Keep the source glottal onset, nasalization, stop voicing/aspiration and local endings. The article presents competing deeper analyses and local deformations; this links the documented family without resolving ultimate origin or cross-IA transmission. Multiword responses and unexplained extra initials or l-extensions are excluded.'),dict(parent='5550',citation='CDIAL[5550]',evidence='Full ḍimba egg gives Assamese ḍimā, Bengali ḍim and Oriya ḍimba directly, supporting eastern dim/dimu/dimb egg forms with dental/retroflex and final-vowel spelling preserved. The possible lump-family connection in the article remains tentative; do not confuse the body or affray homonyms. Cross-IA transmission is qualified.')]
sets=[('egg',{'ọ̃ḍrak','hā̃ḍrūk','ātʰaṛā','aṇṭaṛā','aṭaṛa','aṭaṛā','āṇḍiṛā','ān','ʔon','ʔān','ʔan','anṭṛo','ā̃ṭṛo','ʔā̃ṛa','ʔāṭā','oṇda','āṇḍũ','indo','ā̃ṇṭəṛo','āṇṭəṛa','āṇṭaṛo','ʊɳɖʰaə','əɳɖai','aṇṭa','aṇṭṛa','inḍe','aiṇḍo'}),('egg',{'dim','dimu','dimb'})]
remaining={r['ID'] for r in csv.DictReader((P/'unresearched-records.csv').open())};acc=[]
for r in json.loads((P/'inventory.json').read_text()):
 if r['ID'] not in remaining:continue
 for i,(g,fs) in enumerate(sets):
  if r['Gloss']==g and r['Form'] in fs:
   q=rules[i];acc.append(dict(record=r,parent=q['parent'],family=i,kind='reflex',citation=q['citation'],evidence=q['evidence']+' Exact response: '+r['Form']+'.'))
for suffix,obj in [('rules',rules),('decisions',dict(accepted=acc,held=[]))]:(P/(stem+'-'+suffix+'.json')).write_text(json.dumps(obj,ensure_ascii=False,indent=1)+'\n')
(P/(stem+'_save.py')).write_text((P/'global_sixteenth_save.py').read_text().replace('global-sixteenth',stem));(P/(stem+'_prepare.py')).write_text(Path('/tmp/prepare284.py').read_text());print('accepted',len(acc))

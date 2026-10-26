import csv,json
from pathlib import Path
from etymology_assignments import read_assignments
P=Path('curation/etymology-lab/indo-aryan-diversity-20260922');rs=json.loads((P/'torwali-records.json').read_text());aud={r['Source']:r for r in csv.DictReader(open('data/other/forms/raw_data/20260820-torwali-student-2023-audit.csv'))};assigned={r['Form_ID'] for r in read_assignments() if r['Status']=='accepted'};assigned.update(r['Child_ID'] for r in csv.DictReader(open('cldf/edges.csv')) if r['Rank']=='1')
groups=[
('æṣi','Eye',20,'43','akṣi','Turner explicitly gives Torwali aṣī́ eye under akṣi. Source æʂi retains the retroflex sibilant and final vowel, with its æ vowel preserved.'),
('bāl','Hair',34,'11572','vāla','Turner explicitly gives Torwali bāl hair under vāla. The source ba:l matches directly.'),
('gām','Village',53,'4368','grāma','Turner explicitly gives Torwali gām village under grāma. The p.52 source entry ga:m is distinct from the separate p.177 citation, which is not included in this image-check batch.'),
('yīb','Tongue',88,'5228','jihvā','Turner gives Torwali jīb tongue under jihvā. Source phonetic ji:b is rendered yīb by the existing IPA profile; that transcription-layer difference is preserved, not silently converted back to j.'),
('dān','Tooth',108,'6152','danta','Turner gives Torwali dΛn tooth under danta. The source da:n has the same d-n skeleton with final t absent; its long vowel is retained.'),
('dʰīmi','Smoke',114,'6849-2','dhūmikā','Turner section 2 explicitly gives Torwali dhimī smoke under dhūmikā, with *dhūmiyā/dhūmyā alternatives in the heading. The source dʰi:mi selects existing node6849-2 rather than dhūma section1; the alternative formation remains open.'),
('māś','Fish',198,'9758','matsya','Turner section1 gives Torwali maǰ fish and neighbouring Pashai mās under matsya. The source ma:ʃ preserves its voiceless fricative and long vowel in this regional fish-family assignment; it is not the subsection *matsiya.'),
('mās','Meat, flesh',198,'9982','māṃsa','Turner explicitly gives Torwali mās meat/flesh under māṃsa. The source ma:s is distinct from ma:ʃ fish on the same page; the separate sibilants and senses are preserved.'),
('hīm','Snow',227,'14096','hima','Turner explicitly gives Torwali him snow under hima. The source hi:m preserves the long vowel as printed.')]
a=[]
for f,g,pg,pa,la,ev in groups:
 for r in rs:
  if r['ID'] not in assigned and r['Form']==f and r['Gloss']==g and int(aud[r['Source']]['PDF_Page'])==pg:
   assert not aud[r['Source']]['Relation_Label'];a.append(dict(record=r,parent=pa,kind='reflex',citation=f'CDIAL[{pa}];'+r['Source'],evidence=ev+' Published phonetic notation, source locator and Bahrain dialect label are preserved.'))
(P/'pass021-decisions.json').write_text(json.dumps(dict(accepted=a,held=[]),ensure_ascii=False,indent=2)+'\n');(P/'pass021-source-audit.json').write_text(json.dumps({x['record']['ID']:aud[x['record']['Source']] for x in a},ensure_ascii=False,indent=2)+'\n')
lines=['# Torwali student dictionary: image-check batch','',f'{len(a)} proposed records / {len(a)} reflex rows. No rebuild.','','| # | Torwali | Proposed etymology | Evidence |','|---|---|---|---|']
for i,(f,g,pg,pa,la,ev) in enumerate(groups,1):lines.append(f'| {i} | **{f}** ‘{g.lower()}’ | **{la}**, CDIAL **{pa}** | {ev} |')
lines+=['','Source-audit rows have no explicit variant/inflection relation. The second gām village citation and the milk, belly, ear, nose and hand candidates remain for further source and phonological review.','', 'Source-image completion, validation and save status: pending.'];(P/'pass021-review.md').write_text('\n'.join(lines)+'\n');print(len(a),'proposed')

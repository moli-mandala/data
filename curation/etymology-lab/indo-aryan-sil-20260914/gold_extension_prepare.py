import csv,json
from pathlib import Path
P=Path(__file__).resolve().parent
inventory={r['ID']:r for r in json.loads((P/'inventory.json').read_text())}
remaining={r['ID'] for r in csv.DictReader(open(P/'unresearched-records.csv'))}
remaining.update(x['record']['ID'] for x in json.loads((P/'audit-records.json').read_text()))
# Individually reviewed whole responses. Do not normalize away compound boundaries,
# s/h correspondence, Kalasha rhotic loss or an unexplained suffix.
selected={'swan','sovān','sūān','sūāṇ','śono','śona','śuna','śonnā','sʷəna','sʷona','sun.a','ʃona','sonːa','sonu','sonno','sonnu','saũna','sunu','sʌnə','sʊɳə','sana','sunnu','sonːo','ṣɔn'}
ev='CDIAL 13519 lists the suvarṇa/sauvarṇa gold continuations jointly because the branches are often indistinguishable, including Phal. suāṇ, H. sonā, B. sonā, Or. sunā, WPah. sunnō, jaun. sūnō and G. sɔnũ. These reviewed simple gold responses preserve the corresponding s/ś, rounded-vowel, nasal and geminate variants. Following the user’s explicit sauvarṇa choice, link them to 13519.2; this is an editorial resolution of the ambiguous branch, with possible intra-Indo-Aryan transmission left open.'
q=dict(parent='13519-2',citation='CDIAL[13519.2]',evidence=ev)
acc=[]
for fid in sorted(remaining):
 r=inventory[fid]
 if r['Gloss']=='gold' and r['Form'] in selected:
  acc.append(dict(record=r,family=0,parent=q['parent'],citation=q['citation'],evidence=ev+' Exact source transcription retained; this comparison extends the reviewed gold family and does not revise already linked records.'))
(P/'gold-extension-rules.json').write_text(json.dumps([q],ensure_ascii=False,indent=1))
(P/'gold-extension-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1))
(P/'gold_extension_save.py').write_text((P/'user_resolution_save.py').read_text().replace('user-resolution','gold-extension'))
print('accepted',len(acc))
for x in acc:print(x['record']['Language_ID'],x['record']['Form'])

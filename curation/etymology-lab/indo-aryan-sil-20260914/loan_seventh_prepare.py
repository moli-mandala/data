import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
qs=[dict(parent='f_6sqgvw7opxpy6',gloss=['sky'],words='āsmān|āsmā̃n|asman|asmān|āsamān|asaman|āsmā̃',citation='Platts[p. 53];platts1884[s.v. āsmān]',evidence='Platts p. 53 explicitly records Persian-derived āsmān/asmān sky in Urdu/Hindi. The existing Hindi āsmān lexical entry is used as a supported provisional regional donor. Direct Persian, Pashto or other Indo-Aryan transmission may have occurred locally; the graph does not settle that route.'),dict(parent='f_glk2glwunvrke',gloss=['chest','breast'],words='sina|sīnā|śīnā|sīna|śina|sino|sīnō|sīno|sine|sinə',citation='Platts[pp. 713, 714];liljegren[entry LX001997]',evidence='Platts p. 713 cross-refers Hindi sinā to Persian sīna, defined as breast/bosom/chest on p. 714; this is distinct from the verb sīnā sew. The existing Hindi siina entry provides the immediate regional donor hypothesis. Survey s/ś and final-vowel variation is preserved, with intervening Indo-Aryan transmission unresolved.'),dict(parent='f_dbuiao2qkdgv4',gloss=['whole'],words='sabut|sābūt|sābut|sabūt|sʌbut|sʌbʊt',citation='Platts[p. 368];platts1884[s.v. sabut]',evidence='Platts p. 368 explicitly gives s̤ubūt, vernacular s̤abūt, with adjective entire. The existing Hindi s̤abūt whole/entire donor therefore supports these sabut responses. Regional transmission is provisional, without assuming direct Arabic borrowing by each survey language.')]
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};acc=[];held=[]
for rr in csv.DictReader(open(P/'unresearched-records.csv')):
 r=raw[rr['ID']];w=norm(r['Form'])
 for i,q in enumerate(qs):
  if r['Language_ID']=='H' or r['Gloss'] not in q['gloss'] or w not in {norm(z) for z in q['words'].split('|')}:continue
  if re.search('uncertain|unresolved|illegible|unreadable',r['Tags']+' '+r['Description'],re.I) and r['Language_ID']!='Rana':held.append(dict(record=r,families=[i],reason='Source uncertainty needs lexical-reading verification.',passNumber=48));continue
  acc.append(dict(record=r,family=i,parent=q['parent'],kind='borrowed',citation=q['citation'],evidence=q['evidence']+' Exact survey form '+r['Form']+' is preserved.'))
(P/'loan-seventh-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'loan-seventh-decisions.json').write_text(json.dumps(dict(accepted=acc,held=held),ensure_ascii=False,indent=1));(P/'loan_seventh_save.py').write_text((P/'week_mother_save.py').read_text().replace('week-mother','loan-seventh'))
a=json.loads((P/'platts-loan-seventh-research.json').read_text())+json.loads((P/'platts-chest-research.json').read_text());p={q['parent']:[x for x in a if x['word']==w] for q,w in zip(qs,['آسمان','سينه','ثبوت'])};(P/'loan-seventh-primary-articles.json').write_text(json.dumps(p,ensure_ascii=False,indent=1))
print('accepted',len(acc),'held',len(held))
for i in range(len(qs)):print(i,len([x for x in acc if x['family']==i]),sorted({x['record']['Form'] for x in acc if x['family']==i}))

import json,csv,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
raw={r['ID']:r for r in json.loads((P/'inventory.json').read_text())};eligible={r['ID'] for r in csv.DictReader(open(P/'unresearched-records.csv'))}
parent='f_pqghvk5xbgj24'
ev='Platts p. 530 s.v. do/du expressly gives do-pahar noon (Sanskrit dvi-prahara). The Hindi Rohili survey records dupahar (kannauji p. 84). These whole do-/du-pahar responses are linked to the existing Hindi word as a provisional regional route; the actual immediate Indo-Aryan transmitter remains uncertain. The donor already has an older graph link to *dva-prahara; that reconstruction differs from Platts’s dvi-prahara and is flagged in noon-upstream-audit.json, not silently treated as verified.'
qs=[dict(parent=parent,citation='Platts[p. 530];kannauji[p. 84]',evidence=ev)];acc=[]
for fid in sorted(eligible):
 r=raw[fid];w=norm(r['Form'])
 if r['Gloss']!='noon' or r['Language_ID']=='H':continue
 if re.fullmatch(r'd[ouaəe]+p+h?[aeuəʌ]*(?:h[aeuəʌ]*)?[rṛɽ][aiāã]*',w) or w in {'dupuri','dupur'}:
  acc.append(dict(record=r,family=0,parent=parent,kind='borrowed',citation=qs[0]['citation'],evidence=ev+' Exact survey form '+r['Form']+' is retained.'))
(P/'noon-rules.json').write_text(json.dumps(qs,ensure_ascii=False,indent=1));(P/'noon-decisions.json').write_text(json.dumps(dict(accepted=acc,held=[]),ensure_ascii=False,indent=1));(P/'noon_save.py').write_text((P/'loan_fourth_save.py').read_text().replace('loan-fourth','noon'))
(P/'noon-upstream-audit.json').write_text(json.dumps(dict(existingDonor=parent,existingParent='f_jg7fkl55nyhjm',existingParentForm='*dva-prahara',primaryAlternative='dvi-prahara (Platts p. 530)',action='Older ancestry unchanged; new links identify the complete attested Hindi word, with this upstream reconstruction qualification retained.',newNodesAdded=False),ensure_ascii=False,indent=2))
a=json.loads((P/'platts-compound-research.json').read_text());a=next(x for x in a if x['word']=='do-pahar');(P/'noon-primary-articles.json').write_text(json.dumps({parent:[a]},ensure_ascii=False,indent=1))
print(len(acc));print(sorted({x['record']['Form'] for x in acc}))

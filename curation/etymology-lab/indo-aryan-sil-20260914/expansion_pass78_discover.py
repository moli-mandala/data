"""Discover remaining whole-form matches with equivalent elicitation glosses. No assignments."""
import json,csv,collections,re
from pathlib import Path
P=Path(__file__).resolve().parent
exec((P/'sixth_prepare.py').read_text().split('qs=[]')[0])
ledger=json.loads((P/'pass-ledger.json').read_text());accepted=[x for f in ledger['decisionFiles'] for x in json.loads((P/f).read_text())['accepted']];current={r['Form_ID'] for r in csv.DictReader(open(P.parents[2]/'data/etymology-assignments.csv')) if r['Status']=='accepted'}
# Exact editorial equivalences; retain pronominal number and special nominal senses in the report.
groups={
'eat':['eat','eat!','he ate','eat/he ate','eat!, he ate','to eat'],
'see':['see','look','look!','he saw','look!, he saw','he sees/he saw','see/he saw','to see'],
'hear':['hear','listen','listen!','he heard','listen!, he heard','he hears/he heard','to hear'],
'walk':['walk','walk!','he walked','walk!, he walked','walk!; he walked','walk/he walked','to walk'],
'drink':['drink','drink!','he drank','drink/he drank','drink!, he drank','to drink'],
'sleep':['sleep','sleep!','he slept','he sleeps','he sleeps; he slept','he sleeps/he slept','sleep/he slept','to sleep'],
'bite':['bite','bite!','he bit','bite/he bit','bite!; he bit','bite!, he bit','to bite'],
'kill':['kill','kill!','he killed','kill!; he killed','kill (the bird)','don’t kill/he killed',"don't kill!, he killed",'to kill'],
'die':['die','he died','He died.','don’t die/he died',"don't die!, he died",'to die'],
'sit':['sit','sit down','sit down!','he sat down','sit down/he sat down','sit down; he sat do','sit down!, he sat down','to sit'],
'lie':['lie down','lie down!','he lay down','lie down/he lay down','lie down!; he lay down','lie down!, he lay down','to lie down'],
'come':['come','come!','he came','come/he came','come!, he came','(you) come','to come'],
'go':['go','go!','he went','go/he went','go!, he went','to go'],
'give':['give','give!','he gave','give/he gave','give!; he gave','give!, he gave','(you) give!','to give'],
'run':['run','run!','he ran','run/he ran','run!; he ran','to run'],
'speak':['speak','speak!','he spoke','speak/he spoke','speak!, he spoke','to speak'],
'burn':['burn','it burns','it burned','it burns/it burned','it burns; it burned','to burn'],
'fly-verb':['to fly','it is flying','fly/it flies','it flies; it flew'],
'I':['I','I (1st sg)'],
'we':['we','we (1st pl, exclusive)','we (1st pl, inclusive)','we (incl.)','we (excl.)','we (inclusive)','we (exclusive)'],
'you':['you','you (2nd pl)','you (2nd sg, formal)','you (2nd sg, informal)','you (plural)','you (singular)'],
'he':['he','he (3rd sg, masculine)'],
'she':['she','she (3rd sg, feminine)'],
'he/she':['he/she','he/she (formal)','he/she (informal)'],
 'they':['they','they (3rd pl)'],
'hundred':['hundred','one hundred','100'],
'who':['who','who?'], 'what':['what','what?','what thing'], 'how many':['how many','how many?'],
'horn':['horn','horns'],'tooth':['tooth','teeth'],
'wheat':['wheat','wheat (husked)'],'millet':['millet','millet (husked)'],
'cold':['cold','cold (weather)','cold (water)'], 'hot':['hot','hot (water)','hot (weather)'],
'lightning':['lightning','bolt of lightning'],'big':['big','large'],'tumeric':['tumeric','turmeric']}
alias={x.lower():k for k,v in groups.items() for x in v}
def senses(g):
 g=g.strip();key=alias.get(g.lower())
 if key:return {key}
 return {alias.get(x.strip().lower(),x.strip().lower().rstrip('?!.')) for x in g.split(';') if x.strip()}
idx=collections.defaultdict(list)
for x in accepted:
 if not x['parent'][0].isdigit():continue
 r=x['record'];w=norm(r['Form'])
 if re.search(r'[,;/()\[\]]',w) or len(w)<2:continue
 idx[w].append(x)
cs=[]
for rr in csv.DictReader((P/'unresearched-records.csv').open()):
 if rr['ID'] in current:continue
 w=norm(rr['Form']);g=senses(rr['Gloss']);xs=[x for x in idx[w] if g and g<=senses(x['record']['Gloss'])]
 if not xs:continue
 by=collections.defaultdict(list)
 for x in xs:by[x['parent']].append(x)
 cs.append(dict(record=rr,parents=sorted(by),comparanda={k:v[:3] for k,v in by.items()},canonicalSenses=sorted(g)))
(P/'expansion-pass78-candidates.json').write_text(json.dumps(cs,ensure_ascii=False,indent=1));(P/'expansion-pass78-gloss-equivalences.json').write_text(json.dumps(groups,ensure_ascii=False,indent=1));grouped=collections.defaultdict(list)
for x in cs:grouped['/'.join(x['parents'])].append(x)
with (P/'expansion-pass78-review.txt').open('w') as f:
 for k,xs in sorted(grouped.items(),key=lambda kv:-len(kv[1])):f.write(k+' ('+str(len(xs))+'): '+'; '.join(sorted({x['record']['Language_ID']+' '+x['record']['Form']+' «'+x['record']['Gloss']+'»' for x in xs}))+'\n')
print(len(cs),len(grouped))

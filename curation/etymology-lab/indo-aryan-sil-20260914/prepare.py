"""Prepare review candidates; never writes the accepted overlay.
Only typographic equivalence is normalized. Semantic aliases are explicit.
"""
import csv,json,re,unicodedata,collections
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
def read(p):return list(csv.DictReader(p.open()))
def norm(s):
 s=unicodedata.normalize('NFC',s).strip().lower()
 for a,b in [('tʃʰ','ch'),('tʃh','ch'),('tʃ','c'),('dʒ','j'),('ɖ','ḍ'),('ʈ','ṭ'),('ɳ','ṇ'),('ɽ','ṛ'),('ɭ','ḷ'),('ɾ','r'),('ʃ','ś'),('ɦ','h'),('ʰ','h'),('ʱ','h'),('aː','ā'),('iː','ī'),('uː','ū'),('eː','ē'),('oː','ō')]:s=s.replace(a,b)
 s=unicodedata.normalize('NFC',s)
 # Candidate discovery folds vowel quantity and schwa, never consonant place.
 s=s.translate(str.maketrans('āīūēōəʌɪʊɛɔ','aiueoaaiueo'))
 s=''.join(c for c in unicodedata.normalize('NFD',s) if c != '\u0303')
 return unicodedata.normalize('NFC',s)
families=json.loads((P/'previous-families.json').read_text())
aliases={
'go!, he went':['go','go!','he went','go (imperative)','to go'],
'come!, he came':['come','come!','he came','to come'],
'give!, he gave':['give','give!','he gave','to give'],
'eat!, he ate':['eat','eat!','he ate','to eat'],
'drink!, he drank':['drink','drink!','he drank','to drink'],
'sleep!, he slept':['sleep','sleep!','he slept','to sleep'],
'lie down!, he lay down':['lie down','lie down!','he lay down','to lie down'],
'walk!, he walked':['walk','walk!','he walked','to walk'],
'speak!, he spoke':['speak','speak!','he spoke','to speak'],
'listen!, he heard':['listen','listen!','he heard','hear','to hear'],
'whole/all':['whole','all','complete'],
'sister (older/younger)':['sister','older sister','younger sister','elder sister'],
'brother (older/younger)':['brother','older brother','younger brother','elder brother'],
'child, son/boy, daughter/girl':['child','son','boy','daughter','girl'],
'son/daughter, boy/girl':['son','daughter','boy','girl'],
'I (1 person singular)':['I','I (1st person singular)','I (first person singular)'],
'I (1st person singular)':['I','I (1 person singular)'],
'you (2 person singular, informal)':['you','you (singular)','you (singular informal/formal)'],
'you (singular informal/formal)':['you','you (singular)','you (2 person singular, informal)'],
'we (inclusive/exclusive)':['we','we (inclusive)','we (exclusive)'],
'small; short':['small','short'],
'one hundred':['hundred','100'],
'horns':['horn','horns'],
'head':['head','forehead'],
'yesterday; tomorrow':['yesterday','tomorrow'],
'finger':['finger','fingers'],
'who?':['who'],
'what?':['what'],
'how many?':['how many','how much'],
'water':['water'],
'belly':['belly','stomach'],
'mouth':['mouth','face'],
'same':['same (like)'],
'whole':['whole','full'],
'old':['old (object)','old (thing)'],
'cold':['cold (weather)'],
'hot':['hot (weather)'],
'wheat':['wheat (husked)'],
'chicken':['chicken','hen','cock'],
'leaf':['leaf','leaves'],
'eye':['eye','eyes'],
'ear':['ear','ears'],
'leg':['leg','foot'],
'rope':['rope','string'],
'bone':['bone','bones'],
'fish':['fish','fishes'],
'tooth':['tooth','teeth'],
'arm':['arm','hand'],
'belly':['belly','stomach'],
}
extensions=json.loads((P/'word-extensions.json').read_text())
for i,q in enumerate(families):
 q['forms']=list(dict.fromkeys(q['forms']+extensions.get(str(i),'').split('|')))
 q['senses']={q['gloss'].lower(),*(s.lower() for s in aliases.get(q['gloss'],[]))}
 q['words']={norm(w) for w in q['forms'] if not re.search(r'[,;/ ()]',w)}
inv=json.loads((P/'inventory.json').read_text())
# Discover supported repeated families. These are candidates until manually reviewed.
out=[]
for r in inv:
 if r['previously_linked']:continue
 senses={s.strip().lower() for s in r['Gloss'].split(';')}
 word=norm(r['Form'])
 if re.search(r'[,;/ ()]',word):continue
 matches=[i for i,q in enumerate(families) if word in q['words'] and senses<=q['senses']]
 if matches:out.append({'record':r,'families':matches})
(P/'family-candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=1))
print('candidate records',len(out),'unique language/form/gloss',len({(x['record']['Language_ID'],x['record']['Form'],x['record']['Gloss']) for x in out}))
with (P/'family-candidate-review.txt').open('w') as f:
 for i,q in enumerate(families):
  rs=[x['record'] for x in out if i in x['families']]
  f.write(f"\n{i}. {q['parent']} {q['gloss']} ({len(rs)} records)\n")
  f.write('Evidence: '+' '.join(q['evidence'])+'\n')
  f.write('; '.join(l+': '+', '.join(sorted({r['Form'] for r in rs if r['Language_ID']==l})) for l in sorted({r['Language_ID'] for r in rs}))+'\n')
# Exact primary article/section content from compiled forms, with referenced main article.
ids={q['parent'] for q in families}|{q['parent'].split('-')[0] for q in families}
parents={r['ID']:r for r in csv.DictReader((ROOT/'cldf/forms.csv').open()) if r['ID'] in ids}
(P/'primary-parents.json').write_text(json.dumps(parents,ensure_ascii=False,indent=1))
with (P/'primary-parents.txt').open('w') as f:
 for k,r in parents.items():f.write(k+' '+r['Form']+' '+re.sub('<[^>]+>',' ',r['Etymology'])+'\n')

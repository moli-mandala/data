"""Complete source-first literal proposal; requires independent original review."""
from pathlib import Path
import json,hashlib,unicodedata,collections
H=Path(__file__).resolve().parent
P=H.parent
inventory=[json.loads(x) for x in (P/'scope/source-unit-inventory.jsonl').read_text().splitlines()]
data={}
def w(form,gloss='',language='Aranadan',tags=(),note='',uncertainty=None):
    return dict(form=unicodedata.normalize('NFC',form),gloss=gloss,language=language,tags=list(tags),note=note,typed_uncertainty=uncertainty)
def put(page,section,number,*forms):data[f'nair2013aranadan:p{page}:{section}:{number}']=list(forms)
for n,form in enumerate(['aranaaṭan','Aranaaṭan','araṇṭan','eṛanaaṭan'],1):
    put(9,'ethnonyms',n,w(form,note='Printed ethnic name occurrence. Source tentatively connects the names to Eranadu; this is an attributed proposal, not a validated derivation.'))
pairs={
 'nasalization':[('beḷḷẽ','veḷḷam','water'),('ñãã','ñaan','I'),('kõṭe','kuṭam','waterpot')],
 'medial-length':[('ulaakke','ulakka','pestle'),('ureelu','ural','mortar'),('avaanu','avan','he')],
 'final-vowel':[('aame','aama','tortoise'),('erumee','eruma','buffalo'),('talee','tala','head')],
 'initial-consonant':[('baṇṭi','vaṇṭi','cart'),('baḷe','vaḷe','net'),('cebi','cevi','ear')],
 'medial-consonant':[('Kooyi','kool̲i','fowl'),('naaji/naayi','naal̲i','a measure'),('puje/puye','pul̲a','stream')],
 'geminate-consonant':[('occa','oṛṛa','single'),('kayaccam','kayaṛṛam','steep ascent'),('necci','neṛṛi','forehead')],
}
for section,rows in pairs.items():
    for n,(target,control,gloss) in enumerate(rows,1):
        forms=[]
        for alt in target.split('/'):
            tags=['pron','1','sg'] if gloss=='I' else ['pron','3','sg'] if gloss=='he' else []
            forms.append(w(alt,gloss,tags=tags,note='Explicit slash-separated target alternatives.' if '/' in target else ''))
        forms.append(w(control,gloss,'Malayalam',note='Explicit Malayalam comparison; not an Aranadan attestation.',uncertainty={'type':'transcription','reason':'Provisional literal underlined l; native text omits printed lower stroke, independent notation review pending.'} if section=='medial-consonant' else None))
        put(10 if section=='geminate-consonant' else 9,section,n,*forms)
for n,form in enumerate(['naam','yem','n̲in'],1):
    note=['Used when there is a large group of people; source explicitly denies an inclusive/exclusive distinction.','Used for a small group; source explicitly denies an inclusive/exclusive distinction.','Source explicitly calls this the second singular form; underline on initial n retained literally.'][n-1]
    put(10,'pronoun-prose',n,w(form,tags=['pron','1' if n<3 else '2','pl' if n<3 else 'sg'],note=note))
put(10,'pronoun-prose',4,w('n̲iinu',language='Kannada',note='Explicit Ka. comparator; underline retained as printed, no inferred palatal or retroflex value.'))
expressions={
 'pronoun-examples':[('n̲iin eppoo pooggaa','when do you go?'),('n̲iin naaḷe baa','you come tomorrow'),('n̲iin kalaŋgate','you do not speak')],
 'inanimate-plurals':[('kuṛe maram','trees'),('kuṛe kallu','stones')],
 'dative-examples':[('niina beeṇoo','do you want'),('niina pooga','you may go')],
 'genitive-examples':[('nim pajj','your cow'),('aaṭu kuuṭu','goat’s pen'),('tooṭu arike','near the channel')],
}
for section,rows in expressions.items():
    for n,(form,gloss) in enumerate(rows,1):
        tags=['multiword-expression']
        if section in ('pronoun-examples','dative-examples'):tags+=['sentential']
        if section=='pronoun-examples' and n==1:tags+=['interr']
        put(10,section,n,w(form,gloss,tags=tags,note='Source illustrates '+{'pronoun-examples':'second-person singular usage','inanimate-plurals':'inanimate quantification','dative-examples':'absence of the dative suffix after pronouns','genitive-examples':'deleted genitive marking'}[section]+'.'))
put(10,'plural-prose',1,w('–kaḷ',language='Malayalam',tags=['suffix','pl'],note='Source says Malayalam uses this suffix with both animate and inanimate nouns.'))
put(10,'plural-prose',2,w('–kaḷ',tags=['suffix','pl'],note='Source restricts Aranadan use to animate nouns.'))
put(10,'plural-prose',3,w('kuṛe',note='Called a quantitative adjectival form; used to denote plural inanimate nouns. No separate English lexical definition supplied.'))
put(10,'dative-prose',1,w('–kku',tags=['suffix','dat'],note='Explicitly stated NOT to be used after pronouns; preserve the source restriction, not an invented pronoun inflection.'))
for n,(form,tense) in enumerate([('-um','fut'),('–uppe','pst'),('-ippe','pst'),('–ppe','pst'),('–uge','pres'),('-utu','pres'),('atu','pres')],1):
    put(10,'tense-suffixes',n,w(form,tags=['suffix',tense],note='Explicit tense suffix; printed boundary-mark presence preserved. Future is compared with Malayalam; no donor edge inferred.'))
table=[
 ('tinj','to eat','tinuppe','tinjuge','tinjum'),('keeyu','to cry','keejuppe','keejuge','keejum'),
 ('ceyyu','to do','ceyyuppe','ceyyuge','ceyyum'),('poovu','to go','pooyppe','poovuttu','poogum'),
 ('koll','to kill','konduppe','kondatu','kollum'),('tall','to beat','talluppe','tallutu','tallum'),
 ('biiyu','to fall','biinduppe','biivitu','biiyum'),('kalaŋg','to talk','kalŋguppe','kalaŋgtu','kalaŋgum.'),
]
for n,(stem,gloss,*forms) in enumerate(table,1):
    for col,form in zip(('stem','past','present','future'),[stem,*forms]):
        tags=['verb']+({'past':['pst'],'present':['pres'],'future':['fut'],'stem':[]}[col])
        put(10,'verb-paradigm',f'{n}:{col}',w(form,gloss.removeprefix('to '),tags=tags,note=f'Printed {col} column; lexical sense supplied by the stem row. No person/number inferred.'))
for n,(form,gloss) in enumerate([('ceeṭṭe','housefly'),('cadaḷenne','dandruff'),('kummam','cold'),('ceŋŋane','ear-wax'),('arccu','fin of fishes'),('kummen','hunch back')],1):
    put(11,'vocabulary',n,w(form,gloss,note='Disease sense explicitly specified by source.' if n==3 else ''))
records=[]
for unit in inventory:
    key=unit['source_unit_key']
    forms=data.pop(key,[])
    assert len(forms)==unit['expected_physical_witness_slots'],(key,len(forms),unit['expected_physical_witness_slots'])
    for i,form in enumerate(forms,1):form['witness_key']=key+f':w{i}'
    records.append(dict(**unit,witnesses=forms,raw_source_page_sha256=hashlib.sha256((P/'scope'/unit['raw_evidence']).read_bytes()).hexdigest(),
        first_reading_status='literal proposal awaiting independent review' if forms else 'documentary/context symbols preserved in raw page and positioned spans',
        accepted_for_installation=False))
assert not data
output=H/'complete-first-reading.jsonl'
content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records)
if output.exists():assert output.read_text()==content,'Preserve immutable first reading; use correction derivative.'
else:output.write_text(content)
report=dict(status='complete first literal proposal, not accepted transcription',units=len(records),
    witness_slots=sum(len(r['witnesses']) for r in records),languages=dict(collections.Counter(w['language'] for r in records for w in r['witnesses'])),
    inventory_sha256=hashlib.sha256((P/'scope/source-unit-inventory.jsonl').read_bytes()).hexdigest(),
    first_reading_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),source_keys_equal=True,
    notes=['Whole expressions retained, no synthetic word lemmas.','PUA interpreted provisionally from original glyphs and independent notation controls; no phonemic inference.','Contexts retain complete raw page evidence; no context symbol invented as a lexical lemma.','Every glyph/metadata decision requires independent review before emission.'])
(H/'first-reading-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))

"""Source-English grammatical analysis, never internal Kagani segmentation.
Raw interlinear strings remain in audit and source Notes when normalized here.
"""
# Explicit constructions and translations; neither spelling identity nor language
# comparison creates an etymology or Variant_Of edge.
EXACT={
 'is':('be',['verb','copula','pres']), 'am':('be',['verb','copula','pres','first-person','sg']),
 'was':('be',['verb','copula','pret']), 'were':('be',['verb','copula','pret']),
 'will-go':('go',['verb','fut']), 'will-say':('say',['verb','fut']),
 'will-take-off':('take off',['verb','fut']), 'will-be':('be',['verb','copula','fut']),
 'goes':('go',['verb','pres','third-person','sg']),
 'remainest':('remain',['verb','pres','second-person','sg']),
 'makest':('make',['verb','pres','second-person','sg']),
 'may-make':('make',['verb','subjunctive']),
 'they-may-say':('say',['verb','subjunctive','third-person','pl']),
 'we-may-eat':('eat',['verb','subjunctive','first-person','pl']),
 'dying-am':('die',['verb','pres','first-person','sg']),
 'eating-are':('eat',['verb','pres']),
 'come-is':('come',['verb','copula','pres']),
 'being-is':('be',['verb','copula','pres']),
 'become-is':('become',['verb','pres']),
 'coming-is':('come',['verb','pres']),
 'going-is':('go',['verb','pres']),
 'covered-is':('covered',['participle','copula','pres']),
 'good-is':('good',['adj','copula','pres']),
 'strong-is':('strong',['adj','copula','pres']),
 'mine-is':('mine',['pron','gen','copula','pres']),
 'thine-is':('thine',['pron','gen','copula','pres']),
 'has-been-taken':('take',['verb','perfect']),
 'being-found':('find',['verb','participle','pres']),
 'causing-to-be-put-on':('put on',['verb','participle','caus']),
 'getting-made':('make',['verb','participle']),
 'by-me':('I',['pron','first-person','sg','erg']),
 'by-thee':('thou',['pron','second-person','sg','erg']),
 'by-him':('he',['pron','third-person','sg','erg']),
 'by-them (him)':('they (he)',['pron','third-person','erg']),
 'by-whom':('who',['pron','erg']),
 'him-to':('him',['pron','third-person','sg','dat']),
 'me-to':('me',['pron','first-person','sg','dat']),
 'this-to':('this',['pron','demonstrative','dat']),
 'you-of':('you',['pron','second-person','gen']),
 'by-son':('son',['noun','erg']), 'father-by':('father',['noun','erg']),
 'one':('one',['num']), 'two':('two',['num']),
}
for form,base in [('became','become'),('began','begin'),('came','come'),('collected','collect'),('found','find'),('gone','go'),('met','meet'),('remained','remain'),('thought','think'),('went','go')]:
    EXACT[form]=(base,['verb','pret'])
for ambiguous in ['collected','found']:
    EXACT[ambiguous]=(EXACT[ambiguous][0],['verb'])
EXACT['gone']=('go',['verb','participle'])
for form,base in [('coming','come'),('dividing','divide'),('dying','die'),('falling','fall'),('going','go'),('grazing','graze'),('joining','join'),('living','live'),('making','make'),('playing','play'),('singing','sing'),('spending','spend'),('walking','walk')]:
    EXACT[form]=(base,['verb','participle','pres'])
for form in ['asked','done','given','heard','left','ruined','said','seen']:
    base={'done':'do','given':'give','heard':'hear','left':'leave','said':'say','seen':'see'}.get(form,form.removesuffix('ed'))
    if form=='ruined':base='ruin'
    if form=='asked':base='ask'
    EXACT['was-'+form]=(base,['verb','pret'])
for form,base in [('arisen','arise'),('brought','bring'),('called','call'),('gone','go'),('made','make'),('run','run'),('seized','seize'),('taken','take')]:
    EXACT['having-'+form]=(base,['verb','conjunctive-participle'])
EXACT['having-been-sated']=('sate',['verb','conjunctive-participle'])
for base in ['be','make','persuade','say']:
    EXACT['to-'+base]=(base,['verb','inf'])

def analyze(raw):
    gloss,tags=EXACT.get(raw.lower(),(raw,[]))
    # Case is validated against the keyed source paradigm in the builder;
    # English by alone is not an ergative declaration.
    if raw.lower().startswith('by-') or raw.lower()=='father-by':
        tags=[t for t in tags if t!='erg']
    return gloss,tags


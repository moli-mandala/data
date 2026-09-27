"""Conservative metadata from this source's explicit interlinear labels.

English relation words do not determine a unique Holiya case; those functions
remain observations. English bare plural nouns do not establish a plural suffix.
"""
PRONOUNS = {
'i':['1sg'],'me':['1sg'],'my':['1sg','poss'],'mine':['1sg','poss'],'my-own':['1sg','poss','refl'],
'we':['1pl'],'our':['1pl','poss'],'thou':['2sg'],'thee':['2sg'],'thy':['2sg','poss'],'thine':['2sg','poss'],
'you':['second-person'],'your':['second-person','poss'],'he':['3sg'],'him':['3sg'],'his':['3sg','poss'],
'himself':['3sg','refl'],'they':['3pl'],'them':['3pl'],'anybody':['indef'],'anyone':['indef'],
'anyone-even':['indef','emph'],'this':['demonstrative'],'that':['demonstrative'],'all-this':['demonstrative'],
'such':['demonstrative'],'what':['interr'],'which':['relative']}
NOUNS=set('anger bread brother cash cloth clothes command convict country dance days deed difficulty door enquiry famine father feast food friend friends goat god gold goods gōmā gōpāḷā hunger husk inhabitant interest kid kiss loan man master men merriment money music name neck night noise order ornaments pity place pot property report ring riotousness room rupees rāmā scarcity sense servant servants service share shares shoe shoes sin son sons suspicion swine trace wealth word work years'.split())
ADJECTIVES=set('alive all another best big distressed elder eldest far good great half happy heavy living many merry more proper safe safe-and-sound small some wicked worthy young'.split())
TEMPORAL=set('again always at-night before-yesterday-the-day ever in-the-morning morning-at now then the-day-before-yesterday till-now to-day when'.split())
SPATIAL=set('anywhere before far-from near near-from nearer out there where together'.split())
CONJ=set('and but because if that-is-to-say therefore'.split())
POSTP=set('about for in like to towards with'.split())
PAST=set('appeared asked ate became became-hungry became-not broke-not called came consented-not devoured did did-not did-not-enter died embraced entreated feasted fell found gave gave-not gavest gavest-not heard it-appeared kept made neglected-not opened ran said saw sent shut spent stand-not stayed transgressed-not wasted went went-not'.split())
SIMPLE=set('bring consider die do eat eats give go keep make makes put put-on say see'.split())
PARTICIPLES=set('arising asking becoming bringing coming dancing dividing eating giving happening keeping rattling running saying seeing singing sleeping stealing taking thinking'.split())
NUMERALS=set('eight hundred one twenty two'.split())
NOUN_BASES=NOUNS|{'body','belly','day','field','finger','foot','friday','hand','harlots','house','man','matter','neighbourhood','senses','theft','time','village'}

def classify(unit):
    g=unit.get('emission_gloss',unit['gloss']).lower().replace('(?)','')
    tags=[]; observations=[]
    if g in {'gōmā','gōpāḷā','rāmā'}:tags=['noun','proper-noun']
    elif g in PRONOUNS:tags=['pron']+PRONOUNS[g]
    elif g in NUMERALS:tags=['num']
    elif g in NOUNS:tags=['noun']
    elif g in ADJECTIVES:tags=['adj']
    elif g in TEMPORAL:tags=['adv','temporal']
    elif g in SPATIAL:tags=['adv','spatial']
    elif g in CONJ:tags=['conj']
    elif g in POSTP:tags=['postp']
    elif g in {'o'}:tags=['interj','voc']
    elif g in {'so','so-also','thus','how','well','approximately'}:tags=['adv','manner']
    elif g in {'even'}:tags=['part','emph']
    elif g=='not':tags=['part','neg']
    elif g in {'is','was','were','art','were-not'}:tags=['verb','copula','pret' if g.startswith(('was','were')) else 'pres']
    elif g in PAST:tags=['verb','pret']
    elif g in SIMPLE:tags=['verb']
    elif g in PARTICIPLES:tags=['verb','participle']
    elif g.startswith('having-'):tags=['verb','conjunctive-participle']
    elif g.startswith('to-') and g not in {'to-him','to-the-father','to-day-of'}:tags=['verb','inf']
    elif g.startswith(('will-','i-will-')):tags=['verb','fut']
    elif g.startswith(('has-','have-','had-','would-have-')):tags=['verb','perfect']
    elif g.startswith(('was-','were-')):tags=['verb','pret']
    elif g.startswith(('is-','it-may-','may-')):tags=['verb','pres']
    elif g.endswith(('-was','-is')):tags=['verb','pret' if g.endswith('-was') else 'pres']
    elif g in {'broken','eaten','given','given-if','known'}:tags=['verb','participle']
    elif g in {'caused-to-eat'}:tags=['verb','caus','pret']
    elif g in {'let-us-make'}:tags=['verb','1pl','impv']
    elif g in {'go-would-not'}:tags=['verb','neg']
    elif g.startswith(('he-','i-','we-','thou-')):
        person={'he':'3sg','i':'1sg','we':'1pl','thou':'2sg'}[g.split('-')[0]]
        if g=='he-also':tags=['pron',person,'emph']
        else:tags=['verb',person]+(['pret'] if g.split('-',1)[1] in PAST|{'madest','arose'} else [])
    elif g.startswith('the-') and g[4:] in NOUNS:tags=['noun']
    elif g in {'the-small','the-younger'}:tags=['adj']
    elif g in {"father's",'goat’s'}:tags=['noun','poss']
    elif g in {'word-played','spent-on'}:tags=['verb','pret']
    elif g in {'a-dog','places-in'}:tags=['noun']
    elif g=='there-and':tags=['adv','spatial','conj']
    elif g in {'going-on','it-seems'}:tags=['verb','pres']
    else:
        # Transparent English relational labels give lexical class but no
        # invented unique source case, suffix boundary or person morphology.
        prefix=g.split('-')[0]
        if prefix in PRONOUNS:tags=['pron']+PRONOUNS[prefix]
        elif prefix in NOUN_BASES or g.startswith('the-servants-'):tags=['noun']
        elif prefix in {'worthy'}:tags=['adj']
        elif g in {'hundred-in'}:tags=['num']
        elif g in {'so-many'}:tags=['quantifier']
        elif g in {'to-him'}:tags=['pron','3sg']
        elif g in {'to-the-father'}:tags=['noun']
        elif g in {'to-day-of'}:tags=['adv','temporal']
        elif g in {'by-name','name-by'}:tags=['noun']
        elif g in {'with-friends','with-hunger','in-service'}:tags=['noun']
        elif g in {'room-'}:observations.append('Physical continuation atom; emitted with following -dā.')
        else:observations.append('Lexical class needs source-specific review; no guessed tag.')
        if tags:observations.append('Printed English relational/function label retained; no unique Holiya case inferred from English alone.')
    if g.endswith('-not') or '-not-' in g:tags.append('neg')
    if g in {'was-given','was-seen','was-found','was-not-found','has-been-found','is-got','is-prepared'}:tags.append('pass')
    if 'may-' in g:observations.append('Printed English expresses possibility; no specific Holiya mood form inferred.')
    if g in {'thine-indeed'}:tags.append('emph')
    return list(dict.fromkeys(tags)),observations

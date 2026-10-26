"""Reproduce manually read grammar evidence; not an installation importer."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
rows = []


def add(page, block, form, gloss, route='gumsur-bod-standard', note='', tags=(), witness='', uncertainty=''):
    n = 1 + sum(r['printed_page'] == page and r['block'] == block for r in rows)
    tags = list(tags)
    for person in ('1','2','3'):
        if person in tags:
            number = 'sg' if 'sg' in tags else 'pl'
            tags = [t for t in tags if t not in (person, number)] + [person+number]
    if '1sg' in tags:
        tags = [t for t in tags if t not in ('incl','excl')]
    tag_map = {'masc':'m','fem':'f','neut':'n','prs':'pres','imp':'impv',
               'part':'participle','incl':'inclusive','excl':'exclusive','dem':'demonstrative'}
    tags = [tag_map.get(t,t) for t in tags]
    rows.append(dict(source_unit_key=f'grierson1906kui:p{page}:{block}:{n}', printed_page=page,
                     physical_page=page+20, block=block, form=form, gloss=gloss,
                     route_key=route, source_note=note, tags=list(tags), witness=witness,
                     uncertainty=uncertainty, status='first reading; independent review pending'))


add(457, 'self-name', 'Ku', 'name used by the people themselves', 'unlocalized-kui',
    'Source tentatively relates this self-name to Kōī, while explicitly distinguishing the Kōī dialect of Gondi from Kui.')
add(457, 'name-comparison', 'Kōī', 'one of the names which the Gonds use to denote themselves', 'koi-gondi-control',
    'Lexical self-name comparison, not an identification of the two languages.')

note = 'Source contrasts standard short a with a broader Khondmals sound, probably like English all; specimens often spell it o.'
add(459, 'pronunciation-a', 'raṭā', 'big', note=note)
add(459, 'pronunciation-a', 'roṭā', 'big', 'khondmals', note)
note = 'Letchmajee reports shortening of long ā before ĩ and h or a double consonant; Grierson says the specimens do not confirm this rule.'
for f,g in [('aĩ','I come'),('ādi','thou comest'),('massē','I was')]:
    add(459, 'letchmajee-shortening', f,g,note=note,witness='Lingum Letchmajee')
add(459, 'specimen-length', 'massē', '',note='Repeated comparison spelling in the explanation of consonant simplification.',witness='Lingum Letchmajee')
for f,g in [('māsē','I was'),('āĩ','I am'),('ēāñju','he')]:
    add(459, 'specimen-length',f,g,note='Source says specimens simplify the consonant and preserve the long vowel.',witness='specimens (not individually localized)')
for f,w in [('essē','Lingum Letchmajee'),('ēssē','Major Smith'),('ēsē','specimens (not individually localized)')]:
    add(459, 'e-length',f,'',note='Source contrasts length of e before a double consonant; no separate gloss supplied.',witness=w)
add(459,'e-length','māchē','I am','kalahandi','Source says Kalahandi replaces the double ss by ch; preserve this printed gloss, distinct from later I was.')
for f,g,r in [('gōḍā','horse','gumsur-bod-standard'),('gōṛā','horse','khondmals'),('pāḍu','milk','gumsur-bod-standard'),('pālu','milk','chinna-kimedi'),('iḍḍu','house','gumsur-bod-standard'),('illu','house','chinna-kimedi'),('nēgāli','a good woman','kalahandi'),('nēgāri','a good woman','gumsur-bod-standard')]:
    add(459,'liquid-comparison',f,g,r,'Explicit source comparison: Khondmals often has ṛ for ḍ; Chinna Kimedi l instead; Kalahandi often l for r.')
for f,g,r in [('bēstēñju','he said','orissa-central-provinces'),('vesteñju','he said','gumsur-bod-standard'),('ēbāñju','he','khondmals'),('ēbāru','they','khondmals'),('ēāñju','he','gumsur-bod-standard'),('ēāru','they','gumsur-bod-standard')]:
    add(459,'v-b-comparison',f,g,r,'Source describes v becoming b regionally; the standard pronoun is also reported in Kalahandi.')
add(459,'cognate-comparison','vāḍu','he','external-Telugu','Source compares these forms to argue that standard Kui dropped v.')
add(459,'cognate-comparison','avanu','he','external-Kanarese','Source compares these forms to argue that standard Kui dropped v.')
add(459,'cognate-comparison','ēāñju','he',note='Source tentatively compares this form with Kanarese avanu.')
add(459,'cognate-comparison','avanu','he','external-Kanarese','Repeated source comparison with ēāñju.')
add(459,'possessive-comparison','ēāñji','his','kalahandi','Source describes substitution of ñj for n in connected languages.')
add(459,'possessive-comparison','ēāni','his',note='Standard comparison with Kalahandi ēāñji.')

add(460,'gender','gināri','the woman, or animal, that does',note='Source states that feminine and neuter singular have the same form, and third-person feminine/neuter plurals also coincide.',tags=['part','fem','neut','sg'])
for f,t in [('ru',['masc','pl']),('wi',['fem','neut','pl']),('ga',['fem','neut','pl']),('ska',['fem','neut','pl'])]:
    add(460,'plural-suffix',f,'',note='Explicitly identified plural suffix; ru is used for men, the others for feminine and neuter.',tags=['suffix']+t)
for f,r in [('ru','external-Kanarese-Tamil'),('gaḷ','external-Kanarese-Tamil'),('ōr','external-Gondi'),('ṅg','external-Gondi'),('k','external-Gondi')]:
    add(460,'plural-comparison',f,'',r,'Explicit source comparison of plural suffixes; Kanarese/Tamil shared attribution is not arbitrarily resolved to one language.',tags=['suffix','pl'])
for f,g,r in [('siṅgi','five','gumsur-bod-standard'),('sajgi','six','gumsur-bod-standard'),('añju','five','external-Tamil'),('saiyūṅg','five','external-Gondi'),('ār̤u','six','external-Tamil'),('sārūṅg','six','external-Gondi')]:
    add(460,'numeral-comparison',f,g,r,'Source reports older numerals being replaced by Aryan forms, but Dravidian forms still used through seven.',tags=['num'],uncertainty='Tamil r bears a lower diacritic; precise encoding requires typography-key review.' if f=='ār̤u' else '')
add(460,'inclusive','āju tinā','we shall eat','khondmals','Source calls this ordinary inclusive plural in the first Khondmals specimen, unlike Letchmajee’s dual description. Inclusive plural is said apparently absent in Kalahandi.',tags=['multiword-expression','verb','1pl','incl','fut'])
add(460,'adverbial-morphology','nā','',note='Source says adding nā to the verbal participles forms adverbial phrases; preserve the quoted morpheme without supplying a lexical gloss.',tags=['suffix'])
add(460,'adverbial-expression','pāginā-vio','when beating',note='Source compares these constructions with Telugu relative participles, and says ordinary Kui relative participles are slightly different.',tags=['verb','part'])
add(460,'adverbial-expression','pāgān-āi','having become a beater, having beaten',note='Two source paraphrases for the same printed expression; no unattested base reconstructed.',tags=['verb','part'])
add(460,'smith-comparison','mrānū-gāṭāñju','the owner of the mangoes',note='Grierson contrasts Smith’s longer vowels and simplified consonants with Letchmajee, and says his added tale interlinear translation is uncertain in one or two places.',witness='Major Smith')
add(460,'smith-comparison','mrānu-gaṭṭañju','the owner of the mangoes',note='Letchmajee comparison quoted by Grierson; preserve literal length and doubling rather than normalize to Smith.',witness='Lingum Letchmajee')

for f,g,t in [('ru','',['suffix','masc','pl']),('ābā','father',['noun','sg']),('ābāru','fathers',['noun','pl']),('tinañju','an eater',['noun','sg']),('tināru','eaters',['noun','pl']),('skā','',['suffix','pl']),('kā','',['suffix','pl']),('gā','',['suffix','pl']),('āṅgi-skā','sisters',['noun','pl']),('vihā-kā','bundles of straw',['noun','pl']),('kōḍi-ṅgā','cows',['noun','pl']),('lāveñju','a young man',['noun','masc','sg']),('lāveṅgā','young men',['noun','masc','pl']),('āri','',['suffix','fem']),('āwi','',['suffix','fem','pl']),('tināri','she who eats',['part','fem','sg']),('tināwi','they who eat',['part','fem','pl'])]:
    add(462,'noun-prose',f,g,note='Source noun-number explanation; plural glosses follow the explicitly paired paradigm rather than supplying new lexical senses.',tags=t)
for f,g in [('lāveñju','a young man'),('dādā','elder brother')]:
    add(462,'noun-table-head',f,g,tags=['noun'])
for case, forms in [('nom',['lāveñju','lāveṅgā','dādā','dādāru']),('acc',['lāvēni','lāveṅgāni','dādāni','dādāri']),('dat',['lāvēniki','lāveṅgāniki','dādāki','dādāriki']),('gen',['lāvēni','lāveṅgāni','dādāni','dādāri'])]:
    for col,f in enumerate(forms,1):
        add(462,f'noun-table-{case}-c{col}',f,'young man' if col<3 else 'elder brother',
            note='Literal cell under noun lemma and singular/plural heading; case meaning carried by structured labels, not invented English paraphrase.',
            tags=['noun',case,'sg' if col%2 else 'pl'],
            uncertainty='')
for f,g in [('tini','to'),('taṅgi','to'),('tiṅgi','to'),('tiki','to'),('tai','from'),('dai','from'),('ṭoṭi','from'),('ṭāktā','from'),('tanni','in'),('tā','in'),('lai','in'),('lai-tā','in'),('wāhā-tā','near'),('kā','with')]:
    add(462,'noun-margin-postpositions',f,g,note='Explicitly called postpositions and said to attach to the same form as dative suffix ki.',tags=['postp'],uncertainty='Fine lower marks and consonant cluster need independent confirmation.' if f in ['ṭoṭi','ṭāktā'] else '')
add(462,'noun-margin-dative','ki','',note='Explicitly called the dative suffix in the postposition explanation.',tags=['suffix','dat'])
for f,g in [('ĩju',''),('lāveñju','a young man'),('dādā','elder brother')]:
    add(462,'noun-margin-shape',f,g,note='Source states nouns ending in ĩju form singular as lāveñju, other nouns as dādā; isolated ending retained as quoted morphology.',tags=['suffix'] if f=='ĩju' else ['noun'])
add(462,'adjective-morphology','gaṭṭa','',note='Source says adjectives are indeclinable and may be formed from nouns with gaṭṭa.',tags=['suffix'])
add(462,'adjective-example','dāṭā-gaṭṭa','strength-having, strong',note='Source explicitly exemplifies adjective formation by gaṭṭa.',tags=['adj'])

pron_cols=[('I',['1sg']),('we (inclusive)',['1pl','incl']),('we (exclusive)',['1pl','excl']),('thou',['2sg']),('you',['2pl']),('who?',['interr'])]
for case,forms in [('nom',['ānu','āju','āmu','īnu','īru','imbāi']),('dat',['naṅgē','ammaṅgē','maṅgē','niṅgē','miṅgē','imbāriki']),('gen',['nā, naĩ','ammāni','mā, maĩ','nī','mī','imbāri'])]:
    for col,(f,(g,t)) in enumerate(zip(forms,pron_cols),1):
        add(462,f'pronoun-table1-{case}-c{col}',f,g,tags=['pron',case]+t,note='Literal source cell; comma-separated alternatives retained together pending explicit source-aware expansion.')

pron_cols=[('he',['3sg','masc']),('she, it',['3sg','fem','neut']),('they',['3pl','masc']),('they',['3pl','fem','neut']),('self',['refl','sg']),('selves',['refl','pl'])]
for f,g,t in [('nanna','me','1sg'),('mamma','us','1pl'),('ninna','thee','2sg'),('mimma','you','2pl')]:
    add(462,'pronoun-table1-margin',f,g,'chinna-kimedi','Separate accusatives explicitly attributed to Chinna Kimedi; source says Bod and Gumsur use the dative instead.',tags=['pron','acc',t])
for f in ['umbāi','imbāri','imbāru']:
    add(462,'pronoun-table1-margin',f,'who?',note='Source explicitly gives additional forms, without defining their individual gender/number here.',tags=['pron','interr'])
for case,forms in [('nom',['ēañju','ēri, ērā','ēāru','ēwi, ēwa (-skā)','tānu','tāru']),('acc',['ēāni','ērāni','ēāri','ēwaskāni','tānā','tārā']),('dat',['ēāniki','ērāniki','ēāriki','ēwaskāniki','tānāki','tārāki']),('gen',['ēāni','ērāni','ēāri','ēwaskāni','tānā','tārā'])]:
    for col,(f,(g,t)) in enumerate(zip(forms,pron_cols),1):
        add(462,f'pronoun-table2-{case}-c{col}',f,g,tags=['pron',case]+t,note='Literal source cell; gender and number apply only through the corresponding source column. Parenthetical -skā remains literal pending scope review.')
for f,g,t in [('ēañju','he',['pron','3sg','masc']),('ēwañju','he',['pron','3sg','masc']),('oañju','he',['pron','3sg','masc']),('Tānu','self',['pron','refl']),('trai','self (feminine)',['pron','refl','fem']),('trā','of self',['pron','refl','gen']),('tānā','of self',['pron','refl','gen'])]:
    add(462,'pronoun-table2-margin',f,g,note='Source gives alternate forms and says the genitive often has trā instead of tānā.',tags=t)
for f,g,t in [('ēañju','he',['pron']),('iañju','this man',['pron','dem']),('estañju','who?',['pron','interr']),('Innā','what?',['pron','interr']),('Innāri','what?',['pron','interr']),('ēri','it',['pron']),('ā','that',['pron','dem']),('ē','that',['pron','dem']),('ō','that',['pron','dem']),('i','this',['pron','dem']),('esti','which?',['pron','interr'])]:
    add(462,'pronoun-prose',f,g,note='Source groups iañju/estañju with ēañju inflection; Innā is indeclinable, Innāri inflected like ēri; final five are adjectival pronouns.',tags=t)

verb_columns=[('beat','pāga'),('see','mehpa'),('cut','kōpa'),('become','āwa'),('do','giwa, gipka')]
principal462=[
    ('verbal-noun',['pāga','mehpa','kōpa','āwa','giwa, gipka'],['verb']),
    ('infinitive-purpose',['pāgboṇḍi','mehboṇḍi','kōboṇḍi','āboṇḍi','giboṇḍi'],['verb','inf']),
    ('conjunctive-present',['pāgi','mehpi','kōi','aĩ','gi(pk)i'],['verb','part','prs']),
    ('conjunctive-past',['pāga','mēha','kōa','āja','gia'],['verb','part','pret']),
    ('relative-present-future',['pāgini','mehni','kōni','āni','gini'],['verb','part','prs','fut']),
    ('relative-present-future-negative',['pāgāni','mēhāni','kōāni','āāni','giāni'],['verb','part','prs','fut','neg']),
    ('relative-past',['pāgiti','mehti','kōti','āti','giti'],['verb','part','pret']),
    ('relative-past-negative',['pāgāti','mēhāti','kōāti','āāti','giāti'],['verb','part','pret','neg']),
    ('present-future',['pāgiĩ','mehĩ','kōĩ','aĩ','giĩ'],['verb','prs','fut']),
    ('present-future-negative',['pāgānu','mēhānu','kōānu','āānu','giānu'],['verb','prs','fut','neg']),
]
for label,forms,tags in principal462:
    for col,(f,(g,head)) in enumerate(zip(forms,verb_columns),1):
        add(462,f'principal-{label}-c{col}',f,g,note=f'Source row: {label}; lexical column: {head}, to {g}. Preserve literal cell punctuation/optional segment notation until reviewed expansion.',tags=tags,
            uncertainty='The source consonant after me has a lower mark; h is a provisional transcription, not a phonemic identification. Stable crop: tmp/kui-grammar-see-review.png.' if col==2 else '')

principal463=[
    ('verbal-noun',['pāga','mehpa','kōpa','āwa','giwa, gipka'],['verb']),
    ('past',['pāgitē','mehtē','kōtē','ātē','gitē'],['verb','pret']),
    ('past-negative',['pāgātē (-nu)','mēhātē(-nu)','kōātē(-nu)','āātē(-nu)','giātē(-nu)'],['verb','pret','neg']),
    ('imperative',['pāgāmu','mehmu','kōmu','āmu','gimu'],['verb','imp']),
    ('imperative-negative',['pāgā','mēhā','kōā','āā','giā'],['verb','imp','neg']),
]
for label,forms,tags in principal463:
    for col,(f,(g,head)) in enumerate(zip(forms,verb_columns),1):
        add(463,f'principal-{label}-c{col}',f,g,note=f'Source row: {label}; lexical column: {head}, to {g}. Parenthesized -nu retained literally, no guessed scope expansion.',tags=tags,
            uncertainty='Consonant after me unresolved as in corresponding printed462 cell.' if col==2 else '')

person_rows=[
 ('1sg',['1sg'],['pāgiĩ','pāgānu','pāgitē','pāgātē(-nu)','','']),
 ('2sg',['2sg'],['pāgidi','pāgādi','pāgiti','pāgāti','pāgāmu','pāgā']),
 ('3sg-masc',['3sg','masc'],['pāgineñju','pāgañju','pāgitēñju','pāgātēñju','','']),
 ('3sg-fem-neut',['3sg','fem','neut'],['pāginē','pāgē','pāgitē','pāgātē','','']),
 ('1pl-incl',['1pl','incl'],['pāgina','pāgassu','pāgita','pāgātassu','','']),
 ('1pl-excl',['1pl','excl'],['pāgināmu','pāgāmu','pāgitāmu','pāgātāmu','','']),
 ('2pl',['2pl'],['pāgidēru','pāgēru','pāgitēru','pāgātēru','pāgātu','pāgātu']),
 ('3pl-masc',['3pl','masc'],['pāginēru','pāgēru','pāgitēru','pāgātēru','','']),
 ('3pl-fem-neut',['3pl','fem','neut'],['pāginu','pāju','pāgitu','pāgātu','','']),
]
for label,tags,forms in person_rows:
    for col,f in enumerate(forms,1):
        tense=['prs','fut'] if col<3 else ['pret'] if col<5 else ['imp']
        add(463,f'personal-{label}-c{col}',f,'beat' if f else '',
            note='Literal personal inflection cell; no forms supplied for empty source positions.',
            tags=['verb']+tags+tense+(['neg'] if col%2==0 else []) if f else [],
            uncertainty='Printed third-plural feminine/neuter present-negative cell looks pāju; damaged medial glyph requires independent review, no paradigm-based repair.' if f=='pāju' else '')
        if not f: rows[-1]['disposition']='source-empty-cell'

add(463,'irregular-head','Manba','to be',tags=['verb'])
copula=[('1-incl',['1','incl'],['','manna','','massa']),('1-excl',['1','excl'],['maĩ','mannāmu','massē','massāmu']),('2',['2'],['māñji','māñjēru','massi','massēru']),('3-masc',['3','masc'],['manneñju','mannēru','masseñju','massēru']),('3-fem-neut',['3','fem','neut'],['mannē','mannu','massē','massu'])]
for label,tags,forms in copula:
    for col,f in enumerate(forms,1):
        add(463,f'copula-{label}-c{col}',f,'be' if f else '',tags=['verb']+tags+(['sg'] if col%2 else ['pl'])+(['prs'] if col<3 else ['pret']) if f else [],note='Source copula cell; two singular-inclusive positions are explicitly dotted blanks, not recoverable omitted forms.')
        if not f: rows[-1]['disposition']='source-empty-cell'; rows[-1]['raw_cell']='......'
for f,g,t in [('mannu','be',['verb','imp','sg']),('mañju','be',['verb','imp','pl']),('siḍānu','I am not',['verb','neg','prs','1sg']),('siḍātē','I was not',['verb','neg','pret','1sg'])]:
    add(463,'copula-margin',f,g,note='Source gives imperative and corresponding negative verb, the latter regularly inflected.',tags=t)
add(463,'personal-margin','du','',note='Source says positive imperative plural ends in du when singular does not end in āmu.',tags=['suffix','imp','pl'])
add(463,'personal-margin','āmu','',note='Explicit quoted imperative ending used in the plural formation condition.',tags=['suffix','imp','sg'])
add(463,'personal-margin','gidu','do ye',note='Explicit example of the imperative plural rule.',tags=['verb','imp','2pl'])
for f,g,t in [('punba','to know',['verb']),('venba','to hear',['verb']),('tinba','to eat',['verb']),('puñji','know',['verb','part','prs']),('veñji','hear',['verb','part','prs']),('tiñji','eat',['verb','part','prs']),('puñja','know',['verb','part','pret']),('veñja','hear',['verb','part','pret']),('tiñja','eat',['verb','part','pret']),('salba','to go',['verb']),('sajji','go',['verb','part','prs']),('salla','go',['verb','part','pret']),('sajja','go',['verb','part','pret']),('saĩ','go',['verb','prs','fut']),('sajji','go',['verb','prs','fut','2sg']),('sāndru','go',['verb','prs','fut','2pl']),('sassē','go',['verb','pret']),('salmu','go',['verb','imp','sg']),('sajju','go',['verb','imp','sg']),('saldu','go',['verb','imp','pl'])]:
    add(463,'irregular-prose',f,g,note='Explicitly quoted irregular-verb paradigm forms; grammatical row context retained and English lexical gloss inherited only from its named paradigm.',tags=t)
add(463,'compound-head','manba','to be',note='Source forms compound tenses by adding this verb to present and past conjunctive participles.',tags=['verb'])
for f,g in [('gipki-maĩ','I am doing'),('gipki-massē','I was doing'),('gipkissē','I was doing'),('gia-maĩ','I have done')]:
    add(463,'compound-example',f,g,note='Source compound-tense examples; gipkissē is explicitly the Gumsur form.',tags=['verb','1sg'])
for f,t in [('kānu',['1sg']),('kādi',['2sg']),('kāñju',['3sg','masc']),('kāri',['3sg','fem','neut']),('kāmu (kāsu)',['1pl']),('kādu',['2pl']),('kāru',['3pl','masc']),('kāwi',['3pl','fem','neut'])]:
    add(463,'precative-morphology',f,'',note='Source calls this a kind of precative, formed by adding the listed person-number forms to the verbal noun. No independent verb gloss inferred.',tags=['suffix']+t)
add(463,'precative-example','salba-kāru','let them go',note='Explicit source precative example.',tags=['verb','3pl','masc'])
add(463,'conditional-morphology','ākā','',note='Source denotes condition by adding this to the past relative participle.',tags=['suffix','conditional'])
add(463,'conditional-example','ānu gitākā','if I do, or did',note='Two source temporal paraphrases; preserve full construction.',tags=['multiword-expression','verb','conditional','1sg'],uncertainty='Medial contraction in gitākā needs independent narrow-crop verification.')
add(463,'causal-morphology','Āki','',note='Source says this is added to the past relative participle to denote cause.',tags=['suffix'])
add(463,'causal-morphology','āwā','',note='Source says this is added to the verbal noun to denote cause.',tags=['suffix'],uncertainty='Quoted suffix reading and corresponding example require independent close comparison.')
add(463,'causal-example','ihē-igi gitāki','because (they) did so',note='Source causal example with past relative participle.',tags=['multiword-expression','verb'],uncertainty='Fine vowels of the complete expression need independent confirmation.')
add(463,'causal-example','giwāwā','because (they) did so',note='Source alternate causal construction; no free-variation edge inferred solely from shared paraphrase.',tags=['verb'],uncertainty='Fine vowels and suffix scope require independent confirmation.')
add(463,'ability','Muĩ','I can',note='Source adds this to the verbal noun.',tags=['verb','1sg'])
add(463,'ability','kuĩ','I will not',note='Source adds this to the verbal noun.',tags=['verb','1sg','neg','fut'])
add(463,'ability-example','āmu giwa mudāmu','we cannot do',note='Printed negative-ability construction; preserve source translation without inferring a new independent meaning for the quoted Muĩ.',tags=['multiword-expression','verb','1pl','neg'])
for f,t in [('tinañju',['masc']),('tināri',['fem','neut'])]:
    add(463,'verbal-noun-example',f,'an eater',note='Source derives verbal nouns from relative participles by adding third-person personal-pronoun endings.',tags=['noun']+t)
add(463,'relative-morphology','nā','',note='Source says a second set of relative participles is formed by adding nā to present and past conjunctive participles, always used when participles stand by themselves.',tags=['suffix'])
for f,g in [('pāginā-vio','when beating'),('ānā-vōā','when coming'),('pāgānai','having beaten'),('pāgānaṅga','having beaten'),('pāgānaikō','having beaten')]:
    add(463,'relative-adverbial-example',f,g,note='Source combines participles with particles to form adverbial clauses; shared translation alone does not establish free spelling variation.',tags=['verb','part'],uncertainty='Final adverbial particle vowels require independent close reading.' if f=='ānā-vōā' else '')
for f in ['Aĩ','ājā','ai-kō']:
    add(463,'old-conjunctive-participle',f,'',note='Explicitly called old conjunctive participles of āwa, to become; retain quotation without inventing an independently supplied translation.',tags=['verb','part'])
add(463,'old-conjunctive-head','āwa','to become',tags=['verb'])
add(463,'negative-conjunctive-morphology','ār-aṅgā','',note='Added to the past conjunctive participle to form the negative conjunctive compound.',tags=['suffix','neg'])
add(463,'negative-conjunctive-morphology','ān-aṅgā','','chinna-kimedi','Explicitly attributed Chinna Kimedi form used in the negative conjunctive compound.',tags=['suffix','neg'])
add(463,'negative-conjunctive-example','pāgār-aṅgā','not having struck',note='Source example of the compound negative conjunctive participle.',tags=['verb','part','neg'])
add(463,'passive-head','āwa','to become',note='Source says passive, not in common use, is formed by adding āwa to the verbal noun.',tags=['verb'])
add(463,'passive-example','giwa-āti','it is done',note='Explicit source passive analysis; preserve the qualification that it is not in common use.',tags=['verb','pass'])

for f,g,t in [('nanna','me','1sg'),('mamma','us','1pl'),('ninna','thee','2sg'),('mimma','you','2pl')]:
    add(469,'chinna-accusative',f,g,'chinna-kimedi','Letchmajee-derived notes: separate accusative forms, said identical to Kanarese.',tags=['pron','acc',t],witness='Lingum Letchmajee')
for f,r,t in [('di','gumsur-bod-standard','2sg'),('du','gumsur-bod-standard','2pl'),('ri','chinna-kimedi','2sg'),('ru','chinna-kimedi','2pl'),('ru','external-Telugu','2pl')]:
    add(469,'chinna-person-termination',f,'',r,'Source says present endings di/du are sometimes changed to ri/ru respectively in Chinna Kimedi; compares Telugu ru.',tags=['suffix','prs',t])
add(469,'chinna-unelided','ma','', 'chinna-kimedi','Source explicitly isolates ma in massē and says it is not elided in compound tenses; quoted component, not an independently glossed lexeme.')
add(469,'chinna-unelided','massē','I was','chinna-kimedi','Quoted full form in source explanation of unelided ma.',tags=['verb','pret','1sg'])
for f,g,r in [('gipki-massē','I was doing','chinna-kimedi'),('gipkissē','I was doing','gumsur-bod-standard'),('giā-massē','I had done','chinna-kimedi'),('giassē','I had done','gumsur-bod-standard')]:
    add(469,'chinna-compound',f,g,r,'Source explicitly contrasts the Chinna Kimedi unelided construction with the usual contracted form; the negative comparison is dialectal, not an unattested reconstructed word.',tags=['verb','1sg','pret'])
add(469,'chinna-negative-participle','ēāni suḍinā-aṅga','without having seen him','chinna-kimedi','Letchmajee-derived example, explicitly a negative conjunctive participle construction different from Gumsur; no continuous Chinna Kimedi specimen was available.',tags=['multiword-expression','verb','part','neg'])
for f,g,r in [('ilu','house','kalahandi'),('iḍḍu','house','gumsur-bod-standard'),('salāṅga','angry','kalahandi'),('suḍāṅgi','angry','gumsur-bod-standard'),('aḍā','a goat','kalahandi'),('māchē','I was','kalahandi'),('massē','I was','gumsur-bod-standard')]:
    add(469,'kalahandi-sound-comparison',f,g,r,'Source compares Kalahandi with standard and Chinna Kimedi; ḍ becomes l but aḍā retains ḍ. Source only conjectures that ch may be spelling of s to prevent sh pronunciation.')
for f,g in [('ilā','this'),('nēgāli','a good woman')]:
    add(469,'kalahandi-liquid',f,g,'kalahandi','Source says l often corresponds to r in other Kui dialects.')
for f,g in [('ēāñji','him'),('ēāñju','his')]:
    add(469,'kalahandi-nasal',f,g,'kalahandi','Source says ñj occurs where other Kui dialects have n; preserve his for this occurrence without homogenizing later he.')
add(469,'kalahandi-v-b','ēāñju','he, his','kalahandi','Source says the form agrees with standard in not pronouncing w between ē and ā.',tags=['pron'])
add(469,'kalahandi-v-b','bēstēñju','he said','kalahandi','In other cases w has become b, as in Khondmals.',tags=['verb'])
add(469,'kalahandi-accusative','i','','kalahandi','Explicit accusative suffix; this accusative form is commonly also used as dative.',tags=['suffix','acc'])
add(469,'kalahandi-accusative','ābāi','the father','kalahandi','Explicit accusative example, commonly used as dative as well.',tags=['noun','acc'])
add(469,'kalahandi-accusative','ēāñji','him','kalahandi','Explicit accusative example, commonly used as dative as well.',tags=['pron','acc'])
add(469,'kalahandi-dative','ilutu','to the house','kalahandi','Source proposes either Oriya influence or comparison with Telugu forms; no definite donor etymology asserted.',tags=['noun','dat'])
add(469,'kalahandi-pronoun','nāṅgē','me','kalahandi','Source says accusative personal-pronoun form identical to dative.',tags=['pron','1sg','acc','dat'])
add(469,'kalahandi-pronoun','Amu','we','kalahandi','Gloss and analysis continue on printed470: also used including the addressee; source expressly allows that this could be a mistake in the specimen, otherwise probably Oriya influence. No definite inclusive morphology asserted.',tags=['pron','1pl'])
rows[-1]['additional_printed_pages']=[470]

if __name__ == '__main__':
    out = ROOT / 'grammar-first-reading.jsonl'
    out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(f'{len(rows)} provisional occurrence records; no canonical data changed')

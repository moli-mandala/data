"""Independent source-first readings, assigned pp92–94 and97–101 pre-text.
No installation; raw print and explicitly supported expansions remain separate.
"""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE='tmp/pdfs/bailey1920/bailey1920.pdf'

def page92():
    out=[]
    def add(section,cell,raw,forms,gloss,tags,note='',basis=''):
        out.append(dict(source_unit_key=f'bailey1920kagani:p92:{section}:{cell}',printed_page=92,pdf_page=118,
          section=section,raw_source=raw,forms=forms.split(';') if forms else [],gloss=gloss,tags=tags.split(),
          source_analysis=note,expansion_basis=basis,source_pdf=SOURCE,
          review='First visual reading of full page3x and bounded6–7x crops; independent review pending'))
    def paradigm(section,stem,gloss,rows):
        for case,sg,pl,esg,epl in rows:
            for number,raw,expanded in [('sg',sg,esg),('pl',pl,epl)]:
                add(section,case+'-'+number,raw,expanded,gloss,'noun '+case+' '+number,
                    'Source heading Agent.' if case=='erg' else '',
                    ('Printed hyphen abbreviation expanded only with the explicit '+stem+' stem.' if '-' in raw else ''))
    paradigm('bear','rĭcch','bear',[
      ('nom','rĭcch','rĭcch','rĭcch','rĭcch'),
      ('gen','rĭcch-e dā','rĭcch-ā̃ dā','rĭcche dā','rĭcchā̃ dā'),
      ('dat','-e ko','-ā̃ ko','rĭcche ko','rĭcchā̃ ko'),
      ('abl','-e kōlo or thī','-ā̃ kōlo or thī','rĭcche kōlo;rĭcche thī','rĭcchā̃ kōlo;rĭcchā̃ thī'),
      ('erg','-e, -e sŭṇ','-ā̃, -ā̃ sŭṇ','rĭcche;rĭcche sŭṇ','rĭcchā̃;rĭcchā̃ sŭṇ')])
    for c,raw,forms,tags in [('nom-sg','pŭttŭr','pŭttŭr','nom sg'),('gen-sg','pŭttŭre dā','pŭttŭre dā','gen sg'),('nom-pl','pŭttŭr','pŭttŭr','nom pl'),('gen-pl','pŭttŭrā̃ dā','pŭttŭrā̃ dā','gen pl')]:
        add('son',c,raw,forms,'son','noun '+tags,'Printed prose example. Source does not print further cells represented by etc.; no additional forms inferred.')
    paradigm('man','jăṇ','man',[
      ('nom','jăṇ-ā','-e','jăṇā','jăṇe'),('gen','-e dā','-ĕā̃ dā','jăṇe dā','jăṇĕā̃ dā'),
      ('erg','-e, -e sŭṇ','-ĕā̃, -ĕā̃ sŭṇ','jăṇe;jăṇe sŭṇ','jăṇĕā̃;jăṇĕā̃ sŭṇ')])
    add('man','phonology','ṇ in jăṇā','ṇ','', '','Bailey describes ṇ here as less cerebral than in Panjabi. Source sound-symbol context, not an additional lexical head.')
    out[-1]['scope']='source_phonetic_context'
    paradigm('water','pāṇ','water',[
      ('nom','pāṇ-ī','-ī','pāṇī','pāṇī'),('gen','-īe dā','-īā̃ dā','pāṇīe dā','pāṇīā̃ dā'),('erg','-īe','-īā̃','pāṇīe','pāṇīā̃')])
    for cell,raw,forms,gloss,tags in [
      ('nom-sg','ghăr','ghăr','house','noun nom sg'),('nom-pl','ghăr','ghăr','house','noun nom pl'),
      ('in-sg','ghăre','ghăre','in the house','noun loc sg'),('in-pl','ghărā̃ bĭcc','ghărā̃ bĭcc','in the houses','noun loc pl'),
      ('from-sg','ghăro','ghăro','from the house','noun abl sg'),('from-pl','ghărā̃ thī or kōlo','ghărā̃ thī;ghărā̃ kōlo','from the houses','noun abl pl'),
      ('erg-sg','ghăre','ghăre','house','noun erg sg'),('erg-pl','ghărā̃','ghărā̃','house','noun erg pl')]:
        add('house',cell,raw,forms,gloss,tags,'Source places both in/from expressions under Loc.; the English from explicitly indicates ablative function.' if cell.startswith('from') else '')
    add('house','suffix','-o','-o','','suffix sg abl','Source says suffix -o is used only with singular; corresponding Panjabi -ō is a comparison, not a Kagani form. Plural uses a preposition (thī, kōlo).')
    add('house','panjabi-control','-ō','-ō','','suffix','Explicit Panjabi comparison; excluded from Kagani output.');out[-1]['scope']='comparison_control';out[-1]['source_language']='Panjabi'
    add('house','plural-prepositions','thī, kōlo','thī;kōlo','from','prep','Explicitly repeated plural analytic construction controls; same lexical readings also attested in the house paradigm.')
    for kin,gloss,stem,nom in [('father','father','pīū','pē'),('brother','brother','bhrū','bhrā-')]:
        for case,ending in [('nom',''),('gen',' dā'),('dat',' ko'),('abl',' kōlo, thī'),('erg','')]:
            raw=nom if case=='nom' else (stem+ending if kin=='father' else '-ū'+ending)
            forms=nom.rstrip('-') if case=='nom' else (stem+ending if case!='abl' else stem+' kōlo;'+stem+' thī')
            add(kin,case,raw,forms,gloss,'noun '+case+' sg',
                'Source identifies father and brother as irregular. Columns now distinguish nouns, not singular/plural.'+(' Source explicitly places accent on ī.' if kin=='father' and case=='gen' else ''),
                'Explicit bhrā- / -ū substitution.' if kin=='brother' and case!='nom' else '')
    paradigm('woman','trīmt','woman',[
      ('nom','trīmt','trīmt-ā̃','trīmt','trīmtā̃'),('gen','trīmtī dā','-ā̃ dā','trīmtī dā','trīmtā̃ dā'),
      ('erg','trīmtī, trīmtī sŭṇ','-ā̃, -ā̃ sŭṇ','trīmtī;trīmtī sŭṇ','trīmtā̃;trīmtā̃ sŭṇ')])
    for row in out:
        if row['section']=='woman':row['tags'].append('f')
        if any(' ' in x for x in row['forms']):row['source_analysis']+=' Some proposed answers are multiword; apply multiword-expression per emitted answer.'
        if 'ā̃' in ''.join(row['forms']):row['source_glyph_note']='Compressed paired upper marks retained as ā plus nasal tilde; physical stack is sometimes filled in scan. Not a new phonological analysis.'
    return out

def page93():
    out=[]
    def add(section,cell,raw,forms,gloss,tags,note='',basis=''):
        out.append(dict(source_unit_key=f'bailey1920kagani:p93:{section}:{cell}',printed_page=93,pdf_page=119,
            section=section,raw_source=raw,forms=forms.split(';') if forms else [],gloss=gloss,tags=tags.split(),source_analysis=note,
            expansion_basis=basis,source_pdf=SOURCE,review='Full page3x plus bounded7x crops; independent review pending'))
    for case,sg,pl,esg,epl in [
      ('nom','kŭṛh-ī','-īā̃','kŭṛhī','kŭṛhīā̃'),('gen','-ī dā','-īā̃ dā','kŭṛhī dā','kŭṛhīā̃ dā'),
      ('erg','-ī, -ī sŭṇ','-īā̃, -īā̃ sŭṇ','kŭṛhī;kŭṛhī sŭṇ','kŭṛhīā̃;kŭṛhīā̃ sŭṇ')]:
        for number,raw,forms in [('sg',sg,esg),('pl',pl,epl)]:add('girl',case+'-'+number,raw,forms,'girl','noun f '+case+' '+number,'Continues preceding feminine nouns.','Explicit printed kŭṛh- stem plus printed endings.')
    add('kinship','inflection','-ū','-ū','','suffix obl','Source explicitly says mother/sister/daughter inflect in -ū; no other inflections invented.')
    add('kinship','sister-alternatives','bhēhṇ or bhēṇ','bhēhṇ;bhēṇ','sister','noun f','Source offers the two head alternatives. Subsequent printed paradigm uses bhēhṇ only.')
    for section,head,gloss,oblique,agentlast in [('mother','mā','mother','māū','sŭṇ'),('sister','bhēhṇ','sister','bhēhṇū','sun'),('daughter','dhī','daughter','dhīū','sŭṇ')]:
        for case,raw,forms in [('nom',head,head),('gen',oblique+' dā',oblique+' dā'),('erg',oblique+', '+oblique+' '+agentlast,oblique+';'+oblique+' '+agentlast)]:
            add(section,case,raw,forms,gloss,'noun f sg '+case,'Sister agent postposition is printed plain sun in this occurrence, unlike adjacent sŭṇ; retain literal difference.' if section=='sister' and case=='erg' else '')
    add('eye','singular','ăkkh','ăkkh','eye','noun sg')
    add('eye','plural','ăkkhīā̃','ăkkhīā̃','eye','noun pl')
    tables=[('first','first-person','I','we',[
      ('nom','mē̃','ŭsī̃'),('gen','măṛā, mŭṛhā','ăsdā'),('dat acc','mŭ kŏ, mŭ kă','ăsā̃ ko'),
      ('abl','măṛe kōlo, mŭṛhe kōlo','ăsā̃ kōlo'),('erg','mē̃','ŭsā̃, ăsā̃ sŭṇ')]),
      ('second','second-person','thou','you',[
      ('nom','tū̃','tŭsī̃'),('gen','tŭhṛā, tŭṛā','tŭs dā'),('dat acc','tŭ ko','tŭsā̃ ko'),
      ('abl','tŭhṛe kōlo, tŭṛe kōlo','tŭsā̃ kōlo'),('erg','tŭdh','tŭsā̃, tŭsā̃ sŭṇ')]),
      ('third-proximal','third-person','this; he, she, it','these',[
      ('nom','ē, ēh','ē'),('gen','ĭs dā','ēhnā̃ dā'),('dat acc','ĭs ko','ēhnā̃ ko'),
      ('abl','ĭs de kōlo','ēhnā̃ kōlo'),('erg','ĭs, ĭs sŭṇ','ēhnā̃, ēhnā̃ sŭṇ')])]
    for section,person,sggloss,plgloss,rows in tables:
        for case,sg,pl in rows:
            for number,raw,gloss in [('sg',sg,sggloss),('pl',pl,plgloss)]:
                add(section,case.replace(' ','-')+'-'+number,raw,raw.replace(', ',';'),gloss,
                    'pron '+person+' '+number+' '+case+(' demonstrative' if section=='third-proximal' else ' personal'),
                    'Source prints singular/plural and case/person headings; shared dat./acc. functions retained. Lexical gloss follows the heading, not inferred word decomposition.')
    for row in out:
        if any(' ' in x for x in row['forms']):row['source_analysis']+=' Apply multiword-expression only to emitted answers containing spaces.'
        if any(x in ''.join(row['forms']) for x in ['ā̃','ē̃','ī̃','ū̃']):row['source_glyph_note']='Paired nasal/length marks can be compressed or filled; encoded with both layers provisionally, pending independent original-glyph review.'
    return out

def page94():
    out=[]
    def add(section,cell,raw,forms,gloss,tags,note=''):
        out.append(dict(source_unit_key=f'bailey1920kagani:p94:{section}:{cell}',printed_page=94,pdf_page=120,
            section=section,raw_source=raw,forms=forms.split(';') if forms else [],gloss=gloss,tags=tags.split(),source_analysis=note,
            source_pdf=SOURCE,review='Full original3x plus bounded6–14x crops; independent review pending'))
    for case,sg,pl in [('nom','ō, ōh','ō'),('gen','ŭs dā','ōhnā̃ dā'),('dat acc','ŭs ko','ōhnā̃ ko'),('abl','ŭs de kōlo','ōhnā̃ kōlo'),('erg','ŭs, ŭs sŭṇ','ōhnā̃, ōhnā̃ sŭṇ')]:
        for number,raw,gloss in [('sg',sg,'that; he, she, it'),('pl',pl,'those')]:
            add('third-distal',case.replace(' ','-')+'-'+number,raw,raw.replace(', ',';'),gloss,'pron third-person demonstrative dist '+case+' '+number,'Continues third-person paradigm on93; dat./acc. is one printed shared cell.')
    for section,gloss,extra,rows in [
      ('interrogative','who?','interr',[('nom','kōṇ'),('gen','kăs dā'),('erg','kăs, kŭs sŭṇ')]),
      ('relative','who','relative',[('nom','jehṛā, jē'),('gen','jĭs dā'),('erg','jehṛe')]),
      ('indefinite','anyone; someone','indef',[('nom','kūī'),('gen','kăse dā'),('erg','kăse, kăse sŭṇ')]),
      ('respectful','you','second-person honorific',[('nom','āb-ŭnī'),('gen','āb-ŭnā̃ dā'),('erg','āb-ŭnā̃')])]:
        for case,raw in rows:add(section,case,raw,raw.replace(', ',';'),gloss,'pron '+extra+' '+case,'Columns here distinguish pronoun types, not grammatical singular/plural. Printed source qualifier preserved in Tags.')
    for n,(form,gloss,tags) in enumerate([
      ('kē','what?','pron interr'),('kĭjjh','anything; something','pron indef'),('săb kĭjjh','everything','pron'),('jē kĭjjh','whatever','pron relative'),
      ('ĭtnā','so much','quantifier correlative'),('kĕtnā','how much?','quantifier interr'),('jĭtnā','as much','quantifier relative')],1):
        add('indefinite-quantity',str(n),form,form,gloss,tags,'Source explicitly labels as much relative. So much is the corresponding quantity construction; correlative classification is editorial, not a printed label.' if n in [5,7] else '')
    add('honorific','direct','ŭnī','ŭnī','','honorific pl','Bailey says this is added to nouns to indicate respect, always used in the plural of respect.')
    add('honorific','oblique','ŭnā̃','ŭnā̃','','honorific pl obl','Source explicitly gives the oblique of ŭnī.')
    add('honorific','panjabi-control','horī̃, horā̃','horī̃;horā̃','','honorific','Explicit Panjabi parallels; not target Kagani forms.');out[-1]['scope']='comparison_control';out[-1]['source_language']='Panjabi'
    add('honorific','brother-example','bhrā ŭnī','bhrā ŭnī','(my) brother','noun honorific multiword-expression','Source applies the respect construction; possessive parenthesis is printed.')
    add('honorific','cow-example','sāhb ŭnā̃ dī gā̃','sāhb ŭnā̃ dī gā̃',"the Sahib’s cow",'noun honorific multiword-expression','Exact genitive phrase printed as illustration.')
    add('agent','marker','sŭṇ','sŭṇ','','erg','Used with nouns, third-person pronouns and plurals of first/second persons. Speakers describe it as respectful; Bailey doubts that as its current function because a dog example occurs. Do not assign honorific to every occurrence.')
    for n,(form,gloss) in enumerate([
      ('kŭtte sŭṇ kītai','a dog has done it'),('bhrāū sŭṇ khādhai','my brother has eaten it'),
      ('kŭṛhī sŭṇ āk̲h̲ĕā','the girl said it'),('tŭsā̃ sŭṇ mēlĕā','you milked (the cow)')],1):
        add('agent-examples',str(n),form,form,gloss,'sentential','Illustrates the agent construction, not a claim of honorific meaning. Literal source underlining is retained where printed.')
    add('adjective','ending','-ā','-ā','','suffix adj m sg','Adjectives ending in -ā in masculine singular agree in number, gender and case; no unprinted regular paradigm generated.')
    for n,(form,gloss,tags) in enumerate([
      ('mŭṛhā pē','my father','m sg'),('mŭṛhī bōhṭī','my wife','f sg'),('mŭṛhĕā̃ pŭttŭrā̃ dā','of my sons','m pl gen'),('mŭṛhīā̃ dhīā̃','my daughters','f pl')],1):
        add('adjective-examples',str(n),form,form,gloss,'multiword-expression '+tags,'Complete printed possessive expression illustrating adjective agreement; no isolated unattested adjective stems emitted.')
    for row in out:
        if any(' ' in x for x in row['forms']) and 'sentential' not in row['tags'] and 'multiword-expression' not in row['tags']:
            row['source_analysis']+=' Apply multiword-expression to individual multiword answers.'
        if any(x in ''.join(row['forms']) for x in ['ā̃','ē̃','ī̃','ū̃']):row['source_glyph_note']='Compressed upper nasal/length stack represented provisionally with both layers; independent glyph review pending.'
    return out

def add_record(out,page,section,cell,raw,forms,gloss,tags,note=''):
    out.append(dict(source_unit_key=f'bailey1920kagani:p{page}:{section}:{cell}',printed_page=page,pdf_page=page+26,
        section=section,raw_source=raw,forms=forms.split(';') if forms else [],gloss=gloss,tags=tags.split(),source_analysis=note,
        source_pdf=SOURCE,review='Source-first visual full page and narrow glyph crops; independent review pending'))
    return out[-1]


def page97():
    out=[]
    def add(section,cell,raw,forms,gloss,tags,note=''):return add_record(out,97,section,cell,raw,forms,gloss,tags,note)
    for i,(sg,pl) in enumerate([('ā̃','ā̃'),('ē̃','ō'),('ē','ēn, hēn')],1):
        for number,raw in [('sg',sg),('pl',pl)]:
            add('substantive-present',str(i)+'-'+number,raw,raw.replace(', ',';'),'be','verb auxiliary pres '+str(i)+number,'Person/number correspond to standard three-row two-column paradigm order; page omits repeated person headings.')
    past=[('āsā̃','āsī̃','āsē','āsīē'),('āsē̃','āsīē̃','āsĕō','āsīō'),('āsā','āsī','āsē','äsīā̃')]
    for i,forms in enumerate(past,1):
        for (number,gender),form in zip([('sg','m'),('sg','f'),('pl','m'),('pl','f')],forms):
            r=add('substantive-past',str(i)+'-'+number+'-'+gender,form,form,'be','verb auxiliary pret '+str(i)+number+' '+gender,'Feminine labels printed; masculine follows contrast. Person/number follows paradigm row/column order.')
            if form=='äsīā̃':r['source_glyph_uncertainty']={'reading':'äsīā̃','alternative':'āsīā̃','locator':'printed97 Rect165,117,240,158 at12x','reason':'Initial upper mark consists of two separated small points; uncertain diaeresis versus damaged macron. No phonetic claim.'}
    add('substantive-tone','1','āhsā̃','āhsā̃','be','verb auxiliary pret','Source says the first syllable sometimes has high tone, represented here by h; no extra paradigm forms generated from etc.')
    add('substantive-tone','2','āhsī','āhsī','be','verb auxiliary pret','Second explicitly written high-tone alternative; source does not explicitly reassign person/gender here.')
    add('pour','head','ḍōlnā','ḍōlnā','pour out','verb inf')
    add('pour','imperative','ḍōl, ḍōle','ḍōl;ḍōle','pour out','verb impv','Two printed imperative forms; ordinary singular/plural assignment not printed on this line, so no unsupported number tag.')
    add('pour','polite-imperative','ḍōlī̃','ḍōlī̃','pour out','verb impv sg honorific','Explicitly polite singular.')
    for section,tag,pairs in [('subjunctive','pres subjunctive',[('ḍōlā̃','ḍōlā̃'),('ḍōlē̃','ḍōlō'),('ḍōlē','ḍōlĕn')]),('future','fut',[('ḍōlsā̃','ḍōlsā̃'),('ḍōlsī̃','ḍōlsō'),('ḍōlsī','ḍōlsŭn')])]:
        for i,(sg,pl) in enumerate(pairs,1):
            for number,form in [('sg',sg),('pl',pl)]:add('pour-'+section,str(i)+'-'+number,form,form,'pour out','verb '+tag+' '+str(i)+number,'Person/number follows printed paradigm order.')
    for label,form,tags in [('m-sg','ḍōldā','m sg'),('f-sg','ḍōldī','f sg'),('m-pl','ḍōlde','m pl'),('f-pl','ḍōldīā̃','f pl')]:
        add('pour-past-conditional-present-participle',label,form,form,'pour out','verb conditional participle '+tags,'Source dual analysis: Past conditional or present participle. Conditional time is past; participle time is present. No single contradictory tense tag imposed.')
    indicative=[('ḍōldā̃','ḍōldī ā̃','ḍōldĕ ā̃','ḍōldīā̃'),('ḍōldē̃','ḍōldī ē̃','ḍōldĕ o','ḍōldīō'),('ḍōldai','ḍōldī ē','ḍōldĕn','ḍōldīĕn')]
    for i,forms in enumerate(indicative,1):
        for (number,gender),form in zip([('sg','m'),('sg','f'),('pl','m'),('pl','f')],forms):
            add('pour-present-indicative',str(i)+'-'+number+'-'+gender,form,form,'pour out','verb pres ind '+str(i)+number+' '+gender,'Source says present indicative combines present participle and present substantive; written contractions and spaces retained exactly.')
    add('pour','imperfect','ḍōldā āsā̃','ḍōldā āsā̃','pour out','verb pret ipfv multiword-expression','Source says present participle plus past substantive; only this example is printed, followed by etc.; no remaining cells invented.')
    for label,form,tags in [('m-sg','ḍōlĕā','m sg'),('f-sg','ḍōlī','f sg'),('m-pl','ḍōle','m pl'),('f-pl','ḍōlīā̃','f pl')]:
        add('pour-past',label,form,form,'pour out','verb pret '+tags)
    add('pour','perfect','ḍōlĕai (ḍōlĕā e)','ḍōlĕai;ḍōlĕā e','pour out','verb pres perfect','Source gives contracted form and explicit expanded counterpart, analyzing past plus present substantive. Retain final plain e in the expansion as printed; no silent harmonization.')
    add('pour','pluperfect','ḍōlĕā āsā','ḍōlĕā āsā','pour out','verb pret perfect multiword-expression','Heading is Plup., but parenthetical source analysis repeats “past and pres. verb subst.”; preserve this inconsistency, do not silently correct it.')
    add('pour','conjunctive','ḍōlke','ḍōlke','having poured out','verb conjunctive-participle')
    for label,form,tags in [('m-sg','ḍōlĕādā','m sg'),('f-sg','ḍōlīdī','f sg'),('m-pl','ḍōlede','m pl'),('f-pl','ḍōlīdīā̃','f pl')]:
        add('pour-stative',label,form,form,'in the state of having been poured out; poured out','verb participle '+tags,'Explicitly stative participle; the printed plural has plain e, unlike singular ĕā. Do not regularize.')
    add('pour','past-participle','ḍōlĕā','ḍōlĕā','poured','verb pret participle')
    for row in out:
        if row.get('source_glyph_uncertainty'):row['source_glyph_uncertainty']['second_read']='Root independently confirmed two separated upper points from same12x original crop; provisional ä retained without phonetic interpretation.'
        if any(' ' in f for f in row['forms']) and 'multiword-expression' not in row['tags']:row['source_analysis']+=' Apply multiword-expression to each spaced emitted answer.'
    return out

def page98():
    out=[]
    def add(section,cell,form,gloss,tags,note=''):
        return add_record(out,98,section,cell,form,form,gloss,tags,note)
    note='Source passive joins inflected singular masculine past form to go; the inflected singular participle is unchangeable. Preserve literal occurrence quantities.'
    for n,(form,gloss,tags) in enumerate([
      ('ō ḍōle gēā','it was poured out','sentential pass pret m sg'),('ō ḍōle gĕī','it was poured out','sentential pass pret f sg'),
      ('ō ḍōle gĕīā̃','they were poured out','sentential pass pret f pl'),('ḍōle gaisā̃','will be poured out','verb pass fut multiword-expression')],1):
        add('passive',str(n),form,gloss,tags,note)
    add('passive','auxiliary','gŭcchṇā','go','verb inf','Explicitly named auxiliary in the passive construction; later same-page head is separately attested.')
    for form,cond in [('-dā','after a voiced consonant or vowel, including vowel followed by h'),('-tā','after an unvoiced consonant')]:
        add('participle-morphology',form,form,'','suffix participle conditional','Bailey gives this ending for present participle or past conditional '+cond+'. Exception below remains source-qualified.')
    for i,(form,head,gloss) in enumerate([('hŭṭtā','hŭṭṇā','get tired'),('hŭktā','hŭkṇā','be able'),('jŭldā','jŭlnā','go'),('ṭākdā','ṭākṇā','call')],1):
        claim='Present participle/past conditional stem example.'
        if i==4:claim+=' Source explicitly calls this an exception and tentatively suggests ordinary Lahnda influence; preserve as attributed hypothesis, not an etymology/donor edge.'
        add('participle-examples',str(i)+'-inf',head,gloss,'verb inf',claim)
        add('participle-examples',str(i)+'-part',form,gloss,'verb participle conditional',claim)
    def paradigm(section,gloss,records):
        for label,form,tags in records:add(section,label,form,gloss,'verb '+tags,'Only the forms actually printed are recorded; no missing regular tenses generated.')
    paradigm('take','take',[('inf','ghĭnnā','inf'),('past','ghĭdā','pret')])
    paradigm('eat','eat',[('inf','khēṇā','inf'),('imperative1','khā','impv'),('imperative2','khāo','impv'),('polite','khāī̃','impv honorific sg'),
        ('future1','khaisā̃','fut'),('future2','khaisī̃','fut'),('present-participle','khēndā','pres participle'),('past','khādhā','pret')])
    paradigm('come','come',[('inf','ēṇā','inf'),('future','aisā̃','fut'),('past','āyā','pret'),
        ('stative-m-sg','āyādā','participle m sg'),('stative-f-sg','āīdī','participle f sg'),('stative-m-pl','āĭde','participle m pl'),('stative-f-pl','āīdīā̃','participle f pl')])
    paradigm('go-gacch','go',[('inf','gŭcchṇā','inf'),('imperative','gŭcch','impv'),('future','gaisā̃','fut'),('past','gēā','pret'),
        ('stative-m-sg','gēādā','participle m sg'),('stative-f-sg','gĕīdī','participle f sg'),('stative-m-pl','gĕĭde','participle m pl'),('stative-f-pl','gĕīdīā̃','participle f pl')])
    paradigm('go-jul','go',[('inf','jŭlnā','inf'),('imperative','jŭl','impv'),('future','jŭlsā̃','fut')])
    add('go-jul','cross-reference','gŭcchṇā','go','verb','Source says past, stative participle etc. are as for gŭcchṇā. Resolve as reference to that printed paradigm, not newly invented duplicate inflections.')
    out[-1]['scope']='resolved_cross_reference';out[-1]['target_section']='go-gacch'
    for row in out:
        if 'stative-' in row['source_unit_key']:row['source_analysis']+=' Explicit stative participle heading.'
    return out

def page99():
    out=[]
    def add(section,cell,form,gloss,tags,note=''):return add_record(out,99,section,cell,form,form,gloss,tags,note)
    def paradigm(section,gloss,records):
        for label,form,tags in records:
            add(section,label,form,gloss,'verb '+tags,'Explicit stative participle.' if label.startswith('stative') else '')
    paradigm('sit','sit',[('inf','ăjṇā','inf'),('imperative','ăj','impv'),('future','ăjsā̃','fut'),('past','aiṭhā','pret'),
      ('stative-m-sg','aiṭhādā','participle m sg'),('stative-f-sg','aiṭhīdī','participle f sg'),('stative-m-pl','aiṭhede','participle m pl'),('stative-f-pl','aiṭhīdīā̃','participle f pl')])
    add('sit','compound','ăj gŭcch','sit down','verb impv multiword-expression','Explicitly glossed compound within imperative parenthesis.')
    add('sit','hindi-control','baiṭh jā','sit down','verb impv multiword-expression','Explicit Hindi comparison; not Kagani output.');out[-1]['scope']='comparison_control';out[-1]['source_language']='Hindi'
    paradigm('lift','lift',[('inf','cēṇā','inf'),('imperative1','cā','impv'),('imperative2','cāo','impv'),('future','caisā̃','fut'),('present-participle','cēndā','pres participle'),('past','cāĕā','pret')])
    paradigm('bring','bring',[('inf','āṇṇā','inf'),('future','āṇsā̃','fut'),('present-participle','āndā','pres participle'),('past','āndā','pret')])
    out[-1]['source_analysis']='Source explicitly says past has the same form as present participle; preserve distinct grammatical attestations despite identical spelling.'
    paradigm('be-tired','be tired',[('inf','hŭṭṇā','inf'),('present-participle','hŭṭtā','pres participle'),('stative','hŭṭēādā','participle')])
    out[-1]['source_analysis']='Stative participle; source explicitly says accent is on ē. Do not add an unprinted accent glyph to the literal form.'
    paradigm('put-on','put on (clothes, etc.)',[('inf','lēṇā','inf'),('imperative','lā','impv'),('future','laisā̃','fut'),('present-participle','lēndā','pres participle'),('past','lāyā','pret'),('stative','lāyādā','participle')])
    paradigm('be-found','be found; be obtained',[('inf','thēṇā','inf'),('inf-alternate','thhēṇā','inf'),('future','thaisī','fut'),('present-participle','thēndā','pres participle'),('past','thāyā','pret'),('stative','thāyādā','participle')])
    out[-3]['source_analysis']='Source says e is like è in French père. Attributed pronunciation comparison retained; literal ē not replaced by IPA.'
    add('be-found','french-control','père','','','Explicit French sound comparison for e; no target lexical emission.');out[-1]['scope']='comparison_control';out[-1]['source_language']='French'
    add('be-found','tone-pronunciation','thē̩nā','be found; be obtained','verb inf','Source says deep tone sometimes occurs immediately after initial th, specially marked in thhēṇā, pronounced thē̩nā. Combining vertical line below is literal U0329, not IPA syllabicity. Original crop printed99 Rect132,501,176,528 at12x confirms e macron plus lower stroke and plain n (no underdot), unlike the preceding ordinary transcription.')
    out[-1]['tone_encoding']={'above':'macron retained as printed','below':'U0329 combining vertical line below e','literal_plain_n':True}
    return out

def page100():
    out=[]
    def add(section,cell,form,gloss,tags,note=''):return add_record(out,100,section,cell,form,form,gloss,tags,note)
    for label,form,tags in [('inf','hōṇā','inf'),('future','hōsā̃','fut'),('present-participle','hōndā','pres participle'),('past','hōyā','pret'),('stative','hōyādā','participle')]:
        add('be-become',label,form,'be; become','verb '+tags,'Explicit stative participle.' if label=='stative' else '')
    add('habit','auxiliary','kŭrnā','','verb inf auxiliary','Source identifies habitual construction as present participle agreeing in gender and number plus required tense of kŭrnā. The immediate paragraph does not independently gloss the verb; function retained in Notes.')
    add('habit','plural','khēndīā̃ kŭro','make a habit of eating','sentential impv f pl','Explicit feminine plural label applies to the response. Habit construction as described above.')
    add('habit','singular','khēndī kŭrsā̃','I shall make a habit of eating','sentential fut f 1sg','Explicit feminine singular speaker and future English translation; source habit construction.')
    add('continuation','auxiliary','rēhṇā','remain','verb inf auxiliary','Source keeping-on construction uses present participle, agreeing as before, with rēhṇā.')
    add('continuation','masculine','nhēndā rēhā','he continued washing himself','sentential pret m 3sg','Complete source expression; continuation construction uses agreeing present participle plus remain.')
    add('continuation','feminine','nhēndīā̃ rēhīā̃','they (the women) continued washing themselves','sentential pret f 3pl','Printed feminine plural agreement and parenthetical subject preserved.')
    add('ability','auxiliary','hŭkṇā','be able','verb inf auxiliary','Source says ability uses this verb with the root of the required verb.')
    add('ability','masculine','mē̃ nā jŭl hŭktā','I am unable to go','sentential pres neg 1sg','Whole source sentence; inability meaning explicit. No invented infinitive expanded from root jŭl.')
    add('ability','feminine','mē̃ jŭl nā hŭktī','I am unable to go','sentential pres neg 1sg f','Source explicitly labels feminine speaker; preserves different word order.')
    add('necessity','auxiliary','hōṇā','become','verb inf auxiliary','Necessity/advisability uses infinitive with substantive hōṇā.')
    add('necessity','present','mŭ kŏ jŭlnai','I have to go; I ought to go','sentential pres 1sg','Both source English alternatives retained; no added root-level lexical interpretation.')
    add('necessity','future','tŭ ko jŭlnā hōsī','you will have to go','sentential fut second-person','Full explicit source expression; number not inferred from English you alone.')
    add('intention','question','kŭdū̃ jŭlnai','when are you going; when do you intend to go?','sentential interr second-person','Source says the same construction sometimes means intention or even futurity. First word narrow12x crop printed100 Rect35,397,107,414 confirms both u bodies and final macron-plus-tilde.')
    note='Source notes almost invariable combination of present substantive vowel with preceding ā or e. Both printed contracted and expanded witnesses retained; no invented expansions.'
    for i,(contracted,expanded,gloss) in enumerate([
      ('ō jăṇai','ō jăṇā e','he is a man'),('ō jŭldai','ō jŭldā e','he is going')],1):
        add('contraction',str(i)+'-contracted',contracted,gloss,'sentential pres 3sg',note)
        add('contraction',str(i)+'-expanded',expanded,gloss,'sentential pres 3sg',note)
    add('contraction','3-contracted','kŭr gēāden','where have they gone (are in the state of having gone)','sentential interr 3pl',note)
    add('contraction','3-expanded-fragment','gēāde ēn','','multiword-expression verb participle',note+' Source expands only the predicate, omitting where; retain blank gloss and full preceding translation as context, not an invented independently aligned translation.')
    out[-1]['context_translation']='where have they gone (are in the state of having gone)'
    add('contraction','vowel-context','ā; e','','',note+' These are sound-symbol context for the rule, not separately glossed morphemes.');out[-1]['scope']='source_phonetic_context'
    return out

def page101():
    out=[]
    def add(section,cell,form,gloss,tags,note=''):return add_record(out,101,section,cell,form,form,gloss,tags,note)
    for i,(contracted,expanded,gloss,tags) in enumerate([
      ('kē āk̲h̲ĕai','āk̲h̲ĕā e','what did he say','interr pret third-person'),
      ('kŭhthe gēādē̃','gēādā ē̃','where did you go','interr pret second-person'),
      ('mē̃ āk̲h̲tā̃','āk̲h̲tā ā̃','I am saying','pres 1sg')],1):
        add('contraction',str(i)+'-contracted',contracted,gloss,'sentential '+tags,'Continues the source contraction examples on100. Underlined kh preserved literally.')
        row=add('contraction',str(i)+'-expanded-fragment',expanded,'','multiword-expression verb','Source prints only the expanded predicate, omitting preceding question word or pronoun; keep blank independent gloss and the complete sentence translation as contextual evidence.')
        row['context_translation']=gloss
    add('contraction','phonetic-comment','ai','','','Source says ai is almost exactly French è, phonetically often written epsilon. This is attributed commentary; do not substitute an inferred IPA vowel.')
    out[-1]['scope']='source_phonetic_context'
    for i,(form,gloss,tags) in enumerate([('ghŭllīdā','being sent','verb pres participle pass'),('cāhiye','',''),('cāhīdā','','')],1):
        add('panjabi-controls',str(i),form,gloss,tags,'Source says no organic passive such as the Panjabi example was observed, nor words corresponding to cāhiye/cāhīdā; these are contrastive non-target examples, not Kagani attestations.')
        out[-1]['scope']='comparison_control';out[-1]['source_language']='Panjabi'
    for i,(form,gloss,tags) in enumerate([
      ('gŭcchṇā cŭṅai','it is right or proper to go','pres'),('ĭs tŭre kŭrnā cŭṅā nī̃h','one ought not to do this','neg'),('ĭs tŭre kŭrnā bŭrai','it is bad to do this','pres')],1):
        add('evaluation',str(i),form,gloss,'sentential '+tags,'Source says a word meaning good or bad replaces the compared obligation expression; preserve whole phrase, no invented isolated lexical glosses.')
    return out

if __name__=='__main__':
    import hashlib,sys
    # Pure source-stage generator; never touches canonical forms, registry or profiles.
    sys.path.insert(0,str(HERE.parents[5]));from tags import GRAMMATICAL_TAGS,GENDER_TAGS
    counts={};hashes={}
    for page,fn in [(92,page92),(93,page93),(94,page94),(97,page97),(98,page98),(99,page99),(100,page100),(101,page101)]:
        rows=fn()
        assert len({r['source_unit_key'] for r in rows})==len(rows)
        for row in rows:
            assert set(row['tags'])<=GRAMMATICAL_TAGS|GENDER_TAGS,(row['source_unit_key'],row['tags'])
        path=HERE/f'grammar-p{page}-first-reading-20260927.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        counts[page]={'units':len(rows),'literal_or_expanded_candidates':sum(len(r['forms']) for r in rows),'scope_counts':{scope:sum(r.get('scope','target')==scope for r in rows) for scope in sorted({r.get('scope','target') for r in rows})}}
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report={'status':'complete assigned first-reading inventory; independent review pending; no canonical writes',
      'scope':'All grammar/prose on92–94,97–100 and101 above Prodigal Son heading. Parent owns other chapter pages/narratives.',
      'source_pdf':SOURCE,'page_mapping':'zero-based PDF index printed+25; one-based printed+26','counts':counts,'sha256':hashes,
      'policies':['Raw source abbreviations preserved with explicit proposed expansions only; no omitted regular forms manufactured.',
        'Whole source clauses retained; predicate-only expanded fragments have blank gloss plus exact context.',
        'Controls and sound-symbol commentary distinguished from target morphology.',
        'No profile, canonical forms, database or full build modifications.'],
      'independent_targeted_confirmations':['p97 initial ä in äsīā̃: two separated marks, possible damaged macron; rootconfirmed12x, explicit alternative ā retained.',
        'p99 thē̩nā retains e macron and lower vertical stroke, plain n; rootconfirmed12x.'],
      'source_analysis_caveats':['Pluperfect97 retains inconsistent parenthetical present-substantive label.',
        'Agent94 is not blindly tagged honorific; source itself qualifies that claim.',
        'Present participle and past conditional dual analysis preserved in Notes.']}
    (HERE/'assigned-grammar-first-reading-summary-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(counts))

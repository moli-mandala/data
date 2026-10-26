"""Keyed whole Irula chapter recovery, Nair 2013 pp. 22–27.

Literal source Roman forms; no IPA normalization. Source grammatical labels
are retained separately from lexical meanings. Documentary and zero evidence
are inventoried even when no lexical row can be emitted.
"""
import collections
import hashlib
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PDF=HERE.parents[5]/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
PDF_SHA='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
assert hashlib.sha256(PDF.read_bytes()).hexdigest()==PDF_SHA
rows=[]
def add(p,section,form='',gloss='',surface='',role='target',claim='',label='',note=''):
    n=1+sum(r['printed_page']==p and r['section']==section for r in rows)
    rows.append({'unit_key':f'nair2013irula:p{p}:{section}:{n:02d}','printed_page':p,'pdf_page':p,
                 'section':section,'section_index':n,'source_order':len(rows)+1,'form':form,
                 'surface':surface,'gloss':gloss,'role':role,'source_claim':claim,
                 'grammatical_label':label,'note':note,'source_pdf_sha256':PDF_SHA})
def forms(p,section,entries,**kwargs):
    for e in entries: add(p,section,*e,**kwargs)

# Every lexical or grammatical mention in the p22 noun/case section.
add(22,'community-context',role='documentary',claim='Nilgiri and Attappaadi community distribution, 1981 population and migration account are ethnographic context, not elicitation coordinates or lexical forms.')
add(22,'language-context',role='documentary',claim='Nair attributes independent language status and Tamil-Malayalam subgroup to Irula; Caldwell and Zvelebil are historical analyses, not separate target forms.')
forms(22,'plural-markers',[(f,'') for f in ('–mu','-aru/-ru','–ga')],label='plural suffix',claim='Source list; -ga common plural, following sentence prints -mn with pronominal bases.')
add(22,'plural-common-repeat','–ga',label='plural suffix',claim='Repeated explicitly as the common plural suffix in the prose after the first list.')
add(22,'plural-human-marker-repeat','–aru/-ru',label='plural suffix',claim='Repeated after pronominal -mn, conditioned with av-/iv- stems and human substantives.')
add(22,'plural-pronominal-marker','-mn',label='plural suffix',note='Printed -mn differs from preceding –mu; no emendation.',claim='Source says this occurs with pronominal bases.')
forms(22,'plural-human-bases',[(f,'') for f in ('av-','iv-')],label='bound pronominal stem',claim='Stems with human -aru/-ru suffix; no independently supplied meaning.')
forms(22,'plural-prevocalic',[(f,'') for f in ('–m','-ar/-r')],label='plural suffix',claim='Source conditions these forms before a vowel.')
forms(22,'plural-prepause',[(f,'') for f in ('-mu','-aru/-ru','–ga')],label='plural suffix',claim='These variants are conditioned before a pause, repeated source mentions retained.')
add(22,'archaic-plural','–iru',label='plural suffix',claim='Source calls this an additional archaic suffix.')
forms(22,'plural-examples', [('naamu','we'),('nimu','you'),('nammu','us'),('avaru','those persons'),('ivaru','these persons'),('aaru','who'),('maaplega','bridegrooms'),('boṭṭuga','mountains'),('pammuga','fruits'),('aaḷapoṇḍiru','husband and wife')],label='plural examples')
add(22,'gender-categories',role='documentary',claim='Five categories: masculine singular, feminine singular, epicene plural, neuter singular and neuter plural.')
add(22,'masculine-markers','-an',label='masculine suffix',claim='The following printed sentence is incomplete: -an, -ø and denote masculine suffixes.')
add(22,'masculine-markers','-ø',role='zero-exponent',label='masculine zero exponent',claim='Printed zero; not a pronounceable word.')
add(22,'masculine-statement-blank',role='documentary',claim='Missing third item in printed phrase -an, -ø and denote; no supplied form reconstructed.')
forms(22,'feminine-markers',[(f,'') for f in ('–aḷ','tti/ti','-atti','-itti','-acci','-icci','-ecci','-ci','–i')],label='feminine suffix')
add(22,'case-categories',role='documentary',claim='Ten cases: nominative, accusative, instrumental, locative, dative, purposive, sociative, ablative, genitive, vocative.')
add(22,'nominative-zero',role='zero-exponent',claim='Source says no specific nominative marker, without printing a lexical form.')
forms(22,'accusative-markers',[(f,'') for f in ('-ne','-e')],label='accusative suffix')
forms(22,'accusative-examples',[('kuure-ne','house'),('maratte','tree')],label='accusative',claim='Printed gloss adds (Acc); lexical meaning separated from grammatical label.')
forms(22,'instrumental-markers',[(f,'') for f in ('-aale','ii','–ili')],label='instrumental suffix',note='Bare ii printed without a leading hyphen; function explicit from heading.')
forms(22,'instrumental-examples',[('namanaale','by us'),('maratti','with the tree'),('pammili','with the fruit')],label='instrumental')
add(22,'sociative-marker','-ōḍe',label='sociative suffix')
add(22,'sociative-example','pammōḍe','with fruit',label='sociative')
forms(22,'dative-markers',[(f,'') for f in ('-kk','-kke')],label='dative suffix')
forms(22,'dative-examples',[('avarukku','to them'),('baavikke','to the well')],label='dative')

# p23 case continuation and all pronoun paradigms/mentions.
add(23,'purposive-marker','–kkaayi',label='purposive suffix')
add(23,'purposive-example','maṇikkadukkaayi','for speaking',label='purposive')
forms(23,'ablative-markers',[(f,'') for f in ('-ilirund','-irund')],label='ablative suffix')
forms(23,'ablative-examples',[('peenattirundu','out of the pen'),('koottagiri-ilirund','from Kottagiri')],label='ablative')
forms(23,'locative-markers',[(f,'') for f in ('-i','-ili','-kke','-akke')],label='locative suffix')
forms(23,'locative-examples',[('aanetti','in the elephant'),('uurili','in the village'),('pulikke','tiger'),('namakke','we')],label='locative',note='Later two printed glosses lack explicit locative English wording; heading retained without rewriting meanings.')
forms(23,'genitive-markers',[(f,'') for f in ('-a','tt')],label='genitive suffix')
forms(23,'genitive-examples',[('nama','our'),('pammuttu','fruit’s')],label='genitive')
add(23,'vocative-example','kondii','oh! Sister-in-law',label='vocative',claim='Vocative by lengthening final vowel; source does not print a separate abstract suffix.')
for f,l in [('naa','1sg'),('naamu','1pl'),('nii','2sg'),('niinu','2pl'),('taanu','3sg reflexive'),('taamu','3pl reflexive')]:
    add(23,'personal-pronoun-table',f,label='pronoun '+l,claim='Six explicit cells under Sg/Pl and I/II/III P. (Refl.) headings; no invented English gloss.')
forms(23,'remote-demonstratives',[('vaä','that-he'),('ava','that-she'),('adu','that-it'),('avaru','those-they(persons)'),('aduga/ave','those-they (things)')],label='remote demonstrative pronoun')
forms(23,'proximate-demonstratives',[('ivä','this-he'),('iva','this-she'),('idu','this-it'),('ivaru','these-they'),('iduga/iva','these-they (things)')],label='proximate demonstrative pronoun')
forms(23,'masculine-demonstratives',[('amma','that man'),('imma','this man')],label='masculine demonstrative pronoun')
forms(23,'interrogative-morphs',[(f,'') for f in ('ev-','e-','a','-em')],label='interrogative bound morph')
forms(23,'interrogative-examples',[('eva','which woman'),('edu','which'),('äru','who'),('emmä','which man')],label='interrogative pronoun')
add(23,'indefinite-example','ēdō','something',label='indefinite pronoun')
forms(23,'universal-examples',[('aa -rumu','anybody'),('e-dumu','anything'),('ev- ämu','any person')],label='universal pronoun',note='Printed internal spacing retained; no collapsed spelling.')

# p24 modifiers, every adjective and verbal paradigm example.
forms(24,'noun-modifiers',[('ette kede','grandmother story'),('baaviniiru','well water')],label='noun modifier expression')
for f,g,l in [('nalla','good','qualitative'),('pal̲aya','old','qualitative'),('ciṭṭige','little','quantitative'),('veḷḷa','white','colour'),('kaṇappe','red','colour'),('cinna','small','descriptive'),('aa','that','demonstrative'),('ii','this','demonstrative'),('ee','which','interrogative')]:
    add(24,'adjective-categories',f,g,label='adjective '+l)
add(24,'adjectival-marker','-aana',label='adjectivizer suffix',claim='Added to nouns to form adjectives, as in Tamil. No noun base printed for the following examples.')
forms(24,'adjectival-examples',[('al̲agaana','beautiful'),('acingaana','ugly')],label='adjective',claim='Explicit derived adjectives; the isolated bases are not printed.')
forms(24,'appellative-adjectives',[('nalla','good'),('pudiya','new'),('pal̲aya','old')],label='adjective',claim='Source says adjectives from appellative verbs are common, like Tamil; no unprinted verb bases inferred.')
add(24,'noun-participle-adjectives','kaṛuppu','black',label='adjective',claim='Source mentions nouns and relative participles functioning as adjectives, without assigning each example to a specific origin.')
add(24,'noun-participle-adjectives','kїṭṭa','bad',label='adjective',claim='Source mentions nouns and relative participles functioning as adjectives.',note='Text layer uses Cyrillic ї, visually i with diaeresis; unusual encoding retained as source notation, not asserted IPA.')
add(24,'numeral-context',role='documentary',claim='Numerals said to resemble Tamil; no numeral forms printed here.')
add(24,'tense-context',role='documentary',claim='Irula past and non-past contrasted with three tenses in many South Dravidian languages.')
add(24,'past-in-marker','–in',label='past suffix')
forms(24,'past-in-examples',[('poo-in-iri','went-you (Pl.)','pooniri'),('col-in-iri','told-you (Pl.)','conniri')],label='past 2pl verb',claim='Printed segmented form followed by parenthesized realization.')
add(24,'past-nd-marker','-nd-',label='past suffix')
forms(24,'past-nd-examples',[('nel-nd-eem','hanged we','nendeemu'),('var-nd-a','one who came','vanda')],label='past verb',claim='Printed segmented form followed by parenthesized realization.')
add(24,'past-t-marker','-t-',label='past suffix')
forms(24,'past-t-examples',[('keḍ-t- ø','having spoiled','keṭṭu'),('cax-t-aar','died-they','cattaaru'),('tooṛet-em','get defeated-we','tooṛṛeemu')],label='past verb',claim='Source past suffix paragraph; exact arrowed surfaces retained, no modernized tense translation.')
add(24,'past-d-marker','-d-',label='past suffix')
add(24,'past-d-example','cee-d-eemu','worked-we','ceedeemu',label='past verb')
add(24,'nonpast-zero','ø',role='zero-exponent',label='non-past zero exponent')

# p25 personal endings, negation and participles; all repeated forms retained.
forms(25,'nonpast-examples',[('paakkaamu','see-we'),('aḍugeemu','cook-we'),('tingaaru','eat-they'),('colluga','tell-she')],label='non-past verb')
forms(25,'negative-markers',[(f,'') for f in ('–aa','–ood-')],label='negative suffix')
forms(25,'negative-examples',[('keekkaade','without asking'),('maṇikkaade','without speaking')],label='negative verb')
forms(25,'periphrastic-negative-examples',[('varge','I come'),('vargale','do not come'),('vande','I come'),('vandale','did not come')],label='verb',claim='Source introduces periphrastic negation but pairs each printed form/gloss separately; no repeated forms silently harmonized.')
add(25,'negative-verbs','ille',label='negative verb',claim='Source says tenseless-personless; negates existence.')
add(25,'negative-verbs','alla',label='negative verb',claim='Source says tenseless-personless; negates accidence.')
forms(25,'negative-whole-expressions',[('pammu ille','there is no fruit'),('ii pammu vaa pammu all','this fruit is not a banana')],label='negative clause',note='Whole source expression retained, including final all rather than silently replacing with preceding alla.')
forms(25,'negative-comparison-examples',[('vargale','come not'),('vandale','come not'),('paakkale','look not'),('paattale','looked not')],label='negative verb',claim='Source discusses past negative construction; repeated vargale differs in English gloss from earlier occurrence.')
add(25,'negative-construction','-ale',label='negative suffix',claim='Printed formula verb+past+-ale; not an independently attested lexical root.')
forms(25,'old-tamil-comparisons',[('kaṇṇṭileen','I did not see'),('vantileen','I did not come')],role='comparison-control',label='Old Tamil negative verb',claim='Explicit Old Tamil comparisons; not Irula rows.')
forms(25,'infinitive-markers',[(f,'') for f in ('-a','aakku')],label='infinitive suffix')
forms(25,'infinitive-examples',[('paakka','to see'),('naṭakkaakku','in order to walk')],label='infinitive verb')
add(25,'prohibitive-marker','-ille',label='prohibitive suffix',claim='Printed construction verb stem+infinitive -ille; not borrowing inferred from the comparison.')
add(25,'prohibitive-example','maṇikkaakkille','should/must not speak',label='prohibitive verb')
add(25,'personal-ending-context',role='documentary',claim='Personal endings attach to tense morphs without morphophonemic alternations.')
for f,l in [('-e','first-person column-left'),('-oo','first-person inclusive column-right'),('-eenu','first-person exclusive column-right'),('-a','second-person column-left'),('-iri','second-person column-right'),('-a','third-person masculine column-left'),('-a','third-person feminine column-left'),('-anu/aaru','third-person human column-right'),('–udu','third-person neuter column-left'),('-ina\\','third-person non-human column-right')]:
    add(25,'personal-ending-table',f,label='personal suffix '+l,claim='Printed personal-ending table; person/gender and inclusive/exclusive headings retained; no printed Sg/Pl column headings, so number needs an explicitly supported metadata decision.',note='Final backslash after -ina is literally printed; retained as source notation pending purpose review.' if f=='-ina\\' else '')
forms(25,'relative-participle-markers',[(f,'') for f in ('–a','–aada')],label='relative participle suffix',claim='Second suffix is explicitly negative.')
forms(25,'relative-participle-examples',[('vanda','who came'),('paatta','who saw'),('varga','who comes'),('collugaada','who does not/did not tell')],label='relative participle verb')
add(25,'verbal-participle-marker','–u',label='verbal participle suffix',claim='Examples continue on following page.')

# p26 every modal, participial-noun, imperative and contextual counterexample.
forms(26,'verbal-participle-examples',[('keeṭṭu','having asked'),('ceedu','having done')],label='verbal participle')
add(26,'negative-verbal-participle-marker','–aade',label='negative verbal participle suffix')
forms(26,'negative-verbal-participle-examples',[('paakkaade','without seeing'),('vargaade','without coming')],label='negative verbal participle')
forms(26,'permissive-markers',[(f,'') for f in ('–ila','–idi')],label='permissive suffix')
forms(26,'permissive-examples',[('naḍakkila','one may walk'),('collugila','one may tell'),('uudugidi','let someone blow'),('tingidi','let someone eat')],label='permissive verb')
add(26,'hortative-marker','–oo',label='hortative suffix',claim='Source calls this peculiar, with a following comparison to Zvelebil1982:202 rare link-morph construction.')
forms(26,'hortative-examples',[('tingoo','let us eat'),('vargoo','let us come'),('nallacanne ceegoo','let us do (it) well')],label='hortative verb')
add(26,'hortative-comparison-context','–oo',label='hortative suffix',claim='Nair reports Zvelebil1982:202 describing link morph followed by –oo as a rare formation; marker itself already accounted.')
add(26,'relative-noun-expression','veṭṭina katti','knife used for cutting',label='relative participle expression',claim='Cannot be reduced to a noun compound, in source discussion contrasting Tamil/other Dravidian languages.')
add(26,'unlicensed-noun-compound','veṭṭu katti',role='non-attested-control',claim='Source explicitly says the preceding relative-participle expression cannot be changed into this noun compound; not an attested Irula target expression.')
add(26,'participial-noun-marker','–av',label='participial nominalizer suffix',claim='Added to relative participle forms and followed by third-person number/gender suffixes.')
forms(26,'participial-noun-examples',[('ceegavaa','he who does'),('paattava','he who saw'),('vandava','she who came'),('vandavaru','they who came')],label='participial noun')
add(26,'neuter-participial-noun-marker','–adu',label='neuter singular participial nominalizer suffix')
forms(26,'neuter-participial-noun-examples',[('ceedadu','that which did'),('vandadu','that which came')],label='neuter singular participial noun')
forms(26,'optative-markers',[(f,'') for f in ('-udi','–üdi')],label='optative suffix',claim='Source-reported Zvelebil1982:202: -udi after weak verbs, –üdi after strong verbs; no generalized phonetic substitution.')
forms(26,'optative-examples',[('vargudi','lets come'),('koḍukküdi','let’s give')],label='optative verb')
add(26,'polite-imperative-marker','–mba',label='polite imperative suffix',claim='Added to imperative suffix –i; complete examples follow.')
add(26,'polite-imperative-base-marker','–i',label='imperative suffix')
forms(26,'polite-imperative-examples',[('pooyimba','please go'),('tinnimba','please eat')],label='polite imperative verb')
add(26,'imperative-singular-context',role='documentary',claim='Imperative singular described as the verb stem, without separate segment.')
add(26,'imperative-plural-marker','–i',label='plural imperative suffix')
forms(26,'imperative-examples',[('collu','say'),('colli','say-you (Pl.)')],label='imperative verb')
for f,l in [('-vi','singular'),('–bi','singular'),('–viri','plural'),('–biri','plural'),('–ppiri','plural')]:
    add(26,'injunctive-markers',f,label='injunctive suffix '+l,claim='Source defines injunctive as suggestion rather than command, accompanying imperative notion.')
forms(26,'injunctive-examples',[('deppi','why don’t you break?'),('koduppi','why don’t you give?'),('deppiri','why don’t you (Pl.) give?')],label='injunctive verb',note='Printed deppiri meaning give differs from earlier deppi break; no translation repair.')

# p27 continued injunctive, conditionals, temporals, adverbs and bibliography.
add(27,'injunctive-continuation','colluvi','why don’t you say?',label='injunctive verb')
forms(27,'conditional-markers',[(f,'') for f in ('–aa','–aakki')],label='conditional suffix',claim='Explicit freely varying suffix pair; physical mention keys retained.')
forms(27,'conditional-examples',[('vandaa','if someone comes'),('maṇittaa','if someone speaks'),('oḍittaakki','if someone breaks')],label='conditional verb')
add(27,'concessive-marker','–aalu',label='concessive suffix')
add(27,'concessive-example','mä vündaalu','even if it rains',label='concessive clause')
add(27,'temporal-marker','–aaṭṭi',label='temporal suffix')
add(27,'temporal-example','paakkaaṭṭi','when (x) sees',label='temporal verb')
forms(27,'zvelebil-temporal-comparison',[(f,'') for f in ('aadi/adi','–aatti/atti','aakki/akki')],label='temporal suffix',claim='Nair contrasts Zvelebil1982:204 analysis: non-past stem+aadi/adi or –aatti/atti and past+aakki/akki; slash scope retained literally, not speculative expansion.')
forms(27,'additional-injunctive-temporal-examples',[('colluvin','why don’t you (Pl.) say?'),('timbi','why don’t you eat?'),('timbiri','why don’t you (Pl.) eat?'),('tingadi','when someone eats?'),('vargaatti','when someone comes?')],label='source modal/temporal example',note='Question marks retained in exact printed gloss; specific function follows overt translation rather than assuming all five examples identical.')
forms(27,'simple-adverbs',[('ippuyi','now'),('innemi','still')],label='adverb')
add(27,'derived-adverb-marker','–aayi',label='adverbializer suffix',claim='Occurs after some noun stems to derive adverbs.')
forms(27,'derived-adverb-bases',[(f,'') for f in ('veve','neer','rooca')],label='noun stem',claim='Explicitly printed noun stems; no separately supplied lexical meaning or reconstructed output for neer/rooca.')
forms(27,'derived-adverb-examples',[('veeyaayi','fast'),('truuvaa','straight'),('aamaari','like that')],label='adverb')
add(27,'reported-bibliography',role='bibliographic-context',claim='Zvelebil, Kamil V., The Irula (Erla) Language Part III, Otto Harrassowitz, Wiesbaden,1982; R. Perialwar, A Grammar of the Irula Language, Annamalai University, Annamalai Nagar,1978. Reported works not independently collated.')

assert len({r['unit_key'] for r in rows})==len(rows)
(HERE/'physical-units-first-reading.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
(HERE/'first-reading-status.json').write_text(json.dumps({'status':'complete physical first reading pp22–27; independent review pending', 'source_pdf_sha256':PDF_SHA,'total_units':len(rows),'page_counts':dict(collections.Counter(r['printed_page'] for r in rows))},indent=2)+'\n')
print(len(rows))

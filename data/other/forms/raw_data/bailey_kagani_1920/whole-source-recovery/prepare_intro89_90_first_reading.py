"""Source-local exhaustive introduction inventory, printed89–90; no installation."""
import json,hashlib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE='tmp/pdfs/bailey1920/bailey1920.pdf'

def add(out,page,section,key,raw,forms,gloss,tags,note='',scope='target',language='Kagani'):
    row=dict(source_unit_key=f'bailey1920kagani:p{page}:{section}:{key}',printed_page=page,pdf_page=page+26,
        section=section,raw_source=raw,forms=forms.split(';') if forms else [],gloss=gloss,tags=tags.split(),
        source_analysis=note,scope=scope,source_language=language,source_pdf=SOURCE,
        review='Full original4x and bounded7–13x crops; independent review pending')
    out.append(row);return row

def page89():
    out=[]
    def a(section,key,form,gloss,tags,note='',scope='target',language='Kagani'):
        return add(out,89,section,key,form,form,gloss,tags,note,scope,language)
    a('tone','stream','kŭṭṭhā','stream','noun','Continuation of prior-page tone discussion; both t underdots preserved from original9x crop.')
    a('tone','pull','chĭkṇā','pull','verb inf','Printed as an example in the continuing tone discussion.')
    a('tone','be-found-ordinary','thhēṇā','be found','verb inf','Source says first h immediately after t has no effect on tone, but the second has; explicit pronounced form follows.')
    a('tone','be-found-pronounced','thē̩ṇā','be found','verb inf','Original9x crop printed89 Rect23,35,289,158 confirms e macron, lower vertical tone stroke and n underdot. Retain distinction from printed99 thē̩nā with plain n. U0329 encodes the source vertical line, not IPA syllabicity.')
    a('vowel-contraction','substantive','e','','verb pres copula','Explicit e of the verb substantive combines with preceding final -ā. No independent lexical gloss is printed here.')
    a('vowel-contraction','preceding-ending','-ā','','','Final vowel context for contraction, without a separately identified morphological function. Result said to approach French è, whose phonetic symbol is epsilon; no inferred IPA substitution.',scope='source_phonetic_context')
    a('vowel-contraction','french-symbol','è','','','Explicit French pronunciation comparator; not a target word.',scope='comparison_control',language='French')
    a('consonants','cerebral-l','ḷ','','','Bailey says this cerebral l, common in Northern Panjabi, is not found in Kagani. A negative sound-inventory claim, not a Kagani lexical item.',scope='source_phonetic_context')
    a('consonants','affricate-context','c; j; ts; dz','','','Bailey says c and j often tend towards ts and dz; these are source symbol correspondences, not new lexical heads.',scope='source_phonetic_context')
    a('consonants','bear-head','rĭcch','bear','noun','Source gives pronunciation as almost rĭttsh; preserve qualifier almost, not a categorical sound change.')
    a('consonants','bear-pronunciation','rĭttsh','bear','noun','Bailey’s explicitly approximate pronunciation rendering for rĭcch; no IPA claim.')
    for key,form,gloss in [('nine','nō̃','nine'),('fourteen','cōhdā̃','fourteen')]:
        a('vowel-examples',key,form,gloss,'num','Source describes the long vowel as between ō and English aw, generally where Northern Panjabi has au. Preserve as attributed description, not a silent substitution.')
    a('vowel-examples','symbols','ō; aw; au','','','Comparator symbols in the long-vowel discussion; aw explicitly English, au Northern Panjabi. No independent lexical response.',scope='source_phonetic_context')
    for key,form,gloss in [('north','qŭtŭb','north'),('right','hŭqq','right'),('west','qĭblā','west')]:
        a('q-examples',key,form,gloss,'','Source notes q where most districts have k, tentatively relating this to inhabitants’ religion. Preserve as an attributed speculative claim, not a donor/etymological edge.'+(' Narrow13x original Rect191,237,222,253 shows open u in hŭqq; no analogical change to a.' if key=='right' else ''))
    a('q-examples','qibla-context','qĭbla','','','Printed parenthesis explains west as for the qĭbla at Makka. This unglossed contextual comparison is not independently attributed as a Kagani lexical response; do not invent a donor language.',scope='source_context',language='unspecified')
    a('q-examples','symbols','q; k','','','Source letter comparison; separately contextualized in the three directional examples.',scope='source_phonetic_context')
    for key,form,gloss,tags in [('of','dā','of','prep gen'),('to','ko','to','prep dat'),('from1','thī','from','prep abl'),('from2','kōlo','from','prep abl')]:
        a('noun-prepositions',key,form,gloss,tags,'Source calls these prepositions and explicitly gives the two alternatives for from.')
    a('noun-prepositions','agent','sŭṇ','','prep erg','Source calls this the agent preposition: optional because simple oblique suffices, and not used with first/second singular pronouns. No lexical gloss supplied.')
    for key,form,tags in [('oblique-sg-e','-e','suffix obl sg'),('oblique-sg-u','-ū','suffix obl sg'),('oblique-pl','-ā̃','suffix obl pl')]:
        a('noun-endings',key,form,'',tags,'Source says commonest oblique singular ending is -e or -ū; plural always -ā̃. Preserve source generalization; no unattested paradigms generated.')
    a('honorific','direct','ŭnī','','honorific','Source says this is added to a noun or used with āb to indicate respect.')
    a('honorific','oblique','ŭnā̃','','honorific obl','Explicit oblique counterpart of ŭnī.')
    a('honorific','pronoun','āb','','pron honorific','Target honorific pronoun identified by correspondence to explicitly labelled Urdu āp; no unprinted word-level gloss.')
    a('honorific','urdu-control','āp','','pron honorific','Explicitly labelled Urdu comparison, not target Kagani.',scope='comparison_control',language='Urdu')
    a('honorific','panjabi-brother','bhrā horī āe nē̃','my brother has come','sentential','Explicitly introduced by cf. Panjabi. Literal first horī has a single macron; no nasal added by analogy.',scope='comparison_control',language='Panjabi')
    a('honorific','panjabi-command','āp horā̃ dā kī hŭkm e','what is your command','sentential interr','Same explicitly Panjabi comparison scope. Target replacements follow in a parenthesis.',scope='comparison_control',language='Panjabi')
    a('honorific','kagani-brother-fragment','bhrā ŭnī','','honorific multiword-expression','Explicitly labelled Kagani parenthetical fragment corresponding to part of the preceding Panjabi sentence; preserve without importing the full sentence’s translation.')
    out[-1]['context_translation']='my brother has come'
    a('honorific','kagani-command-fragment','āb ŭnā̃ dā','','honorific gen multiword-expression','Explicitly labelled Kagani parenthetical fragment; no invented wordwise alignment or full-clause expansion.')
    out[-1]['context_translation']='what is your command'
    a('numerals','h-insertion','h','','','Bailey says numbers11–19 insert h, employing high tone; contrasts Panjabi where this occurs only in oblique use. Source symbol/tone commentary, not an independently glossed affix.',scope='source_phonetic_context')
    return out

def page90():
    out=[]
    def a(section,key,form,gloss,tags,note='',scope='target',language='Kagani'):
        return add(out,90,section,key,form,form,gloss,tags,note,scope,language)
    a('adverbs','again','bhī̃','again','adv','Source compares the non-nasal counterpart used by Sasis; the historical description of that community is Bailey’s attributed wording, not editorial classification.')
    a('adverbs','sasi-control','bhī bhī rē rē','','multiword-expression','Explicit Sasi comparison phrase, no translation printed.',scope='comparison_control',language='Sasi')
    a('vowel-contraction','context','ā; e; eā; eo; āe; ai; āē̃; ē̃; aa; ee','','','Present substantive vowel generally combines with preceding ā or e; eā and eo do not combine. āe becomes ai (compared to French è); āē̃ becomes ē̃; one vowel dropped in aa and ee. Printed sequences are phonetic context, not independently attested lexical words.',scope='source_phonetic_context')
    a('vowel-contraction','french-control','è','','','Explicit French sound comparator for ai.',scope='comparison_control',language='French')
    a('infinitive','ordinary','-ṇā','','suffix inf','Source says infinitive ends in -ṇā, with -nā after ṛ, r, l and generally s.')
    a('infinitive','conditioned','-nā','','suffix inf','Explicit ending after ṛ, r, l and generally s; no unattested stems generated.')
    for i,form in enumerate(['-ādā','-ĕādā'],1):
        a('stative-participle',str(i),form,'','suffix participle','Explicit stative participle ending; Bailey says this form occurs as far away as Jammu and Baghat near Simla. Preserve geographical comparison as attributed commentary, not extra lect attestations.')
    a('passive','go','gŭcchṇā','go','verb inf','Source says the participle used in the passive with this verb is unchangeable.')
    a('passive','girl-killed','kŭṛhī māre geī','the girl was killed','sentential verb pret pass','Complete printed example illustrating unchanged passive participle. No inferred individual word glosses.')
    a('present-participle','unvoiced','-tā','','suffix participle pres','Explicit ending after an unvoiced or surd letter.')
    a('present-participle','sonant','-dā','','suffix participle pres','Explicit ending after a sonant.')
    a('present-participle','pulling','chĭktai','he is pulling','sentential verb pres third-person sg','Whole translated verbal expression; the source gives this under the present-participle discussion.')
    a('present-participle','eating','khēndai','he is eating','sentential verb pres third-person sg','Whole translated verbal expression; no invented segmentation.')
    a('habit','do','kărnā','do','verb inf','Root independently confirmed closed a with breve and final ā at13x Rect70,375,210,395. Source describes present participle with do as a habit construction, unlike Panjabi or Urdu; continuation, ability and continuance are said to resemble Panjabi.')
    a('habit','looking','dĭk̲h̲tā kărnā','to be in the habit of looking','verb inf multiword-expression','Full printed habit construction; underlining on kh preserved literally, without phonetic reinterpretation.')
    for i,form in enumerate(['jŭlnā','gŭcchṇā'],1):
        a('compound-verbs','go-'+str(i),form,'go','verb inf','Both expressly said to be used in compound verbs; the comparison with Panjabi/Urdu describes the construction, not an exclusion of these target attestations.')
    for i,form in enumerate(['chŭḍḍṇā','chōhṛnā'],1):
        a('compound-verbs','leave-'+str(i),form,'leave','verb inf','Used when meaning is really leave, contrasted with following intensive; source notes resemblance to Panjabi.')
    a('compound-verbs','leave-intensive','chŭṛnā','','verb inf','Explicit intensive use contrasted with lexical leave. No independent gloss assigned to this intensive function.')
    for key,form,gloss,tags in [('do','kărna','do','verb inf'),('cause-do','kărāṇā','cause to be done','verb inf caus'),('graze','cărnā','graze','verb inf'),('cause-graze','cărāṇā','cause to graze','verb inf caus')]:
        r=a('causatives',key,form,gloss,tags,'Paragraph says Like Panjabi, followed by target causative examples; not explicitly labelled comparator-only forms. Exact do infinitive has plain final a. Original13x Rect28,460,295,518 confirms breve rather than macron over initial vowel; Root independently confirmed closed a base at13x, 2026-09-27; preserve occurrence-specific form.')
        r['independent_glyph_review']={'reviewer':'root','decision':'Confirmed a base and breve; final do a plain','crop':[28,460,295,518],'scale':13,'date':'2026-09-27'}
    return out

if __name__=='__main__':
    from collections import Counter
    summary={}
    for page,rows in [(89,page89()),(90,page90())]:
        path=HERE/f'intro-p{page}-first-reading-20260927.jsonl'
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        summary[str(page)]={'units':len(rows),'forms':sum(len(r['forms']) for r in rows),'scope':dict(Counter(r['scope'] for r in rows)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        assert len({r['source_unit_key'] for r in rows})==len(rows)
    (HERE/'intro-p89-90-first-reading-summary-20260927.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

"""Complete source-position census from originals55–58, not accepted glyphs.

Unicode in leads is a reading aid only; raw native page/line/font evidence remains
authoritative until literal/metadata review. No canonical rows are emitted.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = 'nair2013mullukurumba'
units = []


def group(page, section, leads=(), claim='', disposition='lexical_or_morphological', caution=''):
    key = f'{SOURCE}:p{page}:{section}'
    witnesses = []
    for index, (scope, notation, gloss) in enumerate(leads, 1):
        witnesses.append(dict(witness_key=f'{key}:w{index}', source_scope=scope,
            unaccepted_notation_lead=notation, printed_definition_lead=gloss,
            accepted_literal=False, locator=f'printed/PDF p. {page}, {section}, witness{index}'))
    units.append(dict(source_unit_key=key, printed_page=page, pdf_page=page,
        section=section, disposition=disposition, witnesses=witnesses,
        source_claim=claim, caution=caution,
        original_image=f'originals/page-{page}-150dpi.png', status='scope inventory; literal/font/grammar/graph acceptance pending'))


T, M, D, U = 'Mullu Kurumba target', 'explicit Malayalam comparison', 'documentary name', 'language attribution unresolved'
group(55, 'identity-ethnonym-leads', [(U,'Veetuvar','hunters'),(U,'Mullu','thorn')],
    'Community says they are Veetuvar; Mullu Kurumba is a name created by Wayanad Nairs, with a thorn legend attributed to the king of Kottayam.',
    caution='Do not assign endonym/exonym components to a language solely from ethnographic prose.')
group(55, 'settlements', [(D,n,'') for n in ['Kappaala','Netunkootu','Nerivalappu','Oonimuula','Kaappukkunnu','Paticceri','Kalliccaalu','Maatakkunnu','Konnaatu','Teyiyakunni']],
    'Ten Gudalur settlements listed; no particular elicitation place stated.', 'documentary')
group(55, 'neighbour-community-names', [(D,n,'') for n in ['Betta Kurumba','Kattu Naicka','Paniya']], disposition='documentary')
group(55, 'livelihood-and-contact', claim='Agriculture, hunting/fishing; all also speak Malayalam and some Tamil.', disposition='documentary')
group(55, 'classification-authorities', [(D,'Mullu Karumba','')],
    'Thurston/Aiyappan classify as Malayalam dialect; Census1961 separately; Robert Sathya Joseph1982 treats Nilgiri speech as independent. Preserve source spelling and attributed claims, not an automatic merger.', 'documentary')
group(55, 'gender-number-general', claim='Gender/number have no marked variation from Malayalam.', disposition='context')
group(55, 'masculine-suffixes', [(T,'–en',''),(M,'-an','')], 'Explicit most common masculine suffix with Ma. comparison.')
group(55, 'masculine-examples', [(T,'al̲ak-en','handsome man'),(T,'mutt-en','old man')], caution='Visible drawn lowline under l omitted by native extraction; exact glyph review pending.')
group(55, 'feminine-suffixes', [(T,s,'') for s in ['-atti','-i','aaṭṭi','-aḷu','-ooḷu']], 'Source calls these reflexes of Tamil/Malayalam feminine suffixes; this is an attributed relationship, not independently printed donor witnesses.')
group(55, 'feminine-examples', [(T,'piṇṭ-atti','widow'),(T,'mutt-i','old woman'),(T,'kutum-aaṭṭi','wife'),(T,'mak-aḷu','daughter'),(T,'paṇi-kkaar-atti','maid servant')])
group(55, 'plural-suffixes', [(T,'-maaṛ',''),(T,'–kaḷ',''),(M,'-maaṛ',''),(M,'–kaḷ','')], 'Same printed plural suffixes explicitly said to be used in Malayalam and target.', caution='Two physical printed shapes, four language-scoped witness leads; no invented repeated printing.')
group(55, 'plural-examples', [(T,'mutt-en-maar-u','old men'),(T,'panni-kaḷ','pigs')], caution='Preserve occurrence-specific r marks; do not harmonize suffix-list ṛ with plain example r.')
group(55, 'case-overview', claim='Case suffixes show only marginal differences from Malayalam.', disposition='context')
group(55, 'nominative-zero-cells', [(M,'unmarked',''),(T,'unmarked','')], disposition='zero_or_documentary')
group(55, 'accusative-suffix-cells', [(M,'–e',''),(T,'-e','')], 'Two explicit table columns; target example continues p56.')

group(56, 'accusative-example', [(T,'en-e','I (Acc.)')])
group(56, 'instrumental-suffix-cells', [(M,'-aal',''),(T,'-aalu','')])
group(56, 'instrumental-examples', [(M,'kattiyaal',''),(T,'katti-aalu','by the knife')], caution='Original printed gloss occurs only after target expression; any shared comparison gloss alignment must be separately justified.')
group(56, 'dative-suffix-cells', [(M,'–kki',''),(M,'-nu',''),(T,'–akku',''),(T,'-kku',''),(T,'ukku','')])
group(56, 'dative-examples', [(M,'siitaykku','to Sita'),(M,'raamanu','to Raman'),(T,'namakku','to us'),(T,'talaykku','to the head'),(T,'eŋŋaḷukku','to us')])
group(56, 'genitive-suffix-cells', [(M,'–nṛe',''),(M,'-uṭe',''),(T,'–e','')])
group(56, 'genitive-examples', [(M,'raamanṛe','Raman’s'),(M,'siitayuṭe','Sita’s'),(T,'avanu –e','his'),(T,'avane','his'),(T,'avaḷu-e','her'),(T,'avaḷe','her')], 'Target segmented→surface arrows explicitly printed; keep both forms and scope, no default variant/derivation edge.')
group(56, 'locative-suffix-cells', [(M,'-attu',''),(M,'- il',''),(T,'–atti',''),(T,'-li',''),(T,'-li-lu','')])
group(56, 'locative-examples', [(M,'oriṭattu','at a place'),(M,'talayil','in the head'),(T,'en –atti','in me'),(T,'kaṇi-li','in the trap'),(T,'atu-lu','in that')], caution='Do not harmonize printed -li-lu suffix list with -lu exemplar.')
group(56, 'ablative-cells', [(M,'-nnnu',''),(T,'–nu','')], 'Malayalam postposition after locative; target suffix added after locative suffixes.', caution='Malayalam printed triple n is retained as a source anomaly lead.')
group(56, 'ablative-examples', [(M,'marattil ninnu','from the tree'),(T,'en-attinu','from the trap')], caution='Target en-attinu/from the trap differs from preceding en- pronoun context; preserve literal source/form/gloss conflict, no silent correction.')
group(56, 'directive-cells', [(M,'–kka',''),(T,'eekku','')], 'Malayalam after oblique stem, target after locative case. Target printed eekku lacks leading hyphen.')
group(56, 'directive-examples', [(M,'puṛatteeku','towards outside'),(T,'kaaṭṭuleekku','towards forest')])
group(56, 'pronouns-first-singular', [(T,'naanu',''),(T,'en-','')], 'Explicit I P. Sg.; en- before other cases.')
group(56, 'pronouns-first-plural', [(T,'nammaḷ',''),(T,'n̲aŋŋaḷ','')], 'Explicit I P. plural inclusive/exclusive respectively.', caution='Initial drawn lowline in exclusive form is absent from extracted native text.')
group(56, 'pronouns-second-singular', [(T,'niyyu',''),(T,'nin-','')], 'Explicit II P. Sg.; nin- oblique base.')
group(56, 'pronouns-second-plural', [(T,'nuŋŋaḷ','')], 'Explicit II P. Pl.')
group(56, 'pronouns-third-masculine-singular', [(T,'ivanu',''),(T,'avanu','')], 'Mas. singular proximate/distant.')
group(56, 'pronouns-third-feminine-singular', [(T,'ivaḷu',''),(T,'avaḷu','')], 'Fem. singular proximate/distant.')
group(56, 'pronouns-third-neuter-singular', [(T,'itu',''),(T,'atu','')], 'Neut. singular proximate/distant.')
group(56, 'pronouns-third-plural', [(T,'ivaru',''),(T,'ivicca',''),(T,'avaru',''),(T,'avicca','')], 'Proximate/distant plural slash pairs; prose explicitly says free variation.', caution='Physical vertical placement straddles masculine/feminine lines; do not infer separate gender or fill unprinted cells.')
group(56, 'pronouns-third-neuter-plural', [(T,'itukaḷ',''),(T,'atukaḷ','')], 'Explicit neuter plural proximate/distant.')
group(56, 'pronoun-commentary-occurrences', [(T,s,'') for s in ['ivicca','avicca','ivaru','avaru','itukaḷ','atukaḷ']], 'Repeated forms explicitly printed in commentary; same key accounting does not imply separate lexical head duplication.')
group(56, 'v-base-commentary', [(T,'v-base','')], 'Source calls v-base in first-person dative an important variation.', 'context', 'English notation is not a securely printed independent lexical v witness; record claim rather than invent a paradigm.')
group(56, 'interrogative-bases', [(T,'ee',''),(T,'e','')], 'Interrogative pronouns derived from these bases, as in Malayalam; no separately printed Malayalam base list.')

group(57, 'interrogative-pronouns', [(T,'eeven','who-he'),(T,'eevaḷu','who-she'),(T,'eetu','which'),(T,'entu','what')])
group(57, 'numeral-commentary', claim='Only statement of minor phonological differences from Malayalam; no numbered list or numeral witness is printed in this chapter.', disposition='context')
group(57, 'tense-overview', claim='Three-way tense distinction; past forms generally resemble Tamil.', disposition='context')
group(57, 'past-examples', [(T,'ninneen','stood-I'),(T,'eṭutteen','took-I'),(T,'eṇṇiyaan','counted-he'),(T,'toṭṭaan','touched-he')])
group(57, 'present-suffixes', [(T,'–unu',''),(T,'–inu',''),(M,'–unnu','')], 'Target suffixes described as similar to this explicit Malayalam comparator.')
group(57, 'present-examples', [(T,'puḷi-inu','(is)squeezing'),(T,'tupp-unu','(is) spitting'),(T,'pate-inu','(is)saying'),(T,'cooṭṭ-unu','(is)kicking')])
group(57, 'future-suffixes', [(T,'–un',''),(T,'–p-',''),(M,'–um','')], 'Target –un similar to explicit Malayalam –um; target –p- similar to an unspecified Tamil form. No unprinted Tamil witness added.')
group(57, 'future-examples', [(T,'tuḷḷun','will jump-it'),(T,'cukkun','will shrink-it'),(T,'cooṭṭu-p-en','will kick-I'),(T,'coṭṭuven','will kick-I'),(T,'koy-p-en','will harvest-I'),(T,'koyven','will harvest-I')], 'Last two segmented→surface pairs are explicit analyses, not automatic inflection reconstruction or variants.')
group(57, 'relative-participle-marker', [(T,'–a',''),(M,'–a','')], 'One printed shape explicitly as in Malayalam; two scoped leads do not invent another physical printing.')
group(57, 'relative-participle-examples', [(T,'paay-nt-a','who/which ran'),(T,'paañña','who/which ran'),(T,'tuḷḷu-i-a','who/which jumped'),(T,'tuḷḷiya','who/which jumped')], 'Explicit segmented→surface analyses; retain each literal input/output and nasal marks.')
group(57, 'verbal-participle-overview', claim='Verbal participles resemble Malayalam markers, with tense or negative markers added to verbal base.', disposition='context')
group(57, 'verbal-participle-examples', [(T,'pey-i','having gone'),(T,'peyyi','having gone'),(T,'naṭa-ntu','having walked'),(T,'naṭannu','having walked')])
group(57, 'pronominal-suffix-overview', claim='Suffixes bear close resemblance to Tamil; no explicit Tamil forms printed.', disposition='context')
group(57, 'suffix-first-person-cells', [(T,'-en',''),(T,'-on','')], 'Explicit I P. singular/plural.')
group(57, 'suffix-second-person-cells', [(T,'-e',''),(T,'-uutu','')], 'Explicit II P. singular/plural.')
group(57, 'suffix-third-masculine-singular-cell', [(T,s,'') for s in ['-aan','an','ven']], 'Slash alternatives in III P.(Mas) Sg. cell; only first has printed hyphen.')
group(57, 'suffix-third-feminine-singular-cell', [(T,s,'') for s in ['-aaḷu','-aḷu','ooḷu','vaḷu']], 'Slash alternatives in III P.(Fem) Sg. cell; preserve varying printed hyphens.')
group(57, 'suffix-third-plural-cell', [(T,s,'') for s in ['aaru','aatu','aru','atu','eecca']], 'Slash alternatives in plural column on masculine row; feminine plural field appears empty.', caution='Preserve gender scope uncertainty and empty female cell; do not duplicate five forms into a filled feminine paradigm.')
group(57, 'suffix-third-feminine-plural-blank', [(T,'','')], disposition='blank_cell', caution='No printed form; do not expand from masculine plural cell.')
group(57, 'suffix-neuter-zero-cells', [(T,'unmarked (ø)',''),(T,'unmarked (ø)','')], 'Explicit neuter singular/plural zero cells.', 'zero_or_documentary')
group(58, 'reported-bibliography', [(D,'Robert Sathya Joseph, D.','')], 'A Descriptive Study of Mullu Kurumba, Centre of Advanced Study in Linguistics, Annamalai University, Annamalai Nagar,1982. Thesis type not specified on this page; uncollated authority.', 'documentary')
group(58, 'book-colophon', claim='Nair author/address/email and end-of-book rule; no lexical material.', disposition='documentary')

assert len({u['source_unit_key'] for u in units}) == len(units)
witnesses = [w for u in units for w in u['witnesses']]
assert len({w['witness_key'] for w in witnesses}) == len(witnesses)
out = HERE / 'source-position-inventory.jsonl'
out.write_text(''.join(json.dumps(u, ensure_ascii=False) + '\n' for u in units))
native = json.loads((HERE / 'page-native-font-census.json').read_text())
status = dict(status='complete original-page scope inventory; no accepted glyphs or canonical export',
    physical_source_groups=len(units), scoped_witness_leads=len(witnesses),
    chapter_pages=[55,56,57,58], original_page_images_visually_read=True,
    source_pdf_sha256=native['source_pdf_sha256'], inventory_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    counts_by_disposition={d:sum(u['disposition']==d for u in units) for d in sorted({u['disposition'] for u in units})},
    repeated_or_shared_note='Scoped witness leads include repeated commentary occurrences and explicitly shared Malayalam suffix scopes; they are not claimed to be distinct physical word tokens.',
    outside_mentions=native['whole_book_name_mentions'],
    existing_overlap='Registered Mullu Kurumba/mull1244 with blank coordinates; current Jeba2018 numeral-only package. No numeral forms are printed in the Nair chapter; no etymological/lect merge proposed.',
    review_gates=['full literal/font/lowline reading','source-explicit grammar/classification','comparison routing','identity and bibliography','profile/parser','independent original audit','root activation'],
    canonical_changes=False)
(HERE / 'scope-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'groups':len(units),'witness_leads':len(witnesses),'canonical_changes':False}))

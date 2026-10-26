"""Source-scoped editorial first reading of Nair 2013 Cholanaikkan pp.19-21.

Every physical block has a stable key; forms are provisional literal readings and
must receive an independent original-page audit before any canonical use.
"""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
PDF=Path(__file__).resolve().parents[6]/'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
R=[]
def add(p, section, kind, forms=(), glosses=(), claim='', scope='target', note=''):
    key=f'nair2013cholanaickan:p{p}:{section}'
    assert not any(x['unit_key']==key for x in R)
    assert not glosses or len(forms)==len(glosses)
    R.append({'unit_key':key,'printed_page':p,'pdf_page':p,'kind':kind,'source_forms':list(forms),
              'source_glosses':list(glosses),'source_claim':claim,'scope':scope,'note':note,
              'review':'first reading; independent visual audit pending'})
add(19,'ethnonyms','documentary',('coolanaaykkan','coolekkaarŭ','coolanaykkar'),scope='unattributed community names',note='Chapter does not establish these as target-language lexical utterances.')
add(19,'shelter-term','documentary',('aale',),('rock shelter',),scope='unattributed ethnographic term',note='English ethnographic prose says rock shelters called aale; target-language lexical attribution not explicit.')
add(19,'name-marker','morphology',('-nu',),claim='The people affix -nu to their names.')
add(19,'plural-example','grammatical examples',('ondu mara','eeḍu mara'),('one tree','two trees'),claim='Source says Cholanaikkan lacks plural markers; retain whole expressions, not invented tree lemma.')
add(19,'accusative-markers','morphology',('-a','-e','-in-'),claim='Accusative -a sometimes freely varies with -e after inflectional increment -in-.')
add(19,'accusative-example','grammatical examples',('kūsina','kūsine'),('child','child'),claim='Slash-listed alternatives; contextual accusative function, printed gloss child.')
add(19,'instrumental-markers','morphology',('-indu','-indu'),claim='Source visually prints two indistinguishable -indu markers as alleged free variants; apparent typographical inconsistency retained.',note='Do not invent a contrast.')
add(19,'instrumental-example','grammatical example',('kayttindu',),('by knife',))
add(19,'ablative-markers','morphology',('-liddu','-ddu'),claim='Ablative -liddu with -ddu variant.')
add(19,'ablative-examples','grammatical examples',('alliddu','maraliddu'),('from there','from the tree'))
add(19,'genitive-markers','morphology',('-ø','-e'),claim='Genitive zero and -e variation; Muralidharan 1988 is reported as considering it an independent innovation, not independently consulted.')
add(19,'genitive-examples','grammatical examples',('ennu','ennule'),('my','my'),claim='Slash-listed examples.')
add(19,'locative-markers','morphology',('-lu','-kaḍe'),claim='Locative -lu with -kaḍe variant; -lu restricted to non-human nouns.')
add(19,'locative-examples','grammatical examples',('maralu','ennukaḍe'),('on the tree','with me'))
add(20,'purposive-markers','morphology',('-gāgi','-ga'),claim='Purposive -gāgi freely varies with -ga.')
add(20,'purposive-examples','grammatical examples',('manegāgi','manega'),('for the house','for the house'))
add(20,'vocative-markers','morphology',('-ā','-e','-ī','-ē','-annu'),claim='Vocative paragraph: -ā, -e, -ī as listed variants; then prints -ē after -u/-e stems, an e/ē inconsistency; -annu is stem ending condition, not vocative suffix.',note='Do not treat -annu as vocative variant.')
add(20,'personal-pronouns','paradigm',('naanu','nānke','nanke','en','enke','niinu','nīnke','ninke','taan','tan','tanke','tānke','avnu','avru','ivnnu','ivru','ave','ive','adu','idu'),('I','we','we','my','our','you','you (Pl.)','you (Pl.)','oneself','oneself','themselves','themselves','that-he','that-they','this-he','this-they','that-she','this-she','that-it','this-it'),claim='16 printed pronoun cells, four slash cells expanded to 20 attestations; cell grouping retained in audit.')
add(20,'interrogatives','paradigm',('aanu','eennei','evnu','eve','edu','evru','elli','enda','etteku','endu','ētteku','ende'),('who','how many','which man','which woman','which thing','who','where','what kind','which side','when /now','when /now','in which manner'),claim='11 printed cells, one slash cell expanded to 12 attestations; gloss when /now remains source-ambiguous.')
add(20,'masculine-markers','morphology',('-nu','-anu'),claim='Masculine gender markers connected by source to Proto-Dravidian suffixes; no resolvable donor/ancestor edge.')
add(20,'feminine-markers','morphology',('-itti','-i','-e','-ci','-atti','-iti','-ø','-tti','-cci'),claim='Source lists feminine markers and comparison claims to other Dravidian languages; repeated -e is not a new sense by itself.')
add(20,'gender-prefixes','morphology',('gandu','ennu'),claim='Source says gandu/ennu prefixed for male/female respectively.')
add(20,'gender-examples','grammatical examples',('eṇṇukūsu','gaṇḍāṭu'),('female child','male goat'))
add(20,'future-person-markers','morphology',('-mu','-um','-adu','-ø'),claim='Future -mu described as innovation; -um and -mu contrasted; -adu common past/present person ending, absent after future; zero future termination. -adu claim continues p21.',note='Some forms are reported comparison/absent-context markers; do not install all as independently attested target morphemes without review.')
add(20,'tamil-control','comparative control',('va+nt+aann',),('came',),scope='Tamil, not Cholanaikkan')
add(21,'negative-marker','morphology',('-illa',),claim='Negative existential auxiliary added to infinitive/verbal noun forms.')
add(21,'negative-examples','grammatical examples',('baralilla','tinnadilla'),('won’t come','won’t eat'))
add(21,'causative-permissive','morphology',('-picc-','-aku'),claim='Causative and permissive markers respectively.')
add(21,'causative-permissive-examples','grammatical examples',('nooḍpiccum','nillaku','caaḍaku'),('will cause (someone) to see','(one) may stand','(one) may jump'))
add(21,'hortative-marker','morphology',('-il',),claim='Hortative -il stated to be as in Kannada; no separate Kannada form printed.')
add(21,'hortative-examples','grammatical examples',('naḍli','tereli'),('let (someone) plant','left (someone) open'),note='Source prints left, not let, for second; preserve anomaly.')
add(21,'lexical-list','glossary',('iṇḍreci','iiḍubooyi','eme','eru','aviḷikuusu','udiletaaḍi','divvenumkalu','cikku','cinnoonnu','karse','cuvale','kuuṭruseyṭṭi','kaanana','kuuyennu','goolimara','mondu','niiraaḍalu'),('wife','penis','frog’s croak','join','twin','moustache','crescent moon','hiccough','mole','lungs','earthworm','dove','dowry','friend','banyan tree','rainbow','puberty'))
add(21,'bibliography','documentary',claim='Nair cites R. Muralidharan, A Descriptive Study of Cholanaikka, unpublished PhD thesis, Annamalai University, 1988.',scope='citation only')
# Independent full-page typography review establishes a visible horizontal low
# line on selected n glyphs that the PDF text layer drops. Encode the printed
# mark (U+0332) without assigning it a phonetic value.
review=HERE/'independent-literal-review/underline-census-20260927.json'
assert hashlib.sha256(review.read_bytes()).hexdigest()=='214c454247657f34d81ae51c1c071f7fe1627f0e41c8a4505d7f69af348af435'
corrections=json.loads(review.read_text())['corrections']
by={r['unit_key']:r for r in R}
for correction in corrections:
    forms=by[correction['unit_key']]['source_forms']
    for old,new in zip(correction['old'],correction['new']):
        assert forms.count(old)==1 and '\u0332' in new
        forms[forms.index(old)]=new
    by[correction['unit_key']]['note'] += ' Printed low line on selected n glyphs retained as U+0332; phonetic value not inferred.'
assert len(R)==33
assert sum(len(r['source_forms']) for r in R)==116
assert sum(len(r['source_forms']) for r in R if r['scope']=='target')==111
b=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in R).encode()
(HERE/'physical-units-first-reading.jsonl').write_bytes(b)
status={'status':'first-reading-incomplete-independent-review','pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'pages':[19,20,21],'physical_blocks':len(R),'form_mentions_total':116,'target_form_mentions':111,
        'comparison_control_mentions':1,'unattributed_ethnonym_mentions':3,'unattributed_shelter_term_mentions':1,'first_reading_sha256':hashlib.sha256(b).hexdigest(),
        'canonical_changes':False,'notes':['Counts are editorial physical block/form-mention census, not emitted rows.','Bibliography is a documentary block with no form.','Slash alternatives are distinct mentions; exact graph status requires independent review.','No OCR; font/text extraction and original-page render compared.']}
(HERE/'scope-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(status)

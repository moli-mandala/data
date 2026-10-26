"""Complete four-page source-first inventory, before any Jambu conversion.

Literal notation (including drawn lowlines) remains distinct from IPA.
Physical decomposition and result witnesses are both retained, not variants.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
units = []

def w(form, gloss='', language='Kakkala', role='target', analysis='', note=''):
    return dict(form=form, gloss=gloss, language_label=language, role=role,
                source_analysis=analysis, note=note)

def add(page, section, witnesses=(), claim='', note=''):
    key=f'nair2013kakkala:p{page}:{section}'
    assert key not in {u['unit_key'] for u in units}
    ws=list(witnesses)
    for i, witness in enumerate(ws,1):
        witness['witness_key']=f'{key}:w{i}'
    units.append(dict(unit_key=key,printed_page=page,pdf_page=page,
                      witnesses=ws,source_claim=claim,note=note,
                      review='owner original-page first reading; independent review pending'))

def examples(page, section, rows, analysis='', claim=''):
    add(page,section,[w(f,g,analysis=analysis) for f,g in rows],claim)

add(28,'identity-context',[
 w('kaakkaala',language='documentary',role='parenthetical community name'),
 w('Kuravas',language='documentary',role='community name'),
 w('Kulava',language='documentary',role='reported self-name'),
 w('kuḷuva peeccu', 'language name',note='Source explicitly calls their language this; quotation, not inferred phonemic spelling.'),
 w('Kaikkaaṭis',language='documentary',role='comparison community name'),
 w('kuḷu','speech',language='Kaikkaati',role='comparison with unresolved routing'),
], 'Kakkalas mainly Kottayam, also elsewhere in Kerala; another nomadic Dravidian tribe Kaikkaatis in Maharashtra calls its speech kuḷu.')
pronouns=[
 ('emmaṇam','I','1 singular'),('ñaaŋka','we','1 plural; no inclusive-exclusive distinction'),
 ('ummaṇam','you','2 singular'),('niŋka','you','2 plural'),
 ('iccuvaan','he','3 masculine proximal singular'),('ivaŋga','they','3 masculine proximal plural'),
 ('accuvaan','he','3 masculine distal singular'),('avaŋga','they','3 masculine distal plural'),
 ('iccuvaa/iva','she','3 feminine proximal singular; printed slash alternatives'),('ivaŋga','they','3 feminine proximal plural'),
 ('accuva/ava','she','3 feminine distal singular; printed slash alternatives'),('avaŋga','they','3 feminine distal plural'),
 ('iccire','it','3 neuter proximal singular'),('iccire','they','3 neuter proximal plural'),
 ('accire','it','3 neuter distal singular'),('accire','they','3 neuter distal plural'),
]
add(28,'pronoun-paradigm',[w(f,g,analysis=a) for f,g,a in pronouns],
    'Table prints masculine/feminine rows although preceding prose says third person has common human versus neuter, without masculine/feminine distinction. Neuter lacks number distinction.',
    'Retain table and prose conflict; no forced gender harmonization.')
add(28,'malayalam-pronoun-comparison',[w('iva','they','Malayalam','comparison', 'neuter plural'),w('ava','they','Malayalam','comparison','neuter plural')],
    'These feminine Kakkala forms are neuter plural in Malayalam.')
examples(28,'numerals',list(zip(['orumaacci','irumaacci','muumaacci','n̲aamaacci','aimaacci','arumaacci','el̲umaacci','tuṭṭu','tommalu','patumaacci'],
 ['one','two','three','four','five','six','seven','eight','nine','ten'])),'numeral')
examples(28,'fractionals',[('kaamaaḍu','one fourth'),('ara kaṇṭa','one half'),('arakaṇṭa kaamaaḍu','three fourth')],'fractional numeral')
add(28,'past-neuter-marker',[w('-icci',analysis='third-person neuter personal suffix')], 'Before this personal suffix past forms have no tense suffix.')
examples(28,'past-neuter-examples',[('pal̲ak-icci','(it) became old'),('paaḷikkatt-icci','(it) burned with high flames')],'past third-person neuter',
 'Before the third-person neuter suffix -icci no tense suffix is added.')
add(28,'past-markers',[w(f,analysis='past tense suffix') for f in ['-ṛ-','-t','-tt-','-ṭ-','-cc-','-n-','-ṇṇ','-n̲d','nc-','–un̲n̲']],
 'The prose counts eleven past suffixes; this list has ten, following the no-tense-suffix case. nc- has only a trailing hyphen. Preserve drawn n/unn lowlines.')

past=[
 ('per-ṛ-a','peṛṛa','(she) delivered'),('eeṛ-ṛ-en','eeṛṛeen','(I) undertook'),
 ('ey-t-en','eyteen','(I) shot'),('ney-t-en','neyteen','(I) weaved'),
 ('paa-tt-am','paattam','(they) sought'),('tuu-tt-am','tuuttam','(they) rubbed'),
 ('keeṭ-ṭ-e','keeṭṭe','(he) heard'),('cuṭ-ṭ-e','cuṭṭe','(he) burned'),
 ('piṭi-cc-a','piṭicca','(she) caught'),('toḍa-cc-a','toḍacca','(she) wiped'),
 ('unis´i+n̲n̲+en','unis´innen','(I) beat'),('akicci+n̲n̲+en','akiccinnen','(I) cried'),
 ('ta-nd-a','tanda','(she) gave'),('coma-nd-e','comande','(he) carried'),
 ('kol̲i- ñc-icci','kol̲iñcicci','(if) fell'),('mee-ñc-icci','meeññcicci','(it) grazed'),
 ('kalamb-un̲n̲-aaŋk','kalambun̲n̲aaŋka','(they) saw'),('ooṭ-unn- aaŋka','ooṭun̲n̲aaŋka','(they) ran'),
]
add(29,'past-analysis-table',[w(f,g,analysis=f'past analysis; physical row {i}; '+role,
    note='Printed left analysis and right surface are not automatically variants.')
    for i,(a,b,g) in enumerate(past,1) for f,role in [(a,'decomposed'),(b,'surface')]],
    'Eighteen arrow-linked analyses; source irregularities include if/it typo, doubled ñ and unequal suffix-final a. Preserve both sides.')
add(29,'present-markers',[w(f,analysis='present tense suffix') for f in ['-r-','-ut-','-ir-']])
examples(29,'present-examples',[('akicci-r-en','(I) am crying'),('poo-r-en','(I) am going'),('neer-ut-e','(He) is taking a vow'),('tar-ut-e','(he) is giving'),('tuuŋk-ir-a','(she) is sleeping'),('el̲ut-ir-a','(she) is writing')],'present')
add(29,'future-markers',[w(f,analysis='future tense suffix') for f in ['-v','-um','-uv']])
examples(29,'future-examples',[('poo-v-en','(I) will go'),('akicci-v-en','(I) will cry'),('neṛa-kk-um','(it) will fill'),('viiṛ-kk-um','(it) will swell'),('caaṭ-uv-en','(I) will jump'),('kaḷi-kk-uv-en','(I) will play')],'future')
add(29,'negative-markers',[w('-aatt',analysis='negative relative participle suffix'),w('-aat',analysis='negative verbal participle suffix')])
examples(29,'negative-examples',[('varucc-aatt-a','that which will not come'),('ooṭ-aatt-a','that which will not run'),('tuuŋk-aat-e','without sleeping'),('caaṭ-aat-e','without jumping')],'negative participle')
add(29,'prohibitive-marker',[w('-aatu',analysis='prohibitive suffix')])
examples(29,'prohibitive-examples',[('kemm-un̲n̲-aatu','should not tell'),('ooṭ-unn-aatu','should not run')],'prohibitive')
add(29,'purposive-marker',[w('attena',analysis='purposive marker'),w('aan',language='Malayalam',role='comparison',analysis='purposive marker')],
 'Purposive marker is attena (cf. Ma. aan). No printed leading dash.')

examples(30,'purposive-examples',[('kalamb –attena','for seeing'),('naṭakk-attena','for walking')],'purposive')
add(30,'imperative-singular-markers',[w('-u',analysis='imperative singular suffix'),w('–ū',language='Malayalam',role='comparison',analysis='imperative singular and plural marker')],
 'Kakkala singular imperative can lack a suffix or take -u; Malayalam –ū serves both numbers. Zero is described, not printed as a glyph.')
examples(30,'imperative-singular-examples',[('ari','you (Sg) cut'),('ara','you (Sg.) grind'),('kalamb -u','you(sg.) see'),('kattir -u','you (Sg.) burn')],'imperative singular')
add(30,'imperative-plural-markers',[w('-ŋko',analysis='imperative plural suffix'),w('–uŋko',analysis='imperative plural suffix'),w('-a',analysis='stem-final conditioning vowel'),w('-u',analysis='stem-final conditioning vowel'),w('-ŋko',analysis='imperative plural suffix, repeated distributional witness'),w('-uŋko',analysis='imperative plural suffix, repeated distributional witness'),w('-uŋko',language='Tamil',role='comparison',analysis='honorific form')],
 '-ŋko occurs after a/u-final stems; other stems take -uŋko. Conditioning vowels are not imperative affixes.')
examples(30,'imperative-plural-examples',[('ara- ŋko','you (Pl.) grind'),('aṭa- ŋko','you (Pl.) close'),('eṇṇ-uŋko','you (Pl.) count'),('keeṛ-uŋko','You (Pl.) climb')],'imperative plural')
add(30,'potential-marker',[w('-ām',analysis='potential suffix')],'Potential forms similar to Malayalam; no specific Malayalam form printed.')
examples(30,'potential-examples',[('kraaṇṭ-ir-aam','can sing'),('caaṭ-ir-aam','can jump')],'potential')
add(30,'optative-markers',[w('-ṭṭaa',analysis='optative suffix'),w('-aṭṭaa',analysis='optative suffix'),w('-aṭṭe',language='Malayalam',role='comparison',analysis='optative suffix')])
examples(30,'optative-examples',[('varucc-aṭṭaa','let (me) come'),('ooṭ-aṭṭaa-','let (me) run.')],'optative')
add(30,'participle-zero-context',[w('-i',analysis='stem-final conditioning vowel')],'i-final stems take no participial marker; do not invent a printed zero glyph.')
examples(30,'participle-zero-example',[('akkicci','having cried')],'verbal participle')
add(30,'participle-i-marker',[w('-i',analysis='participle suffix after consonant-final stems')])
examples(30,'participle-i-examples',[('kalamb –i','having seen'),('caaṭ –i','having jumped')],'verbal participle')
add(30,'participle-other-markers',[w('-e',analysis='participle suffix after negative -aat'),w('-aat',analysis='negative marker'),w('-u',analysis='participle suffix after other stems')])
examples(30,'participle-other-examples',[('tuuŋk-aat-e','without sleeping'),('ooṭ-aat-e','without running'),('naṭa-n̲d-u','having walked'),('paatt-u','having looked')],'verbal participle')
add(30,'relative-marker',[w('-a',analysis='relative participle marker')])
examples(30,'relative-example',[('kariñc -ir-a','(which) is laughing')],'relative participle')
examples(31,'relative-continuation',[('era-n̲d-a','(which) begged')],'relative participle')
personal=[('-en','first singular'),('a','first plural'),('-e','second singular'),('-i','second plural'),('-e','third masculine singular'),('-a','third masculine plural'),('-a','third feminine singular'),('-a','third feminine plural'),('-icci','third neuter after past tense'),('-i','third neuter after present tense markers'),('-ø','third neuter before future tense marker')]
add(31,'personal-suffixes',[w(f,analysis=a+' personal suffix') for f,a in personal],
 'Neuter zero is explicitly printed ø; first plural a lacks printed leading hyphen. Temporal conditioning is not itself a tense claim for the suffix.')
examples(31,'personal-examples',[
 ('tūňkiren','sleep-I'),('vāyi+cc-a-m','read-we'),('tap-un̲n̲-e','washed-you'),('taŋg-ir-i-ya','stay-you (Pl.)'),
 ('er-nd-e','begged he'),('el̲ut-ir-a','writes-she'),('kuuṭṭ-un̲n̲-a-m-ka','increased-they'),
 ('pal̲akicci','become old(it)'),('toorricci','failed (it)'),('kiṭṭ –ir-i','obtained (it)'),
 ('toḍa –kk-ir-i','wipes (it)'),('kiṭṭ-um','will obtain (it)'),('viriy-um','will bloom (it)')],
 'personal suffix examples','This table contains source spelling variations, literal macrons and ň, preserved without conversion. The source acute in unis´i/is´iŋk is a separate U+00B4 glyph; attachment and phonemic significance are not inferred.')
examples(31,'lexical-items',[
 ('ayittiṭ','to forgot'),('talapoṭ','to poke'),('is´iŋk','to die'),('kaṇ','to approach'),('cenattukkoṇ','to become angry'),
 ('poṭukkan','husband'),('poṭukkacci','wife'),('acciru','that'),('ummaṇam','you'),('kemm','to say'),('accuvaan','he'),('kaavay','dog'),('vaḷakku','news'),('ummaṇḍi','coffee'),('kraanta','song'),('kaḷamb','to see'),('kaayikkal','coconut'),('puutta','child'),('marikkam','meal'),('polumbu','go'),('puuccu','to go'),('pori','house'),('caaṭayam','short'),('is´iŋkalu','death'),('unis´i','to beat')],
 'lexical list', 'A detached Tea, water line occurs below coffee; exact association unresolved and retained separately.')
add(31,'detached-gloss',[w('', 'Tea, water',language='unresolved',role='unpaired printed gloss')],
 'Tea, water printed beneath coffee in left column; no corresponding form in that baseline. Do not invent a head or silently expand coffee.')
add(31,'bibliography',[], 'Abdul Kalam Azad, M., Syntactic Analysis of Kakkala language, unpublished M.A. thesis, University of Kerala, 1980; Sheela V., Verb Morphology of Kakkala Language, same institution, 1980. Reported authorities, not independently consulted.')

sha=hashlib.sha256(PDF.read_bytes()).hexdigest()
for u in units: u['source_pdf_sha256']=sha
(HERE/'first-reading.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
census=[{k:u[k] for k in ('unit_key','printed_page','pdf_page','source_claim')}|{'physical_witness_count':len(u['witnesses'])} for u in units]
(HERE/'scope'/'physical-unit-census.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in census))
status=dict(status='complete chapter owner first reading; notation/dash sweep and independent review pending',source_pdf_sha256=sha,
 pages=[28,29,30,31],physical_blocks=len(units),physical_witnesses=sum(len(u['witnesses']) for u in units),
 page_blocks={p:sum(u['printed_page']==p for u in units) for p in range(28,32)},
 boundary='Kakkala begins p28; p32 begins Kani.',canonical_changes=False)
(HERE/'first-reading-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(status,ensure_ascii=False))

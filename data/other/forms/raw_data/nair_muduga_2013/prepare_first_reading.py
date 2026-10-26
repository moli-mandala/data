"""Reproducible complete source-first reading of Nair's Muduga chapter.

No canonical writer. Literal readings remain distinct from grammatical analysis.
"""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
SOURCE_SHA='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af'
units=[]
def group(page,section,items,claim='',note='',disposition='lexical'):
    key=f'nair2013muduga:p{page}:{section}'
    witnesses=[]
    for i,item in enumerate(items,1):
        form,gloss,analysis,*extra=item
        language=extra[0] if extra else 'Muduga'
        witnesses.append(dict(witness_key=f'{key}:w{i}',form=form,printed_gloss=gloss,
                              source_analysis=analysis,language_scope=language))
    units.append(dict(source_unit_key=key,printed_page=page,pdf_page=page,
                      source_pdf_sha256=SOURCE_SHA,source_claim=claim,note=note,
                      disposition=disposition,witnesses=witnesses))
def pairs(page,section,items,claim=''):
    for i,(mal,mud,gloss) in enumerate(items,1):
        group(page,f'{section}:{i}',[(mal,gloss,'explicit comparator','Malayalam'),(mud,gloss,'target comparison')],claim)
def markers(page,section,forms,analysis,claim=''):
    group(page,section,[(f,'',analysis) for f in forms],claim)
def examples(page,section,items,analysis,claim=''):
    for i,(form,gloss) in enumerate(items,1):
        group(page,f'{section}:{i}',[(form,gloss,analysis)],claim)

group(50,'identity',[('muḍuga','','community name','documentary')],
      'Mudugas are a forest tribe in Attapady; the source calls this Wayanad district and mentions Bhavani river and reported migration from Coimbatore before the fifteenth century.',
      'Community name not automatically a target-language lexical utterance; historical geography remains attributed.', 'documentary')
group(50,'classification',[],
      'Nair reports Rajendran as arguing Muduga is not a dialect of Tamil, Malayalam or Kannada. The enumerated features below are attributed to that report.',disposition='documentary')
features=[('a',['emma'],'first person plural exclusive pronoun'),('b',['nimma'],'second person plural pronoun'),
          ('c',['–icc-','–cc-'],'causative markers'),('d',['–at-'],'negative marker'),
          ('e',['–yi','–iri'],'imperative plural markers'),('f',['–ya','iya'],'purposive markers'),
          ('g',['–aata'],'optative marker'),('h',['–alu','aatte','-aaṭṭi','kaaṭṭi'],'conditional markers'),
          ('i',['–aṇa'],'accusative case marker'),('j',['–u'],'genitive case marker'),
          ('k',[],'first person plural inclusive and exclusive markers; no exponent printed'),('l',['–eeru'],'second person plural marker')]
for letter,forms,analysis in features:
    group(50,f'reported-feature-{letter}',[(f,'',analysis) for f in forms],
          'Reported distinctive feature attributed by Nair to Rajendran.',
          'Literal bare forms and dash conventions retained; repeated later witnesses are not harmonized.',
          'lexical' if forms else 'documentary')
group(50,'phonological-context',[(f,'','quoted phonological symbol','context') for f in ('ñ','ŋ','n','l')],
      'Source says Muduga lacks the listed nasals and fricative continuant, but possesses voiced stops.',
      'Notation is source context, not standalone lexical entries or an independently inferred IPA inventory.','documentary')
pairs(50,'final-a-to-e',[('aṭaykka','aṭekke','arecanut'),('aṭṭa','aṭṭe','leech'),('oṛṛa','oṛṛe','single')],
      'Malayalam final a is said usually to correspond to Muduga e, compared generally with Kannada.')
pairs(50,'medial-a',[('karayaama','kaareeme','turtle'),('narakam','neraka','hell')])
pairs(50,'final-n-m',[('appan','appe','father')], 'Malayalam final n/m often dropped; continues p51.')
pairs(51,'final-n-m',[('appam','appa','rice-cake'),('aṇṇan','aṇṇe','elder brother')])
pairs(51,'l-changes',[('mala','maga','rain'),('vaala','baaga','plantain'),('palam','paga','fruit'),
                     ('kalutti','kavuttu','neck'),('koluppu','koḷuppu','fat'),('kooli','kooyi','fowl')],
      'Source l frequently becomes g, with v, ḷ and y also mentioned.')
pairs(51,'r-changes',[('avaru','avaṛu','they'),('uruḷa','uṛuḷe','a ball of something /lump'),('irupatu','iṛuvatu','twenty')])
pairs(51,'nn-changes',[('innŭ','inṛu','today'),('kannŭ','kanṛu','calf')])
group(51,'phonological-negative-comparison',[],
      'The p→h or v→b changes found in Malayalam–Kannada are said not to be attested in Muduga.',disposition='documentary')
markers(51,'accusative-marker',['–aṇa'],'accusative suffix after pronouns')
acc=[('ennaṇa','I (Acc.)','first singular accusative'),('emmaṇa','We (Excl.) (Acc.)','first plural exclusive accusative'),
     ('nammaṇa','We (Incl.) (Acc.)','first plural inclusive accusative'),('ninnaṇa','You (Sg.) (Acc.)','second singular accusative'),
     ('nimmaṇa','You (Pl.) (Acc.)','second plural accusative')]
group(51,'accusative-examples',acc)
group(51,'sociative-markers',[('–ooṭe','','sociative suffix'),('-ooṭŭ','','sociative comparison','Malayalam')])
examples(51,'sociative-examples',[('avanooṭe','with him'),('ennooṭe','with me')],'sociative example')
markers(51,'genitive-markers',['–u','-tu','–utu'],'genitive suffix')
examples(51,'genitive-examples',[('ennu','my'),('ninnu','your'),('maletu','of the mountain'),('avaḷtu','of her'),('ammutu','of us')],'genitive example')
markers(51,'locative-marker',['–kku'],'locative suffix')
examples(52,'locative-examples',[('kaṭekku','at the shop'),('kuurekku','at the hut'),('vaanikku','at the river')],'locative example')
personal=[('naanu','','first singular'),('En-','','first singular oblique base'),('emma','','first plural exclusive'),('–em-','','first plural exclusive oblique base'),
          ('namma','','first plural inclusive'),('nam-','','first plural inclusive oblique base'),('nii','','second singular'),('nin','','second singular oblique base'),
          ('nimma','','second plural'),('nim','','second plural oblique base')]
group(52,'personal-pronouns',personal,
      'Source generally compares pronouns with Kannada; no individual Kannada form printed here.',
      'Preserve source capital En- and bare versus dash-marked oblique bases.')
dem=[('ive','','proximal masculine singular'),('ivar','','proximal masculine plural-column (Obl.)'),
     ('iva','','proximal feminine singular'),('ivaḷ','','proximal feminine plural-column (Obl.)'),('itu','','proximal neuter singular'),('itu','','proximal neuter plural-column (Obl.)'),
     ('ivaru','','proximal plural brace form spanning masculine/feminine'),('ave','','distal masculine singular'),('avan','','distal masculine plural-column (Obl.)'),
     ('ava','','distal feminine singular'),('avaḷ','','distal feminine plural-column (Obl.)'),('atu','','distal neuter singular'),('atu','','distal neuter plural-column (Obl.)'),
     ('avaru','','distal plural brace form spanning masculine/feminine')]
group(52,'demonstratives',dem,note='Plural (Obl.) heading and brace scope retained explicitly; source table layout must be independently reviewed before final structured metadata. No variant edges inferred.')
markers(52,'transitive-markers',['ḷ','ḍ','ṭ','tt'],'transitive suffix')
trans=[('aaṭ-ḷ-ura','aaṭara','(she) made to dance'),('kaaṇ-ḍ-ura','kaaṭṭra','(she) will make to see'),
       ('niiŋg-ṭ-ina','niikkina','(she) made to move'),('keṭu-tt-ina','keṭuttina','she made to put out')]
for i,(a,b,g) in enumerate(trans,1):
    group(52,f'transitive-analysis:{i}',[(a,g,'source decomposition'),(b,g,'source surface form')],
          note='The source arrow gives decomposition→surface analysis, not a free-variation or ancestry claim.')
markers(52,'causative-markers',['–acc','–icc'],'causative suffix', 'These later forms differ from p50 introductory –icc-/–cc-.')
examples(52,'causative-examples',[('kuṭi-cc-ina','(she) caused to drink'),('nine-cc-ina','(she) caused to think'),('ooṭ-icc-ina','(she) caused to run'),('eeṛ-icc-ina','(she) made to cause to climb')],'causative example')
markers(52,'present-markers',['–ø-','-ur-','–r-'],'present tense suffix')
examples(52,'present-examples',[('aṭi-kk-ø—a','(she) beats')],'present example', 'Zero is printed inside the attested whole expression; isolated zero remains documentary.')
examples(53,'present-examples',[('tuṛa-kk-ø- a','(she) opens'),('viḷaaṭ-ur-a','(she) plays'),('kutt-ur-a','(she) stabs'),('poo-r-a','(she) goes')],'present example')
markers(53,'future-markers',['–v-','–uv-'],'future tense suffix')
examples(53,'future-examples',[('poo-v-a','(she) will go'),('cey-v-a','(she) will do'),('nooṭ-uv-a','(she) will look'),('coll-uv-a','(she) will say')],'future example')
markers(53,'past-markers',['-tt-','–nd-'],'past tense suffix')
examples(53,'past-examples',[('paṛi-tt-a','(she) plucked'),('oḷe-tt-a','(she) called'),('aḷe-nd-a','(she) measured'),('va-nd-a','(she) came')],'past example')
markers(53,'purposive-markers',['–iya','–ya'],'purposive suffix')
examples(53,'purposive-examples',[('nooṭiya','for seeing'),('oḷekkiya','for calling'),('eṛiya','for throwing'),('tiniya','for eating')],'purposive example')
markers(53,'imperative-markers',['–yi','ri','iri'],'imperative plural suffix')
group(53,'imperative-zero-comment',[], 'Imperative singular forms are unmarked; no printed zero glyph supplied here.',disposition='documentary')
imp=[('poo','(You Sg.) Go','second singular imperative'),('aṭi','(You Sg.) beat','second singular imperative'),
     ('taayi','(You Pl.) give','second plural imperative'),('vaayi','(You Pl.) come','second plural imperative'),
     ('vaayiri','(You Pl.) read','second plural imperative'),('nineri','(You Pl.) think','second plural imperative'),
     ('viḷaatiri','(You Pl.) play','second plural imperative'),('eḷutiri','(You Pl.) write','second plural imperative')]
group(53,'imperative-examples',imp)
group(53,'potential-markers',[('-ila','','potential suffix'),('-aam','','potential comparison','Malayalam')])
examples(53,'potential-examples',[('nooṭila','can see'),('tinila','can eat')],'potential example')
markers(53,'optative-marker',['–aata'],'optative suffix')
examples(53,'optative-examples',[('paaṭaata','let me/ sing'),('varaata','may (I) come')],'optative example')
markers(53,'obligatory-markers',['–utoo','–too'],'obligatory suffix','–utoo follows consonant-ending stems; –too vowel-ending stems.')
examples(53,'obligatory-examples',[('nooṭutoo','should look'),('aṭikkutoo','should beat'),('pootoo','should go')],'obligatory example')
markers(54,'conditional-markers',['–aatte','–aaṭṭi'],'conditional suffix')
examples(54,'conditional-examples',[('vandaatte','even if comes'),('poonaatte','even if goes'),('kaaṇaaṭṭi','when saw, while seeing'),('collaaṭṭi','when said, while saying')],'conditional example')
persons=[('-e-','','first singular personal suffix'),('–m','','plural marker in first plural cell'),('-ee-','','first plural personal suffix before plural r'),('–r','','plural marker in first plural cell'),
         ('e-','','second singular personal suffix'),('-ee-','','second plural personal suffix'),('-e','','third masculine singular personal suffix'),('ṛ','','third masculine plural personal suffix'),
         ('-a','','feminine singular personal suffix'),('ø','','non-gender singular zero'),('-t-','','non-gender singular personal suffix'),('-m','','non-gender plural personal suffix'),('r','','non-gender plural personal suffix')]
group(54,'personal-suffixes',persons,
      note='First plural first line supplies a before-plural-marker note with no overt main suffix. Preserve only printed exponents; do not invent a. Feminine plural cell blank; no filled duplicate.')
ex=[('var-uv-e','will come-I'),('nooṭ-in-e','looked –I'),('var-uv-aa-m-u','will come we (Excl.)'),('paaṭ-uṛ-aa-m-u','sing we (Excl.)'),
    ('var-uv-ee-r-u','will come we (Incl.)'),('poo-r-ee-r-u','go-we(Incl.)'),('va-nd-e','come-you (Sg.)'),('poo-r-e','go-you(Sg.)'),
    ('cey-t-ee-r-u','did-you(pl.)'),('paaṭ-uv-ee-r-u','will sing-you(Pl.)'),('va-nd-e','came-he'),('ooṭ-ur-e','runs-he'),
    ('ta-nd-a','gave-she'),('poo-r-a','goes-she'),('poo-n-aa-r-u','went-they'),('ooṭ-ur-aa-r-u','run-they'),('poo-r-a-t-u','goes-it'),
    ('va-nd-aa-m-u','came-we(Exl.)'),('poo-v-ee-r-u','will go we(Exl.)')]
examples(54,'personal-examples',ex,'source personal-suffix example',
         'Preserve source gloss and spelling, including final Exl. versus preceding inclusive form; no paradigm harmonization.')
group(54,'bibliography',[], 'Rajendran, N., Muduga Language, Ennes Publications, Quilon, 1986.',
      'Reported authority; not independently consulted. Chapter continues no further lexical material before Mullu Kurumba p55.', 'documentary')

def main():
    path=HERE/'first-reading.jsonl'
    path.write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
    report=dict(status='complete chapter source-first extraction; fine own second reading, reconciliation and independent review pending',
                pages=[50,51,52,53,54],physical_groups=len(units),witness_records=sum(len(u['witnesses']) for u in units),
                source_pdf_sha256=SOURCE_SHA,first_reading_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                canonical_changes=False,baseline_preserved=151)
    (HERE/'first-reading-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()

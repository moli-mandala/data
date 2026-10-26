"""Finish LSI Toda residual assessment after three homepage discovery batches."""
import csv
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
dec=json.loads((P/'proposed-decisions.json').read_text())
assert len(records)==154 and len(dec['accepted'])==13 and len(dec['held'])==37
for n,count in [('001',50),('002',50),('003',54)]:
    results=json.loads((P/f'discovery-{n}-results.json').read_text())
    assert len(results)==count
    if n!='002':assert all(x['count']==len(x['rows']) for x in results)
    else:assert [(i+51,x['opts']['params']['form']) for i,x in enumerate(results) if x['count']!=len(x['rows'])]==[(56,'ār'),(59,' u'),(62,'ā')]

def add(i,parents,evidence):
    r=records[i-1];parents=list(parents)
    item=dict(record=r,parent=parents[0],kind='component' if len(parents)>1 else 'reflex',
              citation=r['Source']+';'+';'.join(f'dedr[{p[1:]}]' for p in parents),
              evidence=evidence,comparison='Homepage discovery, full DEDR article and exact LSI source row compared; original source spelling preserved.',status='pending-review')
    if len(parents)>1:item['components']=parents
    dec['accepted'].append(item)

def hold(i,reason):dec['held'].append(dict(record=records[i-1],candidates=[],reason=reason))

simple={
 57:('d5151','DEDR 5151 gives Toda e·Q “which person or thing” with personal oblique e·n-. LSI ēn “What?” shares the interrogative stem and is a qualified whole-article comparison.'),
 65:('d474','DEDR 474 gives Toda ï foQ “twenty”. LSI evoθ retains the two-plus-ten shape, with survey vowel and final-fricative notation.'),
 66:('d2826','DEDR 2826 directly gives Toda e boQ “fifty”; LSI ēboθ has the same numeral components in survey spelling.'),
 68:('d5160','DEDR 5160 gives Toda o·n (oblique en-) “I”, and Kannada ān within the same article. LSI ān is a family-level pronoun match, but its vowel matches Kannada more closely than quoted Toda.'),
 69:('d5160','DEDR 5160 gives Toda o·n (oblique en-) “I”; LSI ānu has a broader family match and a final vowel, with dialect/contact status unresolved.'),
 70:('d5160','DEDR 5160 gives Toda first-person oblique en-. LSI yen “My” retains that stem with a survey initial glide; the possessive function is supplied by the source paradigm.'),
 71:('d5160','DEDR 5160 gives Toda first-person oblique en-. LSI yendu “My” uses that stem with a source possessive/case element not independently etymologised.'),
 72:('d5154','DEDR 5154 gives Toda om inclusive and em exclusive “we”; LSI ām is a family-level first-person plural match but has a vowel closer to the Kannada ā m form in this article.'),
 73:('d5154','DEDR 5154 directly gives Toda em “we (exclusive)”; LSI ēm is the same pronoun with vowel length in survey notation.'),
 74:('d5154','DEDR 5154 gives Toda om/em first-person plural. LSI ām “Our” has the same family but a different vowel and possessive function; the source case label is preserved.'),
 75:('d5154','DEDR 5154 gives Toda om/em first-person plural. LSI āmdu “Our” retains the plural stem with a source possessive suffix; its vowel differs from quoted Toda.'),
 76:('d5154','DEDR 5154 gives Toda em “we (exclusive)”. LSI yēm “Our” has an initial glide and possessive survey function; no separate case etymology is claimed.'),
 77:('d5154','DEDR 5154 gives Toda em “we (exclusive)”. LSI yēmdu “Our” retains the stem with initial glide and a source possessive suffix.'),
 78:('d3684','DEDR 3684 gives Toda ni· with oblique nïn-. LSI nin “Thy” is the matching second-person singular oblique stem.'),
 79:('d3684','DEDR 3684 gives Toda ni· with oblique nïn-. LSI nindu “Thy” retains that stem and a source possessive suffix.'),
 80:('d3688','DEDR 3688 gives Toda nïm “you (plural)”; LSI nīv is a family-level plural comparison with a different final consonant, closer to Kannada nīvu in the same article.'),
 82:('d3675','DEDR 3675 gives Toda nïl- “to stand, stand still, stop”. LSI nel “Stand” matches the consonant stem with a survey vowel difference.'),
 84:('d2053','DEDR 2053 gives Toda kwïṛ- “give to a third person”; LSI kor “Give” is a qualified family-level stem comparison, distinct from its second Give row.'),
 85:('d3098','DEDR 3098 gives Toda to·r- with stem ta- “to give to a first or second person or reflexive”. LSI ta “Give” matches that cited Toda stem exactly; keep distinct from DEDR 2053 kor/kwïṛ-.'),
 99:('d4534','DEDR 4534 gives Toda pïy- (pïs-) “to beat”. LSI pui “Beat” has the same p–y stem with a survey rounded vowel.'),
 111:('d4572','DEDR 4572 gives Toda pï·x- and wider Dravidian pō-/pōk- forms “to go”. LSI pōpery “They go” contains the go verb with person/number inflection, though its vowel is closer to Kannada forms within the article.'),
 118:('d3688','DEDR 3688 gives Toda nïm “you (plural)”, also oblique nïm-. LSI nim “Your” is the same plural pronoun stem with survey possessive function.'),
 119:('d3688','DEDR 3688 gives Toda nïm/nïm-. LSI nimdu “Your” retains that stem and a source possessive suffix.'),
 120:('d1','DEDR 1 gives Toda distal demonstrative aQ and plural aQa·m. LSI adum “He” is a qualified whole-article demonstrative/pronominal comparison, with number/gender force supplied by source gloss.'),
 121:('d1','DEDR 1 includes Tamil avaṉ “that man” and Toda aQ “that person or thing”. LSI avan “He” is a family-level match but closer in form to Tamil than quoted Toda; contact/dialect status is unresolved.'),
 122:('d1','DEDR 1 includes Tamil avaṉ and Toda aQ distal forms. LSI avan “His” is the same family with possessive source function; its Tamil-like spelling is flagged.'),
 123:('d1','DEDR 1 gives Toda distal aQ with plural aQa·m. LSI adum “They” is assigned to the demonstrative family at the article level, while its exact number morphology remains source-specific.'),
 124:('d1','DEDR 1 includes Tamil avar “those persons” and Toda plural aQa·m. LSI avar “They” matches the article family but not its quoted Toda spelling; possible contact is flagged.'),
 125:('d1','DEDR 1 includes Tamil avar and Toda plural aQa·m. LSI avar “Their” retains this family with a source possessive gloss; its Tamil-like form is noted.'),
 126:('d1','DEDR 1 includes Tamil avar and Toda plural aQa·m. LSI avardu “Their” adds a source possessive suffix to the same distal plural family; possible contact remains unresolved.'),
 131:('d657','DEDR 657 directly quotes Toda uf “back of trunk of body”, while flagging phonetic uncertainty. LSI uf “Back” is identical in form and gloss; the article’s uncertainty remains a review qualification.'),
 132:('d1278','DEDR 1278 directly gives Toda kabïn “iron”. LSI kabbun preserves the kab–n frame with a different middle/final vowel and geminate-b survey rendering.'),
 144:('d2170','DEDR 2170 gives Toda kwaṯy “cat” and Kannada kotti within the same cat article. LSI kotti is a family-level comparison, with Kannada-like spelling and unverified immediate donor noted.'),
 150:('d3213','DEDR 3213 directly gives Toda tïgïḷ “moon”. LSI tiggal retains the t–g–l stem in survey spelling.'),
}
for i,(parent,evidence) in simple.items():add(i,[parent],evidence)

subjects={'ān':('d5160','Toda o·n “I”, with article-family ān in Kannada'),
          'ni':('d3684','Toda ni· “thou”'),
          'adum':('d1','Toda distal aQ/plural aQa·m, compared at whole-article level'),
          'ēm':('d5154','Toda em “we (exclusive)”'),
          'nīv':('d3688','Toda nïm “you (plural)”, with Kannada nīvu within the article')}
for i in list(range(87,99))+list(range(100,111)):
    form=records[i-1]['Form'];parts=form.split();assert len(parts)==2,(i,form)
    subject=parts[0];pronoun,ptext=subjects[subject]
    if i<99:verb,verbtext='d480','Toda ïr- “sit, remain, live”'
    elif i<106:verb,verbtext='d4534','Toda pïy- (pïs-) “beat”'
    else:verb,verbtext='d4572','Toda pï·x- “go”; wider article pō-/pōk- family'
    caveat=' LSI erź- is not the exact Toda ïr- notation quoted by DEDR; the source tense/person endings are preserved.' if i<99 else (' LSI pok- resembles the wider article more closely than the quoted Toda pï·x-; possible contact/dialect influence remains unresolved.' if i>=106 else ' LSI pui- differs in vowel from quoted Toda pïy-; source person endings are preserved.')
    evidence=f'LSI gives an explicit two-word pronoun-plus-verb response. DEDR {pronoun[1:]} gives {ptext}; DEDR {verb[1:]} gives {verbtext}. Ordered component links retain both lexical stems without inventing suffix nodes.'+caveat
    add(i,[pronoun,verb],evidence)

holds={
 51:'LSI podθci “Far” has no secure Toda DEDR adverb match after homepage and full-article comparison.',
 52:'LSI batxi “Far” has no secure Toda DEDR adverb match; same gloss as podθci does not establish shared ancestry.',
 53:'LSI muddk “Before” is not identical to the DEDR 5020 Toda/Kannada front forms; its exact stem and donor remain uncertain.',
 54:'LSI mucce “Before” has no securely matching Toda article; keep distinct from muddk.',
 55:'LSI pindalk “Behind” resembles DEDR 4205 Toda pïn “afterwards/younger” but its second element -dalk is unresolved; hold the full form.',
 56:'LSI ār “Who?” has a 1,156-result relaxed homepage query capped at 1,000. DEDR 5151 quotes Toda e·Q interrogative, whereas ār resembles other-language interrogatives; no secure Toda donor analysis yet.',
 58:'LSI āed “Why?” may involve DEDR 5151 Toda e·Qïd “because of what”, but initial vowel and exact morphological function need verification.',
 59:'LSI source enclitic -u “And” is one character and its relaxed homepage query hit the 1,000-row ceiling (2,121 total). No verified lexical parent for the clitic.',
 60:'LSI āle “But” has no securely matched Toda connective article; homepage homographs do not establish ancestry.',
 61:'LSI ha “Yes” is not the DEDR 333 Toda oͅ· affirmative cited in the full article; immediate source unresolved.',
 62:'LSI ā “No” has a 3,141-result relaxed homepage query capped at 1,000 and no secure negative-particle parent.',
 63:'LSI atfo “Go” is not the DEDR 4572 Toda pï·x- go stem; homepage atfok under DEDR 1 means “then”.',
 64:'LSI neś kir “Sit” has an unresolved first element and is not the DEDR 480 Toda ïr- “sit” simplex; hold the two-word response.',
 67:'LSI vaddnūr “Hundred” has unresolved first element vadd “one”. A hundred element is not enough to justify a full-compound assignment.',
 81:'LSI vā “Come” resembles Tamil/Kannada forms in DEDR 5270, whereas quoted Toda po·r- differs; immediate contact/dialect status unresolved.',
 83:'LSI keṭṭ fo “Die” does not match DEDR 2426 Toda soy- “to die”; both source words need independent analysis.',
 86:'LSI vōr “Run” has no secure direct Toda article; DEDR 1041 gives Toda wï·ḍ- “run”, a different stem.',
 112:'LSI ān poki ficpini “I went” has DEDR 5160 and 4572 candidates for the first two words, but ficpini is unresolved; hold all three words together.',
 113:'LSI ni poki fici “Thou wentest” has candidate pronoun/go stems but unresolved third word fici; hold full response.',
 114:'LSI adum poki fici “He went” has candidate demonstrative/go stems but unresolved third word fici; hold full response.',
 115:'LSI ēm poki ficpimi “We went” has candidate pronoun/go stems but unresolved third word ficpimi; hold full response.',
 116:'LSI nīv poki fici “You went” has candidate pronoun/go stems but unresolved third word fici; hold full response.',
 117:'LSI adum poki fici “They went” has candidate demonstrative/go stems but unresolved third word fici; hold full response.',
 127:'LSI mituf “Nose” matches a questioned Toda mütuf quoted inside DEDR 5031, a face article; the dictionary marks that comparison with ?, so hold until the family relation is clarified.',
 128:'LSI parś “Tooth” has no secure Toda tooth article; relaxed matches are unrelated paṟy- verbs.',
 129:'LSI maḍḍ “Head” differs from DEDR 3103 Toda tal “head”; maḍ appears incidentally in compounds, not established as an independent head noun.',
 130:'LSI bir “Belly” differs from DEDR 5259 Toda pi·ṟ “belly, womb”; onset and vowel require further evidence.',
 133:'LSI cinna “Gold” lacks a direct Toda DEDR reflex or verified immediate donor; likely contact vocabulary needs provenance.',
 134:'LSI bilti “Silver” differs from DEDR 5496 Toda piḷy “silver” and resembles a regional form; immediate donor and sound path unverified.',
 135:'LSI vokali xan “Cultivator” has two lexical elements with no securely verified Toda parents or immediate loan donors.',
 136:'LSI āḍu kāran “Shepherd” contains āḍu goat closer to Kannada than DEDR 5152 Toda o·ḍ, while kāran is unresolved; hold full response.',
 137:'LSI dēr “God” has no secure native Toda DEDR parent; immediate donor for a possible deva-type loan is unverified.',
 138:'LSI etud put “Devil” has two unresolved elements; no DEDR entry securely etymologises the entire response.',
 139:'LSI ārś “House” has no secure Toda DEDR house noun matching its form; homepage ar(s)- hits are unrelated.',
 140:'LSI kadare “Horse” repeats the Kannada-like horse form held in the first checkpoint; DEDR 1711 quotes distinct Toda kïQïr and the donor path remains unverified.',
 141:'LSI klatte “Ass” resembles Kannada katte in DEDR 1364 more than quoted Toda katy, with unexplained l; contact/transcription unresolved.',
 142:'LSI voṭṭe “Camel” has no verified Toda DEDR reflex or immediate donor.',
 143:'LSI dānām “Cow” repeats the unresolved cow form held in the first checkpoint; no matching Toda article was found.',
 145:'LSI bitti “Bird” differs from DEDR 4319 Toda pïł̣ and lacks a secure Toda article.',
 146:'LSI belṭ “Bird” differs from DEDR 4319 Toda pïł̣; same gloss does not justify a sound correspondence.',
 147:'LSI huja “Cock” lacks a secure Toda reflex; apparent regional bird words need donor verification.',
 148:'LSI bāt “Duck” has no secure Toda DEDR article; homepage baṭ homographs are unrelated.',
 149:'LSI bīrś “Sun” does not match DEDR 2910 Toda nö·ṟ or DEDR 5517 pi·s̱ without unsupported changes.',
 151:'LSI nerp “Fire” has no securely matching Toda fire noun in full DEDR article comparison.',
 152:'LSI nebb “Fire” has no securely matching Toda fire noun; do not merge with nerp solely on source gloss.',
 153:'LSI dilθ “Fire” has no securely matching Toda fire noun; dialect or immediate donor unresolved.',
 154:'LSI diḍd “Fire” has no securely matching Toda fire noun; distinguish it from the other survey fire responses.',
}
for i,reason in holds.items():hold(i,reason)
seen=[x['record']['ID'] for x in dec['accepted']+dec['held']]
assert len(seen)==len(set(seen))==154
assert set(seen)=={r['ID'] for r in records}
needed={p for x in dec['accepted'] for p in x.get('components',[x['parent']])}|{'d5020','d4205','d333','d2426','d1041','d5031','d3103','d5259','d5496','d1364','d4319','d2910','d5517','d1711','d5152'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
 for row in csv.DictReader(stream):
  if row['Entry_ID'] in needed:articles[row['Entry_ID']]=row
assert needed<=articles.keys(),needed-articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'proposed-decisions.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(dec['held'],ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(dec['accepted']),'assignmentRows':sum(len(x.get('components',[x['parent']])) for x in dec['accepted']),'held':len(dec['held']),'unexamined':154-len(dec['accepted'])-len(dec['held'])}))

"""Assess the remaining LSI Tulu records after homepage and full-article discovery."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
dec = json.loads((P / 'proposed-decisions.json').read_text())
assert len(records) == 137 and len(dec['accepted']) == 34 and len(dec['held']) == 16
for n in ('002', '003'):
    results = json.loads((P / f'discovery-{n}-results.json').read_text())
    assert all(x['count'] == len(x['rows']) for x in results)
assert len(json.loads((P / 'discovery-002-results.json').read_text())) == 50
assert len(json.loads((P / 'discovery-003-results.json').read_text())) == 37

def add(i, parents, evidence):
    r = records[i-1]
    parents = list(parents)
    item = dict(record=r, parent=parents[0], kind='component' if len(parents)>1 else 'reflex',
                citation=r['Source']+';'+';'.join(f'dedr[{p[1:]}]' for p in parents),
                evidence=evidence,
                comparison='Homepage discovery, full DEDR article and exact LSI source row compared; source spelling preserved.',
                status='pending-review')
    if len(parents)>1:item['components']=parents
    dec['accepted'].append(item)

def hold(i, reason):
    dec['held'].append(dict(record=records[i-1], candidates=[], reason=reason))

simple = {
 51:('d4572','DEDR 4572 directly gives Tulu pōpini “to go”. LSI pō is the finite/paradigm stem elicited as “Go”.'),
 52:('d1835','DEDR 1835 directly gives Tulu kuḷḷuni “to sit”; LSI kuḷḷ is the corresponding elicited stem.'),
 53:('d474','DEDR 474 directly gives Tulu irva “twenty”. DEDR 3918 independently cites irva as a ten-based numeral, but the direct numeral article 474 is the proposed family.'),
 54:('d2826','DEDR 2826 directly gives Tulu aiva “fifty”; LSI eiva has a slightly different first-vowel spelling.'),
 55:('d5160','DEDR 5160 directly gives Tulu yānu/yēnu and yānů “I”; LSI yāny preserves its survey final-vowel transcription.'),
 56:('d5160','DEDR 5160 gives Tulu yānu/yēnu with oblique ena-/en-. LSI yena “My” contains that oblique stem with a genitive ending.'),
 57:('d3647','DEDR 3647 directly gives Tulu nama “we (inclusive)”; LSI nama “We” is the same form.'),
 58:('d5154','DEDR 5154 directly gives Tulu eṅkuḷů “we (exclusive)”; LSI yeŋkuḷu has an initial glide and the same plural stem.'),
 59:('d3647','DEDR 3647 gives Tulu nama “we (inclusive)”. LSI uses the identical stem under the possessive gloss “Our”; the case function is supplied by the survey paradigm.'),
 60:('d5154','DEDR 5154 directly gives Tulu eṅkuḷů, oblique eṅkuḷe-. LSI yeŋkuḷe “Our” matches the oblique stem with an initial glide.'),
 61:('d3684','DEDR 3684 gives Tulu ī with oblique nin-. LSI nina “Thy” contains this second-person oblique stem and a genitive ending; it is unrelated to DEDR 2921 nina “marrow”.'),
 62:('d3688','DEDR 3688 directly gives Tulu nikuḷu among its second-person plural forms; LSI nikuḷu “You” is exact.'),
 63:('d5270','DEDR 5270 directly gives Tulu barpini “to come”; LSI bar is its elicited verb stem.'),
 64:('d3675','DEDR 3675 gives Tulu nilpuni “to stay, cease” and nilpuḍuni “make stand”, within the broader stand article. LSI nill “Stand” matches the nil- stem, though no identical standalone Tulu imperative is quoted.'),
 65:('d2426','DEDR 2426 directly gives Tulu saipini “to die”; LSI sai is the elicited verb stem.'),
 66:('d2053','DEDR 2053 gives Tulu korpini “to give” and Koraga/Kodagu congeners. LSI koṛ “Give” is assigned to this whole article, with its survey rhotic differing from the cited Tulu spelling.'),
 68:('d4020','DEDR 4020 directly gives Tulu pāruni “to run, fly, escape”; LSI pār is the matching stem and sense.'),
 81:('d3797','DEDR 3797 directly gives Tulu nōpuni “to beat, strike, flog”; LSI nō is its elicited verb stem.'),
 112:('d3688','DEDR 3688 gives Tulu nikuḷu “you”, also used as an oblique stem. LSI nikuḷe “Your” retains that plural second-person stem with a possessive vowel.'),
 113:('d1','DEDR 1 directly gives Tulu āye “that man”; LSI āye “He” is identical in form and deictic-pronominal function.'),
 114:('d1','DEDR 1 gives Tulu āye “that man” and adjectival ā. LSI āya “His” is its survey possessive case form.'),
 115:('d1','DEDR 1 directly gives Tulu ākuḷu “those persons”; LSI ākuḷu “They” is identical in form.'),
 116:('d1','DEDR 1 gives Tulu ākuḷu “those persons”. LSI ākuḷe “Their” retains this plural stem in a possessive case form.'),
 117:('d1479','DEDR 1479 directly gives Tulu kārů “leg, foot”; LSI kāry is the same form with survey final-vowel notation.'),
 119:('d1892','DEDR 1892 directly gives Tulu kujalů “hair of the head”; LSI kujaly “Hair” has the same stem and survey final-vowel notation.'),
 120:('d5488','DEDR 5488 directly gives Tulu beri “the back”; LSI beri is identical in form and meaning.'),
 121:('d1278','DEDR 1278 directly gives Tulu karba “iron” within its wider burning/black family. LSI kaṛba has a different survey rhotic mark but the same lexeme and sense.'),
 125:('d1844','DEDR 1844 directly gives Tulu kurube “shepherd”; LSI kurube is exact in form and gloss.'),
 128:('d3690','DEDR 3690 directly gives Tulu nīrů “water”; LSI nīry preserves the survey final-vowel rendering.'),
 129:('d494','DEDR 494 directly gives Tulu illů “house, dwelling, family”; LSI illy “House” is the survey rendering.'),
 130:('d1364','DEDR 1364 directly gives Tulu katte “ass”; LSI katte is identical in form and gloss. DEDR 1204 also has katte but means “knife”, so is rejected.'),
 132:('d4355','DEDR 4355 directly gives Tulu pūcè/puccè “cat”; LSI pucce is the second cited form.'),
 134:('d4373','DEDR 4373 directly gives Tulu pūñje “cock”; LSI pūñje is identical in form and meaning.'),
}
for i,(parent,evidence) in simple.items():add(i,[parent],evidence)

hold(67,'LSI voḍ “Run” has no verified Tulu runner matching its form. DEDR 5221 Tulu oḍḍuni “flow, run” concerns a flowing/root shape and lacks the survey initial v; DEDR 4020 pāruni is a different stem.')

pronouns = {'yāny':('d5160','yānu/yānů “I”'), 'ī':('d3684','ī “thou”'),
            'āye':('d1','āye “he/that man”'), 'yeŋkuḷu':('d5154','eṅkuḷů “we (exclusive)”'),
            'yēŋkuḷu':('d5154','eṅkuḷů “we (exclusive)”'),
            'eŋga':('d5154','exclusive first-person plural family; exact eŋga is quoted for Kodagu in this article, while Tulu eṅkuḷů is quoted'),
            'nikuḷu':('d3688','nikuḷu “you (plural)”'), 'ākuḷu':('d1','ākuḷu “they/those persons”')}
verbs = [('d710','uppuni “be, exist”','uppuv-',range(69,75)),
         ('d480','ippuni (itt-) “be, exist”','itt-',range(75,81)),
         ('d3797','nōpuni “beat, strike, flog”','nōp-/nōt-',range(82,100)),
         ('d4572','pōpini “go”','pōp-/pōy-',range(100,112))]
for verb, gloss, stem, indices in verbs:
    for i in indices:
        if i in (85,105):continue
        form = records[i-1]['Form']
        assert len(form.split()) == 2,(i,form)
        subject = form.split()[0]
        pronoun, pgloss = pronouns[subject]
        caveat = ''
        if i==103:caveat=' Here LSI eŋga is the exact Kodagu reflex in this DEDR family, whereas its cited Tulu reflex is eṅkuḷů; the Tulu survey assignment is a family-level comparison.'
        if i>=106:caveat=' The pōy- past stem appears in the wider DEDR 4572 family (including Telugu), while its cited Tulu entry gives pōpini; the past tense comparison is family-level.'
        evidence=(f'LSI has an explicitly two-word subject-plus-verb response. DEDR {pronoun[1:]} gives '
                  f'{pgloss}; DEDR {verb[1:]} gives Tulu {gloss}. The source {stem} verb stem and its person/tense ending remain in the exact survey form; ordered components retain both lexical parts without inventing an inflection suffix node.'+caveat)
        add(i,[pronoun,verb],evidence)

hold(85,'The source record glosses yēŋkuḷu “We beat” but has only the pronoun and no beat verb in either Form or Original. DEDR 5154 supports the pronoun, but the elicited predicate is absent.')
hold(105,'The source record glosses ākuḷu “They go” but has only the pronoun and no go verb in either Form or Original. DEDR 1 supports the pronoun; DEDR 4572 cannot be assigned to an absent predicate.')
later_holds={
 118:'LSI kūli “Tooth” is only incidental in DEDR 418 Tulu kūli nīkāvuni “to show the teeth”; that article etymologises the show-verb, not the tooth noun. No direct tooth parent has been verified.',
 122:'LSI bʰaŋgāry “Gold” resembles a regional baṅgāra-type loan, but its immediate donor and etymological parent are not verified.',
 123:'LSI gulāme “Slave” resembles Indo-Aryan/Arabic ghulām, but the immediate donor into Tulu is unverified.',
 124:'LSI reite “Cultivator” resembles regional raiyta/ryot vocabulary, but the immediate donor and form history are unverified.',
 126:'LSI dēvery “God” resembles Sanskrit deva through regional mediation; the immediate donor and suffixal analysis are unverified.',
 127:'LSI piśāci “Devil” resembles Sanskrit piśāca/related Indo-Aryan forms; the immediate donor is unverified.',
 131:'LSI voṇṭe “Camel” has no secure Tulu reflex or borrowed-parent article established by discovery and full-article comparison.',
 133:'LSI pakki “Bird” resembles Sanskrit pakṣi and regional forms, but the immediate donor is unverified; five homepage homographs do not establish a DEDR parent.',
 135:'LSI batty “Duck” has no securely matching Tulu DEDR article or immediate donor.',
 136:'LSI sūrye “Sun” resembles Sanskrit sūrya through regional mediation; the immediate donor is unverified.',
 137:'LSI tē “Fire” is not identical to DEDR 3266 Tulu tū/sū “fire”; the vowel and initial-consonant differences need more evidence before assignment.',
}
for i,reason in later_holds.items():hold(i,reason)

seen=[x['record']['ID'] for x in dec['accepted']+dec['held']]
assert len(seen)==len(set(seen))==137
assert set(seen)=={r['ID'] for r in records}
needed={p for x in dec['accepted'] for p in x.get('components',[x['parent']])}|{'d3918','d5221','d418','d1204','d3266','d2921'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in needed:articles[row['Entry_ID']]=row
assert needed<=articles.keys(),needed-articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'proposed-decisions.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(dec['held'],ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'proposed':len(dec['accepted']),'rows':sum(len(x.get('components',[x['parent']])) for x in dec['accepted']),'held':len(dec['held']),'unexamined':137-len(dec['accepted'])-len(dec['held'])}))

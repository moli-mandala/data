"""Assess the residual LSI Kodagu paradigms as ordered lexical components."""
import csv
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text())
dec=json.loads((P/'proposed-decisions.json').read_text())
assert len(dec['accepted'])==32 and len(dec['held'])==11
assert not ({r['record']['ID'] for r in dec['accepted']} & {r['record']['ID'] for r in dec['held']})

pronouns={
    'nāny':('d5160','Kodagu na·nï “I” and the na·-/en- oblique'),
    'nīny':('d3684','Kodagu ni·nï “thou”'),
    'avaŋg':('d1','the distal third-person a-/av- family (Kodagu avëⁿ; LSI -ŋg is not printed there)'),
    'eŋga':('d5154','Kodagu eŋga “we”'),
    'niŋga':('d3688','Kodagu niŋga “you” plural'),
    'avu':('d1','Kodagu avu/aeŋga in the distal third-person family'),
    'āvu':('d1','Kodagu avu/aeŋga in the distal third-person family (LSI āvu has long initial vowel)'),
}
verb_evidence={
    'd480':'DEDR 480 quotes Kodagu ir- (-ipp-, iñj-) “be in a place”; the present -ipp- and past iñj- are inflections of this verb.',
    'd4534':'DEDR 4534 quotes Kodagu poyy (poyyuv-, pojj-) “beat”; LSI present/future poy- and past pojj- belong to this paradigm. DEDR 4407 has a homophonous Kodagu poyy- “rain”; the survey’s transitive “beat” gloss selects 4534.',
    'd4572':'DEDR 4572 quotes Kodagu po·- (po·p-, po·c-; also po·k-, po·y-) “go”; the source’s present pōp- and past pōc- are family members.',
}

def add_reflex(i,parent,evidence):
    r=records[i-1]
    dec['accepted'].append(dict(record=r,parent=parent,kind='reflex',citation=r['Source']+';dedr['+parent[1:]+']',evidence=evidence,comparison='Full DEDR article compared with exact LSI survey record and transcription.',status='pending-review'))

def add_components(i,verb):
    r=records[i-1]
    parts=r['Form'].split(' ')
    assert len(parts)==2,(i,parts)
    pro,finite=parts
    parent,pro_evidence=pronouns[pro]
    evidence=f'LSI {r["Form"]} “{r["Gloss"]}” contains a separate pronoun and inflected verb. DEDR {parent[1:]} records {pro_evidence}. {verb_evidence[verb]} Ordered links preserve the source word boundary and do not treat the sentence as a single inherited stem.'
    if i==47:
        evidence+=' LSI iddĭ differs from the Kodagu iñj- past quoted in DEDR 480, but Kannada iddu in the same article supports the broader be-family; this tense stem needs review.'
    if i in {77,78,81}:
        evidence+=' LSI pōn- past differs from the Kodagu po·c- past quoted in DEDR 4572; the go-root identification is stronger than the tense-stem match.'
    dec['accepted'].append(dict(record=r,parent=parent,components=[parent,verb],kind='component',citation=r['Source']+';dedr['+parent[1:]+'];dedr['+verb[1:]+']',evidence=evidence,comparison='Source phonemic field marks these phrase components with +; full DEDR articles read.',status='pending-review'))

def hold(i,reason):
    dec['held'].append(dict(record=records[i-1],candidates=[],reason=reason))

for i in range(40,51):add_components(i,'d480')
hold(51,'LSI ākuḷu ittery “They were” has two words, but DEDR 1 lists ākuḷu under Tulu rather than Kodagu and DEDR 480 does not quote Kodagu itter-. The survey response may cross lect or paradigm boundaries, so neither ordered component is assigned yet.')
add_reflex(52,'d4534','DEDR 4534 directly gives Kodagu poyy (poyyuv-, pojj-) “beat”, beside Kota poy- and Kannada poy. LSI poy is the bare elicited verb in survey spelling. Homophonous DEDR 4407 poyy- means “rain”, so its article is excluded.')
for i in range(53,71):
    if i==56:
        hold(i,'LSI eŋga poyipa nōpuva “We beat” is a three-word response with unexplained nōpuva. DEDR 5154 eŋga and DEDR 4534 poyy can be recognized, but the third token needs source review before assigning a complete ordered analysis.')
    else:add_components(i,'d4534')
for i in range(71,83):
    if i==74:
        hold(i,'LSI yeŋla honeo “We go” lacks the ordinary Kodagu eŋga and po·- stems in DEDR 5154 and 4572. Both words need separate verification; a familiar gloss is insufficient to normalize them.')
    else:add_components(i,'d4572')
add_reflex(83,'d3688','DEDR 3688 quotes Kodagu niŋga “you” plural with oblique niŋga-. LSI niŋgaḍa “Your” adds a possessive ending; the whole inflected pronoun remains within this article family.')
add_reflex(84,'d1','DEDR 1 covers the distal a-/av- third-person pronouns and quotes Kodagu avëⁿ “that man”. LSI avaŋg “He” has an additional velar nasal absent from that printed lemma; the family-level comparison is provisional.')
add_reflex(85,'d1','DEDR 1 covers the distal a-/av- third-person family, including Kodagu avëⁿ “that man”. LSI avanḍa “His” has a possessive ending and an -an- stem; this is a family-level link, not an exact dictionary head match.')
add_reflex(86,'d1','DEDR 1 records the distal third-person plural family, including Kannada avaru “those persons” and Kodagu avu/aeŋga. LSI avaru “They” matches the longer plural form, though the exact Kodagu form is not quoted.')
add_reflex(87,'d1','DEDR 1 directly records Kodagu avu/aeŋga in the distal third-person plural family. LSI avu “They” matches the quoted short form.')
hold(88,'LSI ayanḍa “Their” may use a distal a-/ay- oblique, but DEDR 1 quotes neither this Kodagu stem nor this possessive ending. The relation to avaŋg/avu remains tentative.')

ids=[x['record']['ID'] for x in dec['accepted']+dec['held']]
assert len(ids)==len(set(ids))==92
(P/'proposed-decisions.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(dec['held'],ensure_ascii=False,indent=2)+'\n')
articles=json.loads((P/'primary-evidence.json').read_text())
needed={p for x in dec['accepted'] for p in x.get('components',[x['parent']])} | {'d4407'}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
    for row in csv.DictReader(stream):
        if row['Entry_ID'] in needed:articles[row['Entry_ID']]=row
assert needed<=articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
print('proposed',len(dec['accepted']),'rows',sum(len(x.get('components',[x['parent']])) for x in dec['accepted']),'held',len(dec['held']),'unexamined',92-len(ids))

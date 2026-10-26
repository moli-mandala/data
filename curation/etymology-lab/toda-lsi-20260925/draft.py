"""First 50 LSI Toda residual forms, research proposals only."""
import csv
import gzip
import json
from pathlib import Path

P=Path(__file__).resolve().parent
records=json.loads((P/'survey-inventory.json').read_text());assert len(records)==154
discovery=json.loads((P/'discovery-001-results.json').read_text());assert len(discovery)==50
assert all(x['count']==len(x['rows']) for x in discovery)
proposals={
 1:('d2826','DEDR 2826 gives Toda üζ “five”, with Töwfił̣y-dialect üj. LSI original üch, compiled yc, is close to the cited dialect reflex; its transcription is preserved.'),
 2:('d784','DEDR 784 gives Toda öṭ “eight”, beside Tamil eṭṭu. LSI eṭṭ is a family-level eight comparison with a different vowel/stop notation from the quoted Toda form; it may reflect contact or survey transcription.'),
 5:('d3918','DEDR 3918 directly gives Toda pot “ten” and several compound tens. LSI paθ has a different vowel and final fricative rendering but matches the numeral family; homepage paθ under DEDR 3977 means “bird of prey” and is rejected.'),
 6:('d474','DEDR 474 directly gives Toda e·ḍ “two”; LSI eḍ is its compact survey transcription.'),
 7:('d3655','DEDR 3655 gives Toda no·ng “four”. LSI nānk has the same numeral sense and n–velar stem but a different vowel and final nasal/stop representation, so this is a qualified family-level comparison.'),
 21:('d399','DEDR 399 directly gives Toda o·ḷ “man, Toda”, alongside Kannada āḷ. LSI āl “Man” preserves the same lateral stem but differs in vowel from the quoted Toda reflex; contact or survey transcription remains possible.'),
 22:('d1873','DEDR 1873 directly gives Toda ku·x “girl, daughter”. LSI kux “Woman” has the same stem with a broader female-person gloss; the age/kin difference is flagged.'),
 27:('d4616','DEDR 4616 directly gives Toda mox “child, son, male, daughter, woman”. LSI max “Son” has the son sense and velar-fricative stem, with a survey vowel difference. Homepage exact max under DEDR 4717 means “hillside” and is rejected.'),
 29:('d1873','DEDR 1873 directly gives Toda ku·x “girl, daughter”; LSI kux “Daughter” is a close form and exact kin sense.'),
 31:('d1873','DEDR 1873 gives Toda ku·x “girl, daughter”. LSI kūx “Daughters” is a plural-glossed survey form of the same stem; no plural suffix is independently proposed.'),
 45:('d5086','DEDR 5086 directly gives Toda me·l “up, high (in songs)”; LSI mēl “Up” is the same stem and sense.'),
 46:('d5128','DEDR 5128 directly gives Toda muk and Töwfił̣y-dialect mok “up, west”; LSI mok is the exact dialect form and up sense.'),
 47:('d516','DEDR 516 gives Toda eṟk “down, east”. LSI erg “Down” matches the adverb with a voiced final-stop survey rendering.'),
}
holds={
 3:'LSI anpaθ “Nine” resembles Tamil oṉpatu in DEDR 1025 more than its cited Toda wïnboQ. The immediate contact/dialect history is unverified; preserve the survey form and hold.',
 4:'LSI vadd “One” has no securely matching Toda simplex in DEDR 990; its distinct Toda one forms require further reconciliation.',
 8:'LSI eyan “Father” lacks a verified direct Toda DEDR parent; repeated case/number labels do not establish a stem etymology.',
 9:'LSI enin “Father” lacks a verified direct Toda DEDR parent; keep distinct from eyan and unrelated en- pronoun stems.',
 10:'LSI eyan “Of a Father” depends on unresolved eyan father stem.',
 11:'LSI enin “Of a Father” depends on unresolved enin father stem.',
 12:'LSI enin g “To a Father” contains the unresolved father stem; no separate lexical parent is assigned to the case marker.',
 13:'LSI enin edd “From a Father” contains unresolved enin and a case element; neither receives a guessed parent.',
 14:'LSI eyan “Fathers” depends on unresolved eyan father stem.',
 15:'LSI enin “Fathers” depends on unresolved enin father stem.',
 16:'LSI av “Mother” has no secure Toda DEDR family identified from the full articles; similarly short forms are semantically unsafe.',
 17:'LSI ennon etud “Brother” is an explicit two-word response; neither the ennon relation nor etud kin term has a verified parent.',
 18:'LSI ennon kinnud “Brother” is an explicit two-word response with unresolved elements; do not reduce it to a guessed kin stem.',
 19:'LSI akka “Sister” matches Kannada akka in DEDR 23, but its quoted Toda reflex is okn/okok. Its Toda donor/dialect status needs reconciliation, like earlier held Toda survey forms.',
 20:'LSI enor vøt kux “Sister” contains DEDR 1873 Toda ku·x “girl, daughter”, but the first two words are unresolved. Hold the complete three-word response.',
 23:'LSI kuigʸō “Wife” has no securely matched Toda lexical article; DEDR 1670 kwïṭy “wife” differs in stem.',
 24:'LSI kaṭṭvodi kux “Wife” has DEDR 1873 ku·x as a possible female-person element, but kaṭṭvodi is unresolved; hold the full response.',
 25:'LSI pōpan “Child” has no securely matching Toda DEDR parent.',
 26:'LSI enne “Child” has no securely matching Toda DEDR parent; en- pronouns are not a child etymology.',
 28:'LSI kin max “Son” contains DEDR 4616 Toda mox “son”, but kin remains unverified. Hold the full two-word response.',
 30:'LSI tyj max “Daughter” contains DEDR 4616 Toda mox, but tyj remains unverified; hold the complete two-word response.',
 32:'LSI kadare “Horse (male)” resembles Kannada kudare/kadare in DEDR 1711, whereas the cited Toda reflex is kïQïr. Its immediate donor/dialect status is unverified.',
 33:'LSI kadare “Horses” repeats the horse form closer to Kannada than the cited Toda reflex in DEDR 1711; donor/dialect status remains unresolved.',
 34:'LSI hōri “Bull” resembles Kannada hōri under DEDR 4593, which has no matching Toda reflex; immediate donor status is unverified.',
 35:'LSI dānām “Cow” has no securely matching Toda DEDR parent; repeated singular/plural survey forms do not resolve it.',
 36:'LSI dānām “Cows” has no securely matching Toda DEDR parent.',
 37:'LSI ādu “Goats” matches Kannada āḍu in DEDR 5152 more than its distinctive Toda o·ḍ. The earlier pass holds a separate Toda ādu “He goat” for this exact concern; maintain consistent caution.',
 38:'LSI ulle āl “Good man” resembles Kannada oḷḷe plus āḷ more than quoted Toda wiḷy plus o·ḷ (DEDR 1017, 399). The contact/source-label issue is unresolved; hold both words together.',
 39:'LSI ulle āl “Good men” repeats the unresolved source compound; do not infer plural etymology or relabel the first word.',
 40:'LSI ulle kux “Good woman” contains DEDR 1873 ku·x, but ulle differs from DEDR 1017 Toda wiḷy; hold full response pending contact/dialect clarification.',
 41:'LSI ulle kux “Good women” repeats the unresolved qualifier and female stem; hold full response.',
 42:'LSI holāivi max “Bad boy” contains DEDR 4616 mox, but holāivi is not directly the DEDR 1006 Toda walo(Q) “bad”; hold the compound.',
 43:'LSI holāivi kux “Bad girl” contains DEDR 1873 ku·x, but holāivi lacks a secure adjective analysis; hold the compound.',
 44:'LSI nila “High” has no secure Toda article match; DEDR 5086 me·l and 5128 muk/mok are different stems.',
 48:'LSI nerśg “Down” has no verified Toda DEDR match; DEDR 516 eṟk is a different source form.',
 49:'LSI kīhuri “Near” has no securely matching Toda article after homepage and full-article comparison.',
 50:'LSI kexuri “Near” has no securely matching Toda article; do not infer identity with kīhuri by gloss alone.',
}
assert set(proposals).isdisjoint(holds)
assert len(proposals)+len(holds)==50
dec=dict(status='pending-review',accepted=[],held=[])
for i,(parent,evidence) in proposals.items():
 r=records[i-1]
 dec['accepted'].append(dict(record=r,parent=parent,kind='reflex',citation=r['Source']+f';dedr[{parent[1:]}]',evidence=evidence,comparison='Homepage discovery, full DEDR article and exact LSI source row compared; source spelling preserved.',status='pending-review'))
dec['held']=[dict(record=records[i-1],candidates=[],reason=reason) for i,reason in sorted(holds.items())]
needed={parent for parent,_ in proposals.values()}|{'d1025','d990','d23','d1670','d1711','d4593','d5152','d1017','d1006','d3977','d4717'}
articles={}
with gzip.open(P.parents[2]/'data/dedr/entry-texts-audit.csv.gz','rt') as stream:
 for row in csv.DictReader(stream):
  if row['Entry_ID'] in needed:articles[row['Entry_ID']]=row
assert needed<=articles.keys()
(P/'primary-evidence.json').write_text(json.dumps(articles,ensure_ascii=False,indent=2)+'\n')
(P/'proposed-decisions.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n')
(P/'research-holds.json').write_text(json.dumps(dec['held'],ensure_ascii=False,indent=2)+'\n')
(P/'externally-assigned-records.json').write_text('[]\n')
print(json.dumps({'proposed':len(dec['accepted']),'rows':len(dec['accepted']),'held':len(dec['held']),'unexamined':154-50}))

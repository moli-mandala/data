"""Draft review-only Brahui LSI residual decisions from full DEDR articles."""
import csv
import gzip
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
DATA = P.parents[2]
records = json.loads((P / 'survey-inventory.json').read_text())
assert len(records) == 155
results = []
for name in ('001-035', '036-070', '071-105', '106-140', '141-155'):
    results += json.loads((P / f'discovery-{name}-results.json').read_text())
assert len(results) == len(records)
assert all(x['count'] == len(x['rows']) for x in results)

# Each candidate below has a direct Brahui attestation in the entire DEDR article.
families = {
    'd4764': (22, 'DEDR 4764 directly gives Brahui mār “son, boy, lad” and masiṛ “daughter”; the survey masink is the daughter plural, matching Ali and Kobayashi masin-k.'),
    'd1123': (27, 29, 'DEDR 1123 directly lists Brahui xarās “bull, bullock”, beside Kurux kaṛā and Kui gāṛo. The survey plural xarāsk preserves this stem; the article also compares Indo-Aryan material, so broader ancestry remains open.'),
    'd5151': (56, 57, 58, 'DEDR 5151 explicitly lists Brahui dēr “who”, ant “what”, and antei “why” together with Kurux and Malto interrogatives. The survey antae is the same why form with a transcription variant.'),
    'd809': (63, 'DEDR 809 gives Brahui hining (stem in-) “to go”, explicitly distinguishing its kā- present stem under DEDR 1419; survey hin is a matching stem form of hining.'),
    'd3291': (64, 'DEDR 3291 directly lists Brahui tūling (imperative tūḻẖ, past tūs-) “to sit” beside sleep/rest cognates; the survey tūl is the same stem.'),
    'd5160': (69, 'DEDR 5160 lists Brahui ī “I” with oblique kan-; Ali and Kobayashi explicitly give kan-ā “I-GEN”, matching survey kanā “my”.'),
    'd3647': (70, 'DEDR 3647 directly gives Brahui nan “we”; Ali and Kobayashi give nan-ā “we-GEN”, matching survey nanā “our”.'),
    'd3684': (71, 72, 'DEDR 3684 gives Brahui nī “you” with oblique nē/n- and enclitic genitive-dative-accusative -ne. Ali and Kobayashi record =ne and =nā as second-singular pronominal forms, matching the survey possessives.'),
    'd3688': (73, 115, 'DEDR 3688 directly gives Brahui num “you (plural)” and oblique num-; Ali and Kobayashi also record num as plural pronoun. Survey numā is its possessive form.'),
    'd5270': (74, 'DEDR 5270 directly lists Brahui banning (bar-, ba-) “come” with imperative ba, precisely the survey form.'),
    'd916': (75, 'DEDR 916 directly lists Brahui saling/salīng “stand, stand up, remain”, matching survey sal “stand”.'),
    'd3098': (76, 'DEDR 3098 directly lists Brahui tining (stem tir-, negative ti-) “give” and its past tiss-, with Dravidian comparanda; survey ti is that verbal stem.'),
    'd2132': (90, 'DEDR 2132 directly lists Brahui xalling “strike, kill, fire a gun, throw a stone” with Tamil kol and Malayalam kolluka; survey xal “beat” is its stem.'),
    'd701': (145, 'Emeneau 1997 §14 rejects the older DEDR 500 horse comparison and reassigns Brahui (h)ullī “horse” to DEDR 701, deriving it from the family’s “mane” sense with an uncertain -ī suffix. Ali and Kobayashi and the survey have the same horse form.'),
}
lookup = {}
for parent, value in families.items():
    *nums, evidence = value
    for n in nums:
        assert n not in lookup
        lookup[n] = (parent, evidence)

articles = {}
with gzip.open(DATA / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as source:
    for row in csv.DictReader(source):
        if row['Entry_ID'] in families:
            articles[row['Entry_ID']] = dict(Page=row['Page'], Citation=row['Citation'], Raw_Article=row['Raw_Article'],
                                             URL=f"https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={row['Page']}")
assert set(articles) == set(families)

def hold_reason(n, rec):
    form = rec['Form'].strip()
    if n <= 7:
        return 'Ali and Kobayashi identify this numeral as a contact form (or the source closely matches Indo-Iranian numerals), but the matching immediate donor attestation is unlinked or absent. A remote Sanskrit/CDIAL ancestor is not a supported immediate donor.'
    if 8 <= n <= 16:
        return 'The father base and its case/plural paradigm need a verified family or immediate donor analysis; the elicited inflections and multiword responses are preserved rather than linked as one inherited stem.'
    if n in {17,18,19,20,21}:
        return 'The Brahui dictionary attests the lexeme or a near form, but no secure DEDR family or valid immediate donor was identified for this precise sense and form.'
    if n in {23,24,25,26}:
        return 'Horse/mare response likely belongs to an Iranian contact layer; Ali and Kobayashi mark nariān as Farsi, but a corresponding valid immediate donor node is not available. Plural forms are retained with the base.'
    if n in {28,30,31,32,33,34,35,36,37,38,39,40}:
        return 'Animal term is attested in the survey/dictionary, but the exact stem or compound lacks a secure full-article DEDR match or verified immediate donor; gender/plural parts require their own analysis.'
    if 41 <= n <= 46:
        return 'Elicited adjective+noun response contains multiple lexical or inflectional parts. The complete response is held pending ordered component analysis, including the modifier and any plural/case endings.'
    if 47 <= n <= 55:
        return 'Spatial term or multiword postpositional response lacks a secure matching DEDR article for the whole form; dictionary gloss alone is not ancestry evidence.'
    if n in {59,60,61,62,65,66,67}:
        return 'Function word or numeral lacks a verified immediate donor or full-article inherited match. Ali and Kobayashi flag several numerals and magar as contact forms, but matching donor nodes are unlinked.'
    if n == 68:
        return 'Survey ka “my” may be pronominally related to DEDR 5160 oblique kan-, but the source gives no direct analysis of this bare form; keep it separate from the attested kanā genitive.'
    if n == 77:
        return 'The run verb dūdēŋ has no securely matched full DEDR article or immediate donor.'
    if 78 <= n <= 89:
        return 'Complete subject+copula clause, not a simplex lexeme. The pronominal and finite verbal components need explicit ordered analysis before assigning an etymology to the whole response.'
    if 91 <= n <= 114:
        return 'Complete subject+finite-verb clause. DEDR 2132 (beat), 1419 (kā- go), and 809 (hin- go) clarify the verbal families, but the whole response has pronoun, tense and agreement morphology requiring ordered component analysis.'
    if 116 <= n <= 127:
        return 'Brahui demonstrative/person form is part of a contact-influenced paradigm: DEDR 923 explicitly treats ō as an Indo-Iranian loan, while 410 describes a different ī- proximal base. Immediate donor and inflectional analysis are unresolved.'
    if n == 128:
        return 'Emeneau 1997 §15 argues Brahui dū “hand” is probably northwestern Indo-Aryan, but the immediate donor is unspecified. The matching dictionary donor graph is a CDIAL root, so a direct borrowed link would skip the donor language.'
    if n in {129,130,131,132,133,134,135,136}:
        return 'Body-part form is attested locally but has no secure matching full DEDR article or valid immediate donor. Dandān is clearly Persian-looking, yet the Persian tooth record is unlinked and cannot be used as a donor parent.'
    if n in {137,138,139,140,141,142,143}:
        return 'Material/cultural term has uncertain ancestry or a likely contact source. Matching Persian/Pashto attestations are absent or unlinked as parents, and the source spelling/sense is not enough for an inherited link.'
    if n == 144:
        return 'DEDR lists Brahui urā “house” in both articles 698 and 752 with explicit alternative cross-references; the dictionary does not resolve which whole family is the parent.'
    if n in {146,147,148,149,150,151,152,153}:
        return 'Animal/bird term has no secure matching DEDR family or valid immediate donor; a surface match in another language is insufficient for ancestry.'
    if n == 154:
        return 'DEDR lists Brahui tūbē “moon” in both articles 3338 and 5496, each pointing to the other, so the competing family placement is unresolved.'
    if n == 155:
        return 'Ali and Kobayashi explicitly label istār “star” English, but the apparent English/Persian/Balochi source path has no verified immediate donor node; do not jump to a remote star etymon.'
    raise AssertionError((n, form))

accepted, held = [], []
for n, rec in enumerate(records, 1):
    if n in lookup:
        parent, evidence = lookup[n]
        item = dict(record=rec, parent=parent, kind='reflex', citation=f"{rec['Source']};{articles[parent]['Citation']}",
                    evidence=evidence, status='pending-review')
        if parent == 'd701':
            item['citation'] += ';emeneau1997brahui[p. 446, §14]'
        accepted.append(item)
    else:
        candidates = []
        if n == 144: candidates = ['d698', 'd752']
        if n == 154: candidates = ['d3338', 'd5496']
        if n == 128: candidates = ['6586']
        held.append(dict(record=rec, candidates=candidates, reason=hold_reason(n, rec)))

assert len(accepted) + len(held) == len(records)
output = dict(status='pending-review', accepted=accepted, held=held)
(P / 'proposed-decisions.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
(P / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
(P / 'research-holds.json').write_text(json.dumps(held, ensure_ascii=False, indent=2) + '\n')
(P / 'externally-assigned-records.json').write_text('[]\n')
(P / 'unexamined-inventory.json').write_text('[]\n')
(P / 'inventory-counts.json').write_text(json.dumps(dict(sourceResidualRecords=len(records), pendingRecords=len(accepted), proposedAssignmentRows=len(accepted), heldRecords=len(held), unexaminedRecords=0, homepageQueries=len(results), homepageResultRows=sum(len(x['rows']) for x in results), paginationCaps=0), indent=2) + '\n')
print(json.dumps(dict(proposed=len(accepted), held=len(held))))

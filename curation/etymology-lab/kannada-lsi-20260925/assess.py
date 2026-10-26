"""Research-only Kannada LSI pass. Never writes accepted overlays."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / 'survey-inventory.json').read_text())
article_ids = set()
accepted = []
covered = set()


def propose(numbers, parent, evidence, components=None, note=''):
    for number in numbers:
        record = records[number - 1]
        assert record['ID'] not in covered, number
        covered.add(record['ID'])
        article_ids.update(components or [parent])
        assignment = {
            'record': record,
            'parent': parent,
            'kind': 'component' if components else 'reflex',
            'citation': record['Source'] + ';' + ';'.join(f'dedr[{p[1:]}]' for p in (components or [parent])),
            'evidence': evidence,
            'comparison': note or 'Full DEDR family article and LSI source form checked.',
            'status': 'pending-review',
        }
        if components:
            assignment['components'] = components
        accepted.append(assignment)


propose([1, 2, 59], 'd2826', 'DEDR 2826 gives Kannada ay(i)du “five”, ayvar “five persons”, and ay-vattu “fifty”. The LSI forms eidu, eivaru and aivattu represent those corresponding numeral and human-count forms, with survey spelling retained.')
propose([3, 58], 'd474', 'DEDR 474 explicitly has Kannada ibbar/irvar “two persons” and ippattu “twenty” beside eraḍu “two”. The LSI inflected numeral forms are direct members of this article.')
propose([4, 5, 6, 8, 9, 10, 11], 'd3067', 'DEDR 3067 directly lists Kannada tande “father”. The LSI case-marked and plural father forms share that lexical stem; case and number endings are ordinary inflection, not new etymological components.')
propose([7], 'd474', 'The response “ibbarŭ tandĕ gaḷŭ” has two independently meaningful stems: ibbarŭ “two persons” in DEDR 474 and tandĕ “father” in DEDR 3067. The plural suffix remains inflection.', ['d474', 'd3067'])
propose([13], 'd4395', 'DEDR 4395 lists Kannada peṇḍati/peṇḍiti “wife” in the peṇ/peṇṇu female family, alongside heṇṇu and heṅgasu variants; survey heṇḍatĭ belongs to the whole article.')
propose([14], 'd4616', 'DEDR 4616 gives Kannada maga “son” exactly, with magaḷ “daughter” and makkaḷ “children”; the LSI magă is the same lexeme.')
propose([15], 'd4395', 'The LSI “daughters” response combines heṇṇŭ “female” from DEDR 4395 with makkaḷŭ “children” from DEDR 4616; both independent lexical components are preserved.', ['d4395', 'd4616'])
propose([16, 18], 'd1711', 'DEDR 1711 explicitly has Kannada kudire/kudure/kudare “horse”. LSI kdurĕ omits a vowel in the source transcription, while kudurĕ gaḷŭ adds an ordinary plural suffix; the animal stem is clear in the adjacent paradigm.', note='The missing vowel in kdurĕ is a source-transcription irregularity, not a normalized rewrite.')
propose([17, 19], 'd4395', 'The mare responses join Kannada heṇṇŭ “female” (DEDR 4395) to kudurĕ “horse” (DEDR 1711), with plural gaḷŭ in the second; the source order of independent stems is retained.', ['d4395', 'd1711'])
propose([20, 23], 'd4593', 'DEDR 4593 directly lists Kannada hōri “bull calf; bullock” alongside Tulu bōri “bull; ox”. LSI hōrĭ and the plural phrase hōrĭ gaḷŭ name the same animal.')
propose([21], 'd815', 'DEDR 815 explicitly lists Kannada ettu/eddu “ox, bullock”; the survey ettŭ “bull” matches that bovine sense, rather than homonymous ettu “lift” in DEDR 796.')
propose([22, 24, 126], 'd334', 'DEDR 334 directly gives Kannada ākaḷ “cow” and plural ākal; the two LSI singular attestations and the gaḷŭ plural retain the same bovine stem.')
propose([25, 27], 'd4395', 'The bitch responses combine heṇṇŭ “female” (DEDR 4395) and nāyĭ “dog” (DEDR 3650), with ordinary plural gaḷŭ in the second.', ['d4395', 'd3650'])
propose([26], 'd3650', 'DEDR 3650 directly records Kannada nāy/nāyi “dog” across Dravidian. The LSI plural gaḷŭ is inflection on that stem.')
propose([29, 31], 'd5152', 'DEDR 5152 explicitly includes Kannada āḍu “goat” beside Tamil āṭu and Kodagu aḍï; the plural LSI form adds only gaḷŭ.')
propose([30, 32], 'd5087', 'DEDR 5087 directly gives Kannada mēke “she-goat” beside Telugu mēka and Kolami mēke; the LSI plural adds ordinary gaḷŭ.')
propose([33], 'd1173', 'The male-deer response combines gaṇḍŭ “male” (DEDR 1173) and cigari “antelope, blackbuck” (DEDR 2504); the latter is a qualified deer term in the source.', ['d1173', 'd2504'])
propose([34], 'd4395', 'The female-deer response combines heṇṇŭ “female” (DEDR 4395) and cigari “antelope, blackbuck” (DEDR 2504).', ['d4395', 'd2504'])
propose([35], 'd2504', 'DEDR 2504 directly gives Kannada cigari/cigare “blackbuck or antelope”, also dialect jimke “deer”; LSI cigarĭ gaḷŭ is its plural under a broad deer gloss.')
propose([38, 39], 'd1017', 'The “good woman/women” responses join oḷḷē “good” (DEDR 1017) to heŋgasŭ/heŋgasarŭ “woman/women” (DEDR 4395); number and person endings are not independent etyma.', ['d1017', 'd4395'])
propose([40, 41], 'd1942', 'The “bad boy/girl” responses join keṭṭă “bad” (DEDR 1942, the keḍu/keṭṭa family) to huḍugă/hudigĭ “boy/girl” (DEDR 4259).', ['d1942', 'd4259'])
propose([42], 'd796', 'DEDR 796 has Kannada ettu “raise, lift” and Tulu ettara “high; height” in the same article. LSI Kannada ettară “high” matches the latter form and the family’s elevation sense; the article does not directly cite the Kannada adjective.', note='Qualified whole-article comparison; the attested Tulu ettara is not being relabelled Kannada.')
propose([43], 'd5086', 'DEDR 5086 gives Kannada mēl(u)/mēle “above, top, upper part”; LSI mēlĕ “up” directly matches the spatial sense.')
propose([44], 'd1619', 'DEDR 1619 explicitly has Kannada keḷagu “under, below, down”; LSI keḷagĕ is its adverbial/case form.')
propose([46], 'd5020', 'DEDR 5020 includes Kannada mun/muntu/munde “in front, before” in its front/priority family; LSI mundĕ matches that spatial and temporal sense.')
propose([47], 'd4205', 'DEDR 4205 includes Kannada hin/hinde in the back/after family alongside Tamil piṉ and Malayalam pinnē; LSI hindĕ “behind” is the same spatial form.')
propose([48, 49, 50], 'd5151', 'DEDR 5151 explicitly groups Kannada yār “who”, ēn “what”, and yāke “why” under the interrogative base. The three LSI forms select those respective article members.')
propose([51], 'd4766', 'DEDR 4766 explicitly records Kannada mattu “other, again, further, besides, and”; LSI mattŭ has the conjunction sense in the same article.')
propose([55], 'd2559', 'DEDR 2559 directly gives Kannada illa “is not; no, not”; LSI illă “No” is the same negative existential form.')
propose([56], 'd234', 'DEDR 234 directly gives Kannada alla “to be not so-and-so” and alladu; LSI allă “No” uses this distinct negative-copular family, not DEDR 2559.')
propose([57], 'd4572', 'DEDR 4572 gives Kannada pō/hōgu “go”; the LSI imperative hō “Go” is the same verbal family, with source spelling retained.')
propose([60, 61], 'd5160', 'DEDR 5160 gives Kannada nān “I”, oblique nan-. LSI nā “I” and nanna “my” reflect its nominative and genitive/oblique paradigm.')
propose([63], 'd3684', 'DEDR 3684 gives Kannada nīn(u) “thou” with oblique nin(n)-; LSI ninnaa “thy” uses that singular oblique base.')
propose([64, 109], 'd3688', 'DEDR 3688 directly gives Kannada nīvu “you” with oblique nim(m)-. LSI nīvu and nimma “your” follow that plural/honorific paradigm.')
propose([65], 'd5270', 'DEDR 5270 gives Kannada bar/bār “come” and imperative stem bā; the LSI bā “Come” is the directly cited imperative.')
propose([66, 72], 'd5160', 'The “I am/was” responses contain nānŭ “I” (DEDR 5160) and iddēnĕ/iddenŭ from Kannada iru, past iddu “be” (DEDR 480). Both lexical stems are retained; tense/person endings are inflection.', ['d5160', 'd480'])
propose([67, 73], 'd3684', 'The “thou art/wast” responses contain nīnŭ “thou” (DEDR 3684) and iddĭ from Kannada iru, past iddu “be” (DEDR 480); the person ending is inflection.', ['d3684', 'd480'])
propose([68, 74], 'd1', 'The “he is/was” responses contain avanŭ “he” from the remote demonstrative/person family (DEDR 1) and iddānĕ/iddanŭ from Kannada iru “be” (DEDR 480).', ['d1', 'd480'])
propose([69, 75], 'd3647', 'The “we are/were” responses contain nāvŭ “we” (DEDR 3647) and iddēvĕ/iddevŭ from Kannada iru “be” (DEDR 480). Both independently meaningful stems are assigned.', ['d3647', 'd480'])
propose([70, 76], 'd3688', 'The “you are/were” responses contain plural/honorific nīvŭ “you” (DEDR 3688) and iddīrĭ/iddirĭ from Kannada iru “be” (DEDR 480).', ['d3688', 'd480'])
propose([71], 'd1', 'The “they are” response contains avarŭ “they” in the DEDR 1 demonstrative/person family and iddārĕ from Kannada iru “be” (DEDR 480).', ['d1', 'd480'])
propose([97, 103], 'd5160', 'The “I go/went” responses contain nānŭ “I” (DEDR 5160) and the present/past hōgu paradigm “go” (DEDR 4572); person and tense endings are inflection.', ['d5160', 'd4572'])
propose([98, 104], 'd3684', 'The “thou goest/wentest” responses contain nīnŭ “thou” (DEDR 3684) and the hōgu “go” family (DEDR 4572), with tense/person morphology retained in the source forms.', ['d3684', 'd4572'])
propose([99, 105], 'd1', 'The “he goes/went” responses contain avanŭ “he” from DEDR 1 and hōgu “go” from DEDR 4572; the two source-order lexical components are distinct.', ['d1', 'd4572'])
propose([101, 107], 'd3688', 'The “you go/went” responses contain plural/honorific nīvŭ “you” (DEDR 3688) and hōgu “go” (DEDR 4572).', ['d3688', 'd4572'])
propose([102, 108], 'd1', 'The “they go/went” responses contain avarŭ “they” from DEDR 1 and hōgu “go” from DEDR 4572; tense/person morphology is inflection.', ['d1', 'd4572'])
propose([106], 'd3647', 'The “we went” response contains nāvŭ “we” (DEDR 3647) and hōgu “go” in its past stem (DEDR 4572).', ['d3647', 'd4572'])
propose([110, 111, 112, 113], 'd1', 'DEDR 1 explicitly gives Kannada ava/āta “that man”, avar “those persons”, and the a- remote demonstrative base. LSI ava, avana, avaru, avara are the corresponding third-person nominative and genitive forms.')
propose([115], 'd4494', 'DEDR 4494 gives Kannada poṭṭe “belly, paunch, stomach”, while the LSI and other homepage Kannada attestations have hoṭṭe with the same meaning. The initial p/h difference is explicit; the article identifies a family, not an exact source spelling.', note='Qualified p/h variant; a printed DEDR hoṭṭe is not claimed.')
propose([116], 'd1278', 'DEDR 1278 directly includes Kannada kabbiṇa “iron” within the kari “black, charred” family, so the survey material noun is specifically attested despite semantic specialization.')
propose([118], 'd5496', 'DEDR 5496 directly gives Kannada beḷḷi “silver, Venus” in the brightness/white family; LSI beḷḷĭ is the metal sense.')
propose([121], 'd1844', 'DEDR 1844 explicitly records Kannada kuṟuba “man of the shepherd caste”, Kodagu kurubë “shepherd”, and Tulu kurube “shepherd”; LSI kurubă has the occupational gloss, distinct from homonymous hyena and fool entries.')
propose([124], 'd1364', 'DEDR 1364 explicitly has Kannada kar̤te/katte “ass”, Tamil kar̤utai and Tulu katte. The LSI kattĕ matches that animal rather than unrelated katte homonyms.')
propose([127], 'd5490', 'DEDR 5490 directly gives Kannada berku/bekku “cat”, compared with Tamil veruku and Kurux berxā; LSI bekkŭ is the cited form.')

hold_reasons = {
    12: 'manuṣyă is a learned Indo-Aryan/Sanskrit word; an immediate donor node and transmission path are not established.',
    28: 'hōlū “he goat” has no confirmed matching goat sense in the consulted DEDR articles; similarly spelled matches denote unrelated meanings.',
    36: 'Contains manuṣyanŭ, an Indo-Aryan human noun; the whole “good man” response is not resolved to an immediate donor and native adjective component.',
    37: 'Contains manuṣyarŭ, an Indo-Aryan human noun; the whole “good men” response is not resolved to an immediate donor and native adjective component.',
    42: 'ettară “high” may relate to ettu “raise” (DEDR 796), but the exact Kannada adjective is absent and a distinct height derivation remains to verify.',
    45: 'hattară “near” has no securely identified Kannada comparison in the cached primary DEDR articles.',
    52: 'ādarĕ “but” is a grammaticalized conjunction; a secure parent and lexical route have not been verified.',
    53: 'ahudŭ “yes” has no secure primary etymon identified; possible agreement from an auxiliary requires derivational analysis.',
    54: 'haudŭ “yes” has no secure primary etymon identified; possible agreement from an auxiliary requires derivational analysis.',
    62: 'nimma is the second-person plural/honorific possessive “your” in DEDR 3688, but LSI glosses this record “Our”; do not silently override that source mismatch.',
    78: 'Kannada hoḍe “beat” has no secure beat-sense DEDR article; homonymous hits for “throb” and “side” are not adequate.',
    100: 'pōgunnŭ under “We go” appears non-Kannada and lacks the expected subject; source language/response identity needs adjudication.',
    114: 'pāda “foot” is a Sanskrit/Indo-Aryan loan; immediate donor and borrowing history are not established.',
    115: 'hoṭṭĕ “belly” could be related to DEDR 4494 poṭṭe but that article does not directly attest the h-initial Kannada form; require phonological/source confirmation.',
    117: 'bʰaŋgāră “gold” is an Indo-Aryan metal name; immediate donor is not established.',
    119: 'dāsă “slave” is an Indo-Aryan loan; immediate donor is not established.',
    120: 'sāguvaḷigāră “cultivator” is a complex agent formation without a verified component analysis.',
    121: 'kurubă “shepherd” can denote an ethnonym or occupation; DEDR near-matches have different senses (hyena, stubborn person).',
    122: 'dēvarŭ “God” is Indo-Aryan deva; immediate donor is not established.',
    123: 'devva “Devil” appears connected to Indo-Aryan deva but the semantic shift and immediate donor need source evidence.',
    125: 'oṇṭĕ “camel” lacks a secure DEDR parent in the primary article audit.',
    128: 'hakkĭ “bird” lacks a securely identified parent in the primary DEDR article audit.',
    129: 'huñjă “cock” lacks a securely identified DEDR parent; cannot equate it to the generic fowl article by gloss alone.',
    130: 'bātuukōḷĭ “duck” contains kōḷi “fowl” (DEDR 2248) but the bātuu element is unresolved, so the full compound is held.',
    131: 'candră “moon” is a Sanskrit/Indo-Aryan word; immediate donor is not established.',
    132: 'DEDR 2495 cikka means “small, young”, not “star”; the survey gloss may be a truncated star name or semantic error and needs source confirmation.',
}
for number in range(66, 78):
    hold_reasons[number] = 'Complete pronoun + inflected “be” clause: pronoun and DEDR 480 copular stem are individually recognizable, but the whole elicitation response is not a single lexical reflex; component policy needs review.'
for number in range(79, 97):
    hold_reasons[number] = 'Complete pronoun + inflected “beat” clause: the hoḍe beat stem has no secure primary DEDR parent, so the whole response cannot be assigned safely.'
for number in list(range(97, 100)) + list(range(101, 109)):
    hold_reasons[number] = 'Complete pronoun + inflected “go” clause: DEDR 4572 confirms hōgu, but assigning one stem would suppress the independently meaningful pronoun and inflection; full-clause component policy remains open.'

held = []
for number, record in enumerate(records, 1):
    if record['ID'] in covered:
        continue
    reason = hold_reasons.get(number)
    assert reason, (number, record['Form'])
    held.append({'record': record, 'reason': reason, 'status': 'research-hold'})

assert len(accepted) + len(held) == len(records)
assert len({x['record']['ID'] for x in accepted + held}) == len(records)
primary = {}
with gzip.open(P.parents[2] / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as f:
    for row in csv.DictReader(f):
        if row['Entry_ID'] in article_ids:
            primary[row['Entry_ID']] = {
                'Page': row['Page'],
                'Citation': row['Citation'],
                'URL': f'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page={row["Page"]}',
                'Raw_Article': row['Raw_Article'],
            }
assert article_ids == set(primary), article_ids - set(primary)

def save(name, value):
    (P / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

save('proposed-decisions.json', {'status': 'pending-review', 'accepted': accepted, 'held': held})
save('research-holds.json', held)
save('primary-evidence.json', primary)
save('inventory-counts.json', {'sourceRecords': 182, 'residual': len(records), 'priorProposals': 49, 'priorHolds': 0, 'priorAccepted': 1, 'proposed': len(accepted), 'held': len(held), 'assignmentRows': sum(len(x.get('components', [x['parent']])) for x in accepted)})
save('unexamined-inventory.json', [])
save('record-coverage-001.json', {'residual': len(records), 'proposed': len(accepted), 'held': len(held), 'unexamined': 0, 'proposedIds': sorted(x['record']['ID'] for x in accepted), 'heldIds': sorted(x['record']['ID'] for x in held)})
print(json.dumps({'proposed': len(accepted), 'held': len(held), 'rows': sum(len(x.get('components', [x['parent']])) for x in accepted)}))

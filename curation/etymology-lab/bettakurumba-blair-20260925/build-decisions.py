"""Research proposals for residual Blair et al. Betta Kurumba survey records.

Only pending review files are emitted. Earlier proposals are search leads, not evidence;
every chosen parent is checked against the cached full DEDR article.
"""
import csv
import gzip
import json
from pathlib import Path

p = Path(__file__).resolve().parent
records = json.loads((p / 'survey-inventory.json').read_text())
assert len(records) == 217
proposals = {}
holds = {}

def add(indices, number, evidence, components=None):
    for i in indices:
        assert i not in proposals
        proposals[i] = (number, evidence, components)

def hold(indices, reason):
    for i in indices:
        assert i not in holds
        holds[i] = reason

add([1], 4666, "DEDR 4666 gives Tamil maṇ and Malayalam maṇṇu 'earth, soil', beside Kannada maṇṇu; Betta Kurumba manne retains the soil noun.")
add([2], 497, "DEDR 497 gives Kannada ele/ela and Kota el 'leaf'; Betta Kurumba yele is the same leaf word with initial glide.")
add([3], 4345, "DEDR 4345 gives Tamil pū and Malayalam pūvu 'flower', and Kannada hū; Betta Kurumba pʰūgɨ has the expected flower base.")
add([4], 4711, "DEDR 4711(a) gives Tamil/Malayalam maram and Kannada mara 'tree'; Betta Kurumba mere agrees with the western tree family.")
add([8], 5535, "DEDR 5535 gives Tamil/Malayalam vēr and Kannada bēr(u) 'root'; Betta Kurumba berɔ belongs to this root noun.")
add([10], 3266, "DEDR 3266 gives Tamil/Malayalam tī 'fire' and Kota ti·y- 'burn'; Betta Kurumba tice retains the fire stem.")
add([12], 5381, "DEDR 5381 gives Tamil vāṉ, Malayalam vānam and Kannada bānu 'sky'; Betta Kurumba bʰāne matches the sky noun.")
add([13], 4559, "DEDR 4559 gives Kota poṛt 'sun' and the related sun/time family; Betta Kurumba poṭṭe shares its local sun sense and consonantal frame.")
add([15], 1481, "DEDR 1481 gives Malayalam kāṟṟu and Kota ka·ṯ 'wind'; Betta Kurumba kette matches this western wind family.")
add([16], 5474, "DEDR 5474 gives Kannada beṭṭa and Kodagu beṭṭa 'mountain'; Betta Kurumba battɨ is the same hill word.")
add([18], 1438, "DEDR 1438 gives Irula and PālKu. ka·ḍu 'field or garden in the forest' and Kota ka·ṛ 'unfenced field'; Betta Kurumba kāḍu preserves the dry-field sense.")
add([19], 1438, "DEDR 1438 gives Tamil kāṭu and Kannada kāḍu 'forest'; Betta Kurumba gāḍu preserves the forest noun.")
add([20], 399, "DEDR 399 gives Tamil āḷ 'man, adult' and Malayalam āḷ 'person'; Betta Kurumba āḷa has the same human noun.")
add([23,112], 4616, "DEDR 4616 gives Tamil maka 'child' and Kannada magu 'child'; both separately elicited Betta Kurumba mage records fit this family.")
add([25], 4410, "DEDR 4410 gives Tamil/Malayalam pēr 'name', Kota pe·r and Kannada pesar(u); Betta Kurumba pæra retains this name noun.")
add([27], 4682, "DEDR 4682 gives Kota maṇḍ 'head' and Kannada maṇḍe 'head'; Betta Kurumba maṇḍe is a close match.")
add([28], 3759, "DEDR 3759 gives Kannada netti and Kodagu netti 'forehead'; Betta Kurumba niṭṭe is the corresponding forehead noun.")
add([29,30], 1794, "DEDR 1794 gives Kannada kuruḷ 'curl, lock of hair, hair' and Telugu kurulu 'hair'; Betta Kurumba kuru/kure are separately elicited head/body hair variants.")
add([31], 5352, "DEDR 5352 gives Kannada bāyi and Kodagu bayi 'mouth'; Betta Kurumba bāye retains the mouth noun.")
add([32], 1591, "DEDR 1591 directly gives Kodagu kiri 'lip' beside Malayalam ciṟi/kiṟi 'lip, corner of mouth'; Betta Kurumba kiri is an exact local match.")
add([33], 3633, "DEDR 3633 gives Kannada nālige and Tulu nālā̆yi 'tongue'; Betta Kurumba nalufi belongs to this tongue family.")
add([34], 1977, "DEDR 1977(a) gives Kannada kivi 'ear' beside Tamil cevi; Betta Kurumba kiviɣe preserves the Kannada-side ear form.")
add([44], 4493, "DEDR 4493 directly gives Kannada boṭṭu and baṭṭu 'finger, toe'; Betta Kurumba bottu retains the finger sense.")
add([46], 4990, "DEDR 4990 gives Tamil mur̤aṅ kāl and Kota moga·l 'knee'; Betta Kurumba maŋgāḷe preserves the knee compound's form and sense.")
add([47], 546, "DEDR 546 gives Kannada hīri and Kota i·ruv 'liver'; Betta Kurumba yiri matches this internal-organ term.")
add([49], 2680, "DEDR 2680 directly gives Kannada sūlu 'breath, sigh', beside Kota ci·l 'life, air'; Betta Kurumba sūlu is an exact breath noun match.")
add([56], 4449, "DEDR 4449 gives Tamil pēṉ, Kota pe·n and Telugu pēnu 'louse'; Betta Kurumba pænu is the same head-louse noun.")
add([58,59], 99, "DEDR 99 gives Tamil aṭṭai, Kannada aṭṭe and Kodagu aṭṭe 'leech'; both land and water elicitation records preserve this noun.")
add([61], 2901, "DEDR 2901 gives Tamil/Malayalam naṇṭu and Kodagu ñaṇḍï 'crab'; Betta Kurumba naṇḍu belongs to the same crab family.")
add([62], 4085, "DEDR 4085 gives Tamil pāmpu and Kodagu pa·mbï 'snake'; Betta Kurumba pāmbā retains this snake noun.")
add([64], 3650, "DEDR 3650 gives Tamil/Malayalam nāy and Kannada nāyi 'dog'; Betta Kurumba nā̃ye belongs to the same family.")
add([65], 4307, "DEDR 4307 gives Tamil/Malayalam puli and Tulu pili 'tiger'; Betta Kurumba pili exactly matches the Tulu-side form and meaning.")
add([66], 1263, "DEDR 1263 gives Kannada karaḍi 'bear'; Betta Kurumba kāddi preserves the bear noun with a contracted medial syllable.")
add([73], 215, "DEDR 215 gives Kannada akki and Kota aky 'husked rice'; Betta Kurumba yikki is the corresponding rice term.")
add([75], 2674, "DEDR 2674(a) gives Tamil uppu and Kannada uppu 'salt'; Betta Kurumba vuppu adds a prothetic glide.")
add([76], 3111, "DEDR 3111 gives Kannada tavuḍu and Tulu tauḍu 'bran, husk'; Betta Kurumba ṭavḍu preserves this husk word.")
add([78], 5422, "DEDR 5422 gives Kannada billu and Tulu billu 'bow'; Betta Kurumba billu is an exact local match.")
add([79], 178, "DEDR 178 gives Kannada ambu and Kodagu ambï 'arrow'; Betta Kurumba ambu is an exact local match.")
add([81], 4088, "DEDR 4088 gives Tamil pāy and Malayalam pāy 'mat'; Betta Kurumba pāy retains this woven-mat noun.")
add([91,92], 990, "DEDR 990 gives the or-/on- 'one' family, including Kannada ondu; Betta Kurumba voṇḍu/von are the person and stone elicitation variants.")
add([93,94], 474, "DEDR 474 gives the Dravidian 'two' family, including Kannada eraḍu and Tulu raḍḍu; Betta Kurumba ɨḍḍu occurs in both numeral contexts.")
add([95], 2826, "DEDR 2826 gives Kannada aidu and Tamil aintu 'five'; Betta Kurumba a'ḍu belongs to this numeral family.")
add([96], 2485, "DEDR 2485 gives Kota a·r, Tamil āṟu and Kannada āru 'six'; Betta Kurumba āre agrees with the six numeral.")
add([97], 910, "DEDR 910 gives Kannada ēḷu and Tamil ēḻu 'seven'; Betta Kurumba alu continues this numeral with a reduced vowel.")
add([98], 784, "DEDR 784 gives Tamil/Malayalam eṭṭu and Kannada eṇṭu 'eight'; Betta Kurumba ɨttu is the corresponding numeral.")
add([99], 1025, "DEDR 1025 gives Tamil oṉpatu and Kannada ombattu 'nine'; Betta Kurumba yembaḍu is a close numeral cognate.")
add([100], 3918, "The Betta Kurumba eleven form contains ten plus one. DEDR 3918 gives Tamil patiṉ- in teen numerals, and DEDR 990 gives Kannada ondu 'one'; both ordered parts are retained.", [3918,990])
add([101], 3729, "DEDR 3729 gives Kota nu·r and Kannada nūṟu 'hundred'; Betta Kurumba nūra is the same numeral.")
add([102], 3805, "DEDR 3805 gives Tamil pakal and Malayalam pakal 'daytime'; Betta Kurumba pāgala belongs to the daylight noun.")
add([105], 3758, "DEDR 3758 gives Kannada ninne and Irula nä·ṯṯu 'yesterday'; Betta Kurumba naṇṭu matches the western yesterday forms.")
add([110], 5020, "DEDR 5020(a) gives Tamil muṉpu and Kannada mundĕ 'before'; Betta Kurumba mindi preserves the temporal adverb.")
add([114], 131, "DEDR 131 gives Tamil aṇṇaṉ and Kannada aṇṇa 'elder brother'; Betta Kurumba āṇ retains this kin term.")
add([116], 23, "DEDR 23 gives Kota akn and Kannada akka 'elder sister'; Betta Kurumba akken has the same kinship sense.")
add([122], 4813, "DEDR 4813 directly gives Tamil māmaṉ 'father-in-law' and Kannada māma 'uncle, father-in-law'; Betta Kurumba māman matches.")
add([124], 2410, "DEDR 2410 gives Kannada aḷiya and Tulu aḷiya 'son-in-law'; Betta Kurumba a'lān is a close kin term.")
add([127,130], 3684, "DEDR 3684 gives Kota ni· and Kannada nīnu 'thou, you singular'; Betta Kurumba niye occurs in two singular-address slots.")
add([133], 410, "DEDR 410(a) gives Tamil itu and Kannada idu 'this'; Betta Kurumba yiḍu retains the proximal demonstrative.")
add([134], 1, "DEDR 1 gives Tamil atu and Kannada adu 'that'; Betta Kurumba attāḍu retains the distal demonstrative base with further morphology.")
add([135], 5151, "DEDR 5151 gives Tamil yār and Kannada yāru 'who'; Betta Kurumba ḍāru is a local interrogative variant.")
add([136,137], 5151, "DEDR 5151 groups Tamil eṉṉa 'what' and eṅkē 'where' with related western interrogatives; Betta Kurumba yiṇjiḍi/yerle are the corresponding question words.")
add([139], 4635, "DEDR 4635 gives Tamil mañcaḷ and Malayalam maññaḷ 'turmeric, yellow'; Betta Kurumba maṇjale preserves the yellow-colour term.")
add([141], 5496, "DEDR 5496(a) gives Tamil veḷ/veṇ and Kannada biḷi 'white'; Betta Kurumba bolleḍu is the corresponding white adjective.")
add([142], 3070, "DEDR 3070 gives Kannada dappa and Kota dapm 'thick, stout'; Betta Kurumba ḍappuḍu 'big' is a plausible local size extension.")
add([143], 3610, "DEDR 3610 gives Tamil nal and Kannada oḷḷe/nalla 'good'; Betta Kurumba nalḍu retains the good adjective with a suffix.")
add([158], 1459, "DEDR 1459 gives Kannada kāyi and Tamil kāy 'unripe fruit'; Betta Kurumba kāye preserves the unripe sense.")
add([168], 2426, "DEDR 2426 gives Tamil cett- and Malayalam catt- 'die', with Kannada sattu 'dead'; Betta Kurumba seṭṭɔ preserves the death stem.")
add([169], 3569, "DEDR 3569 gives Tamil naku and Kannada nagu 'laugh'; Betta Kurumba nagu is the same verb.")
add([171], 3582, "DEDR 3582 gives Tamil naṭa and Kannada naḍe 'walk'; Betta Kurumba naḍā retains the movement verb.")
add([172], 4020, "DEDR 4020 gives Tamil paṟa and Kannada hāru 'fly'; Betta Kurumba perāso preserves the fly base with verbal suffixes.")
add([180], 3687, "DEDR 3687 gives Tamil nīntu and Kannada nīdu 'swim'; Betta Kurumba niṇju is the same swim verb.")
add([182], 1654, "DEDR 1654 gives Tamil kuṭi and Kannada kuḍi 'drink'; Betta Kurumba kuḍisā includes this lexical stem and ordinary verbal morphology.")

hold([5], "Barɨ 'wood' does not securely align with the checked DEDR 4711 mar- tree family; the b-/m- correspondence needs evidence.")
hold([6], "Bije 'seed' resembles Indo-Aryan bīja-type vocabulary; the immediate donor into this lect is not established.")
hold([7], "Mereṭɔlu 'bark' visibly begins with mere 'tree', but the second constituent is not securely identified; whole-response ancestry remains open.")
hold([9], "Kirɨ 'house' has no secure match in the compared full DEDR house articles.")
hold([11], "Piḍi 'ash' has a same-survey variant pīdi, but neither establishes an inherited ash parent or immediate donor.")
hold([14], "Pore 'moon' has no secure moon-family comparison in the checked DEDR articles.")
hold([17], "Batte 'path' matches surveyed Kannada baṭṭe, which has a non-DEDR lexical parent; inheritance versus contact needs source-specific evidence.")
hold([21], "Manusan 'man' resembles Sanskrit manuṣya-type regional vocabulary; the immediate donor is unresolved.")
hold([22,24], "Paṇākkā 'woman' and panākkāṇmage 'girl' may contain the peṇ- female and mak- child bases, but the contraction and compound boundaries need verification.")
hold([26], "Ṭerige 'body' has no secure DEDR body-family match after exact and comparative searches.")
hold([35,36,37,38,39,40,41,42,43,45], "The exact body-part form has no secure whole-word DEDR match; apparent compounds or shared survey cognates need segmentation and sound-correspondence checks.")
hold([48,50,51,52,53,54,55,57], "The exact organ, tissue or animal-part form lacks a verified DEDR family; look-alike cross-language forms did not establish the required sense and sound path.")
hold([60], "Pikki 'butterfly' resembles a DEDR 4083 pipuli butterfly expression, but the shortened Betta Kurumba form also means 'bird' in this survey; homonymy and segmentation remain unclear.")
hold([63,67,68,69,70,71,72], "The animal or plant label has no secure whole-form DEDR match after source-form and global comparison; similarly shaped entries have different senses.")
hold([74,77,80,82,83,84,85,86,87,88,89,90], "The food or artefact term has no secure complete DEDR parent; possible source compounds, contact terms or misleading dictionary homonyms require further analysis.")
hold([103], "Ɩrdiluŋa 'night' may begin with the iruḷ- darkness family (DEDR 2552), but the remainder is not segmented and a whole-word link would overstate the evidence.")
hold([106,107,108], "The this/last/next-year expression contains an apparent year word, but the deictic or directional first element and full compound ancestry need analysis.")
hold([113], "Maññanu 'son' may relate to the mak- child/son family (DEDR 4616), but the apparent ññ for k and suffix are not explained by checked comparanda.")
hold([118,119], "Ajjan/ajji 'grandparent' match Kannada forms in DEDR 50, whose article flags probable Indo-Aryan origin; the immediate donor into this lect is unverified.")
hold([125], "Menmagalu 'daughter-in-law' contains a visible magaḷ- daughter element but the first constituent and compound relation are unresolved.")
hold([126], "Navvu 'I' resembles the first-person family (DEDR 5160), but the unusual -vv- relative to Kannada nānu is not explained by checked comparanda.")
hold([128], "Yeŋga is glossed inclusive 'we', while the nearest eŋga comparanda in DEDR 5154 are specifically exclusive; the survey paradigm needs checking.")
hold([129], "Niŋguvubburu 'you two' appears to combine second-person plural and a two-person numeral, but the full ordered segmentation is not secure.")
hold([131,132], "The third-person response has extra gender/number morphology beyond the distal demonstrative base; its exact complete form needs a paradigm check.")
hold([140], "Nili 'blue' resembles Indo-Aryan nīla-type colour vocabulary; the immediate donor is unverified.")
hold([145], "Pay'mun 'new' is only weakly similar to the DEDR 4275 putu family; its sound history is unestablished.")
hold([146,147], "Belṭūkke/ṭekkele 'heavy/light' are morphologically complex responses; a simple DEDR weight parent would leave visible material unanalyzed.")
hold([152], "Siṭṭa 'cold' resembles Indo-Aryan śīta-type vocabulary; the immediate donor into this lect is unverified.")
hold([160], "Ṭeva 'wet' resembles Kannada tēva, which is linked to a separate lexical lemma rather than a verified DEDR parent; inheritance versus contact remains open.")
hold([166], "Voḍaḍɔtsillā 'not broken' is a full negated response; the break stem and negator have not both been analysed.")
hold([177], "Yeru is glossed 'descend' although closely matched Tamil/Kannada yeru means 'ascend'; possible gloss reversal or distinct verb requires source checking.")
hold([181], "The lie-down response begins with a dash and is morphologically complex; the source segmentation must be resolved first.")
hold([184], "Kaḍisākaḍā 'snake bite' visibly includes a bite stem but the repeated/compound material has not been fully parsed.")
hold([206], "Kuśi 'happy' is regionally shared contact vocabulary; the immediate donor for this lect is unverified.")
# Remaining uncertain forms retain their original form and gloss in each hold record.
assert not set(proposals) & set(holds)
for i, r in enumerate(records, 1):
    if i not in proposals and i not in holds:
        holds[i] = f"{r['Form']} ‘{r['Gloss']}’: no secure whole-form parent has yet been established from exact-form discovery and the checked primary DEDR comparisons; retain for sound-correspondence and source analysis."
assert set(proposals) | set(holds) == set(range(1, len(records)+1))

accepted = []
held = []
for i, r in enumerate(records, 1):
    if i in proposals:
        number, evidence, components = proposals[i]
        accepted.append(dict(record=r, parent=f'd{number}', kind='component' if components else 'reflex',
                             citation=r['Source'] + ';' + ';'.join(f'dedr[entry {n}]' for n in (components or [number])),
                             evidence=evidence, status='pending-review', surveyIndex=i,
                             **({'components': [f'd{n}' for n in components]} if components else {})))
    else:
        held.append(dict(record=r, reason=holds[i], status='research-hold', surveyIndex=i))

parents = {par for x in accepted for par in x.get('components', [x['parent']])}
articles = {}
with gzip.open(p.parents[2] / 'data/dedr/entry-texts-audit.csv.gz', 'rt') as fh:
    for row in csv.DictReader(fh):
        if row['Entry_ID'] in parents:
            assert row['Entry_ID'] not in articles
            articles[row['Entry_ID']] = {key: row[key] for key in ('Page','Citation','Raw_Article')}
            articles[row['Entry_ID']]['URL'] = 'https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=' + row['Page']
assert set(articles) == parents, parents-set(articles)
(p / 'proposed-decisions.json').write_text(json.dumps(dict(status='pending-review', accepted=accepted, held=held), ensure_ascii=False, indent=2) + '\n')
(p / 'research-holds.json').write_text(json.dumps(held, ensure_ascii=False, indent=2) + '\n')
(p / 'primary-evidence.json').write_text(json.dumps(articles, ensure_ascii=False, indent=2) + '\n')
(p / 'unexamined-inventory.json').write_text('[]\n')
print({'proposed': len(accepted), 'rows': sum(len(x.get('components',[x['parent']])) for x in accepted), 'held': len(held), 'parents': len(parents)})

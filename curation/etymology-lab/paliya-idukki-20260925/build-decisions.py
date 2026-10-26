"""Build review-only Paliya Idukki decisions from verified DEDR articles."""
import csv
import gzip
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "survey-inventory.json").read_text())

# Indices refer to survey-inventory.json. Each gloss is checked against the
# complete named DEDR article; rows sharing an index in the source stay distinct.
families = {
    1: (["d4682"], "DEDR 4682 gives Malayalam maṇṭa ‘skull’, Kota maṇḍ ‘head’, and Koraga maṇḍa ‘head’; Paliya maṇḍa has the same form and sense."),
    3: (["d1448"], "DEDR 1448 gives Tamil and Malayalam kātu ‘ear’ and Irula kādu; Paliya kātɨ preserves the ear term."),
    5: (["d1655"], "DEDR 1655 gives Tamil kuṭi ‘house, town’ and Malayalam kuṭi ‘house, hut, family’; Paliya kuḍi ‘village’ fits this settlement family, distinct from the drinking verb in 1654."),
    6: (["d5393"], "DEDR 5393 explicitly includes Malayalam vīṭu ‘house’ within the viṭu/vīṭu article; Paliya vīḍɨ matches that house sense."),
    10: (["d4867", "d1298"], "Paliya muḷakakallɨ ‘mortar’ is analyzable as muḷaka ‘pepper/chilli’ plus kallɨ ‘stone’: DEDR 4867 gives Malayalam muḷaku ‘pepper’, and 1298 gives Malayalam kallu ‘stone’. The ordered compound names a grinding stone by use."),
    17: (["d3045"], "DEDR 3045 gives Tamil taṇṇīr/taṇṇi and Malayalam taṇṇīr ‘water’; Paliya ta ṇṇi preserves that water word despite the source's internal spacing."),
    18: (["d4753", "d5033"], "Paliya maḷamoḍam ‘cloud’ combines maḷa ‘rain’ with mōḍam ‘clouded sky’: DEDR 4753 has Malayalam mar̤a and Kannada mar̤e ‘rain’; 5033 has Tamil mōṭam and Kannada mōḍa ‘cloud/clouded sky’."),
    21: (["d3266"], "DEDR 3266 gives Tamil tī ‘fire’ and Malayalam tī ‘fire’; Paliya tīyɨ adds a final vowel to this fire term."),
    26: (["d3650"], "DEDR 3650 directly gives Tamil/Malayalam nāy ‘dog’ and wider Dravidian comparanda; Paliya nāyɨ has the same base."),
    28: (["d399", "d4198"], "Paliya āmpuḷapiḷḷa ‘boy’ contains the āmpuḷa male term already identified in this source with DEDR 399, followed by piḷḷa ‘child’, directly in DEDR 4198; both components are retained in order."),
    29: (["d4395", "d4198"], "Paliya pɛṇpuḷapiḷḷa ‘girl’ has peṇ ‘female’ from DEDR 4395 and piḷḷa ‘child’ from DEDR 4198; both are explicit Malayalam/Tamil comparanda in the full articles."),
    35: (["d503"], "DEDR 503 includes Tamil ir̤ukku ‘evil, disgrace, defect’; Paliya iḷukk ‘bad’ fits this negative-evaluation sense, though its lateral and lost final vowel remain visible."),
    36: (["d1670"], "DEDR 1670 gives Tamil kuṭṭam ‘smallness’, kuṭṭai ‘shortness’, and Malayalam kuṭu ‘small, narrow’; Paliya kuṭṭa ‘short’ belongs to this short/small family."),
    40: (["d3178"], "DEDR 3178 gives Malayalam tār̤a/tār̤ē ‘under, below, down’; Paliya tā ːḷe ‘below’ has that local form and meaning."),
    41: (["d5496"], "DEDR 5496(a) gives Tamil veḷ/veṇ and Malayalam veḷḷa ‘white’; Paliya vɛḷḷa is the same color word."),
    42: (["d3918", "d990"], "Paliya padinonnɨ ‘eleven’ is ten plus one: DEDR 3918 lists Tamil patiṉ- and Malayalam patin- ‘ten’ in 11–19, while 990 lists Malayalam onnu ‘one’. The constituent order is preserved."),
    43: (["d5151"], "DEDR 5151 lists Tamil yār and Malayalam yār/ār ‘who’; Paliya yār is a direct match within the interrogative article."),
    44: (["d5151"], "DEDR 5151 lists the e-/en- interrogative base and Malayalam entu ‘what’; Paliya enna ‘what’ preserves that interrogative family, with a different ending."),
    45: (["d5151"], "DEDR 5151 lists Malayalam eṅṅu ‘where’; Paliya eññge ‘where’ has the corresponding nasal interrogative base and local ending."),
    46: (["d5151"], "DEDR 5151 lists Malayalam eṅṅu ‘where’; this is a second distinct compiled Paliya eññge record for the same source item and is retained separately."),
    47: (["d5151"], "DEDR 5151 lists Malayalam e-ppōḷ ‘when’ and Tamil e-ppōtu; Paliya ɛppam ‘when’ shares the epp- temporal interrogative base."),
    48: (["d5151"], "DEDR 5151 gives Tamil ettaṉai and Malayalam ettira/etra ‘how many’; Paliya ɛttina has matching et(t)- interrogative shape and quantity meaning."),
    50: (["d410"], "DEDR 410(a) gives Tamil itu and Malayalam itu ‘this thing’; Paliya itɨ is the proximal demonstrative."),
    51: (["d1"], "DEDR 1 gives Tamil atu and Malayalam atu ‘that thing’; Paliya atɨ is the distal demonstrative."),
    52: (["d410"], "DEDR 410(a) includes the proximal i-/itu paradigm and plural referents. Paliya itɨkallam ‘these’ contains the proximal base with plural marking, an ordinary pronoun inflection."),
    53: (["d1"], "DEDR 1 includes the distal a-/atu paradigm and plural referents. Paliya ata kallam ‘those’ contains the distal base with plural marking; the source space is preserved."),
    54: (["d1"], "DEDR 1 includes the distal a-/atu paradigm and plural referents. This is a second distinct compiled ata kallam ‘those’ record for the same source item."),
    55: (["d990", "d4597"], "Paliya orepola ‘same’ transparently joins ore ‘one/same’ and pola ‘like’: DEDR 990(a) gives Malayalam oru/ōr ‘one, same’; 4597 gives Malayalam pōla/pōlē ‘like’."),
    57: (["d1124"], "DEDR 1124 gives Tamil kaṭi ‘bite’ and Malayalam kaṭikka ‘bite, chew’; Paliya kaḍicac is the inflected ‘he bit’ response."),
    58: (["d3825"], "DEDR 3825(a) gives Tamil paci ‘be hungry’ and Kannada pasi ‘hunger’; Paliya paśikkɨdɨ preserves the paci-/pasi- hunger stem, though the article's Malayalam form is payi/pai and contact cannot be excluded."),
    59: (["d3825"], "DEDR 3825(a) gives Tamil paci ‘be hungry’ and Kannada pasi ‘hunger’; Paliya paśicacɨ is the second tense response with the same hunger stem. Malayalam payi/pai differs in the article."),
    62: (["d707"], "DEDR 707 gives Malayalam uṟaṅṅuka ‘sleep’ and Tamil uṟaṅku; Paliya oraŋŋiyacɨ is an inflected sleep response with the same nasal stem."),
    63: (["d1524"], "DEDR 1524 gives Tamil kiṭa and Malayalam kiṭakka ‘lie, lie down’; Paliya kiḍandāc reflects the inflected lying verb."),
    65: (["d480"], "DEDR 480 gives Tamil iru ‘sit, remain’ and Malayalam irikkuka ‘sit’; Paliya irunnacɨ is the past sit response, retaining the ir- base."),
    66: (["d3098"], "DEDR 3098 gives Malayalam taruka ‘give’ with past tann- and Tamil taru with past tant-; Paliya ta nnacɨ ‘he gave’ matches the Malayalam past stem."),
    67: (["d811"], "DEDR 811 gives Tamil eri ‘burn’ and Malayalam eriyuka ‘burn’; Paliya ɛriyidɨ is its present response."),
    68: (["d811"], "DEDR 811 gives Tamil eri ‘burn’ and Malayalam eriyuka ‘burn’; Paliya ɛriññacɨ is the corresponding past response, preserved separately."),
    71: (["d2132"], "DEDR 2132 gives Tamil kol and Malayalam kolluka ‘kill’; Paliya kollata is the negative imperative response ‘don’t kill’, an ordinary verbal inflection."),
    74: (["d3582"], "DEDR 3582 gives Tamil naṭa ‘walk’ and Malayalam naṭakka; Paliya naḍandac retains the inflected walk stem."),
    75: (["d1041"], "DEDR 1041 gives Tamil ōṭu and Malayalam ōṭuka ‘run’; Paliya ōḍiden is the ‘he ran’ response."),
    76: (["d4572"], "DEDR 4572 gives Tamil pō and Malayalam pōka ‘go’; Paliya poyacɨ is the inflected ‘he went’ response in that family."),
    77: (["d5270"], "DEDR 5270 gives Tamil vā and Malayalam varuka ‘come’, with past vann-; Paliya vandacɨ is an inflected ‘he came’ form in this family."),
    78: (["d2855"], "DEDR 2855 gives Tamil col and Malayalam colluka ‘say, speak’; Paliya colliyac. is the ‘he spoke’ response."),
    79: (["d2017"], "DEDR 2017(a) gives Tamil kēḷ (past kēṭṭ-) and Malayalam kēḷkka ‘hear, listen’; Paliya kɛṭac. is the ‘he heard’ response."),
    80: (["d4091"], "DEDR 4091(a) gives Tamil pār ‘look’ and Malayalam pārkka ‘behold’; Paliya pārɨ is the direct look stem."),
    82: (["d5160"], "DEDR 5160 gives Malayalam ñān ‘I’ and Tamil yāṉ/nāṉ; Paliya ñān matches the Malayalam first-person singular form."),
    83: (["d3684"], "DEDR 3684 gives Malayalam nī ‘you’ (singular) and Tamil nī; Paliya ni is the informal singular pronoun."),
    84: (["d3688"], "DEDR 3688 gives Malayalam niṅṅaḷ and Kannada nīngaḷ ‘you’ plural, with honorific singular usage in Tamil; Paliya niŋgɛ is the formal second-person form of this plural/honorific family."),
    85: (["d1"], "DEDR 1 directly lists Malayalam avan ‘that man/he’ in the distal a- pronoun paradigm; Paliya avan matches it."),
    86: (["d1"], "DEDR 1 directly lists Malayalam avaḷ ‘that woman/she’; Paliya avaḷ matches this distal pronoun."),
    87: (["d1"], "DEDR 1 directly lists Malayalam avaḷ ‘that woman/she’. This is a second distinct compiled Paliya avaḷ record for the same source item."),
    88: (["d3647"], "DEDR 3647 gives Tamil nām with oblique nam(m)- and Malayalam nām ‘we, I and you’; Paliya namma is explicitly inclusive, matching this family."),
    89: (["d5154"], "DEDR 5154 gives Koḍagu naŋga and Malayalam ñaṅṅaḷ for exclusive ‘we’; Paliya naŋga is explicitly exclusive and belongs to this family."),
    90: (["d3688"], "DEDR 3688 gives Malayalam niṅṅaḷ ‘you’ plural; Paliya niŋŋa is the plural counterpart of record 84."),
    91: (["d1"], "DEDR 1 gives the distal a-/av- pronoun base, including plural persons; Paliya avaga ‘they’ is its plural form with local -ga marking."),
}

hold_reasons = {
    2: "The face word munḍźi lacks a secure matching DEDR article; DEDR 4889 mukam ‘face’ is only a semantic match.",
    4: "Nagarɨ ‘fingernail’ may be an Indo-Aryan nakha-type loan, but the immediate donor is unverified.",
    7: "Kurakuḍici seems to contain roof/thatch and hut material (DEDR 1904, 1655), but the segmentation and meaning of the whole form need independent confirmation.",
    8: "Vacapaḍi may combine an entrance word (DEDR 5354) with paḍi ‘threshold’ (3850), but the first constituent is truncated and the source does not establish segmentation.",
    9: "The first element of viḷakamar ‘broom’ has no verified matching article; do not infer a component from sound resemblance.",
    11: "Piḷḷakallɨ ‘pestle’ contains kallɨ ‘stone’, but piḷḷa is not securely established as a size term here; avoid a partial compound analysis.",
    12: "Kiḷḷi ‘knife’ does not match the knife word katti in DEDR 1204; kiḷḷu ‘pinch’ in 1589 is semantically inadequate.",
    13: "DEDR appendix 32 lists Kannada koḍali ‘axe’ but explicitly compares Indo-Aryan CDIAL 3244; borrowing direction and immediate donor remain unresolved.",
    14: "Motiram ‘ring’ is a likely mudrā-type loan; an immediate donor attestation is not verified.",
    15: "Sūriyan ‘sun’ is a likely sūrya-type loan; an immediate donor attestation is not verified.",
    16: "Candaran ‘moon’ is a likely candra-type loan; an immediate donor attestation is not verified.",
    19: "Arjunavillɨ ‘rainbow’ mixes an Arjuna name with a bow word; the donor and exact compound structure are unverified.",
    20: "Rōḍɨ ‘path’ is an English road-type loan, but the immediate donor in Paliya is unverified.",
    22: "Camal ‘ash’ resembles Tamil/Malayalam cāmpal in DEDR 2453, but the missing nasal-stop cluster and vowel difference need corroboration.",
    23: "Kota mbɨ ‘wheat’ appears loan-like; its immediate donor and original spacing are unresolved.",
    24: "Ta kkāḷi ‘tomato’ is a likely regional loan; no immediate donor attestation was verified.",
    25: "Kovīsɨ ‘cabbage’ is a likely recent loan; no immediate donor attestation was verified.",
    27: "Purusan ‘husband’ is a likely puruṣa-type loan; no immediate donor attestation was verified.",
    30: "Maddiyanam ‘noon’ is a likely madhyāhna-type loan; its immediate donor is unresolved.",
    31: "Sāyamkalam ‘evening/afternoon’ includes loan-looking sāyam/kālam elements; the immediate donor and segmentation are unresolved.",
    32: "Vāra m ‘week’ appears loan-like, with an internal source space; immediate donor unverified.",
    33: "Māsam ‘month’ is a likely māsa-type loan; immediate donor unverified.",
    34: "Varśam ‘year’ is a likely varṣa-type loan; immediate donor unverified.",
    37: "Tu ːram ‘far’ is likely a dūra-type loan, but immediate donor is unverified; this source item has a duplicate compiled record.",
    38: "Tu ːram ‘far’ is a second distinct compiled source record; likely dūra-type loan with unverified immediate donor.",
    39: "Lesa ‘light’ (in weight) has no securely identified DEDR family or immediate donor.",
    49: "Ennacayis ‘what kind?’ contains interrogative enna, but the second constituent is not identified; do not collapse the whole phrase onto DEDR 5151.",
    56: "Pūra ‘all’ is loan-like and lacks a verified immediate donor.",
    60: "Ta ṇṇi ta vikkatɨ is a multiword thirsty clause; the water element is known but the predicate's origin and segmentation are unresolved.",
    61: "Ta ṇṇi ta vicēy is a second multiword thirsty clause with a different predicate form; hold the whole record pending analysis.",
    64: "Ukkarɨ ‘sit’ resembles Tamil ukkāru, but that form is not securely covered by the shortlisted DEDR 1628 squat article.",
    69: "Cakapaḍatɨ ‘don’t die’ is a negative predicate; both the construction and the second element need analysis before ancestry assignment.",
    70: "Cattu poṇatɨ ‘he died’ is a two-word die-plus-go predicate; component status and auxiliary semantics need verification.",
    72: "Koṇḍacɨ ‘he killed’ resembles a take/go past rather than the expected Malayalam kill past; stem identification is unresolved.",
    73: "Parandiricɨ ‘it flew’ appears to contain fly plus an auxiliary; do not label the entire expression as one inherited stem.",
    81: "Pattā c ‘he looked’ may be an altered past of pār-, but the missing r and source spacing require corroboration.",
}

ids = {p for v, _ in families.values() for p in v}
articles = {}
with gzip.open(P.parents[2] / "data/dedr/entry-texts-audit.csv.gz", "rt") as f:
    for article in csv.DictReader(f):
        if article["Entry_ID"] in ids:
            articles[article["Entry_ID"]] = article
assert set(articles) == ids, sorted(ids - set(articles))

accepted = []
held = []
for n, record in enumerate(records, 1):
    if n in families:
        parents, evidence = families[n]
        citation = record["Source"] + ";" + ";".join(f"dedr[{p[1:]}]" for p in parents)
        item = {"record": record, "parent": parents[0], "kind": "component" if len(parents) > 1 else "reflex", "citation": citation, "evidence": evidence, "comparison": "Full DEDR article and source record checked; whole-article family identification.", "status": "pending-review"}
        if len(parents) > 1:
            item["components"] = parents
        accepted.append(item)
    else:
        assert n in hold_reasons, n
        held.append({"record": record, "candidates": [], "reason": hold_reasons[n]})

decisions = {"status": "pending-review", "language": "Paliya", "source": "varghese-mathew2015idukki", "accepted": accepted, "held": held}
(P / "proposed-decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
(P / "research-holds.json").write_text(json.dumps(held, ensure_ascii=False, indent=2) + "\n")
(P / "primary-evidence.json").write_text(json.dumps(articles, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"records": len(records), "proposed": len(accepted), "assignmentRows": sum(len(x.get("components", [x["parent"]])) for x in accepted), "held": len(held), "articles": len(articles)}))

"""Assemble the review-only Betta Kurumba research decisions."""
import collections
import json
from pathlib import Path

P = Path(__file__).resolve().parent
records = json.loads((P / "inventory.json").read_text())
by_form = {r["Form"]: r for r in records}
assert len(by_form) == len(records) == 66
articles = json.loads((P / "primary-evidence.json").read_text())
accepted = []


def propose(forms, parents, evidence, caveat="Full DEDR articles checked."):
    if isinstance(forms, str):
        forms = [forms]
    if isinstance(parents, str):
        parents = [parents]
    assert all(p in articles for p in parents), parents
    for form in forms:
        record = by_form[form]
        accepted.append({
            "record": record,
            "parent": parents[0],
            "kind": "component" if len(parents) > 1 else "reflex",
            "citation": record["Source"] + ";dedr[" + ",".join(p[1:] for p in parents) + "]",
            "evidence": evidence,
            "comparison": caveat,
            "status": "pending-review",
            **({"components": parents} if len(parents) > 1 else {}),
        })


propose("nāvə", "d5160",
        "DEDR 5160 has Tamil nāṉ, Kannada nān and Kodagu nā as first-singular pronouns. The independent SIL Betta Kurumba list also writes navvu 'I'; the final -və/-vvu is a local form not explained by the DEDR paradigm.",
        "Do not normalize this to Kannada nāvu 'we'; both Betta sources gloss it 'I'.")
propose("mertōlə", ["d4711", "d3559"],
        "The source's two-part bark word matches mar-/mer- 'tree' in DEDR 4711 plus tōl/toli 'skin, bark' in DEDR 3559; Malayalam marattoli is in the same DravLex cognacy set, and SIL Betta has mereṭɔlu.",
        "Ordered tree + bark/skin components, not one inherited bark stem.")
propose("dappu", "d3070",
        "DEDR 3070 gives Kannada dappa and Tulu dappa 'thick, stout'; SIL Betta independently has ḍappuḍu 'big'. The survey's broad 'big' gloss is a plausible stout/large sense.")
propose("kayri", "d1124",
        "DEDR 1124 directly gives Kota kayṛ- 'bite' alongside Tamil kaṭi and Kannada kaḍi; DravLex Betta kayri has the same y-r sequence and meaning.")
propose("kuyndigi", "d1693",
        "DEDR 1693 gives Tulu guṇḍigè 'breast, heart' and Kannada guṇḍige 'heart'; Betta kuyndigi shares the -ndigi stem and the breast gloss.",
        "Initial k/g and glide y differ; this is a form-and-meaning comparison, not a direct DEDR Betta attestation.")
propose("ben", "d5517",
        "DEDR 5517 has Kannada ben 'heat' beside beyyu (bend-) 'burn', and Kodagu be·y- 'be warmed, burned'. DravLex Betta ben is glossed 'burn' and is in the survey's burn-A set.",
        "The source does not specify whether ben is a verb or heat noun; retain that uncertainty.")
propose("kettu", "d1207",
        "DEDR 1207 lists Kannada kattu 'begin to burn' and Kota kat- 'burn'; an earlier Betta survey kattu is already proposed to this article. The present kettu differs in its vowel but shares k-tt and sense.")
propose("kīyəggərə", ["d2023", "d561"],
        "The form segments as kīyə 'hand' (DEDR 2023, cf. already proposed Betta kīya) plus -ggərə 'nail, claw' (DEDR 561, cf. SIL Betta aguru and Kota kaiyōru in the cognacy set).",
        "Ordered hand + nail components; the geminate boundary and intervening vowels remain source-specific.")
propose("bu", "d5270",
        "DEDR 5270 gives Kannada bar/bā, Kodagu bar/ba· and other reduced imperatives of 'come'; Betta bār and ban are already proposed in this same article. SIL Betta has ba 'come', supporting a short stem beside DravLex bu.",
        "The survey's u vowel is not independently explained; this is a paradigm-family link.")
propose("nāyə", "d3650",
        "DEDR 3650 directly groups Tamil nāy, Kannada nāyi and Kodagu nāy 'dog'; SIL Betta has nãje and DravLex nāyə, retaining nāy-.")
propose("kuyri", "d1654",
        "DEDR 1654 directly gives Kota kuṛy- 'drink' beside Tamil kuṭi and Kannada kuḍi. DravLex Betta kuyri matches the rhotic Kota stem, and Betta kuḍi was already proposed to this article.")
propose("ticca", "d3266",
        "DEDR 3266 gives Tamil tī 'fire' and Kota ti·y-/ti·c- 'be singed/singe'; DravLex groups Betta ticca with its own tissə and the Tamil/Malayalam fire forms. SIL Betta has tɪtʃe.")
propose("nall", "d3610",
        "DEDR 3610 gives Tamil nalla, Malayalam nalla and Kannada nalla/nal 'good'; the DravLex set independently contains Tamil and Malayalam nalla. SIL Betta has nʌlɖu 'good'.")
propose(["passlāngi", "paccəlāngi"], "d3821",
        "DEDR 3821 lists Kannada pacce/pacca, Kodagu pacce and Tulu pacca 'green'; Betta pass-/paccə- are survey variants with the same adjectival -lāngi ending. SIL Betta has pʌtʃtʃɪ.",
        "Only the colour stem is compared; -lāngi remains Betta morphology.")
propose("kə̄ḷ", "d2017",
        "DEDR 2017 gives Tamil kēḷ and Kannada kēḷu 'hear'; the survey's kə̄ḷ matches their consonantal root and the DravLex cross-language set.")
propose("manngālə", ["d4645", "d1479"],
        "DEDR 4645 directly has Kannada maṇikāl 'knee' from the bend family, with kāl 'leg, foot' in DEDR 1479. Survey manngālə and SIL Betta mʌŋgaɭe match the fused maṇi-kāl construction.",
        "Ordered bent/joint + leg components; DEDR 4645 already includes the lexicalized whole.")
propose("gott uṇḍi", ["d1847", "d697"],
        "DEDR 1847 explicitly cites Kodagu gott(ï) uṇḍï 'it is known', matching the Betta phrase; its second word is uṇḍï 'exists' from DEDR 697 (Kodagu uḷḷ-).",
        "Ordered knowledge + existential auxiliary components; preserve both source words.")
propose("gottāg", ["d1847", "d333"],
        "DEDR 1847 explicitly cites Kodagu gott(ï) a·g- 'to become known'; the second verb belongs to DEDR 333 'become'. Betta gottāg contracts the same two elements.",
        "Ordered knowledge + become components despite one-word survey spelling.")
propose("yali", "d497",
        "DEDR 497 groups Tamil ilai, Malayalam ila, Kannada ele and Kota el 'leaf'; the survey set includes these and Betta yali, while SIL Betta has jele.")
propose("bāḍ", "d3852",
        "DEDR 3852 gives Kannada paḍu and Kota paṛ- 'lie down'; Betta bāḍ 'lie' shares the voiced dental/retroflex stem with the survey's longer bāṭṇu īr expression.")
propose("bāṭṇu īr", ["d3852", "d480"],
        "The source response contains bāṭṇu 'lie down' (DEDR 3852 paḍu/paṭṭ-) followed by īr 'be, remain' (DEDR 480 iru/ir).",
        "Ordered lie + auxiliary components; retain the complete two-word elicitation response.")
propose("bəṭṭə", "d5474",
        "DEDR 5474 directly gives Kannada and Kodagu beṭṭa 'hill, mountain', as well as Kota beṭ; DravLex Betta bəṭṭə and Kannada beṭṭa are grouped together.")
propose("pyārə", "d4410",
        "DEDR 4410 groups Tamil peyar/pēr, Malayalam pēr, Kota pēr and Kannada pesar(u) 'name'; Betta pyārə keeps p-y-r and belongs to the same DravLex comparative set. SIL Betta has pærə.")
propose("ila", "d2559",
        "DEDR 2559 gives Tamil illai, Malayalam illa and Kannada illa 'not, does not exist'; Betta ila has the same negative stem and the survey set groups the southern forms.")
propose("ala", "d234",
        "DEDR 234 separately gives Tamil/Malayalam/Kannada alla 'is not so-and-so'; Betta ala matches Kannada alla in the source set. This predicative negative is kept distinct from ila 'not/existential' under DEDR 2559.")
propose("vandə", "d990",
        "DEDR 990 includes Tamil oṉṟu, Malayalam onnu and Kannada ondu 'one'; Betta vandə matches the Kannada nasal-stop form and its DravLex cross-language set. SIL Betta has woṇḍu.")
propose(["āḷə", "āḷu"], "d399",
        "DEDR 399 directly has Tamil āḷ 'man/person', Malayalam āḷ 'person', Kannada āḷ and Tulu āḷů 'person'; the two Betta vowels are preserved as separate source records.")
propose("byārə", "d5535",
        "DEDR 5535 gives Tamil/Malayalam vēr, Kannada bēr(u) and Kodagu be·rï 'root'; Betta byārə matches the b-initial South Dravidian forms, and SIL Betta has berɔ.")
propose(["bissə", "biccə"], "d5401",
        "DEDR 5401 includes Tamil vittu/viccu 'seed', Kannada bittu and Kota vit; Betta bissə/biccə retain the alternate s/c stems, and SIL Betta records bɪdʒe.",
        "DEDR cautions that some Gondi, Kuwi and Malto seed forms may be Indo-Aryan loans; this Betta proposal rests on the South Dravidian forms.")
propose("ə̄ḷ", "d851",
        "DEDR 851 gives Kannada ēr̤u (edd-) 'stand up, rise' and Kodagu ë·ḷ- (ëdd-) 'get up'; Betta ə̄ḷ matches the present stem.")
propose("əddənu īr", ["d851", "d480"],
        "DEDR 851 gives Kodagu ë·ḷ- (ëdd-) 'get up' and Kannada ēr̤u (edd-) 'stand'; Betta əddənu is the matching past/participial stem followed by īr 'be, remain' (DEDR 480).",
        "Ordered get up + be components; the source's two words are preserved.")
propose("pottə", "d4559",
        "DEDR 4559 groups Kannada pottu 'time, sun', Telugu poddu 'sun/day' and Kota poṛt 'sun'; Betta pottə matches this solar/time family, and SIL Betta has poṭṭe.")
propose("nīnḍźu", "d3687",
        "DEDR 3687 gives Tamil nīntu, Malayalam nīntuka, Kota ni·nj- and Tulu nīnduni 'swim'; Betta nīnḍźu matches their nīn- stem, and SIL Betta has nɪɳdʒu.")
propose("adə", "d1",
        "DEDR 1 lists Kannada adu, Kodagu adï and other a-/ad- distal demonstratives; Betta adə 'that' follows the same paradigm, and the survey set contains these comparanda.")
propose("nīyə", "d3684",
        "DEDR 3684 groups Tamil nī, Kannada nīn(u) and Kodagu ni·nï/ni· 'thou'; Betta nīyə and SIL nɪje retain the nī- second-person stem.")
propose("əddu", "d474",
        "DEDR 474 gives Tamil iraṇṭu, Kannada eraḍu and Toda eḍ 'two'; Betta əddu has the reduced -dd- stem, and SIL Betta records ɨɖɖu for both persons and stones.")
propose("nīrə", "d3690",
        "DEDR 3690 directly has Tamil nīr, Kannada nīru and Toda nīr 'water'; Betta nīrə matches them and SIL Betta has nɪru.")
propose("ayindi", "d5151",
        "DEDR 5151 contains the Dravidian interrogative e-/en-/y- family: Malayalam entu, Kannada ēnu and Kodagu ennï 'what'. Betta ayindi and SIL jɪɳdʒɪɖɪ preserve a nasal-stop interrogative form.")
propose("dārə", "d5151",
        "DEDR 5151 includes Kodagu da·rï 'who' and Tamil yār/ār; Betta dārə matches the Kodagu form closely, and SIL Betta has ɖaru.")
propose("manjəlāngi", "d4635",
        "DEDR 4635 gives Tamil mañcaḷ 'turmeric, yellow', Kodagu mañja and Tulu mañjalu 'yellow'; Betta manjə- and SIL mʌɳdʒʌle share that colour stem.",
        "The -lāngi adjective ending is left as Betta morphology.")

holds = {
    "pikki": "The survey flags a loan and compares Kannada pakṣi/pakki, Tulu pakki and other regional forms, but no immediate donor for Betta pikki has been established. Do not jump to Sanskrit pakṣi.",
    "byāḷ": "The survey groups it only with Betta ben 'burn'; DEDR 5517 directly supports ben via Kannada ben 'heat', but byāḷ lacks a demonstrated sound/morphological path to that article.",
    "səyliyə": "DEDR 2408 has Kannada/Tulu saḷi/sali 'cold', but the Betta səjlijə sequence and adjectival ending lack a demonstrated correspondence; SIL Betta instead records sɪʈʈə.",
    "mardōg": "The survey flags a loan and compares Malayalam marikkuka 'die', but mardōg has unexplained -dōg and the immediate donor is not secure.",
    "ayngənu ipədə": "This is a two-word dry response with no parsed components or same-stem comparative family; an inherited edge would flatten its structure.",
    "būmi": "The survey flags a widespread loan and groups regional bhūmi forms; the immediate donor into Betta has not been identified.",
    "muyrgi": "SIL Betta independently has murɪggɪ 'egg', but neither survey nor DEDR identifies a secure root or immediate donor.",
    "tannivi": "SIL Betta ʈenevɪ may be related to this 'fat' form, but the stem has no securely identified DEDR parent.",
    "teriya": "The source distinguishes this second 'fat' form from tannivi; no same-stem DEDR comparison was verified.",
    "tiviya": "DEDR 3393 has Tamil tūvi and Kannada tippur̤ 'feather', and SIL Betta has ʈɪvɪɣe; the Betta tivi- vowel/stem relationship remains unproved.",
    "fullə": "The survey gives a singleton full-F set; SIL Betta uses pərɨʈʈu for 'full', and no matching DEDR family was verified.",
    "tākərə": "The survey has a singleton hair-J set; SIL Betta uses kuru for 'hair', and no sound-matched DEDR article was verified.",
    "tədlə": "SIL Betta has jɪrɪ 'liver' rather than tədlə; DEDR 546 īral is not a secure match to the survey form.",
    "kayriyə": "The singleton many-K form resembles Betta kayri 'bite' but has different meaning; the source does not parse its morphology and SIL uses næruʈʈu.",
    "kppmāḷi": "The source phonemic kp:ma:ɭi 'neck' is unusual; SIL has kɔmʌɭe. A possible transcription issue and root have not been resolved.",
    "pəyymən": "SIL Betta corroborates pʌj'mun 'new', but the survey singleton has no secure inherited or donor parent.",
    "ədlə": "SIL Betta has ɨrdɪluŋə 'night'; DEDR 235 al 'night' is too short and differs in medial consonants to justify this survey form.",
    "tissəgaṇḍāngi": "The first element may relate to Betta tissə 'fire', but the source's red colour phrase is not securely segmented or compared.",
    "baṭṭi": "The survey compares Kodagu bāte and Telugu bāṭa 'road', but DEDR contains unrelated baṭṭi basket/body entries; a possible contact source is unresolved.",
    "bəynsakrangūdə": "The long round-E singleton is unsegmented; no sound-supported DEDR or immediate donor parent was identified.",
    "nayr": "The survey groups it with Kannada nōḍu and Kota nōṭ- 'see' (DEDR 3794), but Betta nayr differs sharply in vowel and rhotic; SIL Betta uses koḷʌl. Hold rather than assign by gloss/set alone.",
    "naṇgə": "The survey glosses 'we' without clusivity. DEDR 5154 groups exclusive nāṅkaḷ/naŋga, while DEDR 3647 is inclusive; source-specific Betta morphology needs adjudication.",
    "yaṇgə": "SIL Betta independently has jeŋgʌ explicitly 'we (inclusive)', whereas the DravLex comparative set includes exclusive Tamil nāṅkaḷ under DEDR 5154. DEDR 3647 is an inclusive alternative; hold the parent choice.",
}
held = [
    {"record": r, "candidates": [], "reason": holds[r["Form"]]}
    for r in records if r["Form"] in holds
]
assert collections.Counter(r["ID"] for r in records) == collections.Counter(
    x["record"]["ID"] for x in accepted + held
)
assert len(accepted) + len(held) == 66
decisions = {"status": "pending-review", "accepted": accepted, "held": held}
(P / "proposed-decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
print({"proposedRecords": len(accepted), "assignmentRows": sum(len(x.get("components", [x["parent"]])) for x in accepted), "held": len(held)})

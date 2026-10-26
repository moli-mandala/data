"""Literal, source-attributed first reading of Nair's full Adiya chapter.

This is deliberately upstream of any 15-column Jambu proposal. It keeps
comparison columns, whole sentences, slash alternatives, and documentary
mentions separate. Source spelling is not treated as phonemic IPA.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PDF = HERE.parents[5] / "tmp/pdfs/tribal-languages-kerala-nair2013.pdf"
census = [json.loads(s) for s in (HERE / "scope/physical-unit-census.jsonl").read_text().splitlines()]
expected = {r["unit_key"] for r in census}
units = []

def w(form, gloss="", language="Adiya", role="target", note=""):
    return {"form": form, "gloss": gloss, "language_label": language,
            "role": role, "note": note}

def add(page, section, witnesses=(), claim="", note=""):
    key = f"nair2013adiya:p{page}:{section}"
    assert key in expected and not any(x["unit_key"] == key for x in units), key
    units.append({"unit_key": key, "printed_page": page, "pdf_page": page,
                  "witnesses": list(witnesses), "source_claim": claim,
                  "note": note,
                  "review": "first reading; exact font marks and output interpretation pending independent audit"})

def table(page, section, langs, rows, claim=""):
    witnesses = []
    for row_no, (forms, gloss) in enumerate(rows, 1):
        assert len(forms) == len(langs)
        for column, (language, form) in enumerate(zip(langs, forms), 1):
            witnesses.append(w(form, gloss, language,
                               "target" if language == "Adiya" else "comparison control",
                               f"printed table row {row_no}, column {column}"))
    add(page, section, witnesses, claim)

# The three names occur in ethnographic prose. Even raavuleeru is a self-name
# claim, not independently glossed as a lexical item of the target language.
add(4, "community-names", [
    w("Adiya", language="documentary", role="community name"),
    w("aṭiya", language="documentary", role="parenthetical community name"),
    w("raavuleeru", language="documentary", role="reported self-name"),
], "Adiya people in Wayanad call themselves raavuleeru; attribution as a spoken target word is not explicitly established.")
table(4, "vowel-a-length-table", ["Malayalam", "Adiya"], [
    (["akattŭ", "akaattu"], "inside"),
    (["arakkŭ", "araakku"], "lac"),
    (["ulakka", "ulaakka"], "pestle"),
], "Malayalam short vowels correspond to long vowels in the Adiya examples.")
table(4, "vowel-a-to-e-table", ["Malayalam", "Adiya"], [
    (["akalam", "akeela"], "distance"),
    (["pakalŭ", "pakeelu"], "day-time"),
    (["kaṭatlŭ", "ka ṭeelu"], "sea"),
], "Source prints a spaced Adiya sea form in the extract; retain exact visible spacing as a typography hold until the original font is checked.")
table(4, "final-vowel-table", ["Malayalam", "Adiya", "Kannada"], [
    (["pittam", "pitte", "pitta"], "jaundice"),
    (["iṣṭam", "iṭṭe", "iṣṭa"], "liking"),
    (["kaakka", "kaakke", "kaage"], "crow"),
    (["aana", "aane", "aane"], "elephant"),
], "Perhaps due to Kannada influence: this is the author's comparison, not a borrowing edge.")
table(4, "rounded-u-table", ["Malayalam", "Adiya"], [
    (["kaṇṇŭ", "kaṇṇu"], "eye"),
    (["mukkŭu", "mukku"], "nose"),
    (["cuulŭ", "cuulu"], "broom"),
], "Source says Adiya final -u is rounded unlike Malayalam final -u; Malayalam nose spelling is unusual and retained literally.")
table(4, "l-to-v-table", ["Malayalam", "Adiya"], [
    (["kal̲uttŭ", "kavuttu"], "neck"),
    (["eel̲u", "eevu"], "seven"),
    (["aal̲am", "aava"], "depth"),
], "Author describes Malayalam l corresponding to Adiya v; source visibly underlines the comparison l, retained as l plus U+0332. Do not infer etymological graph links.")
table(4, "v-to-b-table", ["Malayalam", "Adiya"], [
    (["va ṭi", "ba ṭi"], "stick"),
    (["vala", "bala"], "bangle"),
    (["va ṭṭi", "ba ṭṭi"], "basket"),
], "Author describes Malayalam v corresponding to Adiya b. The source visibly spaces the t-underdot segment in four cells; whether this is a typesetting split rather than a word boundary is held as source-typography uncertainty.")

add(5, "plural-morphology", [
    w("-a", note="Source calls this the productive Adiya plural suffix."),
    w("ava", language="Malayalam", role="comparison control"),
    w("iva", language="Malayalam", role="comparison control"),
    w("-kaḷ", language="Malayalam", role="comparison control"),
], "The Adiya plural suffix -a contrasts with Malayalam -kaḷ, while Malayalam ava/iva are cited as rare -a comparands.")
table(5, "plural-table", ["Malayalam", "Adiya"], [
    (["puccakaḷ", "pucceya"], "cats"),
    (["piḷḷakaḷ", "puḷḷeya"], "children"),
    (["paṇiyattikaḷ", "peṇiyattiya"], "Paniya Women"),
], "Three printed Malayalam–Adiya plural comparisons.")

# A printed paradigm cell, rather than an inferred English lexical definition,
# supplies each grammatical analysis. Repeated era/ayira cells remain distinct.
pronouns = [
    ("n̲aanu", "I", "I nominative singular; source underlines initial n"),
    ("n̲aanka", "we", "I nominative plural exclusive; source underlines initial n"),
    ("en-", "", "I oblique singular stem"),
    ("n̲aamu", "we", "I oblique plural inclusive; source underlines initial n"),
    ("nii", "you", "II nominative singular, first slash option"),
    ("niiyu", "you", "II nominative singular, second slash option"),
    ("iñcu", "", "II oblique singular"),
    ("ña-", "", "II oblique singular, separately printed"),
    ("iññu", "", "II oblique singular, separately printed"),
    ("n̲inka", "you", "II plural; source underlines initial n"),
    ("ooḷu", "she", "III feminine singular"),
    ("avooḷu", "she", "III feminine singular"),
    ("avo", "she", "III feminine singular"),
    ("era", "", "III feminine plural proximal cell"),
    ("ayira", "", "III feminine plural distal cell"),
    ("eenu", "he/it", "III non-feminine singular"),
    ("itu", "he/it", "III non-feminine singular"),
    ("ayinu", "he/it", "III non-feminine singular"),
    ("era", "", "III non-feminine plural proximal cell"),
    ("ayira", "", "III non-feminine plural distal cell"),
    ("itu", "", "honorific singular, first row"),
    ("era", "", "honorific plural, first row"),
    ("atu", "", "honorific singular, second row"),
    ("ayira", "", "honorific plural, second row"),
]
add(5, "pronoun-paradigm", [w(f, g, note=label) for f, g, label in pronouns],
    "Person, oblique, gender, proximity, inclusion and honorific labels follow printed table positions; zero English cells remain blank rather than inferred.")
add(5, "first-person-variation", [
    w("naanu", "I", note="un-underlined prose alternative, separately printed before the example"),
    w("naa", "I", note="second prose alternative, explicitly said to be in free variation with naanu"),
    w("naanu poyuḷey", "I went", note="first printed slash alternative expanded as a whole expression"),
    w("naa poyuḷey", "I went", note="second printed slash alternative expanded as a whole expression"),
], "Source explicitly calls the separately printed pronouns naanu and naa free variants, then prints a slash-group whole expression. Both occurrence types are retained.")
add(5, "first-person-oblique", [
    w("en-", note="Repeated oblique base in prose, already appears in pronoun table."),
    w("enṛa kuḷḷili uppuḷa", "there is salt in my hut"),
    w("enaakku beeṇṭa", "I don't want"),
], "The case suffixes are added to en-; two printed whole clauses follow.")
add(5, "second-person-distribution", [
    w("nii", note="nominative singular, stated common option"),
    w("niiyu", note="nominative singular, stated alternative"),
    w("innu", note="before accusative"),
    w("na", note="before dative"),
    w("iñcu", note="other oblique environments"),
    w("niinŋka", note="plural and honorific singular; retain extra n in this prose occurrence"),
    w("okka", "all", note="postposition used for plural; source gloss in prose"),
], "Prose distribution repeats and elaborates table forms; source writes niinŋka here, niiŋka in the later address clause, and n̲inka in the table. Preserve each printed witness.")
add(5, "all-of-you-expression", [w("niŋkaḷokkaayi mubari", "all of you come")])
add(5, "kin-address-pronouns", [
    w("niiŋka", note="address to kin through marriage older than speaker"),
    w("nii", note="address to all other kin"),
], "Address usage is source-described, not a distinct lect or speaker attestation.")

add(6, "third-person-gender-examples", [
    w("eenu akkey", "this –he (is a) boy"),
    w("ivan aaṇkuṭṭi", language="Malayalam", role="parenthetical comparison control"),
    w("eenu evuttu", "this (is a) letter"),
    w("itŭ el̲uttŭ", language="Malayalam", role="parenthetical comparison control",
      note="source visibly underlines l in the parenthetical control"),
    w("ayinu cikkey", "that –he (is a) boy"),
    w("avan aaṇkuṭṭi", language="Malayalam", role="parenthetical comparison control"),
    w("ayinu evuttu", "that (is a) letter"),
    w("atu eluttu", language="Malayalam", role="parenthetical comparison control"),
    w("ooḷu", "this –she"),
    w("ivaḷ", language="Malayalam", role="parenthetical comparison control"),
    w("avooḷu", "that –she"),
    w("avaḷ", language="Malayalam", role="parenthetical comparison control"),
], "Source describes Adiya third-person singular as feminine versus non-feminine. Preserve all six attributed Malayalam parentheses separately.")
add(6, "elder-kin-demonstratives", [
    w("atu", note="standalone prose demonstrative; used respectfully for older kin, not neuter"),
    w("itu", note="standalone prose demonstrative; used respectfully for older kin, not neuter"),
    w("atu enṛa appey", "that (he is) my father"),
    w("atu enṛa ammey", "that (she is) my mother"),
], "atu/itu are used respectfully for older kin, not as ordinary neuter pronouns; do not infer an unattested phrase with itu.")
add(6, "younger-kin-demonstratives", [
    w("avooḷu enṛa magaḷu", "she (is) my daughter"),
    w("ayinu enṛa magenu", "he (is) my son"),
], "Source contrasts these younger-kin expressions with respectful atu/itu.")
add(6, "three-tense-examples", [
    w("n̲aanu kaṇṭey", "I saw", note="past example; source underlines initial n"),
    w("n̲aanu kaaṇinṛey", "I am seeing", note="present example; source underlines initial n; r lower mark follows source text glyph"),
    w("n̲aanu kaaṇuvey", "I will see", note="future example; source underlines initial n"),
], "Three complete subject–verb expressions; the source gives no separate isolated tense suffix forms here.")
add(6, "personal-termination-examples", [
    w("n̲aanu bandey", "I came", note="source underlines initial n"),
    w("n̲aaŋga barivoom", "we will come", note="source underlines initial n"),
    w("nii eppoo ban̲ṛe?", "when did you (Sg.) come?",
      note="source underlines n and prints separate lower mark under r"),
    w("niiŋka eppoo bandiri?", "when did you (pl) come?"),
    w("peṇṇu banta", "the girl came"),
    w("ayira bantaaru", "they came"),
], "Complete person-number examples; no isolated personal endings are printed in this passage.")
add(6, "addressee-sex-markers", [
    w("-va", note="suffix for addressing a boy"),
    w("-ma", note="suffix for addressing a girl"),
    w("iŋku baattava", "come here", note="to a boy"),
    w("iŋku baattama", "come here", note="to a girl"),
], "Source says -va/-ma denote sex of the addressee; this is an address contrast, not speaker gender.")
add(6, "affinal-address-mi", [
    w("-mi", note="address to kin through marriage"),
    w("appaa koṭuu", "father, give"),
    w("baava koṭumi", "brother-in-law, give"),
], "-mi is described for verbs addressed to relations through marriage.")

add(7, "affinal-address-i", [
    w("-i", note="source says similar affinal-address usage to preceding -mi"),
    w("appey nii iŋku baa", "father come here"),
    w("baavey niiŋka barii", "brother-in-law come"),
])
add(7, "negative-kaani", [
    w("-kaaṇi", note="commonly used target negator; source contrasts absence in Malayalam/Tamil"),
    w("ii kuḷḷi aarumu kaaṇi", "there is nobody in this hut"),
    w("enaakku katee gottu kaaṇi", "I don't know the story"),
    w("aakki baruva kaaṇi", "elder sister has not come"),
    w("naanonnum paṭeeppa kaaṇi", "I have not learned anything"),
], "The negative follows a verbal participle; source says tense is understood contextually, not overtly marked.")
add(7, "negative-a", [
    w("-a", note="Adiya negative, source also compares classical Tamil and Malayalam"),
    w("aa cikkeey oru paṇimu ceyyaa", "that boy will not do any work"),
    w("naanu koṭaa", "I will not give"),
    w("enaakku par̲r̲aa", "will not suit me",
      note="source draws one continuous underline under both r letters"),
], "Tamil/Malayalam carry a source comparison claim, not printed lexical control forms in this paragraph.")
add(7, "archaic-ula", [
    w("uḷa", note="source calls this archaic verb, common to singular/plural"),
    w("saala kuḷḷu uḷa", "has many houses"),
    w("ayinu uṭe uḷa", "he is here"),
    w("enaakku muun̲ru puḷḷe uḷa", "I have three children",
      note="source underlines n in muunru"),
], "Old Malayalam is a comparison claim without a separately printed old-Malayalam witness.")

kin = [
    ("appa", "father", "first slash alternative"),
    ("appan", "father", "second slash alternative"),
    ("appey", "father", "third slash alternative"),
    ("amma", "mother", "term of reference only"),
    ("avva", "mother", "term of address"),
    ("accappan", "father's father; mother's father", "first slash alternative; also extended to father's mother's sister's husband"),
    ("accappey", "father's father; mother's father", "second slash alternative; same extended relation"),
    ("ittiyamma", "mother's mother; father's mother", "first slash alternative"),
    ("ittiyammey", "mother's mother; father's mother", "second slash alternative"),
    ("accera", "men of the next ascending generation; dead ancestors", "source gives both senses"),
    ("ittiri", "women of the next ascending generation", "female counterpart of accera"),
    ("peerappan", "father's elder brother", "first slash alternative"),
    ("peerappey", "father's elder brother", "second slash alternative"),
    ("peeramma", "father's elder brother's wife", "first slash alternative"),
    ("peeramme", "father's elder brother's wife", "second slash alternative"),
    ("kuḷiyappan", "father's younger brother; mother's younger sister's husband", "first slash alternative"),
    ("kuḷiyappey", "father's younger brother; mother's younger sister's husband", "second slash alternative"),
    ("cittappan", "father's younger brother; mother's younger sister's husband", "alternative term"),
    ("cittappey", "father's younger brother; mother's younger sister's husband", "alternative term"),
    ("aṇṇan", "elder brother; elder cross/parallel cousin", "first slash alternative"),
    ("aṇṇey", "elder brother; elder cross/parallel cousin", "second slash alternative"),
    ("akki", "elder brother's wife; elder sister; elder cross/parallel cousin", "reference term and broader use"),
    ("akkiya", "elder sister", "address term; source extension to elder cousins"),
    ("ileeyey", "younger brother; younger cross/parallel cousin", "source extension"),
    ("ileevo", "younger sister; younger cross/parallel cousin", "source extension"),
    ("ṛaaley", "husband", ""),
    ("ṛaatti", "wife", ""),
    ("kuḷilavaḷu", "wife", "husband's reference to his wife"),
]
add(7, "kinship-terms", [w(f, g, note=n) for f, g, n in kin],
    "All slash forms kept distinct; source-specific extended relationships and address/reference roles retained instead of reduced to a single English headword.")
add(7, "bibliography-heading", claim="The Adiya chapter bibliography begins at the bottom of p7 and continues on p8.")
add(8, "bibliography-continued", claim=(
    "Nair lists four unpublished 1978 University of Kerala M.A. theses: "
    "D. Sankarankutty, Phonology of the Adiya Language; G. S. Syama, Verb Morphology of the Adiya Language; "
    "S. Sarala, A Fragment of Syntax of Adiya Language; Esther Daisy Davidson, Noun Morphology of Adiya. "
    "They are reported authorities, not independently consulted lexical witnesses."
))

assert {u["unit_key"] for u in units} == expected
out = HERE / "first-reading.jsonl"
out.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
(HERE / "first-reading-status.json").write_text(json.dumps({
    "status": "complete provisional source-first reading; independent glyph and metadata review pending",
    "physical_units_accounted": len(units),
    "physical_units_total": len(expected),
    "source_witnesses_accounted": sum(len(u["witnesses"]) for u in units),
    "first_reading_sha256": sha(out),
    "source_pdf_sha256": sha(PDF),
    "canonical_changes": False,
}, ensure_ascii=False, indent=2) + "\n")
print(len(units), "blocks /", sum(len(u["witnesses"]) for u in units), "witnesses")

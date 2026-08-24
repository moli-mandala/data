#!/usr/bin/env python3
"""Install Buddruss's 1979 Grangali supplement to the Atlas questionnaire.

The copyrighted Stanford ILL scan is not redistributed.  ``RAW`` below is the checked-in,
manually collated transcription layer.  Every record was checked against the 300 dpi page render;
OCR was used only to navigate the scan.  One row represents one emitted form or one explicit
non-attestation; repeated Atlas numbers represent genuine variants or distinct elicited senses.
Run from ``data/``; ``--pdf`` optionally verifies the original scan byte-for-byte.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path


SOURCE_ID = "buddruss-grangali1979"
SNAPSHOT_DATE = "2026-08-19"
COLLATION_DATE = "2026-08-20"
CORRECTED_FORM_COUNT = 104
PDF_SHA256 = "175737c74b49630badf88d62f7dca3b199884365556f8345b7f9aebe56e0525a"
PDF_PAGES = 23
ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260819-buddruss-grangali.csv"
AUDIT_OUTPUT = RAW_DIR / "20260819-buddruss-grangali-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260819-buddruss-grangali-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260819-buddruss-grangali-manifest.json"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Atlas_Numbers", "PDF_Page", "Printed_Page", "Raw_Form",
    "Raw_Gloss", "Final_Status", "Final_Form", "Final_Parameter_ID", "Emitted_Key",
    "Resolution", "Review", "Material_Error", "Source", "Record_SHA256",
]

# atlas | printed page | original | English gloss | tags | CDIAL id | source etymology/note
RAW = r"""
1|24|āẓoṛī́|apricot|noun|||
2|24|atā́|hungry|adjective|||
3|24|bugunṓ|lamb|noun|||
4|25|cistalə́|sour|adjective|||source explicitly gives the same form for sour and bitter
5|25|radilā́|sharp|adjective|||
6|25|kacö́r|armpit|noun|||
7|25|ca-|go|verb intr|||
8|25|kandā́|wild almond|noun|||
8|25|bādā́m|edible almond|noun loanword|||from Persian
9|25|cistalə́|bitter|adjective|||source explicitly gives the same form for sour and bitter
10|25|xartī́|donkey|noun|||
11|25|kal|year|noun|||
12|25|tasā́m dēsā́|day after tomorrow|adverb phrase|||
13|26|artə ra paləsā́ arilə́ wusī́sī|rainbow (red, green, and yellow has risen)|phrase|||circumlocution; component etymologies are discussed by Buddruss but the phrase is not directly linked
14|26|kaširə́ zar|silver (white gold)|noun phrase loanword|||calque of Pashto expression
15|26|bis-|sit down|verb intr|||
16|26|ac|today|adverb|||
17|26|sardumə́|autumn|noun|||
18|26|andə́|blind|adjective|||
19|26|darī́|beard|noun|||
20|26|la-|beat|verb tr|||
21|26|kaširə́|white|adjective loanword|||borrowed form
21|26|kasirə́|white|adjective loanword alternate|||accepted alternate pronunciation
22|27|bākəs|box|noun loanword|||English via Pashto
23|27|ā̃nsṭ|mouth|noun|||
23|27|ā̃sṭ|mouth|noun alternate|||
24|27|brəṣpā̃re|birch|noun loanword|||Buddruss treats it as a loan and tentatively discusses a Pashto source
25|27|xantə́|irrigation canal|noun|||
26|27|basə́t|ashes|noun|||
27|27|maλə́k|brain|noun|||
28|27|zanzī́r|chain|noun loanword|||from Pashto
29|27|šutū́r|camel|noun loanword|||from Persian
30|27|pisānsə́|cat|noun|||
31|27|gōrə́|horse|noun|||
32|27|sūrə|dog|noun|||assimilated from šūrə
33|27|rat|sky|noun|||
34|27|ratə anlə́ nāsi|blue sky (there is no cloud in the sky)|phrase|||source circumlocution
35|27|yidə́|heart|noun|||
36|27|bangī́|rooster|noun|||
37|27|ṣín|horn|noun|||
38|28|kērə́|crow|noun|||
39|28|susík|elbow|noun|||
40|28|mis|brass; copper|noun|||
41|28|wisī́|nineteen|numeral|||
42|28|da-|give|verb tr|||
43|28|bas|twelve|numeral|||
44|28|ū|water|noun|||
45|28|lambū́r|lightning|noun|||
46|28|lik-|write|verb tr|||
47|28||foam|excluded|| |explicitly unattested
48|28|list|span (hand measure)|noun|||
49|28|dusmā́n|enemy|noun loanword|||from Persian
50|28|ṣū́r-|hear|verb tr|||
51|28|kondī́r|shoulder|noun|||
52|28|sṓš|summer|noun|||
53|29|āt|flour|noun|||
54|29|tūlakə́|sickle|noun|||
55|29|parī́|fairy|noun loanword|||from Persian
56|29|angā́r|fire|noun|||
57|29|tāu|fever|noun loanword|||from Kabuli Persian
58|29|kəít|fig|noun|||
59|29|zū|girl|noun|||
60|29|puλ|son|noun|||
61|29|kā̃|arrow|noun|||CDIAL 14622 in the commentary belongs to secondary diā 'wall', not this answer
62|29|pimlə́|ant|noun|||
63|29|la|brother|noun|||
64|29|kīlə́|cheese|noun|||
65|29|dum|smoke|noun|||
66|29|tōpā́k|gun|noun|||
67|29|goā́t|animal fat; ghee|noun|||Buddruss's Grangali answer; Grjunberg's comparative form gᵒat remains commentary only
68|29|darím|pomegranate|noun|||
69|29|tutəmarkā́|frog|noun|||
70|30|erā̃|wasp|noun|||
71|30|gā̃s|grass|noun|||
72|30|pau|hedgehog; porcupine|noun|||
73|30|imā́n|winter|noun|||
74|30|šarm-|be ashamed|verb intr loanword|||from Persian
75|30|aṣṭ|eight|numeral|||
76|30|basū́|day (24 hours)|noun|||
76|30|deswā́r|daylight; day|noun|||compared by Buddruss with Lahnda forms and CDIAL 6335, not asserted as a direct reflex
77|30|šudə́|milk|noun loanword|||from Pashto
78|30|zíp|tongue|noun|||
78|30|bāsə́|language|noun|||
79|30|pala-|wash|verb tr|||
80|30|bōčā́r|leopard|noun loanword|||borrowed form
81|30|bistṓ|lip|noun|||
82|30|xᵒat|bed|noun|||source uses a raised o
83|30|dūrwarī́|far|adjective|||
84|30|mas|moon; month|noun|||
84|30|masulī́|moon|noun|||
85|31|ast|hand|noun|||
86|31|gē|house|noun|4251|derived from Sanskrit gēhá-; Buddruss corrects the Wg. form in CDIAL
87|31|xo-|eat|verb tr|||
87|31|xᵒa-|eat|verb tr alternate|||allomorph
88|31|arṣə́|mirror|noun|||from reconstructed *ādarisa
89|31|liλ-|harvest; reap|verb tr|||
89|31|leλ-|harvest; reap|verb tr alternate|||
90|31|dār|mountain|noun|||
91|31|brūt|moustache|noun loanword|||from Persian
92|31|λungalī́|mulberry|noun|||
93|31|ǐm|snow|noun|||Buddruss prints initial i with caron; Ning and Shumashti have im
94|31|kāncə́|black|adjective|||
95|31|turū́|walnut|noun|||
96|31|aṇlə́|cloud|noun|||
97|31|byeλ|night|noun|||
98-99|31|ãc̣|eye|noun|||
100|31|ãr̃ə|egg|noun|||nasalized r is described as an allophone of n
101|31|pē̃|shoulder blade|noun|||
102|31|naṅacə́|fingernail|noun|||
103|31|zas|eleven|numeral|||
104|31|artə zar|gold (red gold)|noun phrase|||calque of Pashto expression
105|32|kõ|ear|noun|||
106|32|zō|barley|noun|||
107|32|anṭī́|bone|noun|||
108-109|32|īc̣|bear|noun|||
110|32||palm of hand|excluded|||explicitly unattested; ast 'hand' is only a cross-reference
111|32|šilaṇtə́|parrot|noun|||
112|32|nasék|granddaughter|noun|||elicited for French petite-fille
113|32|xur|foot|noun|||
114|32|bōt|stone|noun|||
115|32|boṣ|rain|noun|||
115|32|baṣ|rain|noun alternate|||
116|32|mac|fish|noun|||
117|32|manā́|apple|noun loanword|||from Pashto
118|32|zū|louse|noun|||
119|32|dēstə́ angũ|thumb|noun phrase|||
120|32|basə́n|spring (season)|noun|||
120|32|bəsə́n|spring (season)|noun alternate|||
121-122|32|pī́suk|flea|noun|||
123-124|32|cōdḗs|fourteen|numeral|||
125-126|32|cōr|four|numeral|||
127|32|limatə́|tail|noun|||
128|32|lac̣|grape|noun|||
129|32|λawṣ|spleen|noun|||
130|32|ētḗk|kidney|noun|||
131|33|ṣal|jackal (elicited for fox)|noun|12578|Buddruss explicitly says the answer means jackal and derives it from śr̥gālá-
132|33|λō pas-|dream; see a dream|verb phrase|||
133|33|ãs ka-|laugh|verb phrase|||
134|33|nadí|river|noun|||
135|33|tõ|harvested rice|noun|||
136|33|băt|cooked rice|noun|||
137|33|artə́|red|adjective|||
138|33|seóu|sand|noun|||
139|33|lū|blood|noun|||
140|33|umpə́|scorpion|noun|||
141|33|sōrā́s|sixteen|numeral|||
142|33|lõ|salt|noun|||
143|33|õi|snake|noun|||
144|33|ṣu|six|numeral|||
145|34|pas|sister|noun|||
146|34|surí|sun|noun|||
147|34|λõ|sleep|noun|||
148|34|ūc̣|spring; source|noun|||
149|34|mūsõ|mouse|noun|||
150|34|xurík|heel (Buddruss); ankle (Grjunberg)|noun|||Buddruss gives heel; Grjunberg's ankle gloss is retained as comparative provenance
151|34|ṣā̃rə|head|noun|||
152|34|lem|roof|noun|||CDIAL 8730 in the commentary belongs to secondary λəmA- 'forget', not this answer
153|34|λăm|work|noun|||
154|34|λēwā́s|thirteen|numeral|||
155|34|bũzíl|earthquake|noun|||first of two forms printed by Buddruss
155|34|buĩzil|earthquake|noun alternate|||second form printed by Buddruss
156|34|λē|three|numeral|||
157|34|mār-|kill|verb tr|||
158|34|muλ|urine|noun|||
159|34|mondaréi|wind|noun|||
160|35|wōr|belly|noun|||
161|35|andá|meat|noun|||
162|35|lăm|village|noun|||
163|35|-lām|village suffix in place names|suffix|||illustrated by Cukon-lām and Wuriglām
164|35|isí|twenty|numeral|||
165|35|muk|face|noun|||
166|35||Milky Way|excluded|||explicitly unattested
167|35|pas-|see|verb tr|||secondary raxa- means watch or keep watch and is retained only in this audit note
"""

# lect | atlas | printed page | form | actual gloss | tags | note
#
# This is a manual census of the Ningalami and Shumashti forms that Buddruss actually
# prints in the numbered lexical section.  It intentionally does not turn a bare lect
# abbreviation followed by punctuation (e.g. ``Shum., G.B. ...``) into an attestation.
# Forms embedded in commentary are retained with their actual gloss, rather than being
# silently assigned the French Atlas headword (e.g. Ning. mūnda 'neck' under item 5).
COMPARISONS = r"""
Ningalami|2|24|watá|hungry|adjective|
Shumashti|2|24|awata|hungry|adjective|
Shumashti|3|24|bukuník|lamb|noun|
Shumashti|4|25|cuλä|sour; bitter|adjective|Buddruss corrects Fussman's printed culä
Ningalami|5|25|mūnda|neck|noun|comparison embedded in the etymological commentary
Shumashti|7|25|co|go|verb intr imperative|imperative form
Shumashti|10|25|xareṭa|donkey|noun|
Ningalami|10|25|gadə́|donkey|noun|
Shumashti|11|25|kāl|year|noun|
Ningalami|12|25|desāī|tomorrow|adverb|component comparison, not the Atlas phrase 'day after tomorrow'
Shumashti|15|26|nis-|sit down|verb intr|
Shumashti|16|26|nun|today|adverb|
Shumashti|18|26|andá|blind|adjective|
Shumashti|19|26|darí|beard|noun|
Shumashti|20|26|lay-|beat|verb tr|
Ningalami|21|26|käsirə́|white|adjective loanword|
Ningalami|23|27|ā̃sṭ|mouth|noun|
Shumashti|25|27|xā̃ṭṭä|irrigation canal|noun|
Shumashti|30|27|pisā̃sə|cat|noun|
Ningalami|30|27|pisā̃zə́|cat|noun|
Shumashti|31|27|gōro|horse|noun|
Ningalami|32|27|šūrə|dog|noun|
Shumashti|32|27|šūrə|dog|noun|
Ningalami|35|27|yidə́|heart|noun|
Shumashti|35|27|ida|heart|noun|
Shumashti|37|27|ṣín|horn|noun|
Shumashti|38|28|kārə|crow|noun|
Ningalami|39|28|wō̃c̣|elbow|noun|
Shumashti|39|28|susik|shin-bone|noun|actual source gloss differs from Atlas 'elbow'
Shumashti|40|28|mis|brass; copper|noun|
Ningalami|41|28|usí|nineteen|numeral|
Shumashti|42|28|λe-|give|verb tr|
Shumashti|42|28|λi-|give|verb tr alternate|
Ningalami|43|28|bas|twelve|numeral|
Shumashti|43|28|bās|twelve|numeral|
Ningalami|44|28|ū|water|noun|
Shumashti|44|28|wō|water|noun|
Shumashti|48|28|list|span (hand measure)|noun|
Ningalami|50|28|šuni-|hear|verb tr|
Shumashti|53|29|āt|flour|noun|
Shumashti|54|29|tula|sickle|noun|
Ningalami|56|29|angār|fire|noun|
Shumashti|56|29|ãr|fire|noun|
Shumashti|59|29|zū|girl|noun|
Shumashti|60|29|puλ|son|noun|
Ningalami|60|29|zakə́|child|noun|actual source gloss differs from Atlas 'son'
Shumashti|61|29|kõ|arrow|noun|source prints kõ(r), with optional final r
Shumashti|61|29|kõr|arrow|noun alternate|source prints kõ(r), with optional final r
Ningalami|63|29|la|brother|noun|
Shumashti|63|29|lā|brother|noun|
Shumashti|65|29|dūm|smoke|noun|
Shumashti|68|29|dārim|pomegranate|noun|
Shumashti|71|30|gā̃s|grass|noun|
Ningalami|73|30|ēmand|winter|noun|
Shumashti|73|30|yeman|winter|noun|
Ningalami|75|30|õṣṭ|eight|numeral|
Shumashti|75|30|ãṣṭ|eight|numeral|
Ningalami|77|30|šudə́|milk|noun loanword|
Ningalami|77|30|chīr|sour milk|noun|secondary comparison in the commentary
Ningalami|78|30|zip|tongue|noun|
Shumashti|78|30|zīb|tongue|noun|
Shumashti|80|30|bachār|leopard|noun loanword|
Ningalami|81|30|bistō|lip|noun|
Shumashti|81|30|bōstar|lip|noun|
Shumashti|82|30|xāt|bed|noun|
Ningalami|84|30|mas|moon; month|noun|
Ningalami|85|31|wōst|hand|noun|
Shumashti|85|31|ast|hand|noun|
Ningalami|86|31|gē|house|noun|
Ningalami|87|31|xuy-|eat|verb tr|
Shumashti|89|31|leλi-|harvest; reap|verb tr|
Ningalami|90|31|dār-a|mountain|noun inflected|locative form extracted from the quoted utterance utalə dār-a
Shumashti|90|31|dār|mountain|noun|
Shumashti|92|31|λoā̃lī|mulberry|noun|
Ningalami|93|31|im|snow|noun|
Shumashti|93|31|im|snow|noun|
Ningalami|94|31|kācá|black|adjective|
Shumashti|94|31|xacə|black|adjective|
Ningalami|95|31|turū|walnut|noun|
Shumashti|97|31|wyel|night|noun|
Ningalami|98-99|31|wō̃c̣|eye|noun|
Shumashti|98-99|31|aĩc|eye|noun|
Ningalami|100|31|wā̃na|egg|noun|
Shumashti|100|31|ā̃ra|egg|noun|
Shumashti|102|31|naučik|fingernail|noun|
Ningalami|103|31|zas|eleven|numeral|
Shumashti|103|31|zās|eleven|numeral|
Ningalami|105|32|kõ|ear|noun|
Shumashti|105|32|kõr|ear|noun|
Ningalami|106|32|zō|barley|noun|
Shumashti|106|32|zo|barley|noun|
Shumashti|107|32|ãṭhi|bone|noun|
Shumashti|108-109|32|ĩc̣|bear|noun|
Shumashti|112|32|nãwasik|granddaughter|noun|
Ningalami|113|32|xũr|foot|noun|Buddruss questions whether the vowel is long
Shumashti|113|32|xur|foot|noun|
Ningalami|114|32|bōt|stone|noun|
Shumashti|115|32|was|rain|noun|
Ningalami|116|32|mōc|fish|noun|
Shumashti|116|32|māc|fish|noun|
Shumashti|118|32|yū|louse|noun|
Shumashti|119|32|dyēisti aṅur|thumb|noun phrase|
Ningalami|123-124|32|caudēs|fourteen|numeral|
Shumashti|123-124|32|cãudas|fourteen|numeral|
Ningalami|125-126|32|cᵘor|four|numeral|source uses a raised u
Ningalami|128|32|lāc̣|grape|noun|
Shumashti|128|32|lāk|grape|noun|
Shumashti|129|32|plōwa|spleen|noun|
Shumashti|130|32|ãṭeik|kidney|noun|
Shumashti|133|33|āiz-|laugh|verb intr|verbal comparison to Grangali's nominal expression
Ningalami|134|33|nandí|river|noun|
Shumashti|134|33|nādí|river|noun|
Ningalami|135|33|tõ|harvested rice|noun|
Ningalami|136|33|bōt|cooked rice|noun|
Shumashti|136|33|băt|cooked rice|noun|
Ningalami|137|33|wartə́|red|adjective|
Shumashti|137|33|aratə́|red|adjective|
Shumashti|138|33|sīu|sand|noun|
Shumashti|139|33|luí|blood|noun|
Ningalami|141|33|surōs|sixteen|numeral|Buddruss questions the reading of r
Shumashti|141|33|sorās|sixteen|numeral|
Ningalami|142|33|lõ|salt|noun|
Shumashti|142|33|lon|salt|noun|
Ningalami|144|33|ṣo|six|numeral|
Ningalami|145|34|pas|sister|noun|
Ningalami|146|34|surí|sun|noun|
Shumashti|146|34|surí|sun|noun|
Shumashti|147|34|λau|sleep|noun|
Shumashti|148|34|učānik|spring; source|noun|
Shumashti|149|34|mūsõ|mouse|noun|
Shumashti|150|34|xurík|heel|noun|
Shumashti|151|34|šārə|head|noun|
Ningalami|151|34|ṣoũkrə́|head|noun derived|source labels this a derived formation
Shumashti|152|34|lyēmī|roof|noun|
Shumashti|153|34|λăm|work|noun|
Ningalami|154|34|slewās|thirteen|numeral|
Shumashti|154|34|λāwaṣ|thirteen|numeral|
Shumashti|155|34|bõzil|earthquake|noun|
Ningalami|156|34|sle|three|numeral|
Shumashti|156|34|sle|three|numeral|
Shumashti|158|34|muλ|urine|noun|
Ningalami|160|35|wōr|belly|noun|
Shumashti|160|35|war|belly|noun|
Ningalami|161|35|andá|meat|noun|
Ningalami|162|35|lăm|village|noun|
Shumashti|162|35|lăm|village|noun|
Ningalami|164|35|isí|twenty|numeral|
Shumashti|164|35|isí|twenty|numeral|
Shumashti|165|35|dōr|face|noun|
Ningalami|167|35|pas-|see|verb tr|
"""


def records() -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(RAW.strip()), delimiter="|")
    result = []
    seen_counts: Counter[str] = Counter()
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        if len(row) not in {7, 8}:
            raise ValueError(row)
        if len(row) == 8:
            row = row[:6] + [" ".join(row[6:])]
        atlas, page, form, gloss, tags, parameter, note = (cell.strip() for cell in row)
        seen_counts[atlas] += 1
        result.append({
            "atlas": atlas, "printed_page": page, "pdf_page": str(int(page) - 19),
            "form": form, "gloss": gloss, "tags": tags, "parameter": parameter,
            "note": note, "ordinal": str(seen_counts[atlas]),
        })
    return result


def comparison_records() -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(COMPARISONS.strip()), delimiter="|")
    result = []
    seen_counts: Counter[tuple[str, str]] = Counter()
    for row in reader:
        if not row or row[0].startswith("#"):
            continue
        if len(row) != 7:
            raise ValueError(row)
        lect, atlas, page, form, gloss, tags, note = (cell.strip() for cell in row)
        seen_counts[(lect, atlas)] += 1
        result.append({
            "lect": lect, "atlas": atlas, "printed_page": page,
            "pdf_page": str(int(page) - 19), "form": form, "gloss": gloss,
            "tags": tags, "parameter": "", "note": note,
            "ordinal": str(seen_counts[(lect, atlas)]),
        })
    return result


def atlas_numbers(label: str) -> set[int]:
    if "-" in label:
        start, end = map(int, label.split("-"))
        return set(range(start, end + 1))
    return {int(label)}


def key(record: dict[str, str]) -> str:
    suffix = f":form-{record['ordinal']}" if record["ordinal"] != "1" else ""
    return f"{SOURCE_ID}:item-{record['atlas']}{suffix}"


def locator(record: dict[str, str]) -> str:
    return f"{SOURCE_ID}[p. {record['printed_page']}, item {record['atlas']}]"


def comparison_key(record: dict[str, str]) -> str:
    lect = record["lect"].lower()
    suffix = f":form-{record['ordinal']}" if record["ordinal"] != "1" else ""
    return f"{SOURCE_ID}:item-{record['atlas']}:{lect}{suffix}"


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]], header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if header:
            writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path)
    args = parser.parse_args()
    if args.pdf:
        actual = sha256(args.pdf)
        if actual != PDF_SHA256:
            raise ValueError(f"unexpected PDF SHA-256: {actual}")

    raw = records()
    comparisons = comparison_records()
    coverage = set().union(*(atlas_numbers(row["atlas"]) for row in raw))
    assert coverage == set(range(1, 168))
    forms = []
    audit = []
    base_keys: dict[str, str] = {}
    for record in raw:
        emitted = key(record) if record["form"] else ""
        if record["form"]:
            variant = base_keys.get(record["atlas"], "") if "alternate" in record["tags"] else ""
            base_keys.setdefault(record["atlas"], emitted)
            etymology = record["note"]
            if record["parameter"]:
                claim = f"Buddruss derives or directly assigns this form to CDIAL {record['parameter']}."
                etymology = f"{claim} {etymology}".strip()
            forms.append(dict(zip(FORM_FIELDS, [
                "Gng", record["parameter"], record["form"], record["gloss"], "", "",
                "", locator(record), "", etymology, emitted, variant, "", "", record["tags"],
            ])))
        payload = "|".join(record.values()).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Collation_Date": COLLATION_DATE,
            "Atlas_Numbers": record["atlas"],
            "PDF_Page": record["pdf_page"], "Printed_Page": record["printed_page"],
            "Raw_Form": record["form"], "Raw_Gloss": record["gloss"],
            "Final_Status": "installed_form" if record["form"] else "excluded_unattested",
            "Final_Form": record["form"], "Final_Parameter_ID": record["parameter"],
            "Emitted_Key": emitted,
            "Resolution": record["note"] or ("manually collated source transcription" if record["form"] else "explicit non-attestation"),
            "Review": "full manual census against the 300 dpi render",
            "Material_Error": "no", "Source": locator(record),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    comparison_base_keys: dict[tuple[str, str], str] = {}
    for record in comparisons:
        emitted = comparison_key(record)
        pair = (record["lect"], record["atlas"])
        variant = comparison_base_keys.get(pair, "") if "alternate" in record["tags"] else ""
        comparison_base_keys.setdefault(pair, emitted)
        language_id = "Ning" if record["lect"] == "Ningalami" else "Shum"
        tags = record["tags"]
        forms.append(dict(zip(FORM_FIELDS, [
            language_id, "", record["form"], record["gloss"], "", "", record["note"],
            locator(record), "", "", emitted, variant, "", "", tags,
        ])))
        payload = "|".join(record.values()).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Collation_Date": COLLATION_DATE,
            "Atlas_Numbers": record["atlas"], "PDF_Page": record["pdf_page"],
            "Printed_Page": record["printed_page"], "Raw_Form": record["form"],
            "Raw_Gloss": record["gloss"], "Final_Status": "installed_comparison",
            "Final_Form": record["form"], "Final_Parameter_ID": "",
            "Emitted_Key": emitted,
            "Resolution": record["note"] or "manually collated source comparison",
            "Review": "full manual census against the 300 dpi render",
            "Material_Error": "no", "Source": locator(record),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert Counter(row["Final_Status"] for row in audit)["excluded_unattested"] == 3
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(json.dumps({
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": "Buddruss, Georg. 1979. Gṛaṅgali. Ein Nachtrag zum Atlas der Dardsprachen. Münchener Studien zur Sprachwissenschaft 38:21–39.",
        "acquisition": "Stanford Interlibrary Loan request 446721, delivered as a web scan",
        "pdf_sha256": PDF_SHA256,
        "pdf_pages": PDF_PAGES,
        "article_printed_pages": [21, 39],
        "lexical_printed_pages": [24, 35],
        "pdf_redistributed": False,
        "rights": "ILL scan supplied for private study, scholarship, or research; the scan is not checked in.",
        "extraction": {
            "method": "full record-by-record manual collation against 300 dpi page renders; two Tesseract passes were used only for page navigation and comparison",
            "ocr_reproducibility": ["tesseract -l deu+eng --psm 4", "tesseract -l deu+eng --psm 6"],
            "checked_in_layer": "the RAW table in data/other/forms/raw_data/buddruss_grangali_1979.py",
            "atlas_number_coverage": [1, 167],
            "raw_record_count": len(raw),
            "manual_census_count": len(raw) + len(comparisons),
            "comparison_record_count": len(comparisons),
            "comparison_lect_counts": dict(Counter(row["lect"] for row in comparisons)),
            "forms_corrected_after_census": CORRECTED_FORM_COUNT,
            "transcription_uncertainties_remaining": 0,
        },
        "scope": {
            "included": "all main Grangali answers to Atlas questionnaire numbers 1–167, plus every Ningalami and Shumashti form printed in the numbered lexical section, including accurately glossed commentary examples",
            "excluded": "three explicit Grangali non-attestations (47, 110, 166); bare lect abbreviations with no printed form; unnumbered phonological examples outside the lexical section",
            "cdial_policy": "only direct source assignments are linked; comparisons attached to secondary forms or hedged parallels remain prose",
            "language_model": "Grangali (Gng) and Ningalami (Ning) are independent Jambu languages. Both retain Glottolog's umbrella gran1245 because Glottolog does not provide separate codes; Shumashti remains Shum/shum1235.",
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)), "form_count": len(forms),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)), "audit_count": len(audit),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)), "sample_count": len(sample),
        },
        "unresolved": [
            "item 150 has a source-level semantic disagreement: Buddruss glosses the Grangali form as heel, while Grjunberg glosses the comparison as ankle; the transcription itself is resolved",
            "item 24 is certainly called a loan by Buddruss, but his proposed Pashto source is tentative",
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"installed {len(forms)} forms ({len(comparisons)} Ningalami/Shumashti comparisons); "
        f"audited {len(audit)} records covering Atlas 1–167"
    )


if __name__ == "__main__":
    main()

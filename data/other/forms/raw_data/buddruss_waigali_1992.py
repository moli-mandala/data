#!/usr/bin/env python3
"""Install the complete glossary in Buddruss's 1992 Waigali proverbs article.

The copyrighted Stanford ILL scan is not redistributed. ``RAW`` is the checked-in,
manually collated transcription layer. Every printed glossary headword and explicit
headword alternate on pp. 71--78 was checked against the 400 dpi renders; the embedded
OCR and two Tesseract passes were used only for navigation and comparison. Inflected
examples remain source prose unless Buddruss prints them as headwords.

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


SOURCE_ID = "buddruss-waigali1992"
SNAPSHOT_DATE = "2026-08-24"
COLLATION_DATE = "2026-08-24"
PDF_SHA256 = "ca4a146d1b01e5f940e0b49f863fd1ac33ca873bdda1354b93c32bcd85701b8a"
PDF_PAGES = 22
ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260824-buddruss-waigali.csv"
AUDIT_OUTPUT = RAW_DIR / "20260824-buddruss-waigali-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260824-buddruss-waigali-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260824-buddruss-waigali-manifest.json"

TAG_ALIASES = {
    "adjective": "adj", "adverb": "adv", "pronoun": "pron", "numeral": "num",
    "preposition": "prep", "postposition": "postp", "conjunction": "conj",
    "particle": "part", "oblique": "obl", "possessive": "poss",
    "proper": "proper-noun", "demonym": "proper-noun", "historical": "archaic",
    "onomatopoeic": "onomatopoeia", "comparative": "degree", "negative": "neg",
    "past": "pret", "emphatic": "emph", "article": "determiner",
    "genitive": "gen",
}
TAG_DROPS = {"case", "expletive"}

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Unit_ID", "PDF_Page", "Printed_Page",
    "Raw_Form", "Raw_Gloss_German", "English_Gloss", "Final_Status", "Final_Form",
    "Final_Parameter_ID", "Emitted_Key", "Resolution", "Review", "Material_Error",
    "Source", "Record_SHA256",
]

# printed page | head-entry number on page | form | source gloss | English gloss
# | tags | direct Turner/CDIAL id | source/editorial note
RAW = r"""
71|01|a-|Präposition|to; at; before|preposition||source compares Sanskrit ā without assigning a Turner number
71|02|ā|Partizip zu e- kommen|come, arrived|participle|1045|participle of e-; source compares āgata-
71|03|āst'a|gekommen|come, arrived|participle||perfect participle of e-; cross-reference to ā
71|04|āš|Mund|mouth|noun|1533|
71|05|al'i|dies|this|pronoun||first printed headword alternate
71|05|äl'i|dies|this|pronoun alternate||second printed headword alternate
71|06|am'ā|Haus|house|noun|560|
71|07|am'āyter|im Inneren des Hauses|inside the house|spatial||
71|08|am'e|Obliquus zu am'i wir|us|pronoun oblique||oblique of am'i
71|09|Am'eš|Mann aus dem Dorf Amešdes|man from Amešdes|noun demonym||
71|10|am'i|wir|we|pronoun|986|
71|11|änist'a|solch|such|demonstrative|283|
71|12|aṇtal'a|gekreuzt|crossed|adjective||source compares Wamai aṭ'ala
71|13|ara-|hinwerfen, wegwerfen, in Unordnung bringen, zerstören|throw away; disorder; destroy|verb tr||
71|14|at'er|innen, hinein|inside; inward|spatial|357|
71|15|atr'ö|gerade hinauf|straight upward|spatial||opposite of brö
72|01|b-|werden|become|verb intr|9416|
72|02|ba-|können|be able|verb intr|9477|
72|03|-ba|Formans des Genitivs|genitive formative|suffix case||
72|04|-bār|Suffix|abstract-noun suffix|suffix||printed in daṇurab'ār, ištrimačb'ār, ǰentab'ār, trazab'ār
72|05|bašā-|zur Rede stellen, beschuldigen|challenge; accuse|verb tr||source's proposed T. 11589 is explicitly tentative
72|06|ber'am|außen|outside|spatial|9183|formed from ber plus suffix -am
72|07|bernes-|hinausgelangen; zum Frühling gelangen|come out; reach spring|verb intr||
72|08|ber'a|dumm|stupid|adjective|9238|
72|09|bin-|im Zweifel sein, überlegen, nachdenken, bereuen|doubt; deliberate; regret|verb intr||source's comparison to T. 9498 is tentative
72|10|brö|hinab|downward|spatial||opposite of atr'ö
72|11|bur'a|taub|deaf|adjective|9268|
72|12|čin-|schneiden, fällen; entscheiden|cut; fell; decide|verb tr|5046|
72|13|čipičipun'i|Geräusch des Tröpfelns|sound of dripping|noun onomatopoeic||
72|14|čot|Dung, Mist|dung; manure|noun||
73|01|č-|machen, tun, sagen|do; make; say|verb tr|2814|
73|02|čira-|sich verspäten|be late|verb intr|4824|denominative from cira-
73|03|da|Expletivpartikel zur Hervorhebung|emphatic expletive particle|particle||
73|04|dā|Berg|mountain|noun|6793|
73|05|de|Gott kafirisch|pre-Islamic god|noun|6523|
73|06|des|Dorf|village|noun|6547|
73|07|di-|gehen|go|verb intr|227|present stem; source also prints d-
73|07|d-|gehen|go|verb intr alternate|227|explicit alternate stem
73|08|di|auch|also|particle|200|
73|09|dū|zwei|two|numeral|6648|
73|10|duk|Kummer, Leid|sorrow; suffering|noun|6375|
73|11|daṇur'a|schwach, untüchtig, feige, niedrig, arm|weak; incapable; cowardly; lowly; poor|adjective|5524|
73|12|daṇurab'ār|Niedrigkeit|lowliness|noun||derived from daṇur'a with -bār
73|13|e|ein|one|numeral||short form of ew; source gives no Turner number in this entry
73|14|e-|kommen|come|verb intr|2534|
73|15|gā|Kuh|cow|noun|4147|
73|16|gar|Hausstand, Familie|household; family|noun|4428|
73|17|gar'aš|Tag|day|noun|4440|
73|18|go-|Präteritalstamm zu di- gehen|go|verb intr past stem|4008|past stem of di-
73|19|gol|Tal, Land|valley; land|noun|4453|
73|20|ištič'ū|das Tropfen vom Dach ins Haus bei Regen|rain dripping through the roof|noun||first printed alternate
73|20|ištač'ū|das Tropfen vom Dach ins Haus bei Regen|rain dripping through the roof|noun alternate||second printed alternate
73|21|ištr'i|Ehefrau|wife|noun|13734|
74|01|ištri-mač-b'ār|Ehe, Eheleben|marriage; married life|noun||
74|02|ǰat'a|ander|other|adjective||first printed headword alternate
74|02|jat'a|ander|other|adjective alternate||second printed headword alternate; relation to Wamai jad'a is unclear
74|03|ǰay|gut|good|adjective|5190|
74|04|ǰar-|verdauen|digest|verb tr|5304|
74|05|ǰent'a|lebend|alive|adjective|5244|first printed headword alternate
74|05|ǰet'a|lebend|alive|adjective alternate|5244|second printed headword alternate
74|06|ǰentab'ār|Lebendigsein, Leben|being alive; life|noun||
74|07|ǰūt|Leopard|leopard|noun|13969|
74|08|ka|Absolutiv zu č- tun|having done|participle|2814|absolutive of č-
74|09|kāy|irgendwer|someone; anyone|pronoun||source cites both Turner 2694 and 2696, so no single link is imposed
74|10|kan-|lachen|laugh|verb intr|3815|
74|11|-kan|Postposition in, bei|in; at|postposition|2830|
74|12|-kanty'āw|Postposition um willen, für|for; for the sake of|postposition||first printed headword alternate
74|12|-kanty'aw|Postposition um willen, für|for; for the sake of|postposition alternate||second printed headword alternate
74|13|kar'a|hart arbeitend, aktiv|hard-working; active|adjective||
74|14|kaṣ'e|ein Raubvogel|small bird of prey|noun||
74|15|kiš|was, warum|what; why|pronoun||
74|16|kiti|wieviele, einige|how many; some|numeral|3167|
74|17|kō|Obliquus Singular zu kāy wer|whom|pronoun oblique||oblique singular of kāy
74|18|kō|Präteritalstamm zu č- tun|done|verb tr past stem|2814|first printed headword alternate
74|18|krō|Präteritalstamm zu č- tun|done|verb tr past stem alternate|2814|second printed headword alternate
74|19|k'oma|wessen|whose|pronoun||
74|20|kor'ān|Koran, Qur'an|Qur'an|noun loanword||from Arabic-Persian
74|21|kud'āy|Gott|God|noun loanword||from Persian xudāy
75|01|lap'a|Bündel von Fackeln|bundle of torches|noun||
75|02|mā|nicht beim Imperativ und Konjunktiv|not; prohibitive|particle negative|9981|
75|03|māl|Recht|right; entitlement|noun loanword||source considers Arabic-Persian māl or ma'āl
75|04|mač|Ehemann|husband|noun|9888|
75|05|man'aṣ|Mann|man|noun|10049|
75|06|mel'a|Gespräch, Unterhaltung, Wort; Angelegenheit|conversation; word; matter|noun||source marks the comparison to Turner 10331 with a query
75|07|meloḍ'a|Gespräche habend|eloquent; conversational|adjective||
75|08|Melak'an|Personenname|Melakan|noun proper||personal name
75|09|-mili|Postposition zusammen mit|together with|postposition|10133|
75|10|mŕa|Nest|nest|noun|10042|first printed headword alternate
75|10|mŕā|Nest|nest|noun alternate|10042|second printed headword alternate
75|11|mūk|Gesicht; gegenüber, vor|face; opposite; before|noun spatial|10174|
75|12|na|nicht|not|particle negative|6906|
75|13|nām|Name|name|noun|7067|
75|14|n'äri|jetzt|now|adverb||first printed headword alternate; etymology uncertain
75|14|n'ari|jetzt|now|adverb alternate||second printed headword alternate
75|15|nāṭ|Tanz|dance|noun|7580|
75|16|o-|Präsens der Kopula sein|be|verb intr||source relates om to asmi and oš to asi without assigning a Turner number
75|17|oč|Bär|bear|noun|2445|
75|18|opuǰ-|geboren werden|be born|verb intr|1814|
75|19|oṣ'a|Sattheit; genug|satiety; enough|noun particle||
76|01|pa|nur|only|particle||
76|02|pāč|Seite, Richtung|side; direction|noun|8118|
76|03|pāǰ'i|ein großer Raubvogel|large bird of prey; eagle|noun||
76|04|pār|Schlag; mal|blow; time|noun numeral||
76|05|payd'a b-|entstehen|arise; originate|verb intr loanword||from Persian
76|06|pat'om|nachher, später|afterward; later|adverb|7732|first printed headword alternate
76|06|pot'om|nachher, später|afterward; later|adverb alternate|7732|second printed headword alternate
76|07|piš'ä|Katze|cat|noun|8298|
76|08|pōt|Weg|path; road|noun|7785|
76|09|poṭ|roter weicher Stoff aus der Stadt|red soft imported cloth|noun|7700|
76|10|pr-|geben|give|verb tr|8655|
76|11|prü|gleich wie|like; equal to|particle comparative||first printed headword alternate
76|11|prüst'a|gleich wie|like; equal to|particle comparative alternate||second printed headword alternate
76|11|prust'a|gleich wie|like; equal to|particle comparative alternate||third printed headword alternate
76|12|pus'a|Maus|mouse|noun||source tentatively compares *mus'a
76|13|püs b-|verloren gehen, verschwinden|be lost; disappear|verb intr|8310|
76|14|ri|aber|but|conjunction|434|
76|15|ṛa|Formans des Dativs oder Postposition für|dative formative; for|postposition case||
76|16|sa|jener|that|demonstrative|12815|
76|17|sāl|Held; tapfer, tüchtig|hero; brave; capable|noun adjective||source's etymology is tentative
76|18|-sta|Suffix zur Nominalisierung|nominalizing suffix|suffix||
77|01|sun|Gold|gold|noun|13519|
77|02|šāl|Stall|stable|noun|12414|
77|03|šüwal'a|Angehöriger der zweiten Klasse der kafirischen Sklaven|member of the second pre-Islamic slave class|noun historical||
77|04|ṣar|Ziege; Numerativ für Vieh|goat; livestock classifier|noun numeral|12269|first printed headword alternate
77|04|ṣār|Ziege; Numerativ für Vieh|goat; livestock classifier|noun numeral alternate|12269|second printed headword alternate
77|05|ṣay|Kopf|head|noun|12694|
77|06|ṣer'a|blind|blind|adjective|12717|
77|07|ṣoč|Streit|quarrel|noun|13085|
77|08|ta|wenn|if; when|conjunction|5639|
77|09|ta-|aufstellen, einrichten, in Ordnung bringen|set up; arrange; put in order|verb tr|13756|
77|10|tāṭ'i|Vater|father|noun|5754|
77|11|t'ema|Genitiv Plural zu sa jener|of those|pronoun genitive||genitive plural of sa
77|12|ti-|sein, sich befinden, stattfinden|be; be located; take place|verb intr|13768|
77|13|ti|wie|like; as|particle comparative||absolutive of zi- according to source
77|14|to|Obliquus Singular zu sa jener|that one|pronoun oblique||
77|15|traz'a|krank|ill|adjective||source says relation to Ashkun parallels is unclear
77|16|tre|drei|three|numeral|5994|
78|01|tu|du oblique|you|pronoun oblique||
78|02|tü|du|you|pronoun|5889|
78|03|tün|Haufen|heap; pile|noun||etymology uncertain
78|04|tuk'a|Stein zum Werfen oder Schlagen|throwing stone|noun||source offers two possible Turner comparisons
78|05|tük|Stück, Sache, Ding|piece; thing|noun|5466|
78|06|tünür-nāṭ|besonderer kafirischer Tanz in gebeugter Haltung|pre-Islamic dance performed bent over|noun historical||
78|07|ū|Obliquus zu aṇ'a ich|me|pronoun oblique||
78|08|ub'a|Rebhuhn|partridge|noun||
78|09|'uma|mein|my|pronoun possessive||
78|10|watr|Nacht|night|noun||
78|11|wāṭ|Stein|stone|noun|11348|
78|12|weg'ār|Freude, Fröhlichkeit|joy; cheerfulness|noun||
78|13|widiš'ä|Gast|guest|noun|11738|
78|14|wrō|gegessen|eaten|participle||first printed headword alternate; suppletive to yā-
78|14|wrōst'a|gegessen|eaten|participle alternate||second printed headword alternate
78|15|ye|und|and|conjunction||
78|16|yey|Mutter|mother|noun|1351|
78|17|yūd|Kampf|fight|noun|10499|
78|18|zag'a|Sohn|son|noun|14516|
78|19|zö|Herz|heart|noun|14152|
78|20|zor|Milch|milk|noun||source treats etymology as unresolved
"""


def records() -> list[dict[str, str]]:
    result = []
    counts: Counter[tuple[str, str]] = Counter()
    for row in csv.reader(io.StringIO(RAW.strip()), delimiter="|"):
        if not row or row[0].startswith("#"):
            continue
        if len(row) != 8:
            raise ValueError(row)
        page, unit, form, raw_gloss, gloss, tags, parameter, note = (
            cell.strip() for cell in row
        )
        counts[(page, unit)] += 1
        result.append({
            "page": page, "unit": unit, "form": form, "raw_gloss": raw_gloss,
            "gloss": gloss, "tags": tags, "parameter": parameter, "note": note,
            "ordinal": str(counts[(page, unit)]),
        })
    return result


def base_key(record: dict[str, str]) -> str:
    return f"{SOURCE_ID}:p{record['page']}:e{record['unit']}"


def key(record: dict[str, str]) -> str:
    suffix = f":v{record['ordinal']}" if record["ordinal"] != "1" else ""
    return base_key(record) + suffix


def locator(record: dict[str, str]) -> str:
    return f"{SOURCE_ID}[p. {record['page']}, glossary entry {record['unit']}]"


def canonical_tags(value: str) -> str:
    return " ".join(dict.fromkeys(
        TAG_ALIASES.get(tag, tag) for tag in value.split() if tag not in TAG_DROPS
    ))


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
    if args.pdf and sha256(args.pdf) != PDF_SHA256:
        raise ValueError(f"unexpected PDF SHA-256: {sha256(args.pdf)}")

    raw = records()
    forms = []
    audit = []
    for record in raw:
        emitted = key(record)
        etymology = record["note"]
        if record["parameter"]:
            claim = f"Buddruss directly compares or assigns this headword to Turner/CDIAL {record['parameter']}."
            etymology = f"{claim} {etymology}".strip()
        forms.append(dict(zip(FORM_FIELDS, [
            "Wg", record["parameter"], record["form"], record["gloss"], "", "",
            record["raw_gloss"], locator(record), "", etymology, emitted,
            base_key(record) if record["ordinal"] != "1" else "", "", "",
            f"{canonical_tags(record['tags'])} dialect:Wg:nis:Nisheigram",
        ])))
        payload = "|".join(record.values()).encode()
        audit.append({
            "Snapshot_Date": SNAPSHOT_DATE, "Collation_Date": COLLATION_DATE,
            "Unit_ID": f"p{record['page']}:e{record['unit']}",
            "PDF_Page": str(int(record["page"]) - 58), "Printed_Page": record["page"],
            "Raw_Form": record["form"], "Raw_Gloss_German": record["raw_gloss"],
            "English_Gloss": record["gloss"], "Final_Status": "installed_form",
            "Final_Form": record["form"], "Final_Parameter_ID": record["parameter"],
            "Emitted_Key": emitted,
            "Resolution": record["note"] or "manually collated printed glossary headword",
            "Review": "full manual census against the 400 dpi render; embedded OCR and two Tesseract passes compared",
            "Material_Error": "no", "Source": locator(record),
            "Record_SHA256": hashlib.sha256(payload).hexdigest(),
        })

    assert len({row["Entry_Key"] for row in forms}) == len(forms)
    assert {int(row["page"]) for row in raw} == set(range(71, 79))
    write_csv(FORM_OUTPUT, FORM_FIELDS, forms, header=False)
    write_csv(AUDIT_OUTPUT, AUDIT_FIELDS, audit, header=True)
    sample = sorted(audit, key=lambda row: row["Record_SHA256"])[:25]
    write_csv(SAMPLE_OUTPUT, AUDIT_FIELDS, sample, header=True)
    MANIFEST_OUTPUT.write_text(json.dumps({
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": "Buddruss, Georg. 1992. Waigali-Sprichwörter. Studien zur Indologie und Iranistik 16/17:65–80.",
        "acquisition": "Stanford Interlibrary Loan request 446829, delivered by UC Berkeley as a web scan",
        "pdf_sha256": PDF_SHA256,
        "pdf_pages": PDF_PAGES,
        "article_printed_pages": [65, 80],
        "lexical_printed_pages": [71, 78],
        "pdf_redistributed": False,
        "rights": "Copyrighted ILL scan supplied for private study, scholarship, or research; the scan is not checked in.",
        "extraction": {
            "method": "complete record-by-record manual collation against 400 dpi page renders",
            "ocr_reproducibility": [
                "embedded PDF text layer used for navigation",
                "tesseract -l deu+eng --psm 4",
                "tesseract -l deu+eng --psm 6",
            ],
            "checked_in_layer": "the RAW table in data/other/forms/raw_data/buddruss_waigali_1992.py",
            "glossary_headword_record_count": len(raw),
            "transcription_uncertainties_remaining": 0,
        },
        "scope": {
            "included": "every printed headword and explicit headword alternate in the complete glossary on pp. 71–78",
            "excluded": "the 25 proverb texts and their running translations; inflected examples not promoted by Buddruss to headword status; comparative Kamviri, Wamai, Tregami, Pashai, and other-language forms in etymological commentary",
            "cdial_policy": "direct unambiguous Turner/CDIAL assignments are linked; hedged, alternative, and secondary comparisons remain prose",
            "language_model": "all forms belong to canonical Waigali (Wg) and carry the registered Nisheigram dialect tag (nis); informants remain provenance",
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)), "form_count": len(forms),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)), "audit_count": len(audit),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)), "sample_count": len(sample),
        },
        "unresolved": [],
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"installed and audited {len(forms)} Waigali glossary headword records")


if __name__ == "__main__":
    main()

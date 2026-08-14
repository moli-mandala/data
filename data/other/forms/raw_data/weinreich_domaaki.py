"""Extract Weinreich's Nager/Hunza Domaaki vocabulary comparison.

The article's section 8 contains 48 numbered vocabulary comparisons.  The
lexical citation forms are curated below because the PDF uses a private-use
glyph for the retroflex affricate and mixes headwords with complete inflectional
paradigms.  The extractor independently recovers every numbered source entry
for the audit, so omissions and page-locator drift remain testable.

Run from ``data/``::

    uv run --with pdfplumber python \
      data/other/forms/raw_data/weinreich_domaaki.py \
      /path/to/Weinreich-TwoVarietiesomaak-2008.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


SOURCE_ID = "weinreich2008"
RICH_COLUMNS = 15
PDF_PAGES = range(12, 18)  # article pp. 309--314 in the JSTOR wrapper
PRIVATE_RETROFLEX_AFFRICATE = "\ue466"


@dataclass(frozen=True)
class Form:
    lect: str
    value: str
    gloss: str
    turner_ids: tuple[str, ...] = ()


def N(value: str, gloss: str, *turner_ids: str) -> Form:
    return Form("domaaki_nager", value, gloss, turner_ids)


def H(value: str, gloss: str, *turner_ids: str) -> Form:
    return Form("domaaki_hunza", value, gloss, turner_ids)


# Citation forms and explicitly offered alternatives in section 8.  Distinct
# synonyms are independent rows rather than graph variants; the article does
# not claim that one derives from another.
ITEMS: dict[int, tuple[Form, ...]] = {
    1: (N("baríš", "year", "11392"), N("baríša", "year", "11392"), H("baríša", "year", "11392")),
    2: (N("č̣éedoos", "day after tomorrow", "5994", "6333"), H("č̣éedo", "day after tomorrow", "5994", "6333")),
    3: (N("biẓoón", "rainbow", "12052"), H("bíiẓoi biẓoóni", "rainbow", "12052")),
    4: (N("kháša", "mouth"), H("khašá", "mouth")),
    5: (N("phúla", "ashes"), H("phulá", "ashes")),
    6: (
        N("búuši", "cat", "8298"), H("phitíiši", "cat"), H("pitiší", "cat"),
        N("phitíiši", "fairy"),
    ),
    7: (N("bambulaá", "tomcat"), H("bambuláaỵ", "tomcat")),
    8: (N("looyá riíl", "brass", "11135", "10752"), H("looyá haliẓá", "brass", "11135", "13990")),
    9: (N("támlam", "lightning"), H("bíčuṣ", "lightning")),
    10: (N("phóok", "shoulder", "13839", "13840"), N("phaaká", "shoulder", "13839", "13840"), H("phaaká", "shoulder", "13839", "13840")),
    11: (N("yaaỵá", "summer"), N("yaaỵé", "summer"), H("yaaỵé", "summer")),
    12: (N("tíir", "arrow"), N("kóon", "arrow", "3023"), H("kóon", "arrow", "3023")),
    13: (N("biráaya", "brother", "9661"), N("biróoi", "brother", "9661"), H("biráaya", "brother", "9661")),
    14: (N("tumóq", "rifle; pistol"), H("tubóq", "rifle")),
    15: (N("bičíl", "pomegranate"), H("daanú", "pomegranate", "6254")),
    16: (N("manúuko", "frog", "9746"), N("ġúrkuċ", "frog"), H("miník", "frog", "9746")),
    17: (N("ṣiqáal", "wasp"), N("šiqáal", "wasp"), H("iṣqáara", "wasp")),
    18: (
        N("hundekuná", "winter", "14164"), N("hundá", "winter", "14164"), N("hundé", "winter", "14164"),
        H("hundé", "winter", "14164"),
    ),
    19: (
        N("ačimóo õõṭo", "upper lip", "2563"), N("miníino õõṭo", "lower lip", "2563"),
        H("aċímo õõṭo", "upper lip", "2563"), H("minéenio õõṭo", "lower lip", "2563"),
    ),
    20: (N("khoṭ", "bed frame", "3781"), H("khaṭ", "bed", "3781")),
    21: (N("miniindeenáaŋa", "bedding"), H("ʌtsideˑni", "upper bedding")),
    22: (N("č̣onč̣", "moon"), N("č̣ónč̣a", "moon"), H("ċonč̣", "moon")),
    23: (N("áaino", "mirror"), N("áaina", "mirror"), H("ayína", "mirror")),
    24: (N("phúuŋi", "moustache", "9083"), H("salát", "moustache")),
    25: (
        N("khakaí", "walnut"), N("akhóo", "walnut", "48"),
        H("akhóu", "walnut", "48"), H("akhóoỵ", "walnut", "48"),
    ),
    26: (N("ač(h)", "eye", "43"), N("ač(h)a", "eye", "43"), H("ač", "eye", "43")),
    27: (N("číi", "pine tree", "4837"), N("číiya", "pine tree", "4837"), H("číiỵ", "pine tree", "4837")),
    28: (N("póo", "foot; leg", "8056"), H("póo", "foot; leg", "8056")),
    29: (N("bóot", "big flat stone", "11348"), H("bóot", "big flat stone", "11348")),
    30: (N("gíri", "big stone; boulder", "4161"), N("gíiri", "big stone; boulder", "4161"), H("gíiri", "big stone; boulder", "4161")),
    31: (N("hangúṭ", "thumb", "137"), N("baḍí agúla", "thumb", "11225", "135"), H("báṛi agúla", "thumb", "11225", "135")),
    32: (N("čõúndei", "fourteen", "4605"), H("čaundéi", "fourteen", "4605")),
    33: (N("p(h)ačoó", "tail", "8249"), H("čipóoỵ", "tail", "4818")),
    34: (N("ló(o)i", "fox", "11142"), H("láač", "fox", "11003")),
    35: (N("suuná", "dream", "13481"), N("suuné", "dream", "13481"), H("suuná", "dream", "13481")),
    36: (N("briyú(u)", "rice", "12233"), H("bras", "rice")),
    37: (N("sáu", "sand"), H("baalí", "sand", "11580")),
    38: (N("ṣõõi", "sixteen", "12812"), H("ṣõwéi", "sixteen", "12812")),
    39: (N("kirmá", "worm", "3438"), H("kirmá", "snake", "3438"), N("jon", "snake", "5110")),
    40: (N("tóo", "sun", "5767"), H("tóo", "sun", "5767")),
    41: (
        N("gúuwo", "heel", "4479"), N("píni", "heel; lower leg; calf; instep", "8168"),
        H("píni", "heel", "8168"),
    ),
    42: (N("bóbok", "buttock; thigh"), H("bóbok", "calf")),
    43: (N("gíṭa", "vagina"), N("giṭ", "vagina"), H("čut", "vagina", "4860")),
    44: (
        N("širooṭá", "head", "12452"), H("ċhúṭo", "head"),
        N("iċhúṭi", "tuft of hair on top of the head"),
    ),
    45: (N("kom", "work", "2892"), H("krom", "work", "2892")),
    46: (N("čaagá", "bad (masculine)", "4564"), H("ʌčaˑga", "bad (masculine)", "4564")),
    47: (
        N("šóo", "one hundred", "12278"), N("poi-bíiš", "one hundred", "7655", "11616"),
        H("põĩ bíiš", "one hundred", "7655", "11616"),
    ),
    48: (N("núu", "nine", "6984"), N("nũũ", "nine", "6984"), H("náu", "nine", "6984")),
}


def clean_text(text: str) -> str:
    text = text.replace(PRIVATE_RETROFLEX_AFFRICATE, "č̣")
    text = re.sub(r"(?<=\w)-\s+(?=\w)", "", text)
    return unicodedata.normalize("NFC", re.sub(r"\s+", " ", text).strip())


def extract_entries(pdf_path: Path) -> dict[int, tuple[int, str]]:
    entries: dict[int, tuple[int, str]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) != 19:
            raise ValueError(f"expected the 19-page JSTOR article, got {len(pdf.pages)} pages")
        for pdf_page in PDF_PAGES:
            printed_page = pdf_page + 297
            # Exclude JSTOR's download banner while retaining the article's
            # own footnotes, which sit above this crop boundary.
            text = pdf.pages[pdf_page - 1].crop((20, 45, 465, 700)).extract_text() or ""
            matches = list(re.finditer(r"(?m)^8\.(\d+)\.\s+", text))
            for index, match in enumerate(matches):
                number = int(match.group(1))
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                raw = text[match.end() : end]
                raw = raw.split("This content downloaded from", 1)[0]
                if number == 5:
                    raw = re.split(r"(?m)^6 Where not specially indicated", raw, maxsplit=1)[0]
                if number == 48:
                    raw = raw.split("As a postscript", 1)[0]
                entries[number] = (printed_page, clean_text(raw))
    missing = set(range(1, 49)) - set(entries)
    if missing:
        raise ValueError(f"missing vocabulary items: {sorted(missing)}")
    return entries


def build_rows(entries: dict[int, tuple[int, str]]) -> tuple[list[list[str]], list[dict[str, str]]]:
    rows: list[list[str]] = []
    audit: list[dict[str, str]] = []
    for number, forms in ITEMS.items():
        printed_page, raw_entry = entries[number]
        source = f"{SOURCE_ID}[p. {printed_page}, § 8.{number}]"
        counts: dict[str, int] = {}
        for form in forms:
            counts[form.lect] = counts.get(form.lect, 0) + 1
            suffix = counts[form.lect]
            parameters = form.turner_ids or ("",)
            for link_number, parameter in enumerate(parameters, 1):
                key = f"weinreich-domaaki:8.{number}:{form.lect}:{suffix}:link:{link_number}"
                analysis = (
                    f"Weinreich cites Turner {parameter} for orientation, not as a claim "
                    "about the immediate origin of this Domaaki form."
                    if parameter else ""
                )
                rows.append([
                    form.lect, parameter, unicodedata.normalize("NFC", form.value), form.gloss,
                    "", "", "", source, "", analysis, key, "", "", "",
                    "uncertain" if parameter else "",
                ])
        audit.append({
            "Item": f"8.{number}",
            "Printed_Page": str(printed_page),
            "Nager_Forms": "|".join(form.value for form in forms if form.lect == "domaaki_nager"),
            "Hunza_Forms": "|".join(form.value for form in forms if form.lect == "domaaki_hunza"),
            "Turner_IDs": "|".join(dict.fromkeys(
                parameter for form in forms for parameter in form.turner_ids
            )),
            "Raw_Entry": raw_entry,
        })
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/other/forms/20260813-weinreich-domaaki.csv"),
    )
    parser.add_argument(
        "--audit", type=Path,
        default=Path("data/other/forms/raw_data/20260813-weinreich-domaaki-audit.csv"),
    )
    args = parser.parse_args()
    rows, audit = build_rows(extract_entries(args.pdf))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    print(f"wrote {len(rows)} form rows from {len(audit)} vocabulary comparisons")


if __name__ == "__main__":
    main()

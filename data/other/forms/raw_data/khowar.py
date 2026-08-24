"""Extract Elena Bashir's Khowar lexicon and its CDIAL links.

The dictionary body is a two-column, born-digital PDF. Main entries begin in
bold Arial at a fixed column margin; bold forms inside an entry are subentries
and are intentionally kept as part of the parent entry rather than emitted as
independent dictionary headwords. Turner references such as ``(T8000)`` link a
Khowar form to the corresponding CDIAL parameter.

Run from ``data/``::

    uv run --with pdfplumber python data/other/forms/raw_data/khowar.py \
      "/path/to/Khowar-English-Dictionary.pdf"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from dialects import dialect_tag


SOURCE_ID = "bashir2023"
LANGUAGE_ID = "Kho"
FIRST_DICTIONARY_PAGE = 14  # one-based PDF page; printed page 1
LAST_DICTIONARY_PAGE = 173  # one-based PDF page; printed page 160
COLUMNS = ((45, 306, 54), (306, 570, 324))

TURNER_REFERENCE = re.compile(r"(?<![\w])T\s*:?[\s-]*(\d+[a-z]?)(?![\w])")

SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

POS_TAGS = {
    "adj": "adj",
    "adv": "adv",
    "conjunction": "conj",
    "interjection": "interj",
    "modal particle": "part",
    "n": "n",
    "n.": "n",
    "particle": "part",
    "pl n": "pl n",
    "postposition": "postp",
    "pron": "pron",
    "suffix": "suffix",
    "vi": "intr verb",
    "vt": "tr verb",
    "vintr": "intr verb",
    "vtr": "tr verb",
    "vtr, vintr": "tr intr verb",
    "vintr, vtr": "intr tr verb",
}

# The front matter's Table 1 maps contributor codes to people and home regions.
# People are provenance, never dialects.  A contributor therefore selects a
# canonical place-based dialect below, while the exact source label remains in
# Notes and in the audit.
SOURCE_PERSON = {
    "AR": "Abdur Rauf", "AK": "Adina Khan", "A": "Amanullah",
    "ARC": "Amin ur Rahman Chughtai", "AKM": "Amir Khan Mir",
    "BA": "Baba Ayub", "BM": "Babu Muhammad", "BKA": "Bulbul Khan Ayub",
    "CKT": "Changez Khan Tareqi", "DAT": "Dinar Ali Taj",
    "FQ": "Fazal Qayyum", "GMKH": "Gul Murad Khan Hasrat",
    "GNK": "Gul Nawaz Khaki", "HAS": "Haider Ali Shah", "HS": "Hasil Shah",
    "HUR": "Hidayat ur Rahman", "IWK": "Ibrahim Wali Kamil",
    "IA": "Inayatullah Aseer", "ICS": "Inayatullah Chishti Sabri",
    "IF": "Inayatullah Faizi", "IFM": "Inayatullah Faizi's mother",
    "IS": "Islam Shah", "IWA": "Ismail Wali Akhgar",
    "MHH": "Mahbub ul Haq Haqqi", "MNN": "Maula Nigah Nigah",
    "MA": "Mir Ahmed", "MAK": "Muhammad Arap Khan",
    "MII": "Muhammad Irfan Irfan", "MYS": "Muhammad Yousuf Shahzad",
    "MY": "Muhammad Younus", "MS": "Mukarram Shah",
    "MWT": "Murad Wali Taj", "MK": "Mustafa Kamal", "N": "Naseer",
    "NKN": "Naji Khan Naji", "NR": "Naqibullah Razi",
    "RAKR": "Rahmat Akbar Khan Rahmat",
    "RAKRW": "Rahmat Akbar Khan Rahmat's wife",
    "RKB": "Rahmat Karim Baig", "RK": "Rozgar Khan",
    "SN-M": "Saeed Nazir", "SN-C": "Sahib Nadir", "S": "Safitullah",
    "SG": "Samad Gul", "SH": "Sardar Hussain",
    "SSM": "Shahzada Sikandar ul Mulk", "SAS": "Sher Akbar Saba",
    "SWKA": "Sher Wali Khan Aseer", "TMF": "Taj Muhammad Figar",
    "TMFW": "Taj Muhammad Figar's wife", "TMFD": "Taj Muhammad Figar's daughter",
    "WUR": "Wali ur Rahman", "WSiC": "woman storyteller in Chapali",
    "WSiM": "woman storyteller in Mastuj", "ZP": "Zafarullah Parwaz",
    "ZHD": "Zahoor ul Haq Danish", "ZHDM": "Zahoor ul Haq Danish's mother",
    "ZMZ": "Zakir Muhammad Zakhmi", "ZK": "Zarkoti Khan",
}

SOURCE_HOME = {
    "AR": "Parwak, Tehsil Mastuj",
    "AK": "Chapali, Tehsil Mastuj",
    "A": "Khost, Mulkhow",
    "ARC": "Drosh, Tehsil Drosh",
    "AKM": "Chumurkun, Tehsil Chitral",
    "BA": "Chumurkun, Tehsil Chitral",
    "BM": "Mroi, Tehsil Chitral",
    "BKA": "Mulkhow",
    "CKT": "Shogram, Tehsil Mulkhow",
    "DAT": "Pasum, Tehsil Mastuj",
    "FQ": "Chitral town",
    "GMKH": "Parkusap, Tehsil Mastuj",
    "GNK": "Singoor, Tehsil Chitral",
    "HAS": "Chitral town",
    "HS": "Chitral town",
    "HUR": "Jughoor, Tehsil Chitral",
    "IWK": "Mastuj town",
    "IA": "Chitral town",
    "ICS": "Chitral town",
    "IF": "Balim, Laspur",
    "IFM": "Balim, Laspur",
    "IS": "Mulkhow",
    "IWA": "Mastuj town",
    "MHH": "Zondrangram, Terich, Mulkhow",
    "MNN": "Zondrangram, Terich, Mulkhow",
    "MA": "Rayin, Torkhow",
    "MAK": "Sor Rech, Torkhow",
    "MII": "Chitral town",
    "MYS": "Sor Laspur",
    "MY": "Sor Laspur",
    "MS": "Warijun, Mulkhow",
    "MWT": "Pasum, Tehsil Mastuj",
    "MK": "Uzhnu, Torkhow",
    "N": "Shyaqotek, Tehsil Chitral",
    "NKN": "Shagram, Torkhow",
    "NR": "Drosh, Tehsil Drosh",
    "RAKR": "Chapali, Tehsil Mastuj",
    "RAKRW": "Chapali, Tehsil Mastuj",
    "RKB": "Zondrangram, Terich, Mulkhow",
    "RK": "Sor Rech, Torkhow",
    "SN-M": "Madaglasht, Tehsil Drosh",
    "SN-C": "Chitral town",
    "S": "Sonoghor, Tehsil Mulkhow",
    "SG": "Mogh, Lutkoh",
    "SH": "Booni",
    "SSM": "Mastuj",
    "SAS": "Thingshen, Proper Chitral",
    "SWKA": "Bang, Yarkhun",
    "TMF": "Zargarandeh, Chitral town",
    "TMFW": "Zargarandeh, Chitral town",
    "TMFD": "Zargarandeh, Chitral town",
    "WUR": "Chitral town",
    "WSiC": "Chapali",
    "WSiM": "Mastuj",
    "ZP": "Booni",
    "ZHD": "Zondrangram, Terich, Mulkhow",
    "ZHDM": "Zondrangram, Terich, Mulkhow",
    "ZMZ": "Tehsil Torkhow",
    "ZK": "Mahrting, Yarkhun",
}

# Canonical source-locality records.  Several contributors share these points;
# they must consequently share a dialect tag.  Coordinates are reviewed
# source-locality/gazetteer points (quality B), not speaker-specific points.
PLACE_COORDINATES = {
    "Ayun": (35.72168, 71.77158),
    "Balim": (36.07012, 72.43959),
    "Bang": (36.52283, 72.76388),
    "Booni": (36.25392, 72.22284),
    "Chapali": (36.33570, 72.60138),
    "Chitral town": (35.850889, 71.79019),
    "Chumurkun": (35.79784, 71.78801),
    "Drosh": (35.56163, 71.79756),
    "Jughoor": (35.82708, 71.78633),
    "Karimabad": (35.99193, 71.81522),
    "Khairabad": (36.78961, 73.04180),
    "Khost": (36.30044, 72.21299),
    "Khot": (36.50216, 72.53267),
    "Laspur": (36.04784, 72.46796),
    "Lutkoh": (36.01231, 71.65609),
    "Lower Chitral": (35.75000, 71.78000),
    "Madaglasht": (35.77558, 72.03137),
    "Mahrting": (36.49059, 72.72128),
    "Mastuj": (36.28356, 72.51942),
    "Meragram": (36.26364, 72.37142),
    "Mogh": (36.01231, 71.65609),
    "Mroi": (35.93000, 71.82000),
    "Mulkhow": (36.30044, 72.21299),
    "Parkusap": (36.28880, 72.52920),
    "Parwak": (36.27759, 72.39010),
    "Pasum": (36.30426, 72.55493),
    "Proper Chitral": (35.850889, 71.79019),
    "Rayin": (36.39233, 72.37763),
    "Reshun": (36.15365, 72.09928),
    "Shagram": (36.398835, 72.277695),
    "Shogram": (36.32251, 72.17305),
    "Shyaqotek": (35.85000, 71.79000),
    "Singoor": (35.89778, 71.79791),
    "Sonoghor": (36.30000, 72.18000),
    "Sor Laspur": (36.04784, 72.46796),
    "Sor Rech": (36.542845, 72.49219),
    "Thingshen": (35.8512345, 71.78495),
    "Terich": (36.39444, 72.22829),
    "Torkhow": (36.45309, 72.42228),
    "Upper Chitral": (36.33000, 72.29000),
    "Uthul": (36.30442, 72.18117),
    "Uzhnu": (36.49925, 72.44552),
    "Warijun": (36.30044, 72.21299),
    "Yarkhun": (36.52283, 72.76388),
    "Zargarandeh": (35.8507445, 71.791345),
    "Zondrangram": (36.3631067, 72.22319),
}

PLACE_ALIASES = {
    "Chitral Museum": "Chitral town", "Chitral Town": "Chitral town",
    "Karimabad valley": "Karimabad", "Lotkoh": "Lutkoh",
    "Mastuj town": "Mastuj", "Rayin Torkhow": "Rayin",
    "Sonogor": "Sonoghor", "Tehsil Torkhow": "Torkhow",
}

REGIONS = (
    "Upper Chitral", "Lower Chitral", "Proper Chitral", "Chitral town",
    "Torkhow", "Mulkhow", "Laspur", "Yarkhun", "Mastuj", "Drosh",
    "Lutkoh", "Madaglasht", "Parwak", "Chapali", "Booni", "Terich",
    "Sonoghor", "Shagram", "Shogram", "Sor Rech", "Zondrangram",
    "Bang", "Warijun", "Balim", "Pasum", "Uzhnu", "Rayin", "Reshun",
    "Uthul", "Karimabad valley", "Khairabad", "Khot", "Chitral Museum",
    "Mastuj town", "Chumurkun", "Mroi", "Khost", "Parkusap", "Singoor",
    "Jughoor", "Mogh", "Thingshen", "Zargarandeh", "Mahrting", "Sor Laspur",
    "Ayun", "Meragram",
)

DONOR_LANGUAGE = {
    "Eng": "Eng",
    "Ur": "H",
    "Prs": "Pers",
    "Ar": "Ar",
    "Bur": "Bur",
    "Wakhi": "Wkh",
    "Turkic": "TurkicUnspec",
}

RICH_COLUMNS = 15


@dataclass
class Entry:
    form: str
    pdf_page: int
    printed_page: int
    lines: list[str] = field(default_factory=list)
    bold_spans: list[tuple[str, bool]] = field(default_factory=list)
    sequence: int = 0

    @property
    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.lines)).strip()


def _font(char: dict) -> str:
    return char.get("fontname", "").split("+")[-1]


def _bold_spans(line: dict) -> list[tuple[str, bool]]:
    """Return bold lexical spans, preserving genuine spaces from PDF gaps.

    Main headwords and subentries are bold Arial; alternate pronunciations are
    bold italic Arial.  Superscript aspiration and retroflex dots may be stored
    as separate glyphs, so adjacent bold glyphs are joined regardless of size.
    """
    spans: list[tuple[str, bool]] = []
    current: list[str] = []
    current_italic = False
    previous = None

    def finish() -> None:
        nonlocal current
        value = _normalize_headword("".join(current).strip())
        if value:
            spans.append((value.translate(SUPERSCRIPT_DIGITS), current_italic))
        current = []

    for char in line.get("chars") or []:
        font = _font(char)
        bold = "Arial" in font and "Bold" in font
        italic = "Italic" in font or "Oblique" in font
        if not bold:
            if current:
                finish()
            previous = char
            continue
        if current and italic != current_italic:
            finish()
        if not current:
            current_italic = italic
        elif previous is not None and char.get("x0", 0) - previous.get("x1", 0) > 2.0:
            current.append(" ")
        current.append(char.get("text", ""))
        previous = char
    if current:
        finish()
    return spans


def _normalize_headword(form: str) -> str:
    """Repair PDF ordering/spacing around the retroflex ``c-dot`` glyph.

    The dot is a separately positioned glyph. Text extraction can place it
    after the following vowel or superscript ``h`` and can synthesize spaces
    around it even though the printed headword has none.
    """
    # A few fallback-font glyphs extract under visually similar but incorrect
    # Unicode code points. These repairs are confirmed against the rendered PDF.
    form = (
        form.replace("ṷ", "ʋ")
        .replace("ı", "i")
        .replace("I", "í")
        .replace("ɦ", "h")
        .replace("ȷ̌ị́", "ǰí")
        .replace("cṇ", "c̣n")
        .replace("cḷ", "c̣l")
        .replace("zehcpayán", "zehčpayán")
    )
    form = unicodedata.normalize("NFD", form)
    form = (
        form.replace("ȷ\u030ci\u0323\u0301", "j\u030ci\u0301")
        .replace("cn\u0323", "c\u0323n")
        .replace("cl\u0323", "c\u0323l")
        .replace("cγ\u0323", "c\u0323y")
        .replace("cg\u0323", "c\u0323y")
    )
    # The PDF occasionally assigns the retroflex-c dot to a fallback ``k``.
    form = form.replace("ck\u0323", "c\u0323").replace("k\u0323", "c\u0323")
    form = re.sub(r"\s*\u0323\s*", "\u0323", form)
    form = form.replace("ch\u0323", "c\u0323h")
    form = re.sub(
        r"c([aeiou])\u0323([\u0300-\u036f]*)",
        "c\u0323" + r"\1\2",
        form,
    )
    # Remaining false word breaks occur immediately after the displaced glyph.
    form = re.sub(
        r"(c\u0323[aeiou][\u0300-\u036f]*)\s+(?=[bcdfgjklmnpqrstvwxyzṭḍṣẓžšǰɫγ])",
        r"\1",
        form,
    )
    form = re.sub(r"c\u0323h\s+(?=[aeiou])", "c\u0323h", form)
    form = re.sub(r"c\u0323y\s+(?=[aeiou])", "c\u0323y", form)
    form = unicodedata.normalize("NFC", form)
    # The alternative in the muc̣ entry loses its separately positioned dot
    # after the closing slash; visually it is the same retroflex form.
    if form == "muc":
        form = "muc̣"
    return form


def _headword(line: dict, column_start: float) -> str | None:
    """Return a main-entry headword from a pdfplumber text line."""
    if not line.get("chars") or abs(line["x0"] - column_start) > 1.5:
        return None
    chars = line["chars"]
    if "Arial" not in _font(chars[0]) or "Bold" not in _font(chars[0]):
        return None

    # Spaces are synthesized in ``line['text']`` and are absent from ``chars``.
    # Count the initial bold PDF characters, then consume the same number of
    # non-space characters from the reconstructed line. This preserves genuine
    # multiword headwords while correctly rejoining fallback-font fragments.
    bold_count = 0
    for char in chars:
        font = _font(char)
        if char["text"] != "/" and "Arial" in font and "Bold" in font:
            bold_count += 1
        else:
            break
    if not bold_count:
        return None

    consumed = 0
    result = []
    for char in line["text"]:
        if not char.isspace():
            consumed += 1
        result.append(char)
        if consumed == bold_count:
            break
    form = _normalize_headword("".join(result).strip())
    # The PDF prints homonym/variant indices as small baseline-raised digits.
    # Preserve their distinction without treating them as ordinary numerals.
    form = form.translate(SUPERSCRIPT_DIGITS)
    form = re.sub(
        r"\s+\((?:adj|adv|n|pl n|pron|vi|vintr|vt|vtr)\)$", "", form
    )
    return form or None


def _part_of_speech(text: str, form: str) -> str:
    prefix = text[len(form) :]
    quote = prefix.find("‘")
    if quote >= 0:
        prefix = prefix[:quote]
    for raw in re.findall(r"\(([^()]*)\)", prefix):
        normalized = re.sub(r"\s+", " ", raw.strip().lower())
        if normalized in POS_TAGS:
            return POS_TAGS[normalized]
    return ""


def _gloss(text: str, form: str) -> str:
    """Extract the first English definition, excluding later example text."""
    remainder = text[len(form) :]
    match = re.search(r"‘([^’‘]+)[’‘]", remainder)
    if not match:
        match = re.search(r"'([^']+)'", remainder)
    if not match:
        match = re.search(r"‘([^{}]+?)(?=\s*\{)", remainder)
    if not match:
        return ""
    gloss = re.sub(r"\s+", " ", match.group(1)).strip(" ;,")
    return _normalize_headword(gloss)


def _key_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = re.sub(r"[¹²³⁴⁵⁶⁷⁸⁹⁰]", "", text)
    return re.sub(r"[^\w\u0300-\u036f]+", "", text)


def _plausible_lexical_form(text: str) -> bool:
    return bool(_key_text(text)) and not re.search(r"[‘’{}\[\]]", text)


def _entry_key(entry: Entry) -> str:
    digest = hashlib.sha1(
        f"{entry.printed_page}:{_key_text(entry.form)}".encode()
    ).hexdigest()[:10]
    return f"bashir:khowar:{digest}"


def _bracket_etymology(text: str) -> str:
    return " ".join(f"[{value.strip()}]" for value in re.findall(r"\[([^\[\]]+)\]", text))


def _morphology_notes(text: str) -> list[str]:
    notes = []
    for value in re.findall(r"\(([^()]*)\)", text):
        if (
            "+" in value
            or "←" in value
            or re.search(
                r"\b(?:causative|compound|derived|derivation|formation|participle)\b",
                value,
                re.I,
            )
        ):
            value = re.sub(r"\s+", " ", value).strip()
            if value not in notes:
                notes.append(value)
    return notes


def _source_tokens(text: str) -> list[str]:
    values = []
    for group in re.findall(r"\{([^{}]+)\}", text):
        for token in group.split(","):
            token = re.sub(r"\s+(?:19|20)\d{2}$", "", token.strip())
            if token and token not in values:
                values.append(token)
    return values


def _slug(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value[:48] or hashlib.sha1(value.encode()).hexdigest()[:10]


def _canonical_place(value: str) -> str:
    value = re.sub(r"^(?:Village|Tehsil)\s+", "", value.strip(), flags=re.I)
    value = re.sub(r"\s+women$", "", value, flags=re.I)
    value = value.strip(" ()?,.;")
    value = PLACE_ALIASES.get(value, value)
    if value in PLACE_COORDINATES:
        return value
    first = PLACE_ALIASES.get(value.split(",", 1)[0].strip(), value.split(",", 1)[0].strip())
    return first if first in PLACE_COORDINATES else value


def _register_place(place: str, registry: dict[str, dict]) -> str:
    place = _canonical_place(place)
    did = f"Kho-Bashir-place-{_slug(place)}"
    latitude, longitude = PLACE_COORDINATES.get(place, ("", ""))
    registry[did] = {
        "ID": did,
        "Name": f"Khowar: {place}",
        "Glottocode": "khow1242",
        "Latitude": latitude,
        "Longitude": longitude,
        "Clade": "Chitrali",
        "Location": place,
        "Quality": "B" if latitude != "" else "C",
    }
    return did


def _speaker_codes(text: str) -> list[str]:
    alternatives = "|".join(re.escape(code) for code in sorted(SOURCE_HOME, key=len, reverse=True))
    return list(dict.fromkeys(
        match.group(0)
        for match in re.finditer(rf"(?<![\w-])(?:{alternatives})(?![\w-])", text)
    ))


def _source_dialect_ids(text: str, registry: dict[str, dict]) -> list[str]:
    """Map a curly-brace source label to places without promoting people/texts."""
    codes = _speaker_codes(text)
    if codes:
        return list(dict.fromkeys(
            _register_place(SOURCE_HOME[code], registry) for code in codes
        ))
    places = _regions_in(text)
    return list(dict.fromkeys(_register_place(place, registry) for place in places))


def _provenance_notes(tokens: list[str]) -> list[str]:
    codes = []
    locality_labels = []
    residual_labels = []
    for token in tokens:
        found = _speaker_codes(token)
        codes.extend(code for code in found if code not in codes)
        if found:
            stripped = token
            for code in found:
                stripped = re.sub(
                    rf"(?<![\w-]){re.escape(code)}(?![\w-])", "", stripped
                )
            if stripped.strip(" ()?,.;:-"):
                residual_labels.append(token)
        elif _regions_in(token):
            locality_labels.append(token.strip())
        else:
            residual_labels.append(token.strip())

    notes = []
    if codes:
        notes.append(
            "contributors: "
            + ", ".join(f"{SOURCE_PERSON[code]} ({code})" for code in codes)
        )
    if locality_labels:
        notes.append("source localities: " + ", ".join(dict.fromkeys(locality_labels)))
    if residual_labels:
        notes.append("source labels: " + ", ".join(dict.fromkeys(residual_labels)))
    return notes


def _regions_in(text: str) -> list[str]:
    return [region for region in REGIONS if re.search(rf"\b{re.escape(region)}\b", text, re.I)]


def _register_region(region: str, registry: dict[str, dict]) -> str:
    return _register_place(region, registry)


def _entry_languages(entry: Entry, registry: dict[str, dict]) -> list[str]:
    languages = []
    for token in _source_tokens(entry.text):
        languages.extend(_source_dialect_ids(token, registry))
    gloss = _gloss(entry.text, entry.form)
    is_place_definition = re.search(
        r"\b(?:village|town|region|valley|stream|place|tribe|clan)\b", gloss, re.I
    )
    regional_context = [] if is_place_definition else _regions_in(gloss)
    for region in _regions_in(entry.text):
        if re.search(rf"(?:used|usage|form|pronounc\w*)[^.;]{{0,55}}\b{re.escape(region)}\b", entry.text, re.I):
            regional_context.append(region)
    languages.extend(_register_region(region, registry) for region in dict.fromkeys(regional_context))
    return list(dict.fromkeys(languages)) or [LANGUAGE_ID]


def _other_pronunciation_blocks(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", match.group(1)).strip()
        for match in re.finditer(
            r"/Other pronunc(?:iation|iations|):?\s*(.*?)/", text, re.I
        )
    ]


def _variant_block(form: str, blocks: list[str]) -> str:
    key = _key_text(form)
    for block in blocks:
        if key and key in _key_text(block):
            return block
    return ""


def _variant_forms(entry: Entry, blocks: list[str]) -> list[tuple[str, str]]:
    output = []
    for span, italic in entry.bold_spans:
        if not italic:
            continue
        span = span.strip(" /,;:")
        for form in re.split(r"\s*(?:;|,|\bor\b)\s*", span):
            form = re.sub(
                r"\s+(?:sometimes\s+)?in\s+(?:Upper|Lower|Proper)?\s*"
                r"(?:Chitral|Torkhow|Mulkhow|Laspur|Yarkhun|Mastuj|Drosh|Lutkoh|Warijun).*$",
                "",
                form,
                flags=re.I,
            )
            form = re.sub(r"\s*\([^()]*\)\s*$", "", form).strip(" /,;:")
            form = _normalize_headword(form)
            if (
                _plausible_lexical_form(form)
                and _key_text(form) != _key_text(entry.form)
                and (block := _variant_block(form, blocks))
                and (form, block) not in output
            ):
                output.append((form, block))
    return output


def _find_span(text: str, form: str, start: int = 0) -> tuple[int, int] | None:
    pattern = re.escape(form).replace(r"\ ", r"\s+")
    match = re.search(pattern, text[start:])
    if not match:
        return None
    return start + match.start(), start + match.end()


def _subentries(entry: Entry) -> list[dict]:
    """Extract bold, POS-marked subentries as local derived lexical nodes."""
    regular = []
    skipped_main = False
    for form, italic in entry.bold_spans:
        if not skipped_main and _key_text(form) == _key_text(entry.form):
            skipped_main = True
            continue
        if (
            not italic
            and _key_text(form) != _key_text(entry.form)
            and _plausible_lexical_form(form)
            and len(_key_text(form)) >= 2
        ):
            regular.append(form)

    found = []
    cursor = 0
    for index, form in enumerate(dict.fromkeys(regular), 1):
        span = _find_span(entry.text, form, cursor)
        if not span:
            continue
        next_positions = [
            candidate[0]
            for other in regular[index:]
            if (candidate := _find_span(entry.text, other, span[1]))
        ]
        end = min(next_positions) if next_positions else len(entry.text)
        segment = entry.text[span[0]:end]
        pos = _part_of_speech(segment, form)
        if not pos:
            cursor = span[1]
            continue
        gloss = _gloss(segment, form)
        morph = _morphology_notes(segment[:300])
        found.append({"form": form, "pos": pos, "gloss": gloss, "morphology": morph, "text": segment})
        cursor = span[1]
    return found


def _direct_donor(etymology: str) -> tuple[str, str, str] | None:
    """Return a conservative (Language_ID, form, gloss) direct-loan analysis."""
    for note in re.findall(r"\[([^\[\]]+)\]", etymology):
        match = re.match(r"\s*<\s*(Eng|Ur|Prs|Ar|Bur|Wakhi|Turkic)\.?\s+(.*)", note)
        if not match:
            continue
        code, rest = match.groups()
        # Multiple named donor languages do not establish a unique immediate source.
        before_quote = re.split(r"[‘']", rest, 1)[0]
        if re.search(r"\b(?:Eng|Ur|Prs|Ar|Bur|Wakhi|Turkic)\.?\b", before_quote):
            continue
        rest = rest.strip()
        gloss = ""
        if rest.startswith(("‘", "'")):
            if code != "Eng":
                continue
            quote = re.match(r"[‘']([^’']+)[’']", rest)
            if not quote:
                continue
            form = quote.group(1).strip()
        else:
            form = re.split(r"\s*[‘'(;]", rest, 1)[0].strip(" ,.;")
            quote = re.search(r"[‘']([^’']+)[’']", rest)
            if quote:
                gloss = quote.group(1).strip()
        if (
            not form
            or len(form.split()) > 5
            or re.search(r"[?+<>]", form)
            or re.search(r"\b(?:or|possibly|probably|root|suffix)\b", form, re.I)
        ):
            continue
        return DONOR_LANGUAGE[code], _normalize_headword(form), gloss
    return None


def _rich_row(
    language: str,
    parameter: str,
    form: str,
    gloss: str,
    notes: str,
    *,
    etymology: str = "",
    entry_key: str = "",
    variant_of_key: str = "",
    borrowed_from_key: str = "",
    derivation_parent_keys: list[str] | tuple[str, ...] = (),
    tags: list[str] | tuple[str, ...] = (),
    source: str = SOURCE_ID,
) -> list[str]:
    return [
        language, parameter, form, gloss, "", "", notes, source, "",
        etymology, entry_key, variant_of_key, borrowed_from_key,
        "|".join(dict.fromkeys(filter(None, derivation_parent_keys))),
        " ".join(dict.fromkeys(filter(None, tags))),
    ]


def _finish(entry: Entry | None, rows: list[list[str]], valid_cdial: set[str]) -> None:
    """Legacy single-entry helper retained for focused parser tests."""
    if entry is None:
        return
    text = entry.text
    cited = list(dict.fromkeys(TURNER_REFERENCE.findall(text)))
    valid = [number for number in cited if number in valid_cdial]
    invalid = [number for number in cited if number not in valid_cdial]
    pos = _part_of_speech(text, entry.form)
    gloss = _gloss(text, entry.form)
    source = f"{SOURCE_ID}[p. {entry.pdf_page} (printed p. {entry.printed_page})]"

    def append_row(parameter_id: str, note: str) -> None:
        rows.append(
            _rich_row(
                LANGUAGE_ID, parameter_id, entry.form, gloss, note,
                etymology=_bracket_etymology(text), entry_key=_entry_key(entry),
                tags=[pos] if pos else [], source=source,
            )
        )

    for number in valid:
        append_row(number, "")
    if not valid:
        unresolved = ""
        if invalid:
            unresolved = "unresolved Turner citation(s) " + ", ".join(
                f"T{number}" for number in invalid
            )
        append_row("", unresolved)


def read_cdial_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {
            row[0]
            for row in csv.reader(stream)
            if row and re.fullmatch(r"\d+[a-z]?", row[0])
        }


def extract_entries(pdf_path: Path) -> list[Entry]:
    entries: list[Entry] = []
    current: Entry | None = None
    with pdfplumber.open(pdf_path) as pdf:
        if len(pdf.pages) < LAST_DICTIONARY_PAGE:
            raise ValueError(
                f"expected at least {LAST_DICTIONARY_PAGE} PDF pages, got {len(pdf.pages)}"
            )
        for pdf_page in range(FIRST_DICTIONARY_PAGE, LAST_DICTIONARY_PAGE + 1):
            page = pdf.pages[pdf_page - 1]
            for x0, x1, column_start in COLUMNS:
                crop = page.crop((x0, 60, x1, 720))
                lines = crop.extract_text_lines(layout=False, return_chars=True)
                previous_ended_bold = False
                for line in sorted(lines, key=lambda item: item["top"]):
                    form = _headword(line, column_start)
                    if form:
                        if current is not None:
                            entries.append(current)
                        current = Entry(form, pdf_page, pdf_page - 13, sequence=len(entries) + 1)
                        previous_ended_bold = False
                    if current is not None:
                        current.lines.append(line["text"])
                        spans = _bold_spans(line)
                        chars = line.get("chars") or []
                        begins_bold = bool(chars) and "Bold" in _font(chars[0])
                        if previous_ended_bold and begins_bold and spans and current.bold_spans:
                            previous, italic = current.bold_spans.pop()
                            first, first_italic = spans.pop(0)
                            if italic == first_italic:
                                current.bold_spans.append((previous + " " + first, italic))
                            else:
                                current.bold_spans.extend(((previous, italic), (first, first_italic)))
                        current.bold_spans.extend(spans)
                        previous_ended_bold = bool(chars) and "Bold" in _font(chars[-1])
    if current is not None:
        entries.append(current)
    return entries


def _existing_source_index(path: Path) -> dict[tuple[str, str], str]:
    if not path.exists():
        return {}
    candidates: dict[tuple[str, str], list[str]] = {}
    with path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if SOURCE_ID in (row.get("Source") or "").split(";"):
                continue
            key = (row.get("Language_ID", ""), _key_text(row.get("Form", "")))
            if key[0] and key[1]:
                candidates.setdefault(key, []).append(row["ID"])
    return {key: values[0] for key, values in candidates.items() if len(set(values)) == 1}


def _resolved_morph_parents(
    notes: list[str], head_by_form: dict[str, str], own_key: str
) -> list[str]:
    parents = []
    for note in notes:
        compact = _key_text(note)
        for form_key, entry_key in sorted(head_by_form.items(), key=lambda item: -len(item[0])):
            if len(form_key) >= 3 and form_key in compact and entry_key != own_key:
                parents.append(entry_key)
    return list(dict.fromkeys(parents))


def build_rows(
    entries: list[Entry],
    valid_cdial: set[str],
    existing_sources: dict[tuple[str, str], str] | None = None,
) -> tuple[list[list[str]], dict[str, dict], list[dict]]:
    existing_sources = existing_sources or {}
    rows: list[list[str]] = []
    dialects: dict[str, dict] = {}
    audit: list[dict] = []
    generated_sources: dict[tuple[str, str], tuple[str, list[str]]] = {}

    head_candidates: dict[str, list[str]] = {}
    for entry in entries:
        head_candidates.setdefault(_key_text(entry.form), []).append(_entry_key(entry))
    head_by_form = {
        form: keys[0] for form, keys in head_candidates.items() if len(set(keys)) == 1
    }

    def donor_key(language: str, form: str, gloss: str, note: str) -> str:
        lookup = (language, _key_text(form))
        if lookup in existing_sources:
            return f"id:{existing_sources[lookup]}"
        if lookup in generated_sources:
            return generated_sources[lookup][0]
        digest = hashlib.sha1(f"{language}:{lookup[1]}".encode()).hexdigest()[:10]
        key = f"bashir:donor:{language}:{digest}"
        source_row = _rich_row(
            language, "", form, gloss,
            f"Donor form cited in Bashir's Khowar etymology: [{note}]",
            etymology=f"Source form cited by Bashir: [{note}]",
            entry_key=key, tags=["source-form"],
        )
        generated_sources[lookup] = (key, source_row)
        return key

    for entry in entries:
        text = entry.text
        key = _entry_key(entry)
        cited = list(dict.fromkeys(TURNER_REFERENCE.findall(text)))
        valid = [number for number in cited if number in valid_cdial]
        invalid = [number for number in cited if number not in valid_cdial]
        parameters = valid or [""]
        pos = _part_of_speech(text, entry.form)
        gloss = _gloss(text, entry.form)
        source_ref = f"{SOURCE_ID}[p. {entry.pdf_page} (printed p. {entry.printed_page})]"
        source_tokens = _source_tokens(text)
        languages = _entry_languages(entry, dialects)
        morph = _morphology_notes(text)
        etymology = _bracket_etymology(text)
        if morph:
            etymology = "; ".join(filter(None, [etymology, "Morphology: " + "; ".join(morph)]))
        morph_parents = _resolved_morph_parents(morph, head_by_form, key)
        donor = _direct_donor(text)
        borrowed_from = ""
        if donor:
            donor_language, donor_form, donor_gloss = donor
            note = next(
                (b for b in re.findall(r"\[([^\[\]]+)\]", text) if donor_form in b),
                etymology,
            )
            borrowed_from = donor_key(donor_language, donor_form, donor_gloss, note)

        notes = _provenance_notes(source_tokens)
        if invalid:
            notes.append("unresolved Turner citation(s) " + ", ".join(f"T{x}" for x in invalid))
        tags = [pos] if pos else []
        if borrowed_from:
            tags.append("loanword")
        if morph_parents:
            tags.append("compound" if any("+" in value for value in morph) else "derived")

        for parameter in parameters:
            parents = list(morph_parents)
            if borrowed_from and parameter:
                parents.append(f"id:{parameter}")
            for language in languages:
                rows.append(
                    _rich_row(
                        language, parameter, entry.form, gloss, "; ".join(notes),
                        etymology=etymology, entry_key=key,
                        borrowed_from_key=borrowed_from,
                        derivation_parent_keys=parents, tags=tags, source=source_ref,
                    )
                )
        audit.append({
            "Role": "head", "Entry_Key": key, "Parent_Key": borrowed_from,
            "Form": entry.form, "Language_IDs": "|".join(languages), "POS": pos,
            "Gloss": gloss, "Etymology": etymology, "PDF_Page": entry.pdf_page,
            "Printed_Page": entry.printed_page,
            "Source_Labels": "|".join(source_tokens),
            "Contributor_Codes": "|".join(
                dict.fromkeys(
                    code for token in source_tokens for code in _speaker_codes(token)
                )
            ),
        })

        blocks = _other_pronunciation_blocks(text)
        for number, (form, block) in enumerate(_variant_forms(entry, blocks), 1):
            variant_key = f"{key}:variant:{number}"
            variant_languages = _source_dialect_ids(block, dialects)
            variant_languages.extend(
                _register_region(region, dialects) for region in _regions_in(block)
            )
            variant_languages = list(dict.fromkeys(variant_languages)) or list(languages)
            variant_etymology = f"Other pronunciation of {entry.form}: {block}"
            for language in variant_languages:
                rows.append(
                    _rich_row(
                        language, "", form, gloss,
                        "",
                        etymology=variant_etymology, entry_key=variant_key,
                        variant_of_key=key, derivation_parent_keys=[key],
                        tags=["sound-variant", "variant"], source=source_ref,
                    )
                )
            audit.append({
                "Role": "variant", "Entry_Key": variant_key, "Parent_Key": key,
                "Form": form, "Language_IDs": "|".join(variant_languages), "POS": pos,
                "Gloss": gloss, "Etymology": variant_etymology,
                "PDF_Page": entry.pdf_page, "Printed_Page": entry.printed_page,
            })

        for number, subentry in enumerate(_subentries(entry), 1):
            form = subentry["form"]
            sub_key = f"{key}:subentry:{number}"
            sub_morph = subentry["morphology"]
            parents = [key] + _resolved_morph_parents(sub_morph, head_by_form, key)
            sub_etymology = f"Subentry of {entry.form}"
            if sub_morph:
                sub_etymology += "; Morphology: " + "; ".join(sub_morph)
            sub_languages = list(languages)
            sub_languages.extend(
                _register_region(region, dialects)
                for region in _regions_in(subentry["text"][:250])
            )
            sub_languages = list(dict.fromkeys(sub_languages))
            sub_tags = [subentry["pos"], "derived"]
            if any("causative" in value.casefold() for value in sub_morph):
                sub_tags.append("caus")
            if any("+" in value or "compound" in value.casefold() for value in sub_morph):
                sub_tags.append("compound")
            for language in sub_languages:
                rows.append(
                    _rich_row(
                        language, "", form, subentry["gloss"],
                        "",
                        etymology=sub_etymology, entry_key=sub_key,
                        derivation_parent_keys=parents, tags=sub_tags, source=source_ref,
                    )
                )
            audit.append({
                "Role": "subentry", "Entry_Key": sub_key, "Parent_Key": "|".join(parents),
                "Form": form, "Language_IDs": "|".join(sub_languages),
                "POS": subentry["pos"], "Gloss": subentry["gloss"],
                "Etymology": sub_etymology, "PDF_Page": entry.pdf_page,
                "Printed_Page": entry.printed_page,
            })

    # Donor nodes precede borrowers so ancestry ordering remains intuitive.
    source_rows = [value[1] for _, value in sorted(generated_sources.items())]
    return source_rows + rows, dialects, audit


def extract(
    pdf_path: Path,
    valid_cdial: set[str],
    existing_sources: dict[tuple[str, str], str] | None = None,
) -> list[list[str]]:
    rows, _, _ = build_rows(extract_entries(pdf_path), valid_cdial, existing_sources)
    return rows


def sync_dialects(path: Path, dialects: dict[str, dict]) -> None:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or [
            "ID", "Tag", "Language_ID", "Source_Language_ID", "Name", "Glottocode",
            "Latitude", "Longitude", "Clade", "Location", "Quality",
        ]
        rows = [
            row for row in reader
            if not row["Source_Language_ID"].startswith("Kho-Bashir-")
        ]
    for key in sorted(dialects):
        source = dialects[key]
        name = source["Name"].split(": ", 1)[1]
        rows.append({
            "ID": source["ID"],
            "Tag": dialect_tag(LANGUAGE_ID, source["ID"], name),
            "Language_ID": LANGUAGE_ID,
            "Source_Language_ID": source["ID"],
            "Name": name,
            "Glottocode": source["Glottocode"],
            "Latitude": source["Latitude"],
            "Longitude": source["Longitude"],
            "Clade": source["Clade"],
            "Location": source["Location"],
            "Quality": source["Quality"],
        })
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def migrate_existing(output: Path, audit_path: Path, dialect_path: Path) -> tuple[int, int]:
    """Reclassify an installed extract without needing the source PDF again."""
    with dialect_path.open(encoding="utf-8", newline="") as stream:
        old_dialects = {
            row["Source_Language_ID"]: row for row in csv.DictReader(stream)
        }
    with output.open(encoding="utf-8", newline="") as stream:
        old_rows = list(csv.reader(stream))

    dialects: dict[str, dict] = {}
    migrated = []
    source_labels_by_key: dict[str, str] = {}
    for original in old_rows:
        row = list(original)
        if row[0].startswith("Kho-Bashir-"):
            metadata = old_dialects.get(row[0], {})
            label = metadata.get("Name", row[0]).split(" (", 1)[0]
            mapped = _source_dialect_ids(label, dialects)
            if not mapped:
                mapped = [LANGUAGE_ID]
        else:
            mapped = [row[0]]

        already_expanded = any(
            f"{name} ({code})" in row[6]
            for code, name in SOURCE_PERSON.items()
        )
        if row[6].startswith("contributors: ") and not already_expanded:
            raw_labels = row[6].removeprefix("contributors: ")
            source_labels_by_key.setdefault(row[10], raw_labels)
            row[6] = "; ".join(
                _provenance_notes([part.strip() for part in raw_labels.split(",")])
            )
        for language in mapped:
            copy = list(row)
            copy[0] = language
            migrated.append(copy)

    # Several contributor-specific rows now intentionally converge on one
    # place attestation.  Preserve the first occurrence's deterministic order.
    migrated = list(dict.fromkeys(tuple(row) for row in migrated))
    with output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(migrated)

    with audit_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        audit_rows = list(reader)
        fields = list(reader.fieldnames or [])
    for field in ("Source_Labels", "Contributor_Codes"):
        if field not in fields:
            fields.append(field)
    for item in audit_rows:
        mapped = []
        for language in item["Language_IDs"].split("|"):
            if language.startswith("Kho-Bashir-"):
                metadata = old_dialects.get(language, {})
                label = metadata.get("Name", language).split(" (", 1)[0]
                targets = _source_dialect_ids(label, dialects)
                mapped.extend(targets or [LANGUAGE_ID])
            else:
                mapped.append(language)
        item["Language_IDs"] = "|".join(dict.fromkeys(mapped))
        labels = source_labels_by_key.get(item["Entry_Key"], "")
        item["Source_Labels"] = labels.replace(", ", "|")
        item["Contributor_Codes"] = "|".join(_speaker_codes(labels))
    with audit_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)

    sync_dialects(dialect_path, dialects)
    return len(old_rows), len(migrated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path, nargs="?")
    parser.add_argument(
        "--migrate-existing", action="store_true",
        help="normalize the installed CSV/audit without re-extracting the PDF",
    )
    parser.add_argument(
        "--cdial-params", type=Path, default=Path("data/cdial/params.csv")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/other/forms/20260725-bashir-khowar.csv"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("data/other/forms/raw_data/20260725-bashir-khowar-audit.csv"),
    )
    parser.add_argument(
        "--languages", type=Path, default=Path("cldf/languages.csv")
    )
    parser.add_argument(
        "--dialects", type=Path, default=Path("cldf/dialects.csv")
    )
    args = parser.parse_args()
    if args.migrate_existing:
        before, after = migrate_existing(args.output, args.audit, args.dialects)
        print(f"migrated {before} rows to {after} place-based provenance rows")
        return
    if args.pdf is None:
        parser.error("pdf is required unless --migrate-existing is used")
    existing = _existing_source_index(Path("cldf/forms.csv"))
    entries = extract_entries(args.pdf)
    rows, dialects, audit = build_rows(entries, read_cdial_ids(args.cdial_params), existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    sync_dialects(args.dialects, dialects)
    if any(row[0] == "TurkicUnspec" for row in rows):
        with args.languages.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            language_fields = reader.fieldnames
            language_rows = list(reader)
        if not any(row["ID"] == "TurkicUnspec" for row in language_rows):
            language_rows.append({
                "ID": "TurkicUnspec", "Name": "Turkic (unspecified)", "Glottocode": "",
                "Latitude": "", "Longitude": "", "Clade": "Other",
                "Location": "", "Quality": "C",
            })
            with args.languages.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=language_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(language_rows)
    linked = sum(bool(row[1]) for row in rows)
    unresolved = sum("unresolved Turner" in row[6] for row in rows)
    roles = {role: sum(item["Role"] == role for item in audit) for role in ("head", "variant", "subentry")}
    donors = sum(item["Role"] == "head" and bool(item["Parent_Key"]) for item in audit)
    print(
        f"wrote {args.output} ({len(rows)} rows; {roles}; {donors} direct loans; "
        f"{linked} linked rows; {len(dialects)} dialect labels; "
        f"{unresolved} unresolved citations)"
    )


if __name__ == "__main__":
    main()

"""Extract entries from Trail & Cooper's Kalasha dictionary.

The PDF uses a legacy SIL font without a Unicode map.  pdfplumber therefore
returns the font's old keystrokes (``a5`` for ``á``, ``}`` for ``š``, etc.).
This script identifies dictionary entries from their typography, decodes the
roman headwords, and emits Jambu source rows for linked and unlinked entries.

Run from ``data/``:

    uv run --with pdfplumber python data/other/forms/raw_data/kalasha.py \
      "/path/to/Kalasha Dictionary.pdf"
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber


SOURCE_ID = "trail-cooper1999"
LANGUAGE_ID = "Kal"
DEFAULT_DIALECT_ID = "bumb"
FIRST_DICTIONARY_PAGE = 31  # one-based PDF page; printed page 1
LAST_DICTIONARY_PAGE = 376  # one-based PDF page; printed page 346

POS = {
    "Adj.",
    "Adv.",
    "Aux.",
    "Cnj.",
    "Id.",
    "Intj.",
    "KT.",
    "N.",
    "Num.",
    "Pfx.",
    "Ppl.",
    "Pron.",
    "Rel.",
    "RootVb.",
    "Sfx.",
    "V.",
    "Vpfx.",
    "Vsfx.",
}

POS_NOTES = {
    "Adj": "adj",
    "Adv": "adv",
    "Aux": "auxiliary",
    "Cnj": "conj",
    "Id": "idiom",
    "Intj": "interj",
    "KT": "kinship term",
    "N": "noun",
    "Num": "num",
    "Pfx": "prefix",
    "Ppl": "participle",
    "Pron": "pron",
    "Rel": "relator",
    "RootVb": "verb root",
    "Sfx": "suffix",
    "V": "verb",
    "Vpfx": "verbal prefix",
    "Vsfx": "verbal suffix",
}

# The delimiter lookahead keeps ``T-111?`` out of the secure-reference set;
# it is captured separately and linked with an ``uncertain`` tag.
TURNER_REFERENCE = re.compile(r"\bT-\s*(\d+[a-z]?)(?=[^\w?]|$)")
UNCERTAIN_TURNER_REFERENCE = re.compile(r"\bT-\s*(\d+[a-z]?)\s*\?(?=[^\w]|$)")

DIALECT_IDS = {
    "Bumburet": "bumb",
    "Rumbur": "rumb",
    "Birir": "bir",
    "Urtsun": "urt",
}
LOAN_LANGUAGE_IDS = {
    "Arabic": "Ar",
    "Persian": "Pers",
    "Khowar": "Kho",
    "English": "Eng",
    "Urdu": "H",
    "Pashto": "Psht",
    "Kati": "SSNP-chitral-KAT",
    "Turkish": "Tr",
    "Turkic": "Turk",
}
STRUCTURED_LABELS = (
    "Able", "Actor", "Adult", "Anim", "Ant", "Caus", "Child", "Comp", "CPart",
    "Create", "Degrad", "Dim", "Do", "Female", "Future", "Gen", "Group", "Idiom",
    "Imp", "Inan", "Instr", "Loc", "Male", "Mat’l", "Max", "Object", "Onset", "Part",
    "Past", "Phase", "Pos", "Product", "Q", "Resident", "Reversive", "Rmot", "RootVb",
    "Seed", "Sequence", "Sound", "Spec", "Species", "Subadult", "Syn", "Unit", "Use",
    "Whole", "Morph", "Variant", "From", "Etym", "Prdm", "See also", "Restrict",
)
_FIELD_STOP = "|".join(re.escape(label) for label in sorted(STRUCTURED_LABELS, key=len, reverse=True))


@dataclass
class Entry:
    form: str
    part_of_speech: str
    pdf_page: int
    printed_page: int
    key: str = ""
    gloss: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)
    gloss_done: bool = False


def decode_legacy(text: str) -> str:
    """Convert SIL Doulos legacy keystrokes used in the PDF to Unicode."""
    # Multigraph consonants must be handled before the generic diacritic keys.
    text = (
        text.replace("ch4", "čh")
        .replace("c,4", "č̣")
        .replace("c4", "č")
        .replace("j$", "ǰ")
    )
    text = text.translate(
        str.maketrans(
            {
                "D": "ḍ",
                "J": "j̣",
                # The pronunciation guide identifies this glyph as the velar
                # nasal (English ``ng`` in *sing*), not a retroflex nasal.
                "K": "ŋ",
                "L": "ḷ",
                "R": "ṛ",
                "S": "ṣ",
                "T": "ṭ",
                "Z": "ẓ",
                "{": "ž",
                "}": "š",
                "Œ": "i",
                # A second legacy glyph for r occurs in a few headwords.
                "|": "r",
                "—": "-",
            }
        )
    )
    # These use a second legacy spelling in a few entries and examples.
    text = text.replace("s7", "ś").replace("z7", "ź")

    # In this old font the keystroke for a vowel diacritic can follow the
    # entire syllable even though the glyph is drawn over/under its vowel
    # (e.g. ``atik5`` is atík and ``citroy54ak`` is citrọ́yak). PDF extraction
    # retains keystroke order, so attach each mark to the most recent vowel.
    marks = {
        "‡": "\N{COMBINING ACUTE ACCENT}",
        "5": "\N{COMBINING ACUTE ACCENT}",
        "%": "\N{COMBINING ACUTE ACCENT}",
        "†": "\N{COMBINING TILDE}",
        "3": "\N{COMBINING TILDE}",
        "‚": "\N{COMBINING DOT BELOW}",
        "4": "\N{COMBINING DOT BELOW}",
    }
    decoded: list[str] = []
    last_vowel: int | None = None
    for char in text:
        if char in "aeiou":
            last_vowel = len(decoded)
            decoded.append(char)
        elif char in marks and last_vowel is not None:
            decomposed = unicodedata.normalize("NFD", decoded[last_vowel])
            if marks[char] not in decomposed:
                decoded[last_vowel] = decomposed + marks[char]
        else:
            decoded.append(char)
            if char.isspace() or char in ",;./()":
                last_vowel = None
    return unicodedata.normalize("NFC", "".join(decoded))


def _font(word: dict) -> str:
    return word.get("fontname", "").split("+")[-1]


def _lines(words: list[dict]) -> list[list[dict]]:
    """Cluster words sharing a visual baseline.

    Kalasha, English, and Urdu fonts have slightly different top coordinates,
    so pdfplumber's exact line grouping is insufficient here.
    """
    result: list[dict] = []
    for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
        center = (word["top"] + word["bottom"]) / 2
        match = None
        for line in reversed(result[-4:]):
            if abs(center - line["center"]) < 4.4:
                match = line
                break
        if match is None:
            result.append({"center": center, "words": [word]})
        else:
            match["words"].append(word)
            match["center"] = sum(
                (item["top"] + item["bottom"]) / 2 for item in match["words"]
            ) / len(match["words"])
    return [sorted(line["words"], key=lambda item: item["x0"]) for line in result]


def _entry_start(line: list[dict], column_start: float) -> tuple[str, str, int] | None:
    # The nominal 108/315 coordinates can arrive a tiny fraction below their
    # printed value because of PDF floating-point transforms.
    if not line or not (column_start - 1 <= line[0]["x0"] <= column_start + 18):
        return None
    if _font(line[0]) != "SILDoulosNPBold":
        return None

    pos_index = next(
        (
            i
            for i, word in enumerate(line)
            if _font(word) == "TimesNewRoman,Italic" and word["text"] in POS
        ),
        None,
    )
    if pos_index is None:
        # Cross-references, kinship terms, idioms, and a few particles omit a
        # conventional POS label. Their Urdu headword is still set in the
        # dedicated RamnaKLS font, which distinguishes them from examples.
        pos_index = next(
            (i for i, word in enumerate(line) if _font(word).startswith("RamnaKLS")),
            None,
        )
        if pos_index is None:
            return None
        part_of_speech = ""
        gloss_start = max(
            i for i, word in enumerate(line) if _font(word).startswith("RamnaKLS")
        )
    else:
        part_of_speech = line[pos_index]["text"].rstrip(".")
        gloss_start = pos_index

    headwords: list[dict] = []
    for word in line[:pos_index]:
        font = _font(word)
        if font == "SILDoulosNPBold":
            headwords.append(word)
        elif font.startswith("Ramna"):
            break
        elif font == "TimesNewRoman" and word.get("size", 10) < 8:
            continue  # superscript homonym number
        else:
            break
    if not headwords:
        return None
    # pdfplumber sometimes splits a word immediately after a zero-width legacy
    # diacritic.  A real inter-word space is about 4.3 PDF units; the false
    # split has overlapping/touching bounding boxes.
    raw_form = headwords[0]["text"]
    for previous, word in zip(headwords, headwords[1:]):
        separator = "" if word["x0"] - previous["x1"] < 2 else " "
        raw_form += separator + word["text"]
    return decode_legacy(raw_form), part_of_speech, gloss_start


def _add_gloss(entry: Entry, words: list[dict], start: int = 0) -> None:
    if entry.gloss_done:
        return
    for word in words[start:]:
        font = _font(word)
        token = word["text"]
        if font.startswith("Ramna"):
            entry.gloss_done = True
            break
        if font == "TimesNewRoman,Italic":
            entry.gloss_done = True
            break
        if font == "SILDoulosNPBold":
            entry.gloss_done = True
            break
        if font == "TimesNewRoman":
            entry.gloss.append(token)
            if ";" in token:
                entry.gloss_done = True
                break


def _clean_gloss(entry: Entry) -> str:
    gloss = " ".join(entry.gloss)
    gloss = re.sub(r"\s+([,.;:!?])", r"\1", gloss).strip()
    gloss = gloss.rstrip(";")
    return gloss


def _decoded_word(word: dict) -> str:
    """Decode only Kalasha-font tokens, leaving English labels/prose intact."""
    return decode_legacy(word["text"]) if _font(word).startswith("SILDoulos") else word["text"]


def _decoded_line(words: list[dict]) -> str:
    """Decode a visual line without introducing spaces at split diacritics."""
    if not words:
        return ""
    output = _decoded_word(words[0])
    for previous, word in zip(words, words[1:]):
        joined_kalasha = (
            _font(previous).startswith("SILDoulos")
            and _font(word).startswith("SILDoulos")
            and word["x0"] - previous["x1"] < 2
        )
        output += ("" if joined_kalasha else " ") + _decoded_word(word)
    return output


def _field(text: str, label: str) -> str:
    match = re.search(
        rf"(?<![A-Za-z]){re.escape(label)}:\s*(.*?)(?=(?:{_FIELD_STOP}):|$)",
        text,
    )
    if not match:
        return ""
    value = re.sub(r"\s+([,.;:!?])", r"\1", match.group(1)).strip()
    return value.rstrip("; ")


def _turner_ids(entry: Entry, valid_cdial: set[str]) -> tuple[list[str], list[str]]:
    text = " ".join(entry.text)
    secure = []
    for number in TURNER_REFERENCE.findall(text):
        if number in valid_cdial and number not in secure:
            secure.append(number)
    uncertain = [
        number for number in dict.fromkeys(UNCERTAIN_TURNER_REFERENCE.findall(text))
        if number in valid_cdial
    ]
    return secure, uncertain


def _row(
    entry: Entry,
    parameter_id: str,
    notes: str,
    *,
    language_id: str,
    form: str | None = None,
    key: str | None = None,
    variant_of_key: str = "",
    borrowed_from_key: str = "",
    etymology: str = "",
    derivation_parent_keys: str = "",
    tags: str = "",
) -> list[str]:
    return [
        language_id,
        parameter_id,
        form or entry.form,
        _clean_gloss(entry),
        "",
        "",
        notes,
        SOURCE_ID,
        "",  # Cognateset
        etymology,
        key or entry.key,
        variant_of_key,
        borrowed_from_key,
        derivation_parent_keys,
        tags,
    ]


def _finish(
    entry: Entry | None,
    rows: list[list[str]],
    valid_cdial: set[str],
    metadata: dict | None = None,
) -> None:
    if entry is None:
        return
    ids, uncertain_ids = _turner_ids(entry, valid_cdial)
    metadata = metadata or {}
    pos_note = POS_NOTES.get(entry.part_of_speech, entry.part_of_speech)
    base_note = (
        (f"{pos_note}; " if pos_note else "")
        + f"Trail & Cooper PDF p. {entry.pdf_page} "
        + f"(printed p. {entry.printed_page})"
    )

    details = metadata.get("details", [])
    if details:
        base_note += "; " + "; ".join(details)

    def append_row(parameter_id: str, notes: str) -> None:
        rows.append(_row(
            entry,
            parameter_id,
            notes,
            language_id=metadata.get("language_id", DEFAULT_DIALECT_ID),
            borrowed_from_key=metadata.get("borrowed_from_key", ""),
            etymology=metadata.get("etymology", ""),
            derivation_parent_keys="|".join(metadata.get("parents", [])),
            tags=" ".join(metadata.get("tags", [])),
        ))

    for number in ids:
        append_row(number, f"{base_note}; Turner etymology T-{number}")

    # Preserve uncertain Turner citations as qualified Origin_ID edges. Each
    # candidate gets its own row because Parameter_ID is scalar.
    if uncertain_ids:
        for number in uncertain_ids:
            append_row(
                number,
                f"uncertain; {base_note}; uncertain Turner etymology T-{number}?",
            )
    elif not ids:
        append_row("", base_note)


def _head_forms(form: str) -> list[str]:
    return [part.strip() for part in form.split(",") if part.strip()]


def _form_key(form: str) -> str:
    return unicodedata.normalize("NFC", form).casefold().strip(" -.,;:")


def _first_related_form(value: str) -> str:
    value = value.split("‘", 1)[0].split("“", 1)[0]
    value = re.split(r"[.;]", value, maxsplit=1)[0]
    return value.strip(" -,:[]")


def _metadata(entries: list[Entry]) -> dict[str, dict]:
    """Parse source-local morphology, etymology, loans, and relation targets."""
    lookup: dict[str, str] = {}
    for entry in entries:
        for form in _head_forms(entry.form):
            lookup.setdefault(_form_key(form), entry.key)

    result: dict[str, dict] = {}
    for entry in entries:
        text = " ".join(entry.text)
        morph = _field(text, "Morph")
        paradigm = _field(text, "Prdm")
        variants = _field(text, "Variant")
        source = _field(text, "From")
        etym = _field(text, "Etym")
        root = _field(text, "RootVb")
        caus = _field(text, "Caus")
        comp = _field(text, "Comp")
        details = []
        tags = []
        for label, value in (
            ("Morph", morph), ("Prdm", paradigm), ("Variant", variants),
            ("From", source), ("RootVb", root), ("Caus", caus), ("Comp", comp),
        ):
            if value:
                details.append(f"{label}: {value}")
        if morph:
            tags.append("morphology")
        if paradigm:
            match = re.search(r"\bClass\s+([1-4])\b", paradigm)
            if match:
                tags.append(f"Kalasha-class-{match.group(1)}")
        loan_name = next((name for name in LOAN_LANGUAGE_IDS if re.match(rf"{name}\b", source)), "")
        borrowed_from_key = f"tc-loan-{loan_name}" if loan_name else ""
        if loan_name:
            tags.append(f"loan:{loan_name}")
        language_id = DEFAULT_DIALECT_ID
        dialect_definition = re.search(r"\((Bumburet|Rumbur|Birir|Urtsun) dialect\)", text)
        if dialect_definition:
            language_id = DIALECT_IDS[dialect_definition.group(1)]

        parents = []
        # RootVb names the parent of this entry.
        if root:
            parent = lookup.get(_form_key(_first_related_form(root)))
            if parent and parent != entry.key:
                parents.append(parent)
        # A compositional Morph analysis contributes all resolvable components;
        # a single undivided form is an underlying representation, not derivation.
        if morph and ("-" in morph or " " in morph):
            for piece in re.split(r"[-+\s]+", re.sub(r"[()]", " ", morph)):
                parent = lookup.get(_form_key(piece))
                if parent and parent != entry.key and parent not in parents:
                    parents.append(parent)
        elif morph:
            tags.append("underlying-form")

        result[entry.key] = {
            "details": details,
            "tags": tags,
            "etymology": f"Trail & Cooper Etym: {etym}" if etym else "",
            "borrowed_from_key": borrowed_from_key,
            "language_id": language_id,
            "parents": parents,
            "caus": _first_related_form(caus) if caus else "",
            "comp": _first_related_form(comp) if comp else "",
        }

    # Caus and Comp point from the base entry to a derived/complex target.
    for entry in entries:
        base = result[entry.key]
        for relation in ("caus", "comp"):
            target_key = lookup.get(_form_key(base.get(relation, "")))
            if target_key and target_key != entry.key:
                parents = result[target_key]["parents"]
                if entry.key not in parents:
                    parents.append(entry.key)
                tag = "causative" if relation == "caus" else "compound"
                if tag not in result[target_key]["tags"]:
                    result[target_key]["tags"].append(tag)
    return result


def _dialect_variants(
    entry: Entry,
    valid_cdial: set[str],
    metadata: dict,
) -> list[list[str]]:
    variants = _field(" ".join(entry.text), "Variant")
    if not variants:
        return []
    ids, uncertain = _turner_ids(entry, valid_cdial)
    params = [(number, False) for number in ids] + [(number, True) for number in uncertain]
    if not params:
        params = [("", False)]
    output = []
    index = 0
    for segment in variants.split(";"):
        dialects = [name for name in DIALECT_IDS if re.search(rf"\b{name}\b", segment)]
        # Parenthetical (M) forms quote Morgenstierne's separate transcription
        # and need that source's own sound profile, not Trail orthography.
        if not dialects or re.search(r"\(M\)", segment):
            continue
        form = segment.split("(", 1)[0].rsplit(".", 1)[-1]
        form = re.sub(r"^Variant:\s*", "", form).strip(" .,:;[]")
        # Reject residual legacy/control glyphs rather than publishing a form
        # the Kalasha profile cannot interpret.
        if (
            not form
            or _form_key(form) == _form_key(entry.form)
            or re.search(r"[0-9A-Z]", form)
        ):
            continue
        for dialect in dialects:
            index += 1
            for parameter_id, is_uncertain in params:
                note = (
                    ("uncertain; " if is_uncertain else "")
                    + f"dialect variant of {entry.form}; {dialect}; Trail & Cooper PDF p. "
                    f"{entry.pdf_page} (printed p. {entry.printed_page}); Variant: {variants}"
                )
                output.append(_row(
                    entry,
                    parameter_id,
                    note,
                    language_id=DIALECT_IDS[dialect],
                    form=form,
                    key=f"{entry.key}-variant-{index}",
                    variant_of_key=entry.key,
                    borrowed_from_key=metadata.get("borrowed_from_key", ""),
                    etymology=metadata.get("etymology", ""),
                    tags="dialect-variant" + (" uncertain" if is_uncertain else ""),
                ))
    return output


def read_cdial_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8", newline="") as stream:
        return {row[0] for row in csv.reader(stream) if row and re.fullmatch(r"\d+[a-z]?", row[0])}


def extract(pdf_path: Path, valid_cdial: set[str]) -> list[list[str]]:
    entries: list[Entry] = []
    current: Entry | None = None
    with pdfplumber.open(pdf_path) as pdf:
        for pdf_page in range(FIRST_DICTIONARY_PAGE, LAST_DICTIONARY_PAGE + 1):
            page = pdf.pages[pdf_page - 1]
            all_words = page.extract_words(extra_attrs=["fontname", "size"])
            # Reading order is left column, right column. Main entries begin at
            # x=108/315; subentries at x=115.2/322.2.
            for low, high in ((100, 306), (306, 560)):
                words = [
                    word
                    for word in all_words
                    if low <= word["x0"] < high and 120 <= word["top"] <= 670
                ]
                for line in _lines(words):
                    column_start = 108 if low == 100 else 315
                    start = _entry_start(line, column_start)
                    if start:
                        if current is not None:
                            entries.append(current)
                        form, pos, pos_index = start
                        current = Entry(
                            form, pos, pdf_page, pdf_page - 30,
                            key=f"tc-entry-{len(entries) + 1}",
                        )
                        current.text.append(_decoded_line(line))
                        _add_gloss(current, line, pos_index + 1)
                    elif current is not None:
                        current.text.append(_decoded_line(line))
                        _add_gloss(current, line)
    if current is not None:
        entries.append(current)

    metadata = _metadata(entries)
    rows: list[list[str]] = []
    loan_names = sorted({
        key.removeprefix("tc-loan-")
        for item in metadata.values()
        if (key := item.get("borrowed_from_key", ""))
    })
    # The source gives the donor language but usually no donor lexeme. One
    # explicit category node per language makes the borrowing relation
    # navigable without fabricating an unattested donor word for every loan.
    for loan_name in loan_names:
        language_id = LOAN_LANGUAGE_IDS[loan_name]
        placeholder = Entry(
            loan_name,
            "N",
            18,
            -12,
            key=f"tc-loan-{loan_name}",
            gloss=[f"loan-source category for Kalasha ({loan_name}); donor lexeme unspecified"],
        )
        rows.append(_row(
            placeholder,
            "",
            "Trail & Cooper loan-source category; exact donor forms are not supplied",
            language_id=language_id,
            tags="loan-source",
        ))

    for entry in entries:
        item = metadata[entry.key]
        _finish(entry, rows, valid_cdial, item)
        rows.extend(_dialect_variants(entry, valid_cdial, item))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument(
        "--cdial-params",
        type=Path,
        default=Path("data/cdial/params.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/other/forms/20260725-kalasha-trail-cooper.csv"),
    )
    args = parser.parse_args()
    rows = extract(args.pdf, read_cdial_ids(args.cdial_params))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    linked = sum(bool(row[1]) for row in rows)
    uncertain = sum(row[6].startswith("uncertain;") for row in rows)
    print(
        f"wrote {args.output} ({len(rows)} rows: {linked} linked, "
        f"{len(rows) - linked} unlinked, {uncertain} uncertain)"
    )


if __name__ == "__main__":
    main()

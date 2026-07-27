#!/usr/bin/env python3
"""Parse the remaining SSNP appendix wordlists.

The files in this directory are plain-text extractions of several volumes of
the *Sociolinguistic Survey of Northern Pakistan*.  Five use three-concept
table blocks; Ushojo uses one numbered entry per line and then an unnumbered
supplement.  PDF text extraction also moved superscript letters onto following
lines and, occasionally, joined a following location code to the preceding
form.

The appendices use a legacy SIL phonetic font. PDF extraction exposes its
keystrokes (for example ``V`` for /ʌ/, ``K`` for /ŋ/, and a following ``3``
for nasalisation), not the IPA glyphs visible on the page.  This script decodes
those keystrokes to Unicode IPA while retaining the untouched extraction in
``ssnp_wordlists.csv`` for audit. Rows whose table boundary cannot be
determined uniquely are retained and marked ``heuristic`` in the audit table.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_BY_FILE = {
    "chitral": "decker1992",
    "hindko": "rensch-hallberg-oleary1992",
    "gojri": "rensch-hallberg-oleary1992",
    "kohistani": "rensch-decker-hallberg1992",
    "indus kohistani": "rensch-decker-hallberg1992",
    "ushojo": "rensch-decker-hallberg1992",
}

TABLE_SOURCES = {
    "chitral": "KSW KIS KPN KTR KGC KDR BBK BRK URK ASP BIP PUP SSS GWB DML SHK KAT YDG MNJ".split(),
    "hindko": "JAM BAL SIG PES SHP MAN KOH ATK PAG WAP DIK TAL".split(),
    "kohistani": "TOR CHL DSH KAL USH THL LAM RAJ KLK".split(),
    "gojri": "KUN CHT DIR SSW TSW GLT KGH SHZ NAK CAK SAK GJW IND".split(),
    "indus kohistani": "KAN DUB SEO PAT JIJ CHM CHJ GAB BAT".split(),
}

PAGE_LINE = re.compile(
    r"^(?:\d+\s+)?(?:APPENDIX\b|Appendix\b|HINDKO SURVEY DATA\b|"
    r"[BD]\.1\s+(?:Hindko|Swat Kohistan|Indus Kohistan)\s+Word Lists\b|"
    r"\d+\s*$|.*\bSurvey Data\s*$|Hindko Word Lists\s*$|Location Code,)", re.I
)
HEADER_ITEM = re.compile(r"(\d{1,3})\.\s*(.*?)(?=\s+\d{1,3}\.\s*|$)")
NUMBERED = re.compile(r"^\s*(\d{1,3})\.\s*(.*)$")

# Unlike the numbered list, the short Ushojo supplement has no delimiter
# between an English multiword gloss and its form. These are the multiword
# headings occurring in that appendix; all other headings occupy one token
# (plus an immediately following parenthetical usage label).
USH_MULTIWORD_GLOSSES = {
    "fortnight; half", "funeral bier", "handle of ax", "internal organs",
    "lame, dwarf", "pine cone", "post, pillar", "sheep (fat tail)",
    "shin bone", "sieve (for grain)", "skin bag", "son’s wife",
    "sticks on rafters", "stool (small)", "sugar cane", "tree (cedar)",
    "waterchute (mill)", "web; nest", "wife’s brother", "wife’s mother",
}


@dataclass
class Entry:
    source_file: str
    location: str
    number: str
    gloss: str
    form: str
    split: str
    raw: str


def decode_legacy(text: str) -> str:
    """Decode the SIL-font text layer to the IPA displayed in the PDF.

    Diacritic keystrokes occur after the whole syllable in this encoding, so
    they must be moved back to its latest vowel. This is the same convention
    used by the Trail--Cooper legacy decoder in this repository, with the
    additional vowel glyphs found in the SSNP reprints.
    """
    # PDF extraction sometimes puts a zero-width vowel mark in its own
    # whitespace-delimited fragment. It still belongs to the preceding vowel.
    text = re.sub(r"\s+([3458†‡‚ë])", r"\1", text)
    text = (
        text.replace("r†", "ɽ")
        .replace("t_s", "t͡s")
        .replace("C7", "t͡ɕ")
        .replace("c7", "t͡ɕ")
        .replace("S7", "ʂ")
        .replace("s7", "ɕ")
        .replace("Z7", "ʐ")
        .replace("z7", "ʑ")
    )
    text = text.translate(str.maketrans({
        "à": "ɑ", "V": "ʌ", "F": "ə", "I": "ɪ", "U": "ʊ",
        "E": "ɛ", "O": "ɔ", "Q": "æ",
        "B": "b", "C": "t͡ʃ", "D": "ɖ", "G": "x", "J": "d͡ʒ",
        "K": "ŋ", "L": "ɫ", "M": "ɬ", "N": "ɳ", "P": "f",
        "R": "ɽ", "S": "ʂ", "T": "ʈ", "Z": "ʐ",
        "{": "ʒ", "}": "ʃ", "Œ": "i", "—": "-",
        "`": "ʔ", "0": "", "ƒ": "", "½": "", "7": "",
    }))

    marks = {
        "3": "\N{COMBINING TILDE}",
        "†": "\N{COMBINING TILDE}",
        "4": "\N{COMBINING DOT BELOW}",
        "‚": "\N{COMBINING DOT BELOW}",
        "5": "\N{COMBINING ACUTE ACCENT}",
        "8": "\N{COMBINING ACUTE ACCENT}",
        "‡": "\N{COMBINING ACUTE ACCENT}",
        "ë": "ː",
    }
    decoded: list[str] = []
    last_vowel: int | None = None
    vowels = "aeiouɑʌəɪʊɛɔæéú"
    for char in text:
        if char in vowels:
            last_vowel = len(decoded)
            decoded.append(char)
        elif char in marks:
            if last_vowel is not None:
                mark = marks[char]
                if mark == "ː":
                    if not decoded or decoded[-1] != mark:
                        decoded.append(mark)
                else:
                    base = unicodedata.normalize("NFD", decoded[last_vowel])
                    if mark not in base:
                        decoded[last_vowel] = base + mark
        else:
            decoded.append(char)
            if char.isspace() or char in ",;./()":
                last_vowel = None
    return unicodedata.normalize("NFC", "".join(decoded)).strip()


def header_items(line: str) -> list[tuple[str, str]]:
    """Extract numbered concepts from a table heading."""
    return [(number, gloss.strip()) for number, gloss in HEADER_ITEM.findall(line)]


def logical_lines(text: str, codes: list[str], source: str) -> list[str]:
    """Separate location rows which PDF extraction accidentally concatenated."""
    alternatives = "|".join(map(re.escape, codes))
    if source == "gojri":
        # In this extract a lost newline is represented by the preceding
        # aspiration glyph. Requiring it avoids mistaking bEKUN for code KUN.
        marker = re.compile(rf"(?<=ʰ)(?=(?:{alternatives})\s)")
    elif source == "chitral":
        # Several late Chitral tables lost every row break on a page.
        marker = re.compile(rf"(?<!^)(?=(?:{alternatives})\s)")
    else:
        marker = None
    result = []
    for physical in text.splitlines():
        parts = marker.split(physical.rstrip()) if marker else [physical.rstrip()]
        for part in parts:
            if part.strip():
                result.append(part)
    return result


def join_fragments(parts: list[str]) -> str:
    """Rejoin glyph fragments displaced to a continuation line.

    In these extracts an aspirated cluster commonly ends one physical line in
    ``h`` and its vowel starts the next (``ph`` + ``Ur``).  Only line-boundary
    pairs are joined; ordinary spaces in the source remain meaningful.
    """
    tokens: list[str] = []
    for part in parts:
        new = part.strip().split()
        if tokens and new == ["h"]:
            tokens[-1] += "h"
            continue
        if tokens and new and tokens[-1].endswith("h") and re.match(r"^[AEIOUVQaeiouàáë]", new[0]):
            tokens[-1] += new.pop(0)
        tokens.extend(new)
    return " ".join(tokens)


def slash_atoms(text: str) -> list[str]:
    """Treat ``x / y`` as one indivisible alternative-form atom."""
    tokens = text.split()
    atoms: list[str] = []
    index = 0
    while index < len(tokens):
        atom = [tokens[index]]
        index += 1
        while index + 1 < len(tokens) and tokens[index] == "/":
            atom.extend(tokens[index : index + 2])
            index += 2
        atoms.append(" ".join(atom))
    return atoms


def split_cells(text: str, count: int) -> tuple[list[str], str]:
    """Split a row into concept cells, conservatively flagging ambiguity."""
    atoms = slash_atoms(text)
    if len(atoms) == count:
        return atoms, "exact"
    if len(atoms) < count:
        return atoms + [""] * (count - len(atoms)), "short"

    # Extra atoms are most often multiword responses in the first column.  A
    # right-anchored split preserves the two usually single-word later cells.
    cells = [" ".join(atoms[: len(atoms) - count + 1]), *atoms[-(count - 1) :]]
    return cells, "heuristic"


def parse_table(name: str, path: Path, codes: list[str]) -> list[Entry]:
    code_re = re.compile(rf"^\s*({'|'.join(map(re.escape, codes))})\s+(.*)$")
    entries: list[Entry] = []
    concepts: list[tuple[str, str]] = []
    current_code: str | None = None
    fragments: list[str] = []
    pending_header = ""

    def flush() -> None:
        nonlocal current_code, fragments
        if current_code is None or not concepts:
            current_code, fragments = None, []
            return
        raw = " | ".join(part.strip() for part in fragments)
        payload = join_fragments(fragments)
        cells, status = split_cells(payload, len(concepts))
        for (number, gloss), form in zip(concepts, cells):
            if form and form not in {"---", "----"}:
                entries.append(Entry(name, current_code, number, gloss, form, status, raw))
        current_code, fragments = None, []

    for line in logical_lines(path.read_text(encoding="utf-8"), codes, name):
        if pending_header:
            pending_header += " " + line.strip()
            items = header_items(pending_header)
            if len(items) >= 2:
                concepts = items
                pending_header = ""
            continue
        items = header_items(line)
        # A real table heading has at least two concepts. This excludes prose
        # and the numbered page furniture found in some extracts.
        if len(items) >= 2:
            flush()
            concepts = items
            continue
        # Some three-column headings wrap after their first numbered item.
        # Do not append that heading (and its English continuation) to the
        # final location row of the preceding table.
        if NUMBERED.match(line):
            flush()
            pending_header = line.strip()
            concepts = []
            continue
        match = code_re.match(line)
        if match:
            flush()
            current_code, payload = match.groups()
            fragments = [payload]
        elif current_code is not None and not PAGE_LINE.match(line.strip()):
            fragments.append(line)
    flush()
    return entries


def concept_glosses() -> dict[int, list[str]]:
    """Collect source spellings used to split Ushojo's inline records."""
    result: dict[int, set[str]] = {}
    for name in TABLE_SOURCES:
        for line in (HERE / name).read_text(encoding="utf-8").splitlines():
            for number, gloss in header_items(line):
                result.setdefault(int(number), set()).add(gloss)
    return {number: sorted(values, key=len, reverse=True) for number, values in result.items()}


def split_inline(number: int, text: str, glosses: dict[int, list[str]]) -> tuple[str, str]:
    for gloss in glosses.get(number, []):
        if text == gloss:
            return gloss, ""
        if text.startswith(gloss + " "):
            return gloss, text[len(gloss) + 1 :]
    # All numbered concepts occur in the table inventories. Keep an explicit
    # fallback so a future source revision fails visibly rather than crashing.
    first, _, rest = text.partition(" ")
    return first, rest


def parse_ushojo(path: Path) -> list[Entry]:
    glosses = concept_glosses()
    entries: list[Entry] = []
    current: tuple[int, str] | None = None
    continuations: list[str] = []

    def flush() -> None:
        nonlocal current, continuations
        if current is None:
            return
        number, text = current
        gloss, first_form = split_inline(number, text, glosses)
        form = join_fragments([first_form, *continuations])
        if form and form not in {"---", "----"}:
            raw = " | ".join([text, *continuations])
            entries.append(Entry("ushojo", "USH", str(number), gloss, form, "inline", raw))
        current, continuations = None, []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = NUMBERED.match(line)
        if match:
            flush()
            current = (int(match.group(1)), match.group(2).strip())
        elif current is not None:
            if PAGE_LINE.match(line.strip()):
                continue
            # The unnumbered alphabetical supplement begins after item 210.
            # It is intentionally not guessed apart here: without column
            # geometry, plain strings such as "snow him" are underdetermined.
            if current[0] == 210:
                flush()
                break
            continuations.append(line)
    flush()
    entries.extend(parse_ushojo_supplement(path))
    return entries


def parse_ushojo_supplement(path: Path) -> list[Entry]:
    """Parse the alphabetized, unnumbered entries following item 210."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("210.")) + 1
    records: list[list[str]] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or PAGE_LINE.match(stripped):
            continue
        continuation = line[:1].isspace()
        if records and not continuation:
            previous_last = records[-1][-1].split()[-1]
            words = stripped.split()
            continuation = (
                (previous_last.endswith("h") and len(words) == 1 and re.match(r"^[AEIOUVQaeiouàáë]", words[0]))
                or words == ["h"]
            )
        if continuation and records:
            records[-1].append(stripped)
        else:
            records.append([stripped])

    entries = []
    for index, fragments in enumerate(records, 1):
        first = fragments[0]
        gloss = next(
            (candidate for candidate in sorted(USH_MULTIWORD_GLOSSES, key=len, reverse=True)
             if first.startswith(candidate + " ")),
            "",
        )
        if gloss:
            first_form = first[len(gloss) + 1 :]
        else:
            words = first.split()
            gloss_words = [words.pop(0)]
            while words and words[0].startswith("(") and words[0].endswith(")"):
                gloss_words.append(words.pop(0))
            gloss, first_form = " ".join(gloss_words), " ".join(words)
        form = join_fragments([first_form, *fragments[1:]])
        entries.append(Entry(
            "ushojo", "USH", f"supp-{index:03d}", gloss, form,
            "supplement", " | ".join(fragments),
        ))
    return entries


def extract() -> list[Entry]:
    entries = []
    for name, codes in TABLE_SOURCES.items():
        entries.extend(parse_table(name, HERE / name, codes))
    entries.extend(parse_ushojo(HERE / "ushojo"))
    return entries


def language_id(entry: Entry) -> str:
    """Namespace survey codes, which are only unique within an appendix."""
    source = re.sub(r"[^a-z0-9]+", "-", entry.source_file.lower()).strip("-")
    return f"SSNP-{source}-{entry.location}"


def write(entries: list[Entry], audit_path: Path, forms_path: Path) -> None:
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Source_File", "Location_Code", "Concept_Number", "Gloss", "Form", "Split", "Raw"])
        for entry in entries:
            writer.writerow(entry.__dict__.values())

    with forms_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for entry in entries:
            notes = f"SSNP {entry.source_file}, location {entry.location}, item {entry.number}"
            if entry.split in {"heuristic", "short"}:
                notes += f"; {entry.split} automatic table split"
            ipa = decode_legacy(entry.form)
            if ipa:
                writer.writerow([
                    language_id(entry), "", ipa, entry.gloss, "", ipa, notes,
                    SOURCE_BY_FILE[entry.source_file],
                ])


def update_language_coordinates(language_path: Path, coordinate_path: Path) -> None:
    """Apply audited SSNP coordinates without normalizing unrelated CSV lines."""
    with coordinate_path.open(encoding="utf-8") as handle:
        coordinates = {row["Language_ID"]: row for row in csv.DictReader(handle)}
    with language_path.open(encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    if not lines:
        raise ValueError(f"missing header in {language_path}")

    def split_ending(line: str) -> tuple[str, str]:
        if line.endswith("\r\n"):
            return line[:-2], "\r\n"
        if line.endswith(("\r", "\n")):
            return line[:-1], line[-1]
        return line, ""

    header, _ = split_ending(lines[0])
    fieldnames = next(csv.reader([header]))
    seen = set()
    output = [lines[0]]
    for line in lines[1:]:
        body, ending = split_ending(line)
        values = next(csv.reader([body]))
        row = dict.fromkeys(fieldnames, "")
        row.update(zip(fieldnames, values))
        if row["ID"] in coordinates:
            row["Latitude"] = coordinates[row["ID"]]["Latitude"]
            row["Longitude"] = coordinates[row["ID"]]["Longitude"]
            seen.add(row["ID"])
            buffer = io.StringIO(newline="")
            csv.DictWriter(
                buffer, fieldnames=fieldnames, lineterminator=ending or "\n"
            ).writerow(row)
            line = buffer.getvalue()
        output.append(line)
    missing = coordinates.keys() - seen
    if missing:
        raise ValueError(f"coordinate IDs missing from LanguageTable: {sorted(missing)}")
    with language_path.open("w", newline="", encoding="utf-8") as handle:
        handle.writelines(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=HERE / "ssnp_wordlists.csv")
    parser.add_argument(
        "--forms", type=Path, default=HERE.parent / "20260725-ssnp.csv",
        help="forms import consumed by make_cldf.py",
    )
    parser.add_argument(
        "--languages", type=Path, default=HERE.parents[3] / "cldf/languages.csv",
        help="CLDF LanguageTable whose SSNP coordinates should be refreshed",
    )
    parser.add_argument(
        "--coordinates", type=Path, default=HERE / "ssnp_locations.csv",
        help="audited SSNP locality coordinates",
    )
    args = parser.parse_args()
    entries = extract()
    write(entries, args.audit, args.forms)
    update_language_coordinates(args.languages, args.coordinates)
    print(f"wrote {len(entries):,} forms to {args.forms}")


if __name__ == "__main__":
    main()

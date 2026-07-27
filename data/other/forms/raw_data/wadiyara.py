#!/usr/bin/env python3
"""Extract Appendix A (the 1,650-item Wadiyari wordlist) from Zubair's PDF.

The PDF's embedded ToUnicode table maps many perfectly valid Charis SIL glyphs
to spaces.  In particular, ordinary PDF text extraction loses most nasal
vowels.  The embedded TrueType font is sound, however, and its glyph IDs are
the PDF's character IDs.  This script rebuilds the Unicode map from the font's
``cmap`` table before using the stable table geometry to extract the appendix.

Two CSV files are written:

* ``raw_data/wadiyara_wordlist.csv`` keeps one row per numbered prompt and the
  three LRP columns, including hyphens for missing responses.
* ``../20230306-wadiyara.csv`` is the repository's eight-column forms import.
  Identical responses to a prompt are deduplicated. Existing manual CDIAL IDs
  and review notes are retained by exact-form matching or conservative
  within-gloss nearest-form matching for transcription corrections.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import re
import struct
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

import pdfplumber
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


FIRST_PDF_PAGE = 125  # printed page 108
LAST_ITEM = 1650
SOURCE = "zubair"

# Four prompts are split into lettered subitems in the source. One item number
# is visibly mistyped as 8092; its neighbours make the intended 0892 certain.
NUMBER_CORRECTIONS = {"666a": "0666a", "666b": "0666b", "8092": "0892"}

# Boundaries halfway through the gutters between the five printed columns.
COLUMNS = {
    "gloss": (145.0, 280.0),
    "lrp1": (280.0, 355.0),
    "lrp2": (355.0, 436.0),
    "lrp3": (436.0, 540.0),
}


def _ttf_tables(font: bytes) -> dict[bytes, tuple[int, int]]:
    count = struct.unpack_from(">H", font, 4)[0]
    tables = {}
    for index in range(count):
        tag, _checksum, offset, length = struct.unpack_from(
            ">4sIII", font, 12 + 16 * index
        )
        tables[tag] = (offset, length)
    return tables


def _format_4(font: bytes, offset: int) -> dict[int, int]:
    """Return Unicode code point -> glyph ID for a TrueType format-4 cmap."""
    segment_count = struct.unpack_from(">H", font, offset + 6)[0] // 2
    end_offset = offset + 14
    ends = struct.unpack_from(f">{segment_count}H", font, end_offset)
    start_offset = end_offset + 2 * segment_count + 2
    starts = struct.unpack_from(f">{segment_count}H", font, start_offset)
    delta_offset = start_offset + 2 * segment_count
    deltas = struct.unpack_from(f">{segment_count}h", font, delta_offset)
    range_offset = delta_offset + 2 * segment_count
    ranges = struct.unpack_from(f">{segment_count}H", font, range_offset)

    result = {}
    for index, (start, end, delta, glyph_range) in enumerate(
        zip(starts, ends, deltas, ranges)
    ):
        for codepoint in range(start, end + 1):
            if codepoint == 0xFFFF:
                continue
            if glyph_range == 0:
                glyph = (codepoint + delta) & 0xFFFF
            else:
                address = (
                    range_offset + 2 * index + glyph_range + 2 * (codepoint - start)
                )
                glyph = struct.unpack_from(">H", font, address)[0]
                if glyph:
                    glyph = (glyph + delta) & 0xFFFF
            if glyph:
                result[codepoint] = glyph
    return result


def _format_12(font: bytes, offset: int) -> dict[int, int]:
    """Return Unicode code point -> glyph ID for a TrueType format-12 cmap."""
    group_count = struct.unpack_from(">I", font, offset + 12)[0]
    result = {}
    for index in range(group_count):
        start, end, first_glyph = struct.unpack_from(
            ">III", font, offset + 16 + 12 * index
        )
        for codepoint in range(start, end + 1):
            result[codepoint] = first_glyph + codepoint - start
    return result


def _glyph_to_unicode(font: bytes) -> dict[int, int]:
    """Invert the richest Unicode cmap in an embedded TrueType font."""
    cmap_offset, _length = _ttf_tables(font)[b"cmap"]
    subtable_count = struct.unpack_from(">H", font, cmap_offset + 2)[0]
    candidates = []
    for index in range(subtable_count):
        platform, encoding, relative = struct.unpack_from(
            ">HHI", font, cmap_offset + 4 + 8 * index
        )
        offset = cmap_offset + relative
        fmt = struct.unpack_from(">H", font, offset)[0]
        if fmt in {4, 12} and platform in {0, 3}:
            # Prefer Microsoft's full-Unicode format 12, then Unicode-platform
            # maps, then the BMP-only Microsoft map.
            score = (fmt == 12, platform == 0, encoding == 10)
            candidates.append((score, fmt, offset))
    if not candidates:
        raise ValueError("embedded TrueType font has no usable Unicode cmap")
    _score, fmt, offset = max(candidates)
    forward = _format_12(font, offset) if fmt == 12 else _format_4(font, offset)

    # A subset font can expose aliases for a glyph. Prefer IPA and Latin
    # letters/marks over compatibility or private-use code points.
    def preference(codepoint: int) -> tuple[int, int]:
        category = unicodedata.category(chr(codepoint))
        useful = (
            codepoint < 0x3000
            and not 0xE000 <= codepoint <= 0xF8FF
            and category not in {"Cc", "Cf", "Cs", "Co", "Cn"}
        )
        return (useful, -codepoint)

    reverse: dict[int, int] = {}
    for codepoint, glyph in forward.items():
        if glyph not in reverse or preference(codepoint) > preference(reverse[glyph]):
            reverse[glyph] = codepoint
    return reverse


def _unicode_cmap(glyphs: dict[int, int]) -> bytes:
    mappings = []
    for glyph, codepoint in sorted(glyphs.items()):
        if glyph > 0xFFFF or not (0 <= codepoint <= 0x10FFFF):
            continue
        destination = chr(codepoint).encode("utf-16-be").hex().upper()
        mappings.append(f"<{glyph:04X}> <{destination}>")

    chunks = []
    for start in range(0, len(mappings), 100):
        group = mappings[start : start + 100]
        chunks.extend([f"{len(group)} beginbfchar", *group, "endbfchar"])
    body = "\n".join(chunks)
    return (
        "/CIDInit /ProcSet findresource begin\n"
        "12 dict begin\n"
        "begincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n"
        "/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"{body}\n"
        "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    ).encode("ascii")


def _read_unicode_cmap(data: bytes) -> dict[int, int]:
    """Read the simple bfchar/bfrange forms emitted by Microsoft Word."""
    result = {}

    def destination(value: str) -> int:
        text = bytes.fromhex(value).decode("utf-16-be")
        return ord(text) if len(text) == 1 else ord(text[0])

    for raw_line in data.decode("latin-1").splitlines():
        line = raw_line.strip()
        pair = re.fullmatch(r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>", line)
        if pair:
            result[int(pair.group(1), 16)] = destination(pair.group(2))
            continue
        direct = re.fullmatch(
            r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>",
            line,
        )
        if direct:
            start, end = int(direct.group(1), 16), int(direct.group(2), 16)
            first = destination(direct.group(3))
            for code in range(start, end + 1):
                result[code] = first + code - start
            continue
        array = re.fullmatch(
            r"<([0-9A-Fa-f]+)>\s+<([0-9A-Fa-f]+)>\s+\[(.*)\]", line
        )
        if array:
            start, end = int(array.group(1), 16), int(array.group(2), 16)
            values = re.findall(r"<([0-9A-Fa-f]+)>", array.group(3))
            if len(values) == end - start + 1:
                for offset, value in enumerate(values):
                    result[start + offset] = destination(value)
    return result


def repair_unicode_maps(source: Path, destination: Path) -> None:
    """Write a temporary PDF whose Type0 font maps reflect its embedded TTF."""
    reader = PdfReader(source)
    repaired = set()
    for page in reader.pages[FIRST_PDF_PAGE - 1 :]:
        fonts = page.get("/Resources", {}).get("/Font", {})
        for reference in fonts.values():
            font = reference.get_object()
            identity = font.indirect_reference.idnum if font.indirect_reference else id(font)
            if identity in repaired or font.get("/Subtype") != "/Type0":
                continue
            descendants = font.get("/DescendantFonts") or []
            if not descendants:
                continue
            descendant = descendants[0].get_object()
            descriptor = descendant.get("/FontDescriptor")
            descriptor = descriptor.get_object() if descriptor else None
            embedded = descriptor.get("/FontFile2") if descriptor else None
            if not embedded or descendant.get("/CIDToGIDMap") != "/Identity":
                continue
            glyphs = _glyph_to_unicode(embedded.get_object().get_data())
            current = font.get("/ToUnicode")
            if current:
                # Word's map is accurate for ordinary punctuation and letters;
                # only its numerous mappings to U+0020 are corrupt. Keep every
                # informative original mapping and repair the space-mapped CIDs
                # from the embedded font.
                existing = _read_unicode_cmap(current.get_object().get_data())
                glyphs.update(
                    {code: value for code, value in existing.items() if value != 0x20}
                )
            stream = DecodedStreamObject()
            stream.set_data(_unicode_cmap(glyphs))
            font[NameObject("/ToUnicode")] = stream
            repaired.add(identity)

    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    with destination.open("wb") as output:
        writer.write(output)


def _line_text(chars: list[dict]) -> str:
    """Rebuild a field from positioned characters, including wrapped lines."""
    lines: list[list[dict]] = []
    for char in sorted(chars, key=lambda item: (item["top"], item["x0"])):
        if not lines or abs(char["top"] - lines[-1][0]["top"]) > 2.0:
            lines.append([char])
        else:
            lines[-1].append(char)

    output = []
    for line in lines:
        attached: dict[int, list[str]] = defaultdict(list)
        marks = [
            char
            for char in line
            if len(char["text"]) == 1 and unicodedata.combining(char["text"])
        ]
        bases = [char for char in line if char not in marks]
        for mark in marks:
            # The zero-width combining glyph follows its base in the PDF
            # content stream even when its x coordinate overlaps the following
            # consonant. Preserve that relationship before sorting by x.
            preceding = [
                char
                for char in bases
                if char["_seq"] < mark["_seq"]
                and any(unicodedata.category(value).startswith("L") for value in char["text"])
            ]
            if preceding:
                attached[id(max(preceding, key=lambda char: char["_seq"]))].append(mark["text"])
        text = []
        right = None
        for char in sorted(bases, key=lambda item: item["x0"]):
            value = char["text"]
            if not value or value == "\x00":
                continue
            if right is not None and char["x0"] - right > 1.8 and text[-1:] != [" "]:
                text.append(" ")
            text.append(value + "".join(attached[id(char)]))
            right = max(right if right is not None else char["x1"], char["x1"])
        rebuilt = "".join(text).strip()
        if rebuilt:
            output.append(rebuilt)
    return re.sub(r"\s+", " ", " ".join(output)).strip()


def _canonical_form(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s*\.\s*", ".", text.strip())
    text = re.sub(r"\s+", " ", text)
    # Match the repository's existing Wadiyari conversion profile. It uses a
    # precomposed a-tilde but decomposed tilde for the other vowel qualities.
    for composed, decomposed in {
        "ẽ": "e\u0303", "ĩ": "i\u0303", "õ": "o\u0303", "ũ": "u\u0303",
        "Ẽ": "E\u0303", "Ĩ": "I\u0303", "Õ": "O\u0303", "Ũ": "U\u0303",
    }.items():
        text = text.replace(composed, decomposed)
    return text


def extract_wordlist(pdf: Path) -> list[dict[str, str]]:
    records = []
    with tempfile.TemporaryDirectory(prefix="wadiyara-") as directory:
        repaired = Path(directory) / "repaired.pdf"
        repair_unicode_maps(pdf, repaired)
        with pdfplumber.open(repaired) as document:
            for pdf_index in range(FIRST_PDF_PAGE - 1, len(document.pages)):
                page = document.pages[pdf_index]
                words = page.extract_words()
                number_lines: list[list[dict]] = []
                for word in sorted(
                    (word for word in words if 108 <= word["x0"] < 145),
                    key=lambda word: (word["top"], word["x0"]),
                ):
                    if not number_lines or abs(word["top"] - number_lines[-1][0]["top"]) > 2:
                        number_lines.append([word])
                    else:
                        number_lines[-1].append(word)
                numbered = []
                for line in number_lines:
                    text = "".join(word["text"] for word in sorted(line, key=lambda word: word["x0"]))
                    if re.fullmatch(r"(?:\d{4}[ab]?|\d{3}[ab])", text):
                        numbered.append({"top": line[0]["top"], "text": text})
                if not numbered:
                    continue
                anchors = sorted(
                    (
                        float(word["top"]),
                        NUMBER_CORRECTIONS.get(word["text"], word["text"]),
                    )
                    for word in numbered
                )
                first_top, last_top = anchors[0][0], anchors[-1][0]

                buckets: dict[str, dict[str, list[dict]]] = {
                    number: {field: [] for field in COLUMNS}
                    for _top, number in anchors
                }
                for sequence, original_char in enumerate(page.chars):
                    char = dict(original_char, _seq=sequence)
                    if char["top"] < first_top - 20 or char["top"] > last_top + 20:
                        continue
                    field = next(
                        (
                            name
                            for name, (left, right) in COLUMNS.items()
                            if left <= char["x0"] < right
                        ),
                        None,
                    )
                    if field is None:
                        continue
                    _distance, number = min(
                        (abs(char["top"] - top), number) for top, number in anchors
                    )
                    buckets[number][field].append(char)

                for _top, number in anchors:
                    row = {"number": number}
                    for field in COLUMNS:
                        value = _line_text(buckets[number][field])
                        row[field] = _canonical_form(value) if field.startswith("lrp") else value
                    # Item 0668's final right parenthesis is another glyph that
                    # both of the PDF's Unicode maps label as a space. Balanced
                    # gloss parentheses make the intended punctuation explicit.
                    if row["gloss"].count("(") == row["gloss"].count(")") + 1:
                        row["gloss"] += ")"
                    records.append(row)
                    if int(re.match(r"\d+", number).group()) == LAST_ITEM:
                        break
                if records and int(re.match(r"\d+", records[-1]["number"]).group()) == LAST_ITEM:
                    break

    numbers = [int(re.match(r"\d+", record["number"]).group()) for record in records]
    expected = list(range(1, LAST_ITEM + 1))
    if sorted(set(numbers)) != expected:
        missing = sorted(set(expected) - set(numbers))
        raise ValueError(
            f"wordlist numbering is not 0001-{LAST_ITEM:04d}; "
            f"missing={missing[:20]}"
        )
    return records


def read_manual_seed(path: Path) -> dict[str, list[list[str]]]:
    by_form: dict[str, list[list[str]]] = defaultdict(list)
    if not path.exists():
        return by_form
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) == 8 and row[2]:
                by_form[_canonical_form(row[2])].append(row)
    return by_form


def _gloss_key(text: str) -> str:
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    return " ".join(re.findall(r"[a-z]+", text))


def _manual_match(
    form: str, gloss: str, manual: dict[str, list[list[str]]]
) -> tuple[list[str] | None, str | None]:
    exact = manual.get(form)
    if exact:
        return exact[0], form

    source_gloss = _gloss_key(gloss)
    candidates = []
    for manual_form, rows in manual.items():
        for row in rows:
            manual_gloss = _gloss_key(row[3])
            if not manual_gloss or not (
                manual_gloss in source_gloss or source_gloss in manual_gloss
            ):
                continue
            score = difflib.SequenceMatcher(None, form, manual_form).ratio()
            if score >= 0.78:
                candidates.append((score, manual_form, row))
    if not candidates:
        return None, None
    _score, manual_form, row = max(candidates, key=lambda item: item[0])
    return row, manual_form


def write_raw(records: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["number", "gloss", "lrp1", "lrp2", "lrp3"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def write_forms(
    records: list[dict[str, str]], path: Path, manual: dict[str, list[list[str]]]
) -> tuple[int, int]:
    rows = []
    reused_manual = set()
    for record in records:
        consultants: dict[str, list[str]] = {}
        for label in ("LRP1", "LRP2", "LRP3"):
            form = record[label.lower()]
            if not form or form == "-":
                continue
            consultants.setdefault(form, []).append(label)

        for form, labels in consultants.items():
            seed, matched_form = _manual_match(form, record["gloss"], manual)
            seed = seed or ["WK", "", form, "", "", "", "", SOURCE]
            if matched_form:
                reused_manual.add(matched_form)
            provenance = f"Wordlist no. {record['number']}; {', '.join(labels)}"
            review_note = re.sub(
                r"^Wordlist no\. [^;]+; LRP[123](?:, LRP[123])*(?:; )?",
                "",
                seed[6].strip(),
            )
            note = f"{provenance}; {review_note}" if review_note else provenance
            rows.append(
                ["WK", seed[1], form, record["gloss"], seed[4], seed[5], note, seed[7] or SOURCE]
            )

    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    return len(rows), len(reused_manual)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="A-Phonological-Description-of-Wadiyari.pdf")
    parser.add_argument(
        "--raw-output", type=Path, default=here / "wadiyara_wordlist.csv"
    )
    parser.add_argument(
        "--forms-output", type=Path, default=here.parent / "20230306-wadiyara.csv"
    )
    parser.add_argument(
        "--manual-seed",
        type=Path,
        help="optional eight-column manual CSV whose CDIAL IDs/notes take precedence",
    )
    args = parser.parse_args()

    manual = read_manual_seed(args.manual_seed or args.forms_output)
    records = extract_wordlist(args.pdf)
    write_raw(records, args.raw_output)
    row_count, reused_count = write_forms(records, args.forms_output, manual)
    print(
        f"Extracted {len(records)} prompts and {row_count} unique prompt/form rows; "
        f"reused manual metadata for {reused_count} forms."
    )


if __name__ == "__main__":
    main()

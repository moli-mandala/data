"""OCR the appendix vocabularies in Ghatage's Survey of Marathi Dialects.

The scans print one lexical entry per line, with the dialect form and part of
speech in the left column and the English gloss in the right column.  This
importer uses Tesseract's TSV output so that column position and word-level
confidence remain available for auditing.

The currently configured volume is *Marati of Kasargod* (1970), whose
vocabulary occupies PDF pages 144--176 (printed pages 136--168).  Additional
volumes can be added to ``VOLUMES`` after their scans and appendix boundaries
have been verified.

Run from ``data/``::

    uv run python data/other/forms/raw_data/ghatage_survey.py \
      marati-kasargod /path/to/source.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path


RICH_COLUMNS = 15
OCR_LANGUAGE = "script/Latin"
POS_RE = re.compile(
    r"^(?P<form>.+?)\s+(?P<pos>['“]?(?:M[.,:]?\s*F[.,:]?(?:\s*N[.,:]?)?(?:\s*Pl)?|"
    r"M[.,:]?\s*N|F[.,:]?\s*N|N[.,:]?\s*(?:Pl|PI)|WN|Adj|Ady|Adv|Conj|E|F|"
    r"Indi|Interj|M|N|Nu|Part|Postp|Pro|VY|VV|V)[.,:_?]{0,2})"
    r"(?P<trailing>\s+\S{1,4})?$",
    re.I,
)
POS_TAGS = {
    "adj.": ("adj",),
    "adv.": ("adv",),
    "conj.": ("conj",),
    "e.": ("expression",),
    "f.": ("noun", "f"),
    "f,": ("noun", "f"),
    "indi.": ("indeclinable",),
    "interj.": ("interj",),
    "m.": ("noun", "m"),
    "m,": ("noun", "m"),
    "n.": ("noun", "n"),
    "n,": ("noun", "n"),
    "nu.": ("num",),
    "part.": ("part",),
    "postp.": ("postp",),
    "pro.": ("pron",),
    "v.": ("verb",),
    "m.f.": ("noun", "m", "f"),
    "m.n.": ("noun", "m", "n"),
    "m.f.n.": ("noun", "m", "f", "n"),
    "f.n.": ("noun", "f", "n"),
    "n.pl.": ("noun", "pl"),
}


@dataclass(frozen=True)
class Volume:
    slug: str
    source_id: str
    language_id: str
    dialect_id: str
    dialect: str
    first_pdf_page: int
    last_pdf_page: int
    first_printed_page: int
    left_boundary: int = 650
    right_boundary: int = 700
    body_top: int = 220
    body_bottom: int = 2000


VOLUMES = {
    "marati-kasargod": Volume(
        slug="marati-kasargod",
        source_id="ghatage-kasargod1970",
        language_id="M",
        dialect_id="ghatage-kasargod1970",
        dialect="Marati of Kasargod",
        first_pdf_page=144,
        last_pdf_page=176,
        first_printed_page=136,
    ),
}


@dataclass(frozen=True)
class Word:
    left: int
    top: int
    width: int
    height: int
    confidence: float
    text: str

    @property
    def right(self) -> int:
        return self.left + self.width


@dataclass
class Entry:
    pdf_page: int
    printed_page: int
    ordinal: int
    raw_left: str
    form: str
    pos: str
    definition: str
    confidence: float
    low_confidence_tokens: list[str]
    flags: list[str]
    key_override: str = ""
    locator_override: str = ""
    review_reason: str = ""

    def key(self, source_id: str) -> str:
        return self.key_override or f"{source_id}:p{self.printed_page}:e{self.ordinal}"


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return unicodedata.normalize("NFC", text).strip()


def _canonical_pos(value: str) -> str:
    base = re.sub(r"[\s.,:_?'“”]", "", value.casefold())
    canonical = {
        "adj": "Adj.", "ady": "Adj.", "adv": "Adv.", "conj": "Conj.", "e": "E.",
        "f": "F.", "indi": "Indi.", "interj": "Interj.", "m": "M.",
        "n": "N.", "nu": "Nu.", "part": "Part.", "postp": "Postp.",
        "pro": "Pro.", "v": "V.", "vy": "V.", "vv": "V.",
        "mf": "M.F.", "mfpl": "M.F.", "fn": "F.N.",
        "mn": "M.N.", "mfn": "M.F.N.", "npl": "N.Pl.", "npi": "N.Pl.",
        # A recurrent scan/OCR confusion on this volume is N. -> WN,.
        "wn": "N.",
    }
    return canonical[base]


def parse_tsv(tsv: str) -> list[Word]:
    # OCR occasionally emits a bare quote as a noise token.  TSV fields are
    # not quoted, so enabling csv's default quote handling can silently join
    # every following line into that token.
    rows = csv.DictReader(tsv.splitlines(), delimiter="\t", quoting=csv.QUOTE_NONE)
    words = []
    for row in rows:
        if row.get("level") != "5" or not row.get("text", "").strip():
            continue
        words.append(
            Word(
                left=int(row["left"]),
                top=int(row["top"]),
                width=int(row["width"]),
                height=int(row["height"]),
                confidence=float(row["conf"]),
                text=row["text"].strip(),
            )
        )
    return words


def group_lines(words: list[Word], tolerance: int = 14) -> list[list[Word]]:
    """Group TSV words by vertical centre while retaining reading order."""
    lines: list[list[Word]] = []
    for word in sorted(words, key=lambda item: (item.top + item.height / 2, item.left)):
        centre = word.top + word.height / 2
        target = next(
            (
                line
                for line in reversed(lines[-3:])
                if abs(
                    centre
                    - sum(item.top + item.height / 2 for item in line) / len(line)
                )
                <= tolerance
            ),
            None,
        )
        if target is None:
            lines.append([word])
        else:
            target.append(word)
    return [sorted(line, key=lambda item: item.left) for line in lines]


def parse_page(tsv: str, volume: Volume, pdf_page: int) -> list[Entry]:
    printed_page = volume.first_printed_page + pdf_page - volume.first_pdf_page
    words = [
        word
        for word in parse_tsv(tsv)
        if volume.body_top <= word.top <= volume.body_bottom
        and word.width > 5
        and word.confidence >= 0
    ]
    entries: list[Entry] = []
    for line in group_lines(words):
        left_words = [word for word in line if word.right < volume.left_boundary]
        right_words = [word for word in line if word.left > volume.right_boundary]
        raw_left = _clean(" ".join(word.text for word in left_words))
        definition = _clean(" ".join(word.text for word in right_words))
        match = POS_RE.match(raw_left)
        if raw_left and definition:
            relevant = left_words + right_words
            low = [word.text for word in relevant if word.confidence < 60]
            flags = []
            if low:
                flags.append("low-confidence")
            if any(char in raw_left for char in "?{}[]|€£©#$>_"):
                flags.append("suspicious-character")
            if match and match.group("trailing"):
                flags.append("trailing-noise")
            entries.append(
                Entry(
                    pdf_page=pdf_page,
                    printed_page=printed_page,
                    ordinal=len(entries) + 1,
                    raw_left=raw_left,
                    form=_clean(match.group("form") if match else raw_left).strip(" ,;‘’“”'"),
                    pos=_canonical_pos(match.group("pos")) if match else "",
                    definition=definition,
                    confidence=sum(word.confidence for word in relevant) / len(relevant),
                    low_confidence_tokens=low,
                    flags=flags,
                )
            )
        elif (
            definition
            and entries
            and not raw_left
            and not definition.isdigit()
            and not re.search(r"\b(?:GOVERNMENT|CENTRAL|PRESS|BOMBAY)\b", definition, re.I)
        ):
            entries[-1].definition = _clean(f"{entries[-1].definition} {definition}")
            relevant = right_words
            entries[-1].low_confidence_tokens.extend(
                word.text for word in relevant if word.confidence < 60
            )
            if entries[-1].low_confidence_tokens and "low-confidence" not in entries[-1].flags:
                entries[-1].flags.append("low-confidence")
    return entries


def run_tesseract(image: Path) -> str:
    # The generic English model systematically drops the source's central-vowel,
    # retroflex, palatal, and nasal diacritics.  The Latin-script model retains
    # those distinctions while still recognizing the English definition column.
    command = [
        "tesseract", str(image), "stdout", "--psm", "6", "-l", OCR_LANGUAGE, "tsv",
    ]
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout
    except FileNotFoundError as exc:
        raise SystemExit("tesseract is required to run this importer") from exc


def render_page(pdf: Path, page: int, output_prefix: Path) -> Path:
    command = [
        # The reconstructed scans render at their native 1419x2200 pixels at
        # 144 DPI, which is also the coordinate system used by Volume bounds.
        "pdftoppm", "-f", str(page), "-l", str(page), "-r", "144",
        "-singlefile", "-png", str(pdf), str(output_prefix),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise SystemExit("pdftoppm is required to render the source scan") from exc
    return output_prefix.with_suffix(".png")


CORRUPT_FORM_CHARS = set("?{}[]|€£©#$>_�")
SOURCE_FORM_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "əɨɛɔčšśǰṭḍṇñŋḷņțłļãõāēīū"
    ": -()'̃"
)
POS_IN_FORM_RE = re.compile(
    r"(?:^|\s)(?:M|N|F|VY?|VV|Adj|Ady|Adv|Pro|Nu|E)[.,:_?]*(?:\s|$)"
)


def _form_decisions(entry: Entry) -> tuple[list[str], list[str]]:
    if "excluded" in entry.flags:
        return [], [entry.form]
    candidates = [_clean(form).strip(" ,;‘’“”'") for form in re.split(r"\s*~\s*", entry.form)]
    accepted: list[str] = []
    rejected: list[str] = []
    for form in candidates:
        # A full stop is printed as entry punctuation when no POS label follows;
        # it is not part of the lexical form.
        form = form.removesuffix(".").strip()
        has_letter = any(char.isalpha() for char in form)
        corrupt = (
            not form
            or not has_letter
            or any(char in CORRUPT_FORM_CHARS for char in form)
            or any(char.isupper() for char in form)
            or any(char not in SOURCE_FORM_CHARS for char in form)
            or POS_IN_FORM_RE.search(form) is not None
        )
        (rejected if corrupt else accepted).append(form)
    return accepted, rejected


def rich_row(
    entry: Entry,
    volume: Volume,
    *,
    form: str | None = None,
    entry_key: str | None = None,
    variant_of_key: str = "",
) -> list[str]:
    tags = list(POS_TAGS.get(entry.pos.casefold(), ()))
    tags.append(
        f"dialect:{volume.language_id}:{volume.dialect_id}:"
        f"{volume.dialect.replace(' ', '%20')}"
    )
    # None of the OCR heads has been silently promoted to a reviewed
    # transcription.  Keep the typed marker even when confidence is high.
    tags.append("ocr-review")
    if variant_of_key:
        tags.append("variant")
    form = entry.form if form is None else form
    entry_key = entry.key(volume.source_id) if entry_key is None else entry_key
    return [
        volume.language_id,
        "",
        form,
        entry.definition,
        "",
        "",
        "",
        (
            f"{volume.source_id}[{entry.locator_override}]"
            if entry.locator_override
            else f"{volume.source_id}[p. {entry.printed_page}, entry {entry.ordinal}]"
        ),
        "",
        "",
        entry_key,
        variant_of_key,
        "",
        variant_of_key,
        " ".join(tags),
    ]


def rich_rows(entry: Entry, volume: Volume) -> tuple[list[list[str]], list[str]]:
    forms, rejected = _form_decisions(entry)
    if not forms:
        return [], rejected
    main_key = entry.key(volume.source_id)
    rows = [rich_row(entry, volume, form=forms[0], entry_key=main_key)]
    for index, form in enumerate(forms[1:], 1):
        rows.append(
            rich_row(
                entry,
                volume,
                form=form,
                entry_key=f"{main_key}:variant:{index}",
                variant_of_key=main_key,
            )
        )
    return rows, rejected


def extract(volume: Volume, pdf: Path | None, image_dir: Path | None = None) -> list[Entry]:
    all_entries = []
    with tempfile.TemporaryDirectory(prefix=f"{volume.slug}-") as temporary:
        temporary_path = Path(temporary)
        for pdf_page in range(volume.first_pdf_page, volume.last_pdf_page + 1):
            if image_dir is not None:
                image = image_dir / f"{pdf_page:03}.png"
                if not image.exists():
                    image = image_dir / f"{pdf_page:03}.webp"
                if not image.exists():
                    raise SystemExit(f"missing source image: {image}")
            else:
                if pdf is None:
                    raise SystemExit("a PDF or --image-dir is required")
                image = render_page(pdf, pdf_page, temporary_path / f"page-{pdf_page}")
            all_entries.extend(parse_page(run_tesseract(image), volume, pdf_page))
    return all_entries


def apply_corrections(entries: list[Entry], volume: Volume, corrections: Path) -> None:
    """Apply independently reviewable, source-image-backed OCR corrections."""
    if not corrections.exists():
        raise SystemExit(f"missing required correction file: {corrections}")
    by_key = {entry.key(volume.source_id): entry for entry in entries}
    seen: set[str] = set()
    with corrections.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            action = row["Action"].strip().casefold()
            key = row["Entry_Key"].strip()
            if key in seen:
                raise ValueError(f"duplicate correction key: {key}")
            seen.add(key)
            if action == "add":
                entry = Entry(
                    pdf_page=int(row["PDF_Page"]),
                    printed_page=int(row["Printed_Page"]),
                    ordinal=int(row["Entry"] or 0),
                    raw_left="[manual source-image recovery]",
                    form=row["Form"].strip(),
                    pos=row["POS"].strip(),
                    definition=row["Definition"].strip(),
                    confidence=100.0,
                    low_confidence_tokens=[],
                    flags=["human-corrected"],
                    key_override=key,
                    locator_override=row["Locator"].strip(),
                    review_reason=row["Reason"].strip(),
                )
                entries.append(entry)
                by_key[key] = entry
                continue
            if key not in by_key:
                raise ValueError(f"correction target is absent: {key}")
            entry = by_key[key]
            if action == "exclude":
                entry.flags.append("excluded")
            elif action == "replace":
                entry.form = row["Form"].strip()
                entry.pos = row["POS"].strip()
                entry.definition = row["Definition"].strip()
                entry.flags.append("human-corrected")
            else:
                raise ValueError(f"unsupported correction action {action!r} for {key}")
            entry.review_reason = row["Reason"].strip()


def write_outputs(entries: list[Entry], volume: Volume, output: Path, audit: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for entry in entries:
            rows, _ = rich_rows(entry, volume)
            assert all(len(row) == RICH_COLUMNS for row in rows)
            writer.writerows(rows)
    with audit.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Status", "Reason", "PDF_Page", "Printed_Page", "Entry", "Raw_Form_POS", "Form", "POS",
                "Definition", "Mean_Confidence", "Low_Confidence_Tokens", "Entry_Key",
                "Emitted_Keys", "Rejected_Forms", "Review_Flags",
            ]
        )
        for entry in entries:
            rows, rejected = rich_rows(entry, volume)
            if "human-corrected" in entry.flags:
                status = "verified"
                reasons = [entry.review_reason or "verified against source image"]
            elif "excluded" in entry.flags:
                status = "excluded"
                reasons = [entry.review_reason or "excluded after source-image review"]
            else:
                status = "ocr_unreviewed" if rows else "corrupt"
                reasons = ["source transcription requires human verification"]
            if rejected:
                reasons.append("corrupt or unparsed form kept audit-only")
            writer.writerow(
                [
                    status,
                    "; ".join(reasons),
                    entry.pdf_page,
                    entry.printed_page,
                    entry.ordinal,
                    entry.raw_left,
                    entry.form,
                    entry.pos,
                    entry.definition,
                    f"{entry.confidence:.2f}",
                    " | ".join(entry.low_confidence_tokens),
                    entry.key(volume.source_id),
                    "|".join(row[10] for row in rows),
                    "|".join(rejected),
                    " ".join(entry.flags),
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("volume", choices=sorted(VOLUMES))
    parser.add_argument("pdf", type=Path, nargs="?")
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--corrections", type=Path)
    args = parser.parse_args()
    volume = VOLUMES[args.volume]
    output = args.output or Path(f"data/other/forms/20260817-ghatage-{volume.slug}.csv")
    audit = args.audit or Path(
        f"data/other/forms/raw_data/20260817-ghatage-{volume.slug}-audit.csv"
    )
    corrections = args.corrections or Path(
        f"data/other/forms/raw_data/20260817-ghatage-{volume.slug}-corrections.csv"
    )
    entries = extract(volume, args.pdf, args.image_dir)
    apply_corrections(entries, volume, corrections)
    write_outputs(entries, volume, output, audit)
    print(f"wrote {len(entries)} entries to {output}")
    print(f"wrote OCR audit to {audit}")


if __name__ == "__main__":
    main()

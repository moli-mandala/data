"""Parse Emeneau & Burrow's *Dravidian Borrowings from Indo-Aryan*.

The checked-in text is OCR extracted from a two-column scan.  Most dictionary
entries are linear; a small set of entry numbers were placed after their text
by the PDF extractor.  Exact, unique opening phrases repair those boundaries,
and ``parse_audit.csv`` records the result of every extraction and match.

Outputs are the ordinary Jambu raw-data tables plus a conservative map from a
DBIA source ID to a canonical CDIAL entry.  A redirect is emitted only for a
unique normalized headword match (or a uniquely best gloss-disambiguated
homonym); everything else remains an independent ``dbiaN`` entry for review.
"""

from __future__ import annotations

import csv
import html
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
CDIAL_PARAMS = HERE.parent / "cdial" / "params.csv"

BORROWER_LANGUAGES = {
    "Ta": "Tam",
    "Ma": "Mal",
    "Ko": "Kota",
    "To": "Toda",
    "Ka": "Kannada",
    "Kod": "Kodagu",
    "Koḍ": "Kodagu",
    "Tu": "Tulu",
    "Te": "Telugu",
    "Kol": "Kolami",
    "Nk": "Naikri",
    "Pa": "Parji",
    "Ga": "Gadaba",
    "Go": "Gondi",
    "Konda": "Konda",
    "Kui": "Kui",
    "Kuwi": "Kuwi",
    "Kur": "Kurux",
    "Malt": "Malto",
    "Br": "Brahui",
    # The addendum's Koi item is explicitly compared with DED 256 and belongs
    # with the Koya material conventionally represented under Gondi here.
    "Koi": "Gondi",
}

DONOR_LABELS = (
    "Skt", "Pkt", "Pali", "Pa", "Ap", "Mar", "H", "Beng", "Guj",
    "OMar", "Or", "Panj", "Bihari",
)

_BORROWER_RE = re.compile(
    r"(?<![\w])(" + "|".join(map(re.escape, sorted(BORROWER_LANGUAGES, key=len, reverse=True)))
    + r")[.,]\s*"
)
_DONOR_RE = re.compile(
    r"(?<![\w])(" + "|".join(map(re.escape, sorted(DONOR_LABELS, key=len, reverse=True)))
    + r")[.,]?\s+"
)
_ENTRY_RE = re.compile(
    r"(?m)^(\d{1,3})\s*([a-z])?\s+"
    r"(?=(?:" + "|".join(map(re.escape, sorted(BORROWER_LANGUAGES, key=len, reverse=True)))
    + r")\b[.,]?)"
)

# The PDF's two-column text layer displaced these numbers to the end of a neighbouring column.
# The anchors are exact, unique entry openings confirmed against the printed ordering/indexes.
BOUNDARY_REPAIRS = {
    "Ta. arankam island formed": "18 Ta. arankam island formed",
    "Ta. allam, nallam ginger": "22 Ta. allam, nallam ginger",
    "Ta. asti property, wealth": "40 Ta. asti property, wealth",
    "Ta. ukkirāņam storehouse": "43 Ta. ukkirāņam storehouse",
    "Ta. ūmattai, ūmattam": "51 Ta. ūmattai, ūmattam",
    "Ta. kēļi fun, jest": "113 Ta. kēļi fun, jest",
    "Ta. komaram inspiration": "124 Ta. komaram inspiration",
    "Ta. (Irawati Karve": "135 Ta. (Irawati Karve",
    "Ta. tuti (-pp-, -tt-)": "212 Ta. tuti (-pp-, -tt-)",
    "30 Ta. mutta-kkācu": "300 Ta. mutta-kkācu",
}

ENGLISH_STARTERS = {
    "a", "an", "and", "as", "at", "be", "being", "but", "by", "cf",
    "for", "from", "id", "in", "into", "is", "made", "name", "not",
    "of", "on", "or", "perhaps", "the", "to", "used", "with", "without",
    "one", "kind", "state", "act", "that", "which", "this", "same",
}


def folded(value: str) -> str:
    """Accent/punctuation-insensitive key, including recurring OCR glyphs."""
    value = value.casefold().translate(str.maketrans({
        "ſ": "s", "ş": "s", "ș": "s", "ç": "s", "ț": "t", "ņ": "n",
        "ļ": "l", "đ": "d", "ł": "l", "ŋ": "n",
    }))
    value = "".join(
        char for char in unicodedata.normalize("NFD", value)
        if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"[^a-z]+", "", value)


def words(value: str) -> set[str]:
    value = re.sub(r"<[^>]+>", " ", value)
    return {
        word for token in re.findall(r"[^\W\d_]+", value, flags=re.UNICODE)
        if len(word := folded(token)) >= 3
        if word not in ENGLISH_STARTERS
    }


def dictionary_region(raw: str) -> str:
    start = raw.index("\n1 Ta") + 1
    end = raw.index("## p. 63", start)
    region = raw[start:end]
    for source, replacement in BOUNDARY_REPAIRS.items():
        if region.count(source) != 1:
            raise ValueError(f"DBIA boundary repair anchor is not unique: {source!r}")
        region = region.replace(source, replacement, 1)
    return region


def extract_entries(raw: str):
    """Return monotonically numbered entries recoverable from the OCR stream."""
    region = dictionary_region(raw)
    matches = []
    last_key = (0, "")
    for match in _ENTRY_RE.finditer(region):
        number = int(match.group(1))
        letter = match.group(2) or ""
        # Index/citation numbers can begin a physical OCR line.  Actual entry
        # heads are monotonic, so this cheaply rejects those false positives.
        key = (number, letter)
        if last_key < key and number <= 337:
            matches.append(match)
            last_key = key

    entries = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(region)
        text = region[match.end():end]
        text = re.sub(r"^##.*$", " ", text, flags=re.MULTILINE)
        text = text.replace("\f", " ")
        text = re.sub(r"\[\s*\d+\s*\]", " ", text)
        text = re.sub(r"-\s*\n\s*(?=[a-z])", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        page_matches = list(re.finditer(r"## p\.\s*(\d+)", region[:match.start()]))
        page = page_matches[-1].group(1) if page_matches else "9"
        entries.append((int(match.group(1)), match.group(2) or "", page, text))
    return entries


def donor_boundary(text: str) -> int:
    slash = re.search(r"/\s*(?:\?|<)?\s*(?=(?:" + "|".join(DONOR_LABELS) + r")\b)", text)
    if slash:
        return slash.start()
    donor = _DONOR_RE.search(text)
    return donor.start() if donor else len(text)


def first_form(clause: str) -> str:
    """Extract one safe citation form from a semicolon-delimited clause."""
    clause = clause.strip(" /.,")
    clause = re.sub(r"^\([^)]*\)\s*", "", clause)
    match = re.match(r"([?*]?[\w'’:.·āīūēōṛṝḷḹṃṁḥṅñṭḍṇśṣçşșțņļſăäöüãẽĩõũ-]+)", clause)
    if not match:
        return ""
    form = match.group(1).strip("?.,")
    if folded(form) in ENGLISH_STARTERS or len(folded(form)) < 1:
        return ""
    return unicodedata.normalize("NFC", form)


def borrower_forms(text: str):
    boundary = donor_boundary(text)
    borrower_text = text[:boundary]
    markers = list(_BORROWER_RE.finditer(borrower_text))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(borrower_text)
        span = borrower_text[marker.end():end].strip()
        language = BORROWER_LANGUAGES[marker.group(1)]
        # Semicolons are ambiguous in DBIA: they can introduce another lemma, another sense, or
        # ordinary English prose.  Only the first citation form of each explicitly labelled
        # language span is safe to publish automatically; all later material remains searchable
        # in the preserved span and full-entry transcription for future manual expansion.
        form = first_form(span)
        if form:
            yield language, form, span


def donor_candidates(text: str):
    boundary = donor_boundary(text)
    donor_text = text[boundary:]
    candidates = []
    for marker in _DONOR_RE.finditer(donor_text):
        tail = donor_text[marker.end():]
        form = first_form(tail)
        if form and folded(form):
            # The donor gloss may precede the citation ("cf. Skt. X") or follow it.  The whole
            # entry is still a short, coherent semantic unit and disambiguates CDIAL homonyms
            # better than an arbitrary one-sided window.
            candidates.append((marker.group(1), form, text))
    return candidates


def cdial_index():
    by_head = defaultdict(list)
    with CDIAL_PARAMS.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            by_head[folded(row[1].split(",", 1)[0])].append(row)
    return by_head


def reconcile(candidates, index):
    """Return (CDIAL id, donor form, reason), or blanks when unsafe."""
    for _language, form, context in candidates:
        options = index.get(folded(form), [])
        if len(options) == 1:
            return options[0][0], form, "unique normalized headword"
        if len(options) > 1:
            context_words = words(context)
            scored = []
            for option in options:
                overlap = len(context_words & words(option[3]))
                scored.append((overlap, option))
            scored.sort(key=lambda item: item[0], reverse=True)
            if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1][0], form, "homonym disambiguated by gloss"
    return "", candidates[0][1] if candidates else "", "no safe CDIAL match"


def write_outputs():
    raw = (HERE / "dbia.txt").read_text(encoding="utf-8")
    entries = extract_entries(raw)
    index = cdial_index()
    found_numbers = {number for number, _letter, _page, _text in entries}

    params = []
    forms = []
    redirects = []
    audit = []
    for number, letter, page, text in entries:
        entry_id = f"dbia{number}{letter}"
        candidates = donor_candidates(text)
        cdial_id, donor_form, decision = reconcile(candidates, index)
        source = f"dbia[p. {page}, no. {number}{letter}]" if page else f"dbia[no. {number}{letter}]"
        escaped = html.escape(text)
        etymology = f"<html><body><p>{escaped}</p></body></html>"
        params.append([entry_id, donor_form or f"DBIA {number}{letter}", "Indo-Aryan", etymology, ""])
        if cdial_id:
            redirects.append([entry_id, cdial_id, donor_form, decision])

        emitted = 0
        for form_index, (language, form, span) in enumerate(borrower_forms(text), 1):
            key = f"dbia-{number}{letter}-{language}-{form_index}"
            forms.append([
                language, entry_id, form, span, "", "", "", source, "", "",
                key, "", "", "", "loanword",
            ])
            emitted += 1
        audit.append([
            entry_id, number, letter, page, donor_form, cdial_id, decision, emitted,
            " | ".join(form for _lang, form, _context in candidates),
        ])

    for missing in sorted(set(range(1, 338)) - found_numbers):
        audit.append([
            f"dbia{missing}", missing, "", "", "", "", "OCR entry boundary not recoverable", 0, "",
        ])

    def write(name, header, rows):
        with (HERE / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            if header:
                writer.writerow(header)
            writer.writerows(rows)

    write("params.csv", None, params)
    write("forms.csv", None, forms)
    write("cdial_redirects.csv", ["DBIA_ID", "CDIAL_ID", "Headword", "Reason"], redirects)
    write(
        "parse_audit.csv",
        ["DBIA_ID", "Number", "Letter", "Page", "Donor", "CDIAL_ID", "Decision", "Forms", "Candidates"],
        audit,
    )
    print(
        f"wrote {len(params)} entries, {len(forms)} conservative forms, "
        f"and {len(redirects)} CDIAL redirects; {337 - len(found_numbers)} boundaries need review"
    )


if __name__ == "__main__":
    write_outputs()

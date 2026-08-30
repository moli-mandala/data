#!/usr/bin/env python3
"""Install the Domari lexicon of Bruno Herin's 2012 Aleppo grammar sketch.

The article (*Linguistic Discovery* 10.2: 1--52, doi 10.1349/PS1.1537-0852.A.412) is a
grammar sketch of the Domari of Aleppo, based on the author's own 2009--2010 fieldwork.
It is not a dictionary: its lexicon lives in italicised citations inside running prose,
in numbered interlinear examples, and in paradigm tables.

Two representations of the article exist and neither is sufficient alone:

* the journal's XML/HTML rendering (``?htmlAlways=yes``) carries the phonetic
  transcription as real Unicode in ``<font face="Doulos SIL"><i>`` runs, and marks
  glosses with curly double quotes.  This is the canonical text here;
* the publisher PDF carries the printed pagination that citations of the article use,
  but its embedded font subsets drop every non-ASCII glyph from the text layer
  (``Dōm`` extracts as ``  m``).  It is used *only* to recover printed page numbers,
  by aligning the surviving English prose word sequence against the HTML word
  sequence.  No Domari material is read from the PDF.

Neither snapshot is redistributed; both are cached under ``tmp/domari-aleppo-cache``
and pinned by SHA-256.  ``20260825-herin-domari-aleppo-extract.psv`` is the checked-in
deterministic extraction, so the curation, audit and installed CSV can be rebuilt with
``--offline`` and reviewed without the sources.

Raw source records
------------------

Every one of the article's own citation units is a raw record with an explicit audit
status:

``prose``       each curly-quoted gloss in running prose, an abstract paragraph or a
                footnote, together with the italic form group that governs it;
``example``     each italicised word cell of a numbered interlinear example, together
                with the Leipzig-style morpheme gloss printed beneath it;
``translation`` the free translation line of a numbered example;
``paradigm``    each italicised cell of a non-interlinear table.

Editorial policy, applied uniformly and recorded per record in the audit:

* install every Domari item the article glosses, whether it is cited in prose, glossed
  word-by-word in an interlinear example, or printed in a paradigm whose lexeme and
  gloss the article states;
* install a multi-word Domari item only where the article glosses it as a unit -- the
  complex-verb citations (``gā kar-`` "to say"), the adverbial and adpositional lists,
  and the relational-noun phrases -- tagged ``multiword-expression``;
* leave free translations, bare inflectional affixes, phoneme-inventory cells and
  non-Domari comparanda and donor forms out of the installed CSV; they stay in the
  audit with a reason;
* the article's Palestinian/Jerusalem Domari comparanda are Matras's data, not Herin's
  Aleppo attestations, and are excluded rather than mixed into the Aleppo dialect;
* donor statements become ``Etymology`` prose plus a ``loan:*`` tag, never graph edges:
  the article names Arabic, Kurdish, Turkish and Persian donors by spelling only.
  Its Indo-Aryan etyma, which footnote 1 states are taken from Turner (1962--1966),
  are matched to CDIAL only where the printed etymon has a unique CDIAL headword with
  a compatible meaning; every candidate is recorded in the audit.

Three explicit normalizations are applied to the printed text, each visible in the
audit alongside the raw span:

* the article's decomposed combining diacritics are normalized to NFC and its two
  distinct apostrophes for the glottal stop are unified on ``ʾ``;
* optional segments printed in parentheses -- ``-mā(n)``, ``trombīl(ã)``, ``ka(ǧ)ǧã`` --
  are expanded into an explicit head plus alternate rows rather than left as punctuation;
* glosses keep the author's wording but use ``;`` between senses, and a verb glossed
  with a bare English verb is written ``to eat`` so that stems, inflected citations and
  interlinear tokens of one lexeme carry the same gloss.

Run from ``data/``::

    uv run python data/other/forms/raw_data/herin_domari_aleppo_2012.py --refresh
    uv run python data/other/forms/raw_data/herin_domari_aleppo_2012.py --offline --install
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import html
import io
import json
import re
import unicodedata
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

SOURCE_ID = "herin2012domari"
SNAPSHOT_DATE = "2026-08-25"
COLLATION_DATE = "2026-08-25"

ARTICLE_URL = "https://journals.dartmouth.edu/journals/xmlpage/1/article/412?htmlAlways=yes"
PDF_URL = "https://journals.dartmouth.edu/journals/xmlpage/1/document/883"
HTML_SHA256 = "905f18eec71f0fee0565ea1c008ceec5f1b3fda00e1e33c2200f1e45f2711202"
PDF_SHA256 = "ba17d3e2ed8c4d35ff0bb56016088ac913b1d453ba80a53be54cd7f1e393d444"
PDF_PAGES = 53          # cover sheet plus printed pages 1--52
FIRST_PRINTED_PAGE = 1
LAST_PRINTED_PAGE = 52

LANGUAGE_ID = "as"
DIALECT_TAG = "dialect:as:aleppo:Aleppo"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
DEFAULT_CACHE = ROOT / "tmp/domari-aleppo-cache"
FORM_OUTPUT = ROOT / "data/other/forms/20260825-herin-domari-aleppo.csv"
AUDIT_OUTPUT = RAW_DIR / "20260825-herin-domari-aleppo-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260825-herin-domari-aleppo-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260825-herin-domari-aleppo-manifest.json"
EXTRACT_SNAPSHOT = RAW_DIR / "20260825-herin-domari-aleppo-extract.psv"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
EXTRACT_FIELDS = [
    "Sequence", "Unit_ID", "Region", "Section", "Printed_Page", "Raw_Form", "Raw_Phonetic",
    "Raw_Before", "Raw_Gloss", "Raw_After", "Raw_Context",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Unit_ID", "Region", "Section", "Printed_Page",
    "Raw_Form", "Raw_Gloss", "Raw_Context", "Status", "Reason", "Final_Forms", "Final_Gloss",
    "Final_Tags", "Etymology", "Etymon_Candidates", "Parameter_ID", "Emitted_Keys", "Review",
    "Material_Error", "Source", "Record_SHA256",
]

ITAL_A, ITAL_B, CELL = "\x01", "\x02", "\x03"
OPEN_Q, CLOSE_Q = "“", "”"


# --------------------------------------------------------------------------------------
# Snapshots
# --------------------------------------------------------------------------------------

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(cache: Path, name: str, url: str, expected: str, refresh: bool) -> Path:
    path = cache / name
    if refresh or not path.exists():
        if not refresh:
            raise FileNotFoundError(
                f"missing cached snapshot {path}; rerun with --refresh to fetch {url}"
            )
        cache.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "Jambu dictionary importer/1.0 "
                                        "(https://github.com/moli-mandala/data)"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    found = _digest(path)
    if found != expected:
        raise ValueError(f"{path} has SHA-256 {found}, expected {expected}")
    return path


# --------------------------------------------------------------------------------------
# HTML text layer
# --------------------------------------------------------------------------------------

BLOCK_RE = re.compile(
    r'(?is)<h(?P<hl>[1-6])[^>]*class="doc-section-heading-\d"[^>]*>(?P<head>.*?)</h(?P=hl)>'
    r'|<p[^>]*class="(?P<pc>doc-t[fx]|doc-fn|doc-abstract)"[^>]*>(?P<para>.*?)</p>'
    r'|(?P<table><table.*?</table>)')
FOOTNOTE_RE = re.compile(r'(?is)<a name="fn(\d+)"')
SECTION_RE = re.compile(r"((?:\d+\.)*\d+)\s+(.*)")
EXAMPLE_RE = re.compile(r"^\((\d{1,3})\)$")
CAPTION_RE = re.compile(r"^Table (\d{1,2}):\s*(.*)$")
SUBLABEL_RE = re.compile(r"^([a-h])\.$")

# The article writes the glottal stop with several apostrophes, and marks primary stress
# inside its phonetic brackets with a plain ASCII apostrophe.
APOSTROPHES = {"’": "ʾ", "ʼ": "ʾ", "'": "ʾ"}
STRESS = {"’": "ˈ", "ʼ": "ˈ", "'": "ˈ", "ʾ": "ˈ"}


DOULOS_OPEN = re.compile(r'(?is)<font face="Doulos SIL">')
SUPERSCRIPT_RE = re.compile(r"(?is)\s*<sup>(.*?)</sup>")


def _superscript(match: re.Match) -> str:
    """Footnote markers vanish; the article's superscript ``y`` is palatalisation.

    ``lāft<sup>y</sup>ī`` is one word, not two: the 15 superscript ``y`` spans mark a
    palatalised consonant. The remaining non-footnote spans are English ordinal
    suffixes (``19<sup>th</sup> century``) and stay as ordinary letters.
    """
    inner = match.group(1)
    if "<a " in inner.lower():
        return " "
    text = html.unescape(re.sub(r"(?s)<[^>]+>", "", inner)).strip()
    return "ʸ" if text == "y" else text
COMBINING = re.compile("\\s+([\u0300-\u036f])")


def flatten(fragment: str) -> str:
    """One HTML fragment -> text with Doulos SIL runs delimited by ITAL_A/ITAL_B.

    The article sets every phonetic string -- citation forms, bracketed phonetic
    transcriptions and comparanda in other languages -- in Doulos SIL, but its use of
    ``<i>`` is inconsistent (``tətã́`` on p. 5 is upright while the two forms printed
    beside it are italic). The font run is therefore the reliable delimiter of source
    phonetic material; running prose is never set in Doulos SIL.
    """
    fragment = SUPERSCRIPT_RE.sub(_superscript, fragment)
    # Whitespace between a <font> and its <i> is the publisher's line wrapping; whitespace
    # inside the <i> is a real word boundary. ``lāfty-ā`` + ``\n muḥtaším`` are two words,
    # ``pī`` + ``-r-ã`` is one form split across two spans.
    fragment = re.sub(r'(?is)(<font face="Doulos SIL">)\s+(<i>)', r"\1\2", fragment)
    fragment = re.sub(r"(?is)(</i>)\s+(</font>)", r"\1\2", fragment)
    fragment = DOULOS_OPEN.sub(ITAL_A, fragment)
    # Doulos SIL is the only font the article opens, so the next </font> closes one.
    fragment = re.sub(r"(?is)</font>", ITAL_B, fragment)
    fragment = re.sub(r"(?is)<br\s*/?>", f" {CELL} ", fragment)
    fragment = re.sub(r"(?s)<[^>]+>", "", fragment)
    fragment = html.unescape(fragment)
    # The publisher splits some diacritics into their own span, which leaves whitespace
    # between a vowel and its combining mark.
    fragment = unicodedata.normalize("NFC", COMBINING.sub(r"\1", fragment))
    # Merge adjacent spans and drop empty ones.
    fragment = re.sub(f"{ITAL_B}(\\s*){ITAL_A}",
                      lambda m: m.group(1) if m.group(1).strip() else "", fragment)
    fragment = re.sub(f"{ITAL_A}\\s*{ITAL_B}", "", fragment)
    fragment = re.sub(r"\s+", " ", fragment)
    fragment = re.sub(f"\\s*{ITAL_A}\\s*", f" {ITAL_A}", fragment)
    fragment = re.sub(f"\\s*{ITAL_B}\\s*", f"{ITAL_B} ", fragment)
    # Repair again: collapsing whitespace can expose a mark that was separated from its
    # base by the span boundary itself (``kərī`` + ``́`` -> ``kərī ́``).
    fragment = unicodedata.normalize("NFC", COMBINING.sub(r"\1", fragment))
    return re.sub(r" +", " ", fragment).strip()


def read_blocks(markup: str) -> list[dict]:
    """Segment the article body into headed prose, footnote and table blocks."""
    body = markup[markup.index("<body"):]
    blocks: list[dict] = []
    section, number = "Front matter", ""
    for match in BLOCK_RE.finditer(body):
        if match.group("head") is not None:
            heading = flatten(match.group("head")).replace(ITAL_A, "").replace(ITAL_B, "")
            parsed = SECTION_RE.match(heading)
            number, section = (parsed.group(1), heading) if parsed else ("", heading)
            continue
        if match.group("para") is not None:
            text = flatten(match.group("para"))
            if not text.replace(ITAL_A, "").replace(ITAL_B, "").strip():
                continue
            footnote = FOOTNOTE_RE.search(match.group("para"))
            number_of = footnote.group(1) if footnote else ""
            # The article prints its table captions beneath the table they name.
            heading = CAPTION_RE.match(_unmark(text).strip())
            if heading:
                for previous in reversed(blocks):
                    if previous["kind"] == "table":
                        previous["caption"] = (heading.group(1), heading.group(2).strip())
                        break
            blocks.append(dict(kind=match.group("pc"),
                               section=f"Footnote {number_of}" if number_of else section,
                               number="" if number_of else number,
                               text=text, footnote=number_of, rows=[], caption=("", "")))
            continue
        rows = [[flatten(cell) for cell in re.findall(r"(?is)<t[dh][^>]*>(.*?)</t[dh]>", row)]
                for row in re.findall(r"(?is)<tr.*?</tr>", match.group("table"))]
        blocks.append(dict(kind="table", section=section, number=number, text="",
                           footnote="", rows=rows, caption=("", "")))
    return blocks


# --------------------------------------------------------------------------------------
# Printed pagination, recovered from the PDF by prose alignment
# --------------------------------------------------------------------------------------

WORD_RE = re.compile(r"[a-z]{4,}")


def pdf_word_pages(pdf_path: Path) -> tuple[list[str], list[int]]:
    import pypdf

    reader = pypdf.PdfReader(str(pdf_path))
    if len(reader.pages) != PDF_PAGES:
        raise ValueError(f"{pdf_path} has {len(reader.pages)} pages, expected {PDF_PAGES}")
    words: list[str] = []
    pages: list[int] = []
    for index, page in enumerate(reader.pages):
        if index == 0:          # the repository cover sheet carries no printed number
            continue
        for word in WORD_RE.findall((page.extract_text() or "").lower()):
            words.append(word)
            pages.append(index)  # printed page number equals the PDF page index
    return words, pages


def page_index(blocks: list[dict], pdf_path: Path) -> dict[int, int]:
    """Map each block index to the printed page its English prose aligns with.

    The PDF text layer keeps ASCII prose and loses the phonetic glyphs, so the two word
    sequences are aligned on lowercase English words only. Footnotes are printed at the
    foot of the page that references them but collected at the end of the HTML, so they
    are aligned separately against the same global word list.
    """
    pdf_words, pdf_pages = pdf_word_pages(pdf_path)
    flow = [index for index, block in enumerate(blocks) if block["kind"] != "doc-fn"]

    html_words: list[str] = []
    owner: list[int] = []
    for index in flow:
        block = blocks[index]
        text = (block["text"] or " ".join(" ".join(row) for row in block["rows"]))
        for word in WORD_RE.findall(text.replace(ITAL_A, "").replace(ITAL_B, "").lower()):
            html_words.append(word)
            owner.append(index)

    votes: dict[int, Counter] = defaultdict(Counter)
    matcher = difflib.SequenceMatcher(None, html_words, pdf_words, autojunk=False)
    for start_a, start_b, size in matcher.get_matching_blocks():
        for offset in range(size):
            votes[owner[start_a + offset]][pdf_pages[start_b + offset]] += 1

    pages: dict[int, int] = {}
    previous = FIRST_PRINTED_PAGE
    for index in flow:
        if votes[index]:
            previous = min(votes[index])
        pages[index] = previous

    # Footnote blocks: locate the longest matching run of their own words.
    for index, block in enumerate(blocks):
        if block["kind"] != "doc-fn":
            continue
        words = WORD_RE.findall(block["text"].replace(ITAL_A, "").replace(ITAL_B, "").lower())
        matcher = difflib.SequenceMatcher(None, words, pdf_words, autojunk=False)
        tally: Counter = Counter()
        for start_a, start_b, size in matcher.get_matching_blocks():
            for offset in range(size):
                tally[pdf_pages[start_b + offset]] += 1
        pages[index] = tally.most_common(1)[0][0] if tally else previous
    return pages


# --------------------------------------------------------------------------------------
# Raw record extraction
# --------------------------------------------------------------------------------------

QUOTED_RE = re.compile(f"{OPEN_Q}([^{OPEN_Q}{CLOSE_Q}]*){CLOSE_Q}")
ITALIC_RE = re.compile(f"{ITAL_A}(.*?){ITAL_B}")
# Connectives permitted inside one citation. A citation is a chain of Doulos SIL runs
# holding the form itself and, in the phonology sections, its bracketed phonetic
# realisation; the connective between two runs may carry no letters, and its brackets
# must be consistent with which side is bracketed, so that
# ``kû "who" (Aleppo kō), keita "where"`` never reads ``kō ~ keita``.
PLAIN_LINK = re.compile(r"[\s]*(?:~|or)?[\s]*")
# The article often typesets the citation hyphen of a verb root outside the font run:
# ``sāk - "can, be able"``. The hyphen belongs to the form.
TAIL_LINK = re.compile(r"[\s,;:]*(-?)[\s,;:]*")
OPEN_LINK = re.compile(r"[\s,;:~(]*\[[\s]*")
CLOSE_LINK = re.compile(r"[\s]*\][\s,;:~)]*")
INNER_LINK = re.compile(r"[\s,;:~]*|[\s]*\][\s]*\([\s]*~?[\s]*\[[\s]*")
ALTERNATION = " ~ "
CONTEXT = 140
BEFORE = 70
AFTER = 70


class Run:
    """One Doulos SIL span, with the bracket depth of the plain text around it."""

    __slots__ = ("start", "end", "text", "bracketed")

    def __init__(self, start: int, end: int, text: str, bracketed: bool):
        self.start, self.end, self.text, self.bracketed = start, end, text, bracketed


def doulos_runs(text: str) -> list[Run]:
    runs: list[Run] = []
    depth = 0
    cursor = 0
    for match in ITALIC_RE.finditer(text):
        depth += text.count("[", cursor, match.start()) - text.count("]", cursor, match.start())
        runs.append(Run(match.start(), match.end(), match.group(1), depth > 0))
        cursor = match.end()
    return runs


def _joins(text: str, earlier: Run, later: Run) -> bool:
    """Whether two runs belong to one citation."""
    link = text[earlier.end:later.start]
    if re.search(r"[^\W\d_]", link.replace("or", "")):
        return False
    if earlier.bracketed and later.bracketed:
        return bool(INNER_LINK.fullmatch(link))
    if earlier.bracketed:
        # A bracketed realisation belongs to the form before it, never to the one after:
        # ``[ɑː] pɑ̄sṓm [pɑːˈsoːm] "at me"`` cites one form with one realisation.
        return False
    if later.bracketed:
        return bool(OPEN_LINK.fullmatch(link))
    return bool(PLAIN_LINK.fullmatch(link))


def _governing_group(text: str, quote_start: int) -> tuple[str, str, str]:
    """The form group, its bracketed realisations, and the prose that introduces it."""
    runs = [run for run in doulos_runs(text) if run.end <= quote_start]
    if not runs:
        return "", "", ""
    tail = text[runs[-1].end:quote_start]
    hyphen = ""
    if runs[-1].bracketed:
        if not CLOSE_LINK.fullmatch(tail):
            return "", "", ""
    else:
        matched = TAIL_LINK.fullmatch(tail)
        if not matched:
            return "", "", ""
        hyphen = matched.group(1)
    group = [runs[-1]]
    while len(runs) > len(group):
        previous = runs[-len(group) - 1]
        if not _joins(text, previous, group[0]):
            break
        group.insert(0, previous)
    forms = [_clean(run.text) for run in group if not run.bracketed]
    if hyphen and forms and not forms[-1].endswith("-"):
        forms[-1] += hyphen
    phonetics = [_clean(run.text, phonetic=True) for run in group if run.bracketed]
    before = text[max(0, group[0].start - BEFORE):group[0].start]
    before = re.sub(r"\s+", " ", before.replace(ITAL_A, "").replace(ITAL_B, ""))
    return (ALTERNATION.join(item for item in forms if item),
            ALTERNATION.join(item for item in phonetics if item),
            before)


def _unmark(text: str) -> str:
    return text.replace(ITAL_A, "").replace(ITAL_B, "")


def _slug(block: dict) -> str:
    if block["footnote"]:
        return f"fn{block['footnote']}"
    if block["number"]:
        return "s" + block["number"]
    heading = block["section"].lower()
    if heading.startswith("front"):
        return "abstract"
    return re.sub(r"[^a-z0-9]+", "-", heading).strip("-") or "prose"


def _clean(text: str, phonetic: bool = False) -> str:
    text = unicodedata.normalize("NFC", COMBINING.sub(r"\1", text))
    for source, target in (STRESS if phonetic else APOSTROPHES).items():
        text = text.replace(source, target)
    return re.sub(r"\s+", " ", text).strip(" \t.,;")


def prose_units(blocks: list[dict], pages: dict[int, int]) -> list[dict]:
    units: list[dict] = []
    counters: Counter = Counter()
    for index, block in enumerate(blocks):
        if block["kind"] == "table":
            continue
        slug = _slug(block)
        text = block["text"]
        for quote in QUOTED_RE.finditer(text):
            counters[slug] += 1
            start = max(0, quote.start() - CONTEXT)
            context = text[start:quote.end() + 40]
            form, phonetic, before = _governing_group(text, quote.start())
            units.append(dict(
                unit=f"{slug}:q{counters[slug]}",
                block=index,
                offset=quote.start(),
                region="prose",
                section=block["section"],
                page=pages.get(index, FIRST_PRINTED_PAGE),
                raw_form=form,
                raw_phonetic=phonetic,
                raw_before=before,
                raw_gloss=_clean(quote.group(1)),
                raw_after=re.sub(r"\s+", " ",
                                 _unmark(text[quote.end():quote.end() + AFTER])),
                context=_clean(context.replace(ITAL_A, "").replace(ITAL_B, "")),
            ))
    return units


def table_units(blocks: list[dict], pages: dict[int, int]) -> list[dict]:
    """Interlinear example words, their free translations, and paradigm cells.

    A table is one of the article's numbered interlinear examples when it carries an
    example number, a sub-example letter, or a free-translation line printed in a row of
    its own. The verb paradigms also print quoted glosses, but always beside the form
    they gloss, never on a row of their own. Examples continue across tables: the article
    breaks a long example into a second table with no label of its own, and the word
    numbering has to run on.
    """
    units: list[dict] = []
    example, sublabel, word = "", "", 0
    table_counts: Counter = Counter()
    # The article prints example (36) twice: once in 6.1 and again in 6.2, where the
    # numbering runs 37, 38, 36, 39. The second printing keeps a distinct unit id.
    printings: Counter = Counter()
    repeat = ""
    for index, block in enumerate(blocks):
        if block["kind"] != "table":
            continue
        rows = block["rows"]
        page = pages.get(index, FIRST_PRINTED_PAGE)
        labels = [_unmark(cell).strip() for row in rows for cell in row]
        number = next((EXAMPLE_RE.match(label).group(1)
                       for label in labels if EXAMPLE_RE.match(label)), "")
        letter = next((SUBLABEL_RE.match(label).group(1)
                       for label in labels if SUBLABEL_RE.match(label)), "")
        translation = any(QUOTED_RE.search(" ".join(row)) and not any(ITAL_A in cell for cell in row)
                          for row in rows)
        if number or letter or translation:
            if number:
                printings[number] += 1
                repeat = f"-r{printings[number]}" if printings[number] > 1 else ""
                example, sublabel, word = number, "", 0
            if letter:
                sublabel, word = letter, 0
            units.extend(_interlinear(rows, block, index, page, example,
                                      sublabel + repeat, word))
            word = max((int(unit["unit"].rsplit(":w", 1)[1])
                        for unit in units if unit["region"] == "example"
                        and unit["unit"].startswith(f"ex{example}{sublabel}{repeat}:w")),
                       default=0)
            continue
        table_counts[block["number"] or block["section"]] += 1
        ordinal = block["caption"][0] or f'{_slug(block)}:t{table_counts[block["number"] or block["section"]]}'
        units.extend(_paradigm(rows, block, index, page, ordinal))
    return units


LANGUAGE_LABEL_RE = re.compile(r"^\((Arabic|Domari|Kurdish|Turkish|Persian)\)$")


def _interlinear(rows, block, index, page, example, sublabel, word=0) -> list[dict]:
    units: list[dict] = []
    tag = f"ex{example}{sublabel}"
    language = next((LANGUAGE_LABEL_RE.match(_unmark(cell).strip()).group(1)
                     for row in rows for cell in row
                     if LANGUAGE_LABEL_RE.match(_unmark(cell).strip())), "")
    form_row = next((row for row in rows if any(ITAL_A in cell for cell in row)), None)
    if form_row is None:
        return units
    position = rows.index(form_row)
    gloss_row = rows[position + 1] if position + 1 < len(rows) else []
    forms = [(offset, cell) for offset, cell in enumerate(form_row) if ITAL_A in cell
             and not SUBLABEL_RE.match(_unmark(cell).strip())
             and not LANGUAGE_LABEL_RE.match(_unmark(cell).strip())]
    for offset, cell in forms:
        word += 1
        gloss = gloss_row[offset] if offset < len(gloss_row) else ""
        units.append(dict(
            unit=f"{tag}:w{word}",
            block=index,
            offset=word,
            region="example",
            section=block["section"],
            page=page,
            raw_form=_clean(" ".join(ITALIC_RE.findall(cell))),
            raw_phonetic="", raw_before="", raw_after="",
            raw_gloss=_clean(gloss.replace(ITAL_A, "").replace(ITAL_B, "")),
            context=(f"example ({example}{sublabel}) word {word}"
                     + (f"; the article labels this example {language}" if language else "")),
        ))
    for row in rows[position + 1:]:
        for cell in row:
            for quote in QUOTED_RE.finditer(cell):
                units.append(dict(
                    unit=f"{tag}:translation",
                    block=index,
                    offset=999,
                    region="translation",
                    section=block["section"],
                    page=page,
                    raw_form="", raw_phonetic="", raw_before="", raw_after="",
                    raw_gloss=_clean(quote.group(1)),
                    context=f"free translation of example ({example}{sublabel})",
                ))
    return units


def _paradigm(rows, block, index, page, ordinal) -> list[dict]:
    """Cells of a captioned table, keyed by the article's own table number."""
    units: list[dict] = []
    label_prefix = f"t{ordinal}" if str(ordinal).isdigit() else ordinal
    caption = f'Table {ordinal}: {block["caption"][1]}' if block["caption"][0] else "uncaptioned table"
    header = [cell.replace(ITAL_A, "").replace(ITAL_B, "").strip() for cell in (rows[0] if rows else [])]
    for row_index, row in enumerate(rows):
        label = row[0].replace(ITAL_A, "").replace(ITAL_B, "").strip() if row else ""
        for column, cell in enumerate(row):
            if ITAL_A not in cell:
                continue
            for repeat, italic in enumerate(ITALIC_RE.findall(cell), start=1):
                column_label = header[column] if column < len(header) else ""
                suffix = f":{repeat}" if repeat > 1 else ""
                units.append(dict(
                    unit=f"{label_prefix}:r{row_index}c{column}{suffix}",
                    block=index,
                    offset=row_index * 100 + column * 10 + repeat,
                    region="paradigm",
                    section=block["section"],
                    page=page,
                    raw_form=_clean(italic),
                    raw_phonetic="", raw_before="", raw_after="",
                    raw_gloss="",
                    context=f"{caption} | row label {label!r} | column label {column_label!r}",
                ))
    return units



# --------------------------------------------------------------------------------------
# Gloss and tag curation
# --------------------------------------------------------------------------------------

# The article's own list of abbreviations (p. 50) mapped onto Jambu's canonical tags.
# ``SUB`` is Herin's subject marker and ``SUBJ`` his subjunctive; the two must not merge.
GRAMMATICAL = {
    "ABL": "abl", "ACC": "acc", "AD": "ade", "CAUS": "caus",
    "CM": "contextualiser", "COM": "comitative", "COMP": "complementizer",
    "COP": "copula", "COUNT": "counterfactual", "DEF": "definite", "DET": "definite",
    "DEM": "demonstrative", "FUT": "fut", "IMP": "impv", "IMPFV": "ipfv", "IN": "ine",
    "INDEF": "indef", "INSTR": "instr", "NEG": "neg", "OBJ": "obj", "OBL": "obl",
    "PASS": "pass", "PRF": "perfect", "PFV": "pfv", "PROG": "progressive", "REFL": "refl",
    "REL": "relative", "RM": "remoteness", "SUB": "subj", "SUBJ": "subjunctive",
    "SUP": "superessive", "VERS": "versative", "PL": "pl", "SG": "sg",
    "1SG": "1sg", "2SG": "2sg", "3SG": "3sg", "1PL": "1pl", "2PL": "2pl", "3PL": "3pl",
    "3SF": "3sg f", "1SF": "1sg f", "f": "f", "m": "m",
}
# Glosses that name a category instead of a meaning. The lexical value follows from the
# category, not from guesswork; anything not listed here stays unglossed.
CATEGORY_GLOSS = {
    ("1sg",): "I", ("1pl",): "we", ("2sg",): "you (singular)", ("2pl",): "you (plural)",
    ("3sg",): "he, she", ("3pl",): "they", ("demonstrative",): "this, that",
    ("neg",): "not", ("fut",): "future marker", ("copula",): "to be",
    ("refl",): "self", ("relative",): "who, which", ("complementizer",): "that",
    ("obj", "3sg"): "him, her", ("obj", "3pl"): "them", ("subjunctive",): "",
}
# Interlinear glosses whose mechanical split misreads the author's intent. The first
# element is the lexical gloss, the second the tags it also implies.
GLOSS_OVERRIDES = {
    "there.is": ("there is", "verb"),
    "there.is.not": ("there is not", "verb neg"),
    "how.many": ("how many", "interr"),
    "how.much": ("how much", "interr"),
    "foot-ball": ("football", "noun"),
    "in.front.of": ("in front of", "postp"),
    "old.PL": ("old", "adj pl"),
    "far-more=COP": ("far", "degree"),
    "big-more=COP": ("big", "degree"),
    "well-behaved": ("well-behaved", "adj"),
    "no place-OBL-AD": ("place", "noun obl ade"),
    "make.sit.IMPFV.3SG": ("to seat, to make sit", "verb ipfv 3sg caus"),
    "get.up.IMP.2SG": ("to get up", "verb impv 2sg"),
    "go.out.SUBJ.1SG": ("to go out", "verb subjunctive 1sg"),
    "go.out.SUBJ.3SG": ("to go out", "verb subjunctive 3sg"),
    "go.down.SUBJ.1SG": ("to go down", "verb subjunctive 1sg"),
    "get.angry.PFV.3SG": ("to get angry", "verb pfv 3sg"),
    "be.able.IMPFV.1SG": ("to be able, can", "verb ipfv 1sg"),
    "old.woman-INDEF=COP": ("old woman", "noun indef"),
    "old.man-INDEF-OBL-SUP": ("old man", "noun indef obl superessive"),
    "this.OBL": ("this", "demonstrative obl"),
    "DEM.OBL": ("this, that", "demonstrative obl"),
    "NEG.COP.3SG-CM": ("to be", "verb copula neg 3sg contextualiser"),
    "NEG.COP.3SG-RM": ("to be", "verb copula neg 3sg remoteness"),
}
# Verbs are cited by their root plus a bare English verb; ``to`` keeps the root, its
# paradigm cells and its interlinear tokens on one gloss.
VERBAL = re.compile(r"^(?:to\s+)?(?P<verb>[a-z][a-z]*)"
                    r"(?:\s+(?:up|out|down|back|round|away|for|at|off))?$")


def parse_gloss(raw: str) -> tuple[str, list[str], bool]:
    """Split a Leipzig-style morpheme gloss into a lexical gloss and canonical tags.

    The lexical gloss comes from the first morpheme that is not purely grammatical --
    ``NEG-go.PROG.3SG`` is the verb "go" with a negative prefix -- and every morpheme
    after it is affixal. A gloss made only of category labels is resolved through
    ``CATEGORY_GLOSS`` rather than guessed at.
    """
    raw = raw.strip()
    if raw in GLOSS_OVERRIDES:
        gloss, tags = GLOSS_OVERRIDES[raw]
        return gloss, tags.split(), False
    uncertain = raw.endswith("?")
    text = raw.rstrip("?").strip()
    if not text:
        return "", [], uncertain
    # ``speak.PFV.1.SG.`` and ``fear.IMPFV.3.SG.f.`` write person and number apart.
    text = re.sub(r"\b([123])\.(SG|PL)\b", r"\1\2", text)
    # A hyphen between two lower-case English words is part of the gloss, not a morpheme
    # boundary: ``grand-father-OBL-SUP`` is one noun in the oblique superessive.
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", "\x00", text)
    tags: list[str] = []
    identity: list[str] = []
    lexical: list[str] = []
    root_seen = False
    for morpheme in re.split(r"[-=]", text):
        parts = [part.strip() for part in morpheme.split(".") if part.strip()]
        for part in parts:
            if part in GRAMMATICAL:
                mapped = GRAMMATICAL[part].split()
                tags.extend(mapped)
                identity.extend(mapped)
            elif not root_seen:
                lexical.append(part.replace("\x00", "-"))
        # Only the first non-grammatical morpheme is the root; later ones are affixal.
        root_seen = root_seen or bool(lexical)
    gloss = " ".join(lexical).strip()
    if gloss:
        # Only an explicit verbal category makes the host a verb. Nothing here infers a
        # noun from a case suffix: ``trōt-ə`` "small-OBL" is an adjective agreeing in
        # case, and a one-word English gloss cannot tell the two apart.
        if any(tag in tags for tag in VERBAL_TAGS):
            tags.append("verb")
    else:
        core = tuple(dict.fromkeys(tag for tag in identity if tag in CORE_CATEGORIES))
        gloss = CATEGORY_GLOSS.get(core, "")
        if core and core[0] in PRONOUN_GLOSS:
            tags.extend(["pron", "personal"])
    return gloss, list(dict.fromkeys(tags)), uncertain


# Categories that can stand in for a lexical meaning on their own.
CORE_CATEGORIES = ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl", "demonstrative", "neg",
                   "fut", "copula", "refl", "relative", "complementizer", "obj",
                   "subjunctive")
CASE_TAGS = ("acc", "obl", "abl", "ade", "ine", "superessive", "comitative", "versative",
             "instr", "indef")
VERBAL_TAGS = ("pfv", "ipfv", "progressive", "subjunctive", "impv", "perfect", "caus",
               "pass")


def normalize_gloss(gloss: str, verbal: bool = False) -> str:
    """Tidy an English gloss: the article's apostrophe, sense separators, verb citation.

    ``to X`` is written only where the record is independently known to be verbal -- a
    root cited with a final hyphen, a paradigm cell, or an interlinear token carrying a
    verbal category. Otherwise ``nēzək`` "close" (an adjective) and ``mangīš`` "request"
    (a deverbal noun) would be rewritten into verbs they are not.
    """
    gloss = GLOSS_TYPOS.get(gloss, gloss)
    gloss = gloss.replace("ʾ", "’").strip()
    gloss = re.sub(r"\s*;\s*", "; ", gloss)
    if verbal:
        match = VERBAL.match(gloss)
        if match and match.group("verb") in VERB_GLOSSES:
            return "to " + gloss.removeprefix("to ")
    return gloss


# Typographical slips in the author's English glosses, corrected with the raw span kept
# in the audit.
GLOSS_TYPOS = {
    "in the kichen": "in the kitchen",
    "fourty": "forty",
}


# Bare English verbs the article uses to gloss a verb root, normalized to ``to X`` so a
# root, its paradigm cells and its interlinear tokens share one gloss.
VERB_GLOSSES = {
    "arrive", "ask", "be", "become", "believe", "bend", "bring", "buy", "close", "come",
    "cough", "cut", "do", "drink", "eat", "enter", "fear", "feed", "find", "forget",
    "give", "go", "have", "hear", "help", "hide", "hit", "kill", "kiss", "know", "laugh",
    "leave", "let", "like", "live", "look", "lose", "love", "make", "meet", "open",
    "paint", "play", "praise", "prefer", "pray", "push", "quit", "remain", "return",
    "say", "see", "sit", "speak", "start", "stay", "study", "take", "travel",
    "understand", "visit", "wash", "want", "watch", "work", "burn", "disappoint",
    "drive", "put", "read", "sell", "send", "sleep", "throw", "wait", "walk", "write",
}


# --------------------------------------------------------------------------------------
# Structural curation
# --------------------------------------------------------------------------------------

# Donor and etymon statements: ``X "gloss" (< Arabic Y "gloss")``. The named language
# immediately precedes the cited form, so the form is the donor, never the Domari item.
DONOR_RE = re.compile(
    r"(?:^|[\s(<])(?:(?:colloquial|Modern Standard|Standard|Old|Middle|varieties of)\s+)*"
    r"(Arabic|Kurdish|Turkish|Persian|Greek|Armenian|Hebrew|Aramaic|Romani|Indo-Aryan)\s*$")
# The full donor label as printed, so ``Old Indo-Aryan`` is not flattened to ``Indo-Aryan``.
DONOR_LABEL_RE = re.compile(
    r"((?:(?:colloquial|Modern Standard|Standard|Old|Middle)\s+)*"
    r"(?:Arabic|Kurdish|Turkish|Persian|Greek|Armenian|Hebrew|Aramaic|Romani|Indo-Aryan))\s*$")
INDIC_DONORS = {"Indo-Aryan", "Old Indo-Aryan", "Middle Indo-Aryan"}
# A donor statement counts as this citation's etymology only when the article prints it
# beside the form, inside an open parenthesis or after ``<``. Prose that merely names a
# language -- "Matter was taken from Kurdish har", "although Arabic lissa" -- compares
# two languages and says nothing about the preceding Domari word.
PARENTHETICAL_DONOR = re.compile(r"[<]|\([^)]*$")
# The article states an etymology beside the Domari citation, as ``(< colloquial Arabic
# taxt, originally a loan from Persian)`` or ``(Arabic faḍḍal "he preferred")``. Reading
# it from the citation's own trailing text catches donors the article does not gloss.
ETYMOLOGY_HEAD = re.compile(
    r"^\s*\(\s*(?:<\s*)?(?P<label>(?:(?:colloquial|Old|Middle|Modern Standard|Standard)\s+)*"
    r"(?:Arabic|Kurdish|Turkish|Persian|Greek|Armenian|Hebrew|Aramaic|Indo-Aryan))\b")
# One donor form and, when the article gives it, its gloss. Used when the printed
# parenthesis is left unclosed, as after ``faḍḍil kar- "prefer"`` on p. 7.
DONOR_BODY = re.compile(r"[^,;)“”]*(?:“[^”]*”)?")


def etymology_statement(unit: dict) -> tuple[str, str]:
    """The donor or etymon statement the article prints beside a citation.

    Reading it from the citation's own trailing text catches the donors the article
    does not gloss (``taxt "bed" (< colloquial Arabic taxt, originally a loan from
    Persian)``), which a donor form has to be glossed to be seen at all.
    """
    after = unit.get("raw_after", "")
    head = ETYMOLOGY_HEAD.match(after)
    if not head:
        return "", ""
    rest = after[head.end():]
    close, opener = rest.find(")"), rest.find("(")
    if close != -1 and (opener == -1 or close < opener):
        body = rest[:close]
    else:
        body = DONOR_BODY.match(rest).group(0)
    body = re.sub(r"\s+([,;.])", r"\1", re.sub(r"\s+", " ", body)).strip(" ,;")
    label = head.group("label")
    return (f"< {label} {body}".strip(), label) if body else ("", "")


COMPARATIVE_FOOTNOTES = {"6", "11", "12", "13", "14", "15", "16", "21", "26", "29", "32",
                         "33", "34", "37", "38", "39", "43", "48"}
# Captioned tables that print no free lexeme.
EXCLUDED_TABLES = {
    "1": "phoneme-inventory chart, not lexical citations",
    "4": "bound pronoun suffixes, not free words",
    "5": "Layer I case suffixes, not free words",
    "6": "Layer II case markers, not free words",
}
# Tables that do print lexemes, with the gloss and tags the article itself supplies.
PARADIGM_TABLES = {
    "2": dict(gloss="", tags="pron personal"),
    "3": dict(gloss="", tags="pron personal emph"),
    "7": dict(gloss="header", tags="verb pfv"),
    "8": dict(gloss="header", tags="verb ipfv"),
    "9": dict(gloss="header", tags="verb subjunctive"),
    "10": dict(gloss="to do, to make", tags="verb progressive"),
    "11": dict(gloss="to become", tags="verb"),
    "12": dict(gloss="to do, to make", tags="verb"),
    "13": dict(gloss="to be able, can", tags="verb"),
    "14": dict(gloss="to ask, to want", tags="verb"),
    "15": dict(gloss="to be", tags="verb copula"),
}
PERSON_LABELS = {
    "1.SG": "1sg", "2.SG": "2sg", "3.SG": "3sg",
    "1.PL": "1pl", "2.PL": "2pl", "3.PL": "3pl",
    "1": "1sg", "2": "2sg", "3": "3sg",
}
COLUMN_LABELS = {
    "Perfective": "pfv", "Imperfective": "ipfv", "Subjunctive": "subjunctive",
    "Present": "pres", "Past": "pret", "Singular": "sg", "Plural": "pl",
}
PRONOUN_GLOSS = {
    "1sg": "I", "2sg": "you (singular)", "3sg": "he, she",
    "1pl": "we", "2pl": "you (plural)", "3pl": "they",
}
# Complex ("light") verbs are cited as a nominal element plus kar- "do" or h- "become".
COMPLEX_VERB_RE = re.compile(r"\s(?:kar-?|h\s*-|\(h\)\s*-|hr-)$")
# Sections whose multi-word citations are lexical list items rather than phrases.
LIST_SECTIONS = {"s2.7": "num"}

# Per-unit curation overrides, each read from the printed page. ``x-compare`` marks
# another scholar's material or another Dom variety; ``x-donor`` an etymon or a model
# cited in the language it belongs to; ``x-meta`` a reconstruction or an analytic citation.
_COMPARE = "another scholar's material or another Dom variety, cited for comparison"
_DONOR = "cited in its own language as a donor, etymon or structural model, not as Domari"
_META = "cited as an analysis or a reconstruction, not as an attested Aleppo form"
EXCLUSIONS: dict[str, str] = {
    # Forms cited in the language they belong to.
    "s1.2:q68": f"{_DONOR}: Kurdish bê, the source of Domari ʋē",
    "s2.3:q3": f"{_DONOR}: the Arabic phrase maʿ baʿḍ, which the Domari construction parallels",
    "s2.12:q19": f"{_DONOR}: the Arabic preposition ʿala",
    "s2.12:q23": f"{_DONOR}: the Arabic preposition ʿala",
    "s2.12:q28": f"{_DONOR}: the Arabic preposition ʿind",
    "s2.12:q46": f"{_DONOR}: the Arabic verb xāf",
    "s2.12:q47": f"{_DONOR}: the Arabic preposition min",
    "s2.13:q7": f"{_DONOR}: the Persian preposition az",
    "s2.13:q14": f"{_DONOR}: the Arabic preposition ʿind",
    "s2.13:q36": f"{_DONOR}: Arabic maʿ, which the article says did not enter Aleppo Domari",
    "s2.13:q37": f"{_DONOR}: Arabic fi, which the article says did not enter Aleppo Domari",
    "s2.13:q38": f"{_DONOR}: Arabic min, which the article says did not enter Aleppo Domari",
    "s2.13:q39": f"{_DONOR}: Arabic ʿala, which the article says did not enter Aleppo Domari",
    "s2.13:q40": f"{_DONOR}: Arabic la, which the article says did not enter Aleppo Domari",
    "s2.14:q24": f"{_DONOR}: the Kurmanji preposition ji",
    "s2.14:q25": f"{_DONOR}: the Arabic preposition min",
    "s2.14:q28": f"{_DONOR}: Arabic ktīr, the base of borrowed aktar",
    "s2.14:q30": f"{_DONOR}: Arabic ḥasan, the base of borrowed aḥsan",
    "s2.14:q34": f"{_DONOR}: the Arabic preposition qadd, a possible origin of qattã",
    "s2.15:q16": f"{_DONOR}: the Turkish morpheme karşɪ",
    "s2.16:q11": f"{_DONOR}: the colloquial Arabic interrogative kam ~ akamm",
    "s3.2:q3": f"{_DONOR}: the Kurdish light verb kirin",
    "s3.2:q4": f"{_DONOR}: the Kurdish light verb bûn",
    "s3.2:q64": f"{_DONOR}: the Arabic idiom xayyabt amal-i",
    "s3.2:q65": f"{_DONOR}: the Arabic verb xayyab",
    "s3.2:q66": f"{_DONOR}: the Arabic root x-y-b",
    "s3.2:q67": f"{_DONOR}: the Arabic noun amal",
    "s3.2:q69": f"{_DONOR}: the Arabic stem y-xīb",
    "s3.6:q18": f"{_DONOR}: the Levantine Arabic pseudo-verb bidd-",
    "s6.2:q11": f"{_DONOR}: Arabic kull, a component of the replicated kull-mā",
    "fn19:q1": f"{_DONOR}: Arabic waqt, the source of Domari waxti",
    # Other Dom varieties and earlier descriptions.
    "s2.9:q2": f"{_COMPARE}: Palestinian Domari lāfty-ā",
    "s2.9:q3": f"{_COMPARE}: Palestinian Domari kəry-ā",
    "s2.11:q2": f"{_COMPARE}: the article marks this example as from the dialect of Beirut",
    "s2.11:q3": f"{_COMPARE}: the article marks this example as from the dialect of Beirut",
    "s2.15:q32": f"{_COMPARE}: amakera, attested by Macalister but not in Aleppo",
    "5-negation-strategies:q6": f"{_COMPARE}: Palestinian Domari from Matras (1999:31)",
    "5-negation-strategies:q7": f"{_COMPARE}: Palestinian Domari from Matras (1999:31)",
    "5-negation-strategies:q8": f"{_COMPARE}: Palestinian Domari from Matras (1999:31)",
    "7-conclusion:q2": f"{_COMPARE}: Palestinian Domari gara",
    "7-conclusion:q3": f"{_COMPARE}: Palestinian Domari garī",
    # Analytic citations.
    "s3.1:q19": f"{_META}: the auxiliary (a)ččh- Matras posits behind the subjunctive -č-",
    "introduction:q6": f"{_DONOR}: the Arabic nickname akkālīn zēt for one Dōm group",
}
# Printed citations whose extent the surrounding punctuation does not resolve.
FORM_OVERRIDES: dict[str, str] = {
    # p. 15: "ʿan ġafle "suddenly" (also ġafl-ē-ki), faǧʾatan, taqrīban "almost"". The
    # gloss scopes over taqrīban only; faǧʾatan is listed without a gloss of its own.
    "s2.8:q14": "taqrīban",
}
# Aleppo forms the article prints inside an otherwise comparative footnote or section.
INCLUSIONS: dict[str, str] = {
    "fn37:q5": "the article gives this as the corresponding Aleppo form",
    "fn37:q6": "the article gives this as the corresponding Aleppo form",
    # Multi-word citations the article glosses as one lexical item: fixed expressions,
    # and complex predicates whose nominal element is not separately glossed.
    **{unit: "fixed expression glossed as a unit" for unit in (
        "introduction:q8", "s2.4:q4", "s2.8:q10", "s2.8:q13", "s2.9:q8", "s2.15:q36",
        "s2.16:q5", "s2.16:q12", "s2.16:q17", "s2.16:q23", "s2.16:q26", "s2.16:q27",
        "s2.16:q49", "7-conclusion:q16")},
    **{unit: "complex predicate cited and glossed as one verb" for unit in (
        "s1.2:q39", "s1.2:q47", "s1.2:q60", "s1.2:q70", "s2.2:q27", "s3.1:q59",
        "s3.1:q60", "s3.1:q61", "s3.2:q27", "s3.2:q28", "s3.2:q30", "s3.2:q31",
        "s3.2:q33", "s3.2:q54", "s3.2:q55", "s3.2:q56", "s3.2:q59", "s3.3:q23",
        "s3.3:q51", "5-negation-strategies:q9", "5-negation-strategies:q13", "fn46:q1")},
}
UNIT_TAGS: dict[str, str] = {
    # The article introduces these two roots as the only inherited Indo-Aryan modals.
    "s3.5:q1": "verb stem modal",
    "s3.5:q3": "verb stem",
}

EXCLUSION_REASONS = {
    "x-quote": "quoted English prose or a title, not a glossed Domari citation",
    "x-phonetic": "cited only as a phonetic realisation, with no citation form printed",
    "x-donor": "donor or etymon form cited in another language, not a Domari attestation",
    "x-compare": "another scholar's material from a different Dom variety, cited for comparison",
    "x-phrase": "phrase or clause example whose component lexemes are glossed elsewhere",
    "x-translation": "free translation of a numbered example",
    "x-inventory": "phoneme-inventory chart cell",
    "x-affix": "bound affix or case marker, not a free word",
    "x-nolex": "no lexical gloss: the source glosses this token by grammatical category only",
    "x-meta": "metalinguistic citation of a morpheme or a reconstruction, not an attested word",
}



# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------

OPTIONAL_RE = re.compile(r"\(([^()]*)\)")
INLINE_GLOSS_RE = re.compile(r"^\s*\(([^)]*)\)")
SECTION_ID_RE = re.compile(r"^s((?:\d+\.)*\d+)$")
EXAMPLE_ID_RE = re.compile(r"^ex(\d+)([a-h]?)(-r\d+)?:(w\d+|translation)$")
TABLE_ID_RE = re.compile(r"^t(\d+):r(\d+)c(\d+)(?::(\d+))?$")


def locator(unit: dict) -> str:
    """The CLDF locator: printed page plus the article's own citation unit."""
    page, uid = unit["page"], unit["unit"]
    example = EXAMPLE_ID_RE.match(uid)
    if example:
        number, letter, _, part = example.groups()
        where = "free translation" if part == "translation" else f"word {part[1:]}"
        return f"{SOURCE_ID}[p. {page}, example ({number}{letter}) {where}]"
    table = TABLE_ID_RE.match(uid)
    if table:
        # The cell is named by the article's own row and column labels. Neither may carry
        # a semicolon, which separates CLDF citations from one another.
        labels = [part.split(" label ", 1)[1].strip("'")
                  for part in unit["context"].split(" | ")[1:]]
        cell = ", ".join(label for label in labels if label)
        return f"{SOURCE_ID}[p. {page}, Table {table.group(1)}, {cell}]"
    head = uid.split(":")[0]
    section = SECTION_ID_RE.match(head)
    if section:
        return f"{SOURCE_ID}[p. {page}, section {section.group(1)}]"
    if head.startswith("fn"):
        return f"{SOURCE_ID}[p. {page}, footnote {head[2:]}]"
    if head == "abstract":
        return f"{SOURCE_ID}[p. {page}, abstract]"
    if head[0].isdigit():
        return f"{SOURCE_ID}[p. {page}, section {head.split('-')[0]}]"
    return f"{SOURCE_ID}[p. {page}, {head}]"


def expand_optional(form: str) -> list[str]:
    """``(h)ōčəm`` -> the printed reading plus the reading without the optional segment."""
    groups = list(OPTIONAL_RE.finditer(form))
    if not groups:
        return [form]
    if len(groups) > 2:
        return [OPTIONAL_RE.sub(r"\1", form)]
    readings: list[str] = []
    for mask in range(1 << len(groups)):
        index = [0]

        def replace(match: re.Match) -> str:
            keep = not (mask >> index[0]) & 1
            index[0] += 1
            return match.group(1) if keep else ""

        candidate = re.sub(r"\s+", " ", OPTIONAL_RE.sub(replace, form)).strip()
        if candidate and candidate not in readings:
            readings.append(candidate)
    return readings


def split_alternates(form: str, cells: bool = False) -> list[str]:
    """Split a printed alternation. Table cells also separate readings with ``;`` or ``,``."""
    pattern = r"\s*[~;,/]\s*" if cells else r"\s*[~/]\s*"
    return [tidy_form(part) for part in re.split(pattern, form) if tidy_form(part)]


def align_alternates(alternates: list[str]) -> list[str]:
    """Complete an alternation elided across a light verb.

    ``dahn ~ dāhín kar-`` cites two pronunciations of one complex verb, not the bare
    nominal ``dahn``; only the nominal element varies, so the light verb is restored.
    Alternants that are complete on their own -- ``trən ʋīst ʋīst ~ štār ʋīst`` "eighty"
    -- are left exactly as printed.
    """
    if len(alternates) < 2:
        return alternates
    last = alternates[-1].split()
    if len(last) != 2 or not COMPLEX_VERB_RE.search(alternates[-1]):
        return alternates
    return [part if len(part.split()) > 1 else f"{part} {last[-1]}" for part in alternates]


def tidy_form(form: str) -> str:
    form = form.replace("?", "").replace("!", "")
    form = re.sub(r"\s+", " ", form).strip(" ,;:")
    return unicodedata.normalize("NFC", form)


def donor_label(before: str) -> str:
    match = DONOR_LABEL_RE.search(before)
    return match.group(1) if match else ""


def inline_gloss(unit: dict) -> str:
    """The parenthesised morpheme gloss the article often prints after a translation."""
    match = INLINE_GLOSS_RE.match(unit.get("raw_after", ""))
    if not match:
        return ""
    inner = match.group(1).strip()
    if not re.search(r"[A-Z]{2,}|\.\d", inner):
        return ""
    return inner


def classify(unit: dict) -> tuple[str, str]:
    """Assign an audit status and reason to one raw record."""
    uid, region = unit["unit"], unit["region"]
    if uid in EXCLUSIONS:
        return "skipped", EXCLUSIONS[uid]
    if "-r" in uid.split(":")[0]:
        return "skipped", ("second printing of the same numbered example; installed "
                           "from its first occurrence")
    if region == "translation":
        return "skipped", EXCLUSION_REASONS["x-translation"]
    if region == "paradigm":
        if unit["raw_form"].startswith("-"):
            return "skipped", EXCLUSION_REASONS["x-affix"]
        table = uid.split(":")[0]
        if table.startswith("t") and table[1:] in EXCLUDED_TABLES:
            return "skipped", EXCLUDED_TABLES[table[1:]]
        if not TABLE_ID_RE.match(uid):
            return "skipped", "uncaptioned front- or back-matter table"
        return "ingested", ""
    if region == "example":
        label = re.search(r"labels this example (\w+)", unit["context"])
        if label and label.group(1) != "Domari":
            return "skipped", (f"{EXCLUSION_REASONS['x-compare']} "
                               f"(the article labels example {uid.split(':')[0][2:]} "
                               f"{label.group(1)})")
        return "ingested", ""
    head = uid.split(":")[0]
    if head.startswith("fn") and head[2:] in COMPARATIVE_FOOTNOTES and uid not in INCLUSIONS:
        return "skipped", EXCLUSION_REASONS["x-compare"]
    if DONOR_RE.search(unit["raw_before"]):
        return "skipped", EXCLUSION_REASONS["x-donor"]

    if unit["raw_form"].startswith("-"):
        return "skipped", EXCLUSION_REASONS["x-affix"]
    if not unit["raw_form"]:
        if unit["raw_phonetic"] and uid in INCLUSIONS:
            return "ingested", ""
        return ("skipped", EXCLUSION_REASONS["x-phonetic"] if unit["raw_phonetic"]
                else EXCLUSION_REASONS["x-quote"])
    return "ingested", ""


def emit(unit: dict, forms: list[str], gloss: str, tags: list[str], suffix: str = "",
         phonemic: str = "", etymology: str = "", notes: str = "") -> list[dict]:
    """Build the installed rows for one decision, expanding optional segments.

    The first reading is the head; every further reading -- a printed ``~`` alternant or
    a form without its optional segment -- becomes an ``alternate`` row pointing back at
    the head through ``Variant_Of_Key``.
    """
    base = f"{SOURCE_ID}:{unit['unit']}{suffix}"
    readings: list[str] = []
    for form in dict.fromkeys(tidy_form(form) for form in forms):
        for reading in expand_optional(form):
            if reading and reading not in readings:
                readings.append(reading)
    rows: list[dict] = []
    for index, reading in enumerate(readings):
        key = base if index == 0 else f"{base}:variant:{index + 1}"
        rows.append(dict(
            Language_ID=LANGUAGE_ID, Parameter_ID="", Form=reading, Gloss=gloss,
            Native="", Phonemic=phonemic if index == 0 else "", Notes=notes,
            Source=locator(unit), Cognateset="", Etymology=etymology if index == 0 else "",
            Entry_Key=key, Variant_Of_Key="" if index == 0 else base,
            Borrowed_From_Key="", Derivation_Parent_Keys="",
            Tags=" ".join(dict.fromkeys(
                list(tags) + (["alternate"] if index else []) + [DIALECT_TAG])),
        ))
    return rows


def paradigm_record(unit: dict) -> tuple[str, list[str], str]:
    """Gloss, tags and reason for one cell of a captioned paradigm table."""
    table = unit["unit"].split(":")[0][1:]
    spec = PARADIGM_TABLES.get(table)
    if spec is None:
        return "", [], "table is not a curated lexeme paradigm"
    context = unit["context"]
    row_label = re.search(r"row label '([^']*)'", context)
    column_label = re.search(r"column label '([^']*)'", context)
    row_label = row_label.group(1).strip().rstrip(".") if row_label else ""
    column_label = column_label.group(1).strip() if column_label else ""
    tags = spec["tags"].split()
    gloss = spec["gloss"]
    if gloss == "header":
        # Tables 7--9 head each column with the verb root and its gloss.
        printed = QUOTED_RE.search(column_label)
        if not printed:
            return "", [], "column header carries no glossed lexeme"
        gloss = normalize_gloss(printed.group(1), verbal=True)
        if not row_label:
            tags = [tag for tag in tags if tag not in {"pfv", "ipfv", "subjunctive"}]
            tags.append("stem")
    person = PERSON_LABELS.get(row_label, "")
    if person:
        tags.append(person)
    if column_label in COLUMN_LABELS:
        tags.append(COLUMN_LABELS[column_label])
    if table in {"2", "3"}:
        # The free-pronoun tables index person by row and number by column.
        number = COLUMN_LABELS.get(column_label, "sg")
        person = PERSON_LABELS.get(row_label, "")
        person = person[0] + number if person else ""
        tags = [tag for tag in tags if tag not in PERSON_LABELS.values()]
        if person:
            tags.append(person)
        gloss = PRONOUN_GLOSS.get(person, "")
        if not gloss:
            return "", [], "pronoun cell without a resolvable person and number"
    return gloss, list(dict.fromkeys(tags)), ""


def prose_rows(unit: dict) -> tuple[list[dict], str, str, list[str], str]:
    """Curate one prose citation.

    A citation whose translation is followed by a matching word-by-word gloss --
    ``məns-a krī "the house of the man" (man-OBL house)`` -- is an inline interlinear
    example and is emitted one row per word. A single word is an ordinary lexical
    citation. Any other multi-word citation is a phrase or clause example unless the
    article prints it as a lexical item.
    """
    # A citation is verbal when its gloss is: the article cites roots with a final hyphen
    # (``sāk-`` "can, be able") but also bound prepositions the same way (``z-`` "from").
    printed = tidy_form(unit["raw_form"])
    verbal = bool(
        (printed.endswith("-") or COMPLEX_VERB_RE.search(printed))
        and unit["raw_gloss"].split()[:1]
        and unit["raw_gloss"].split()[0].strip(",").lower() in VERB_GLOSSES)
    gloss = normalize_gloss(unit["raw_gloss"], verbal=verbal)
    alternates = align_alternates(
        split_alternates(FORM_OVERRIDES.get(unit["unit"], unit["raw_form"])))
    if not alternates:
        alternates = [tidy_form(split_alternates(unit["raw_phonetic"])[0])] \
            if unit["raw_phonetic"] else []
    if not alternates:
        return [], "skipped", EXCLUSION_REASONS["x-quote"], [], ""
    head = alternates[0]
    words = head.split()
    inline = inline_gloss(unit)
    inline_words = inline.split()
    section = unit["unit"].split(":")[0]
    phonemic = "; ".join(split_alternates(unit["raw_phonetic"]))

    if len(words) > 1 and len(inline_words) == len(words):
        rows: list[dict] = []
        aligned = [part for part in alternates if len(part.split()) == len(words)]
        for index, morpheme in enumerate(inline_words):
            word_gloss, tags, uncertain = parse_gloss(morpheme)
            word_gloss = normalize_gloss(word_gloss, verbal="verb" in tags)
            if not word_gloss:
                continue
            if uncertain:
                tags = tags + ["uncertain"]
            rows.extend(emit(unit, [part.split()[index] for part in aligned], word_gloss,
                             tags, suffix=f":w{index + 1}"))
        if not rows:
            return [], "skipped", EXCLUSION_REASONS["x-nolex"], [], gloss
        return rows, "ingested", "", [], gloss

    tags: list[str] = []
    if inline and len(inline_words) == 1 and len(words) == 1:
        _, tags, uncertain = parse_gloss(inline)
        if uncertain:
            tags = tags + ["uncertain"]
    if section in LIST_SECTIONS:
        tags = tags + [LIST_SECTIONS[section]]
    if verbal and len(words) == 1:
        tags = tags + ["verb"] + (["stem"] if printed.endswith("-") else [])
    if len(words) > 1:
        if COMPLEX_VERB_RE.search(head):
            tags = tags + ["verb", "conjunct-verb"]
        elif section not in LIST_SECTIONS and unit["unit"] not in INCLUSIONS:
            return [], "skipped", EXCLUSION_REASONS["x-phrase"], [], gloss
        tags = tags + ["multiword-expression"]
    if not gloss:
        return [], "skipped", EXCLUSION_REASONS["x-nolex"], [], gloss
    if unit["unit"] in UNIT_TAGS:
        tags = tags + UNIT_TAGS[unit["unit"]].split()
    notes = "" if unit["raw_form"] else "cited only in phonetic transcription"
    return (emit(unit, alternates, gloss, list(dict.fromkeys(tags)), phonemic=phonemic,
                 notes=notes),
            "ingested", "", tags, gloss)


def audit_row(unit: dict, status: str, reason: str, rows: list[dict], gloss: str,
              tags: list[str]) -> dict:
    payload = "|".join([unit["unit"], unit["raw_form"], unit["raw_phonetic"],
                        unit["raw_gloss"], status, reason])
    return dict(
        Snapshot_Date=SNAPSHOT_DATE, Collation_Date=COLLATION_DATE, Unit_ID=unit["unit"],
        Region=unit["region"], Section=unit["section"], Printed_Page=unit["page"],
        Raw_Form=unit["raw_form"], Raw_Gloss=unit["raw_gloss"],
        Raw_Context=unit["context"], Status=status, Reason=reason,
        Final_Forms="; ".join(row["Form"] for row in rows),
        Final_Gloss="; ".join(dict.fromkeys(row["Gloss"] for row in rows)) or gloss,
        Final_Tags="; ".join(dict.fromkeys(row["Tags"] for row in rows)),
        Etymology="; ".join(dict.fromkeys(row["Etymology"] for row in rows if row["Etymology"])),
        Etymon_Candidates="", Parameter_ID="",
        Emitted_Keys="; ".join(row["Entry_Key"] for row in rows),
        Review="", Material_Error="", Source=locator(unit),
        Record_SHA256=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    )


def build_records(units: list[dict]) -> tuple[list[dict], list[dict]]:
    """Curate every raw record into installed rows plus one audit row."""
    records: list[dict] = []
    audit: list[dict] = []
    etymologies: dict[str, str] = {}

    for unit in units:
        status, reason = classify(unit)
        rows: list[dict] = []
        gloss, tags, notes, phonemic = "", [], "", ""
        if status == "ingested":
            if unit["region"] == "paradigm":
                gloss, tags, problem = paradigm_record(unit)
                if problem or not gloss:
                    status, reason = "skipped", problem or EXCLUSION_REASONS["x-nolex"]
                else:
                    rows = emit(unit, split_alternates(unit["raw_form"], cells=True),
                                gloss, tags)
            elif unit["region"] == "example":
                gloss, tags, uncertain = parse_gloss(unit["raw_gloss"])
                gloss = normalize_gloss(gloss, verbal="verb" in tags)
                if not gloss:
                    status, reason = "skipped", EXCLUSION_REASONS["x-nolex"]
                else:
                    if uncertain:
                        tags = tags + ["uncertain"]
                    if " " in tidy_form(unit["raw_form"]):
                        tags = tags + ["multiword-expression"]
                    rows = emit(unit, [unit["raw_form"]], gloss, tags)
            else:
                rows, status, reason, tags, gloss = prose_rows(unit)
        for row in rows:
            records.append(row)
        if status == "ingested" and rows:
            statement, label = etymology_statement(unit)
            if statement:
                etymologies[rows[0]["Entry_Key"]] = statement
        audit.append(audit_row(unit, status, reason, rows, gloss, tags))

    apply_etymologies(records, etymologies)
    link_etyma(records, audit)
    return dedupe(records), audit


def link_etyma(records: list[dict], audit: list[dict]) -> None:
    """Resolve the article's Indo-Aryan etyma against CDIAL and record every candidate."""
    index = cdial_index()
    by_key = {row["Emitted_Keys"].split("; ")[0]: row for row in audit if row["Emitted_Keys"]}
    for record in records:
        if not record["Etymology"]:
            continue
        parameter, candidate = match_etymon(record["Etymology"], index)
        record["Parameter_ID"] = parameter
        row = by_key.get(record["Entry_Key"])
        if row is not None:
            row["Etymology"] = record["Etymology"]
            row["Etymon_Candidates"] = candidate
            row["Parameter_ID"] = parameter


def apply_etymologies(records: list[dict], etymologies: dict[str, str]) -> None:
    """Attach each donor or etymon statement to the Domari row it explains."""
    by_key = {record["Entry_Key"]: record for record in records}
    for key, statement in etymologies.items():
        record = by_key.get(key)
        if record is None:
            continue
        record["Etymology"] = "; ".join(filter(None, [record["Etymology"], statement]))
        label = statement.split()[1] if len(statement.split()) > 1 else ""
        donor = re.match(r"<\s+((?:Old |Middle |colloquial |Modern Standard |Standard )*"
                         r"[A-Z][A-Za-z-]+)", statement)
        name = donor.group(1) if donor else label
        if name.split()[-1] != "Indo-Aryan":
            tags = record["Tags"].split()
            for tag in ("loanword", f"loan:{name.split()[-1]}"):
                if tag not in tags:
                    tags.insert(len(tags) - 1, tag)
            record["Tags"] = " ".join(tags)


def dedupe(records: list[dict]) -> list[dict]:
    """Merge rows that repeat one attestation, keeping every citation and tag.

    A merged row's key survives as an alias so that alternates printed beside it keep
    pointing at the row that absorbed their head.
    """
    merged: dict[tuple[str, str], dict] = {}
    order: list[tuple[str, str]] = []
    alias: dict[str, str] = {}
    for record in records:
        key = (record["Form"], record["Gloss"])
        if key not in merged:
            merged[key] = record
            order.append(key)
            continue
        head = merged[key]
        alias[record["Entry_Key"]] = head["Entry_Key"]
        citations = head["Source"].split(";")
        if record["Source"] not in citations:
            head["Source"] = ";".join(citations + [record["Source"]])
        head["Tags"] = " ".join(dict.fromkeys(head["Tags"].split() + record["Tags"].split()))
        for field in ("Phonemic", "Etymology", "Notes", "Parameter_ID"):
            if not head[field] and record[field]:
                head[field] = record[field]
    surviving = [merged[key] for key in order]
    keys = {record["Entry_Key"] for record in surviving}
    for record in surviving:
        target = record["Variant_Of_Key"]
        while target and target not in keys and target in alias:
            target = alias[target]
        record["Variant_Of_Key"] = target if target in keys else ""
    return surviving



# --------------------------------------------------------------------------------------
# Indo-Aryan etyma
# --------------------------------------------------------------------------------------

CDIAL_CSV = ROOT / "data/cdial/cdial.csv"
# The article's Indo-Aryan etyma are Turner's (footnote 1); Turner prints them as the
# Old Indo-Aryan head, the Sanskrit head or the Prakrit form of a numbered entry.
CDIAL_LANGUAGES = ("Indo-Aryan", "Sk", "Pk")
INDIC_LABEL_RE = re.compile(r"^<\s*(?:Old |Middle )?Indo-Aryan\s+(?P<etymon>\S+)")


def cdial_index(path: Path = CDIAL_CSV) -> dict[str, set[str]]:
    index: dict[str, set[str]] = defaultdict(set)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.reader(stream):
            if len(row) < 3 or row[0] not in CDIAL_LANGUAGES:
                continue
            index[unicodedata.normalize("NFC", row[2])].add(row[1])
    return index


def match_etymon(statement: str, index: dict[str, set[str]]) -> tuple[str, str]:
    """Resolve a printed Indo-Aryan etymon to a CDIAL entry number.

    Only an exact headword match with a single entry number is accepted. Vowel length and
    quality are never normalized away, so ``pēṭṭa`` does not silently become Turner's
    ``*peṭṭa``; near misses are reported as unresolved candidates in the audit instead.
    """
    printed = INDIC_LABEL_RE.match(statement)
    if not printed:
        return "", ""
    etymon = unicodedata.normalize("NFC", printed.group("etymon").strip("*"))
    entries = index.get(etymon, set())
    if len(entries) == 1:
        return next(iter(entries)), f"{etymon} -> CDIAL {next(iter(entries))} (exact head)"
    if not entries:
        return "", f"{etymon} -> no exact CDIAL head; left unlinked"
    return "", f"{etymon} -> CDIAL {', '.join(sorted(entries))} (ambiguous); left unlinked"




# --------------------------------------------------------------------------------------
# Validation and output
# --------------------------------------------------------------------------------------

def canonical_tags(records: list[dict]) -> set[str]:
    return {tag for record in records for tag in record["Tags"].split()
            if not tag.startswith("dialect:") and not tag.startswith("loan:")}


def validate(records: list[dict], audit: list[dict]) -> None:
    keys = [record["Entry_Key"] for record in records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate Entry_Key values: {duplicates[:5]}")
    for record in records:
        if not record["Form"]:
            raise ValueError(f"empty form for {record['Entry_Key']}")
        if not record["Gloss"]:
            raise ValueError(f"empty gloss for {record['Entry_Key']}")
        if "�" in "".join(record.values()):
            raise ValueError(f"replacement character in {record['Entry_Key']}")
        if record["Language_ID"] != LANGUAGE_ID:
            raise ValueError(f"unexpected language for {record['Entry_Key']}")
        if DIALECT_TAG not in record["Tags"].split():
            raise ValueError(f"missing dialect tag on {record['Entry_Key']}")
        if not record["Source"].startswith(f"{SOURCE_ID}["):
            raise ValueError(f"missing locator on {record['Entry_Key']}")
        if record["Variant_Of_Key"] and record["Variant_Of_Key"] not in keys:
            raise ValueError(f"dangling variant target on {record['Entry_Key']}")
    if len(audit) != len({row["Unit_ID"] for row in audit}):
        raise ValueError("audit unit ids are not unique")
    if any(not row["Status"] or (row["Status"] == "skipped" and not row["Reason"])
           for row in audit):
        raise ValueError("every audit row needs a status, and every exclusion a reason")


def summarize(records: list[dict], audit: list[dict]) -> dict:
    statuses = Counter(row["Status"] for row in audit)
    return {
        "source": SOURCE_ID,
        "article_url": ARTICLE_URL,
        "pdf_url": PDF_URL,
        "html_sha256": HTML_SHA256,
        "pdf_sha256": PDF_SHA256,
        "snapshot_date": SNAPSHOT_DATE,
        "collation_date": COLLATION_DATE,
        "printed_pages": [FIRST_PRINTED_PAGE, LAST_PRINTED_PAGE],
        "raw_records": len(audit),
        "raw_records_by_region": dict(Counter(row["Region"] for row in audit)),
        "status_counts": dict(statuses),
        "exclusion_reasons": dict(Counter(
            row["Reason"] for row in audit if row["Status"] == "skipped")),
        "installed_rows": len(records),
        "installed_alternates": sum(1 for record in records
                                    if "alternate" in record["Tags"].split()),
        "rows_with_phonemic": sum(1 for record in records if record["Phonemic"]),
        "rows_with_etymology": sum(1 for record in records if record["Etymology"]),
        "rows_linked_to_cdial": sum(1 for record in records if record["Parameter_ID"]),
        "languages": [LANGUAGE_ID],
        "dialects": [DIALECT_TAG],
        "tags": sorted(canonical_tags(records)),
    }


def sample(records: list[dict], size: int = 25) -> list[dict]:
    """A deterministic spread across the article for the checked-in review sample."""
    if len(records) <= size:
        return records
    step = len(records) / size
    return [records[int(index * step)] for index in range(size)]


def write_outputs(records: list[dict], audit: list[dict], install: bool) -> dict:
    manifest = summarize(records, audit)
    if not install:
        return manifest
    with FORM_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for record in records:
            writer.writerow([record[field] for field in FORM_FIELDS])
    with AUDIT_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit)
    with SAMPLE_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FORM_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sample(records))
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
    return manifest



def extract(cache: Path, refresh: bool) -> list[dict]:
    markup = snapshot(cache, "article-412.html", ARTICLE_URL, HTML_SHA256, refresh)
    pdf = snapshot(cache, "article-412.pdf", PDF_URL, PDF_SHA256, refresh)
    blocks = read_blocks(markup.read_text(encoding="utf-8"))
    pages = page_index(blocks, pdf)
    units = prose_units(blocks, pages) + table_units(blocks, pages)
    # Document order: every unit carries the index of the block it was read from, so the
    # audit, the donor attachments and the emitted rows all follow the printed article.
    units.sort(key=lambda unit: (unit["block"], unit.get("offset", 0)))
    for sequence, unit in enumerate(units, start=1):
        unit["seq"] = sequence
    return units


def write_extract(units: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="|", lineterminator="\n")
        writer.writerow(EXTRACT_FIELDS)
        for unit in units:
            writer.writerow([unit["seq"], unit["unit"], unit["region"], unit["section"],
                             unit["page"], unit["raw_form"], unit.get("raw_phonetic", ""),
                             unit.get("raw_before", ""), unit["raw_gloss"],
                             unit.get("raw_after", ""), unit["context"]])


def read_extract(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="|"))
    return [dict(seq=int(row["Sequence"]), unit=row["Unit_ID"], region=row["Region"],
                 section=row["Section"], page=int(row["Printed_Page"]), raw_form=row["Raw_Form"],
                 raw_phonetic=row["Raw_Phonetic"], raw_before=row["Raw_Before"],
                 raw_gloss=row["Raw_Gloss"], raw_after=row["Raw_After"],
                 context=row["Raw_Context"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--refresh", action="store_true",
                        help="fetch the article snapshots before extracting")
    parser.add_argument("--offline", action="store_true",
                        help="rebuild from the checked-in extraction snapshot")
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()

    if args.offline:
        units = read_extract(EXTRACT_SNAPSHOT)
    else:
        units = extract(args.cache, args.refresh)
        write_extract(units, EXTRACT_SNAPSHOT)
    records, audit = build_records(units)
    validate(records, audit)
    manifest = write_outputs(records, audit, args.install)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Snapshot Fran Woods' Halbi--English Dictionary from its Webonary FLEx export.

Webonary publishes the dictionary as a FLEx XHTML export paginated 25 records to a page under
``browse/browse-vernacular/?letter=<letter>&key=hlb&paged=<n>``.  The markup is the configured
FLEx export rather than prose, so every field this importer reads is a source-authored element,
not a heuristic slice of running text:

* ``div.entry[id]``               -- the FLEx entry GUID, used verbatim as the stable entry key
* ``span.mainheadword``           -- Devanagari headword (``hlb``)          -> ``Native``
* ``span.citationform``           -- Woods' phonemic IPA (``hlb-fonipa``)   -> ``Phonemic``
* ``span.sharedgrammaticalinfo``  -- part of speech shared across senses    -> ``Tags``
* ``span.sense[entryguid]``       -- one sense, attributed to its owning entry by GUID
* ``span.definition``/``definitionorgloss`` -- ``Eng`` and ``Hin`` definitions
* ``span.generalnote``/``encyclopedicinfo``  -- usage notes                 -> ``Notes``
* ``span.semanticdomain``         -- SIL semantic-domain code and label     -> ``Tags``
* ``span.example``/``span.translation``      -- illustrative sentences      -> audit only
* ``span.complexformentryref``    -- ``comp. of`` / ``ph. v. of`` / ``redup. of`` components
* ``span.visiblevariantentryref`` -- this entry's own ``dial. var. of`` / ``borr. fr. Hin`` target
* ``span.variantformentrybackref``-- typed variant back-references, incl. ``Hin. borr.``
* ``span.minimallexreference``    -- ``cf`` cross-references

Relation direction matters and the export encodes it in the class name, not the label.
``complexformentryrefs`` sits on a complex form and names its components (``comp. of``);
``complexformsnotsubentries`` sits on a *component* and names the complex forms built from it
(bare ``comp.``).  Only the former is a derivation parent -- reading the latter would invert
every compound in the dictionary.  The same holds for variants: ``visiblevariantentryrefs`` is
this entry's own relation to its main entry, while ``variantformentrybackrefs`` is the reverse
view rendered on the main entry.

``span.sense`` carries ``entryguid``, so senses are attributed by that attribute rather than by
DOM position.  Without it a subentry's senses are silently read as the head entry's -- ``घड़ूक``
'to flare' would acquire its subentry ``घड़ घड़ूक`` 'to flare up' as a third sense.  Subentries and
the ``minorentrycomplex`` stubs that repeat them are parsed as entries in their own right and
deduplicated on GUID.

What is installed
-----------------
One row per entry that has both a headword and a citation form.  Multiple senses are joined into
one numbered gloss following the house convention already used for the Kullui import, and the
per-sense definitions, notes, examples and domains stay in the audit at full fidelity.

What is deliberately *not* installed
------------------------------------
* Etymologies.  Woods prints none, so every row is installed unetymologised (a lone node) exactly
  as the Gondi survey lists are.  Linking Halbi headwords to CDIAL on shape alone is not
  defensible and is left to a separate reviewed pass.
* ``Borrowed_From_Key``.  Woods states the donor as a language label (``borr. fr. Hin``,
  ``borr. fr. Eng``), not as a cited donor form, so there is no entry key to point at.  The claim
  is kept as a ``borrowed:hin``/``borrowed:eng`` tag and in the audit.  The reverse ``Hin. borr.``
  back-reference asserts nothing about the entry carrying it and is recorded only.
* The ``bgw`` and ``ori`` variant labels.  Both sides of these pairs are tagged ``hlb`` in the
  source, so they are Bhatri- and Odia-associated Halbi variants rather than Bhatri or Odia
  records.  They stay under Halbi with the label preserved and flagged for review.
* Example sentences and their free translations, which are text rather than lexemes.
* Pictures, audio and the semantic-domain browse pages.

Transcription
-------------
``conversion/halbi-woods.txt`` maps Woods' phonemic IPA onto Jambu's house transcription,
following ``conversion/chattisgarhi.txt`` -- the profile for Halbi's nearest neighbour.
``Phonemic`` and ``Original`` keep the source string untouched.

The one decision that needed evidence is vowel length. Woods' Devanagari headwords use only ी
and ू; ि and ु never occur anywhere in the dictionary, and her IPA writes both as plain ``i``
and ``u``. Her transcription therefore encodes quality, not length: ``ə`` is the inherent short
vowel and ``a i u`` are the long ones. They map to ``a`` and ``ā ī ū`` respectively, so घर
``ɡʰər`` becomes ``gʰar`` and अदालत ``ədalət`` becomes ``adālat``. Nasalisation rides on the
converted vowel (``ə̃`` -> ``ã``, ``ã`` -> ``ā̃``).

``ʃ`` and ``ʒ`` never stand alone in this source: all 875 and 854 occurrences are the tails of
``tʃ``/``dʒ``, which become the house palatals ``c``/``j``. ``j`` itself occurs once, in काँजी
``kãji``, where the Devanagari shows it is /dʒ/ rather than the glide Woods otherwise writes
``y``, so it maps to house ``j``.

Usage (run from ``data/``)::

    uv run --with curl_cffi python data/other/forms/raw_data/woods_halbi.py --refresh
    uv run --with curl_cffi python data/other/forms/raw_data/woods_halbi.py --refresh --install
    uv run python data/other/forms/raw_data/woods_halbi.py --offline --install

``--refresh`` needs ``curl_cffi``: webonary.org sits behind a TLS-fingerprinting challenge that
plain ``urllib`` cannot complete.  Responses are cached under ``tmp/webonary-halbi-cache`` keyed
by URL, so an interrupted crawl resumes and ``--offline`` rebuilds the pinned snapshot with no
network access.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from dialects import dialect_tag  # noqa: E402

SOURCE_ID = "woods2019halbi"
SNAPSHOT_DATE = "2026-08-26"
LANGUAGE_ID = "hal"
BASE_URL = "https://www.webonary.org/halbi/browse/browse-vernacular/"
USER_AGENT = "Jambu dictionary importer/1.0 (https://github.com/moli-mandala)"
PAGE_SIZE = 25
MAX_PAGES_PER_LETTER = 60

CACHE_DIR = ROOT / "tmp/webonary-halbi-cache"
RAW_DIR = ROOT / "data/other/forms/raw_data"
FORM_OUTPUT = ROOT / "data/other/forms/20260826-woods-halbi.csv"
AUDIT_OUTPUT = RAW_DIR / "20260826-woods-halbi-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260826-woods-halbi-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260826-woods-halbi-manifest.json"

# Woods worked in one village with one speech community; the whole dictionary is that lect.
DIALECT_SOURCE_ID = "woods2019halbi-BHATPAL"
DIALECT_NAME = "Bhatpal (Woods 2019)"
DIALECT_TAG = dialect_tag(LANGUAGE_ID, DIALECT_SOURCE_ID, DIALECT_NAME)

# The vernacular browse index, in the dictionary's own alphabet order.
LETTERS = [
    "अँ", "अ", "आँ", "आ", "ईं", "ई", "ऊँ", "ऊ", "एँ", "ए", "ओं", "ओ",
    "क", "ख", "ग", "घ", "च", "छ", "ज", "झ", "ट", "ठ", "ड", "ढ",
    "त", "थ", "द", "ध", "न", "प", "फ", "ब", "भ", "म", "य", "र", "ल", "स", "ह",
]

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Entry_Key", "GUID", "Letter", "Page", "Entry_Kind", "Headword", "Homograph",
    "IPA", "Part_Of_Speech", "Sense_Count", "Sense_Number", "Gloss_English", "Gloss_Hindi",
    "Definition_Kind", "Usage_Note", "Semantic_Domains", "Example", "Example_Translation",
    "Complex_Form_Type", "Complex_Form_Parts", "Variant_Of_Type", "Variant_Of_Target",
    "Variant_Of_Key", "Variant_Backref_Type", "Variant_Backref_Target",
    "Cross_Reference_Type", "Cross_Reference_Target", "Status", "Reason", "Record_SHA256",
]

# Forward complex-form labels: printed on the complex form, naming its components. The bare
# labels ("comp.", "ph. v.") are the reverse view and are never read as derivation parents.
DERIVATIONAL_TYPES = {
    "comp. of", "der. of", "ph. v. of", "redup. of", "id. of", "say. of",
}
# Forward variant labels: printed on the variant, naming the entry it varies from.
VARIANT_OF_TYPES = {
    "dial. var. of", "fr. var. of", "sp. var. of", "unspec. var. of",
}
# Forward borrowing labels, whose donor is a language rather than a cited form.
BORROWING_TYPES = {"borr. fr. Hin": "borrowed:hin", "borr. fr. Eng": "borrowed:eng"}
# Devanagari, to catch citation forms that were typed in the wrong writing system.
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# --------------------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------------------

def page_url(letter: str, page: int) -> str:
    url = f"{BASE_URL}?letter={quote(letter)}&key=hlb"
    return url if page == 1 else f"{url}&paged={page}"


def cache_path(url: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:16]}.html"


def fetch(url: str, *, offline: bool, session=None, delay: float = 1.5) -> str | None:
    path = cache_path(url)
    if path.exists():
        return path.read_text(encoding="utf-8")
    if offline:
        return None
    response = session.get(url, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} for {url}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text, encoding="utf-8")
    time.sleep(delay)
    return response.text


def open_session():
    try:
        from curl_cffi import requests
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "--refresh needs curl_cffi (webonary.org uses a TLS-fingerprint challenge):\n"
            "  uv run --with curl_cffi python data/other/forms/raw_data/woods_halbi.py --refresh"
        ) from exc
    return requests.Session(impersonate="chrome", headers={"User-Agent": USER_AGENT})


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------

def nearest_lang(string, stop) -> str | None:
    """The ``lang`` of the innermost element enclosing ``string`` at or below ``stop``."""
    ancestor = string.parent
    while ancestor is not None:
        if ancestor.get("lang"):
            return ancestor.get("lang")
        if ancestor is stop:
            return None
        ancestor = ancestor.parent
    return None


def lang_runs(node, lang: str) -> list[str]:
    """Text of every outermost ``lang`` span in ``node``, taking only that language's characters.

    Two FLEx habits make the naive reading wrong.  A styled run is split into sibling spans of
    the same language -- a drop-capital yields ``<span lang=en>T</span><span lang=en>hose...``
    -- so pieces are joined with no separator, since a space would give ``T hose people``.  And
    a homograph number is a nested ``lang="en"`` superscript *inside* the ``hlb`` headword, so
    strings are kept only when their innermost enclosing language is the one asked for;
    otherwise ``चार`` comes back as ``चार1``.
    """
    runs = []
    for span in node.find_all("span", lang=lang):
        ancestor = span.parent
        nested = False
        while ancestor is not None and ancestor is not node:
            if ancestor.get("lang") == lang:
                nested = True
                break
            ancestor = ancestor.parent
        if nested:
            continue
        text = "".join(
            string for string in span.find_all(string=True)
            if nearest_lang(string, span) == lang
        ).strip()
        if text:
            runs.append(text)
    return runs


def lang_text(node, lang: str) -> str:
    return " ".join(lang_runs(node, lang))


def own_referenced_entries(ref) -> list:
    """The entries ``ref`` points at, excluding those nested under a target's own relations.

    A ``visiblevariantentryref`` whose target is itself a compound renders that target's
    ``primaryentryrefs`` inside it.  Read naively, ``असी`` 'borr. fr. Hin चार कोड़ी' also appears
    to vary from चार and कोड़ी, which are components of its target, not variants of it.
    """
    found = []
    for entry in ref.find_all("span", class_="referencedentry"):
        ancestor = entry.parent
        skip = False
        while ancestor is not None and ancestor is not ref:
            classes = ancestor.get("class", []) or []
            if "referencedentry" in classes or "primaryentryref" in classes:
                skip = True
                break
            ancestor = ancestor.parent
        if not skip:
            found.append(entry)
    return found


def referenced_headword(node) -> str:
    """The headword of a referenced entry, without the headwords of its own components.

    A referenced compound renders its ``primaryentryrefs`` inside the same element, so reading
    every ``hlb`` run under it returns ``चार कोड़ी चार कोड़ी``. Its own ``headword`` span comes
    first and holds only its own text.
    """
    own = node.find("span", class_="headword")
    return lang_text(own if own is not None else node, "hlb")


def homograph_number(node) -> str:
    """The superscript homograph digit FLEx renders as an ``en`` span inside a headword."""
    for span in node.find_all("span", lang="en"):
        text = span.get_text("", strip=True)
        if text.isdigit():
            return text
    return ""


def own_senses(entry, guid: str) -> list:
    """Senses belonging to ``guid``, not to a subentry rendered inside the same element."""
    return [s for s in entry.find_all("span", class_="sense") if s.get("entryguid") == guid]


def sense_number(sense) -> str:
    """The ``sensenumber`` printed beside a sense, which is its ``sensecontent`` sibling."""
    container = sense.parent
    if container is None:
        return ""
    number = container.find("span", class_="sensenumber")
    return number.get_text("", strip=True) if number else ""


def parse_sense(sense) -> dict:
    definition = sense.find("span", class_="definition")
    kind = "definition"
    if definition is None:
        definition = sense.find("span", class_="definitionorgloss")
        kind = "gloss" if definition is not None else ""
    english = lang_text(definition, "en") if definition else ""
    hindi = lang_text(definition, "hi") if definition else ""

    note_node = sense.find("span", class_="generalnote") or sense.find(
        "span", class_="encyclopedicinfo"
    )
    note = lang_text(note_node, "en") if note_node else ""

    examples = []
    for content in sense.find_all("span", class_="examplescontent"):
        example = content.find("span", class_="example")
        translation = content.find("span", class_="translation")
        examples.append(
            {
                "text": lang_text(example, "hlb") if example else "",
                "translation": lang_text(translation, "en") if translation else "",
            }
        )

    domains = []
    for domain in sense.find_all("span", class_="semanticdomain"):
        code = domain.find("span", class_="abbreviation")
        name = domain.find("span", class_="name")
        domains.append(
            {
                "code": code.get_text("", strip=True) if code else "",
                "name": name.get_text("", strip=True) if name else "",
            }
        )

    return {
        "number": sense_number(sense),
        "kind": kind,
        "gloss_en": english,
        "gloss_hi": hindi,
        "note": note,
        "examples": examples,
        "domains": domains,
    }


def guid_of(node) -> str:
    link = node.find("a", href=True)
    if not link:
        return ""
    tail = link["href"].rstrip("/").rsplit("/", 1)[-1]
    return tail if re.fullmatch(r"g[0-9a-f-]{8,}", tail) else ""


def parse_entry(node, guid: str, kind: str) -> dict:
    headword_node = node.find("span", class_="mainheadword") or node.find(
        "span", class_="headword"
    )
    citation = node.find("span", class_="citationform")
    grammar = node.find("span", class_="sharedgrammaticalinfo")

    complex_forms = []
    for ref in node.find_all("span", class_="complexformentryref"):
        ctype = ""
        type_node = ref.find("span", class_="complexformtype")
        if type_node:
            ctype = lang_text(type_node, "en")
        parts = []
        for referenced in own_referenced_entries(ref):
            parts.append(
                {"headword": referenced_headword(referenced), "guid": guid_of(referenced)}
            )
        complex_forms.append({"type": ctype, "parts": parts})

    # This entry's own relation to the main entry it varies from.
    variant_of = []
    for visible in node.find_all("span", class_="visiblevariantentryrefs"):
        vtype = ""
        type_node = visible.find("span", class_="variantentrytype")
        if type_node:
            vtype = lang_text(type_node, "en")
        for ref in visible.find_all("span", class_="visiblevariantentryref"):
            for referenced in own_referenced_entries(ref):
                variant_of.append(
                    {
                        "type": vtype,
                        "headword": referenced_headword(referenced),
                        "guid": guid_of(referenced),
                    }
                )

    # The reverse view: entries that vary from this one.
    variants = []
    for back in node.find_all("span", class_="variantformentrybackrefs"):
        vtype = ""
        type_node = back.find("span", class_="variantentrytype")
        if type_node:
            vtype = lang_text(type_node, "en")
        for ref in back.find_all("span", class_="variantformentrybackref"):
            variants.append(
                {"type": vtype, "headword": referenced_headword(ref), "guid": guid_of(ref)}
            )

    cross_refs = []
    for ref in node.find_all("span", class_="minimallexreference"):
        rtype = ""
        type_node = ref.find("span", class_="ownertype_abbreviation")
        if type_node:
            rtype = lang_text(type_node, "en")
        cross_refs.append(
            {"type": rtype, "headword": referenced_headword(ref), "guid": guid_of(ref)}
        )

    return {
        "guid": guid,
        "kind": kind,
        "headword": lang_text(headword_node, "hlb") if headword_node else "",
        "homograph": homograph_number(headword_node) if headword_node else "",
        "ipa": lang_text(citation, "hlb-fonipa") if citation else "",
        "pos": lang_text(grammar, "en") if grammar else "",
        "senses": [parse_sense(s) for s in own_senses(node, guid)],
        "complex_forms": complex_forms,
        "variant_of": variant_of,
        "variants": variants,
        "cross_refs": cross_refs,
    }


def parse_page(html: str) -> list[dict]:
    """Every entry, minor entry and subentry rendered on one browse page."""
    soup = BeautifulSoup(html, "html5lib")
    found = []
    for div in soup.find_all("div", id=re.compile(r"^g[0-9a-f-]{8,}$")):
        classes = " ".join(div.get("class", []))
        if "entry" not in classes:
            continue
        found.append(parse_entry(div, div["id"], classes.replace(" left", "").strip()))
    for span in soup.find_all("span", class_="subentry"):
        sense = span.find("span", class_="sense")
        guid = sense.get("entryguid", "") if sense else guid_of(span)
        if guid:
            found.append(parse_entry(span, guid, "subentry"))
    return found


def richness(entry: dict) -> tuple:
    """Rank two renderings of the same GUID so the fullest one is kept."""
    return (
        len(entry["senses"]),
        sum(len(s["examples"]) for s in entry["senses"]),
        sum(len(s["domains"]) for s in entry["senses"]),
        len(entry["ipa"]),
        len(entry["complex_forms"])
        + len(entry["variant_of"])
        + len(entry["variants"])
        + len(entry["cross_refs"]),
    )


# --------------------------------------------------------------------------------------
# crawling
# --------------------------------------------------------------------------------------

def crawl(*, offline: bool) -> tuple[dict[str, dict], dict]:
    session = None if offline else open_session()
    entries: dict[str, dict] = {}
    origin: dict[str, tuple[str, int]] = {}
    pages_read = 0
    missing_pages: list[str] = []

    for letter in LETTERS:
        previous_ids: list[str] | None = None
        page = 1
        while page <= MAX_PAGES_PER_LETTER:
            url = page_url(letter, page)
            html = fetch(url, offline=offline, session=session)
            if html is None:
                missing_pages.append(url)
                break
            pages_read += 1
            found = parse_page(html)
            ids = [e["guid"] for e in found]
            # The browse view repeats the final page for an out-of-range ``paged``; an
            # identical GUID list is the end of the letter, not more data.
            if not ids or ids == previous_ids:
                break
            for entry in found:
                current = entries.get(entry["guid"])
                if current is None or richness(entry) > richness(current):
                    entries[entry["guid"]] = entry
                origin.setdefault(entry["guid"], (letter, page))
            previous_ids = ids
            page += 1
        else:
            raise RuntimeError(f"letter {letter} exceeded {MAX_PAGES_PER_LETTER} pages")

    for guid, (letter, page) in origin.items():
        entries[guid]["letter"] = letter
        entries[guid]["page"] = page
    return entries, {"pages_read": pages_read, "missing_pages": missing_pages}


# --------------------------------------------------------------------------------------
# row construction
# --------------------------------------------------------------------------------------

def compose_gloss(senses: list[dict]) -> str:
    """Join senses the way the Kullui import does: ``1) first; 2) second``."""
    glosses = [s["gloss_en"] or s["gloss_hi"] for s in senses]
    glosses = [g for g in glosses if g]
    if not glosses:
        return ""
    if len(glosses) == 1:
        return glosses[0]
    return "; ".join(f"{i}) {g}" for i, g in enumerate(glosses, 1))


def compose_notes(entry: dict) -> str:
    parts = []
    hindi = [s["gloss_hi"] for s in entry["senses"] if s["gloss_hi"]]
    if hindi:
        parts.append("Hindi gloss: " + "; ".join(hindi))
    notes = [s["note"] for s in entry["senses"] if s["note"]]
    if notes:
        parts.append("; ".join(notes))
    for relation in entry["variant_of"]:
        if relation["type"] and relation["headword"]:
            parts.append(f"{relation['type']} {relation['headword']}")
    for variant in entry["variants"]:
        if variant["type"] and variant["headword"]:
            parts.append(f"{variant['type']} {variant['headword']}")
    for ref in entry["cross_refs"]:
        if ref["headword"]:
            label = ref["type"] or "cf"
            parts.append(f"{label}. {ref['headword']}")
    return "; ".join(parts)


def compose_tags(entry: dict) -> str:
    tags = []
    if entry["pos"]:
        tags.append(entry["pos"])
    for sense in entry["senses"]:
        for domain in sense["domains"]:
            if domain["code"]:
                tags.append(f"semdom:{domain['code']}")
    for relation in entry["variant_of"]:
        borrowing = BORROWING_TYPES.get(relation["type"])
        if borrowing:
            tags.append(borrowing)
    # ``bgw``/``ori`` label a variant pair whose members are both tagged hlb. What Woods meant by
    # them is not recoverable from the export, so they are surfaced for review, not interpreted.
    labels = {r["type"] for r in entry["variant_of"]} | {v["type"] for v in entry["variants"]}
    if labels & {"bgw", "ori"}:
        tags.append("uncertain:dialect")
    tags.append(DIALECT_TAG)
    return " ".join(dict.fromkeys(tags))


def entry_key(guid: str) -> str:
    return f"{SOURCE_ID}:{guid}"


def build_rows(entries: dict[str, dict]) -> tuple[list[dict], list[dict], Counter]:
    rows: list[dict] = []
    audit: list[dict] = []
    status_counts: Counter = Counter()

    for guid in sorted(entries):
        entry = entries[guid]
        key = entry_key(guid)
        headword = entry["headword"].strip()
        ipa = unicodedata.normalize("NFC", entry["ipa"].strip())
        gloss = compose_gloss(entry["senses"])

        # Every visible variant reference is a FLEx variant relation; the label only says what
        # kind it is. A borrowing label still means "this entry is a variant of that one".
        variant_of_key = ""
        for relation in entry["variant_of"]:
            if relation["guid"]:
                variant_of_key = entry_key(relation["guid"])
                break

        if not headword:
            status, reason = "excluded", "no headword"
        elif not ipa:
            status, reason = "excluded", "no citation form"
        elif DEVANAGARI.search(ipa):
            # Woods typed the headword into the citation-form field for this record. Converting
            # Devanagari through the IPA profile would invent a reading, so it stays out.
            status, reason = "excluded", "citation form is Devanagari, not IPA"
        elif not gloss and not variant_of_key:
            status, reason = "excluded", "no definition and no variant-of relation"
        else:
            status, reason = "installed", ""
        status_counts[f"{status}:{reason}" if reason else status] += 1

        if status == "installed":
            derivation = []
            for complex_form in entry["complex_forms"]:
                if complex_form["type"] in DERIVATIONAL_TYPES:
                    derivation.extend(
                        entry_key(p["guid"]) for p in complex_form["parts"] if p["guid"]
                    )
            rows.append(
                {
                    "Language_ID": LANGUAGE_ID,
                    "Parameter_ID": "",
                    "Form": ipa,
                    "Gloss": gloss,
                    "Native": headword,
                    "Phonemic": ipa,
                    "Notes": compose_notes(entry),
                    "Source": f"{SOURCE_ID}[entry {guid}]",
                    "Cognateset": "",
                    "Etymology": "",
                    "Entry_Key": key,
                    "Variant_Of_Key": variant_of_key,
                    "Borrowed_From_Key": "",
                    "Derivation_Parent_Keys": " ".join(dict.fromkeys(derivation)),
                    "Tags": compose_tags(entry),
                }
            )

        senses = entry["senses"] or [None]
        for sense in senses:
            example = sense["examples"][0] if sense and sense["examples"] else {}
            record = {
                "Snapshot_Date": SNAPSHOT_DATE,
                "Entry_Key": key,
                "GUID": guid,
                "Letter": entry.get("letter", ""),
                "Page": entry.get("page", ""),
                "Entry_Kind": entry["kind"],
                "Headword": headword,
                "Homograph": entry["homograph"],
                "IPA": ipa,
                "Part_Of_Speech": entry["pos"],
                "Sense_Count": len(entry["senses"]),
                "Sense_Number": sense["number"] if sense else "",
                "Gloss_English": sense["gloss_en"] if sense else "",
                "Gloss_Hindi": sense["gloss_hi"] if sense else "",
                "Definition_Kind": sense["kind"] if sense else "",
                "Usage_Note": sense["note"] if sense else "",
                "Semantic_Domains": "; ".join(
                    f"{d['code']} {d['name']}".strip() for d in sense["domains"]
                ) if sense else "",
                "Example": example.get("text", ""),
                "Example_Translation": example.get("translation", ""),
                "Complex_Form_Type": "; ".join(
                    c["type"] for c in entry["complex_forms"] if c["type"]
                ),
                "Complex_Form_Parts": "; ".join(
                    " + ".join(p["headword"] for p in c["parts"])
                    for c in entry["complex_forms"]
                ),
                "Variant_Of_Type": "; ".join(
                    r["type"] for r in entry["variant_of"] if r["type"]
                ),
                "Variant_Of_Target": "; ".join(
                    r["headword"] for r in entry["variant_of"] if r["headword"]
                ),
                "Variant_Of_Key": variant_of_key,
                "Variant_Backref_Type": "; ".join(
                    v["type"] for v in entry["variants"] if v["type"]
                ),
                "Variant_Backref_Target": "; ".join(
                    v["headword"] for v in entry["variants"] if v["headword"]
                ),
                "Cross_Reference_Type": "; ".join(
                    r["type"] for r in entry["cross_refs"] if r["type"]
                ),
                "Cross_Reference_Target": "; ".join(
                    r["headword"] for r in entry["cross_refs"] if r["headword"]
                ),
                "Status": status,
                "Reason": reason,
            }
            payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
            record["Record_SHA256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            audit.append(record)

    return rows, audit, status_counts


def symbol_inventory(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        for character in row["Phonemic"]:
            counts[character] += 1
    return counts


# --------------------------------------------------------------------------------------
# outputs
# --------------------------------------------------------------------------------------

def write_forms(rows: list[dict]) -> None:
    with FORM_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for row in rows:
            writer.writerow([row[field] for field in FORM_FIELDS])


def write_audit(audit: list[dict]) -> None:
    with AUDIT_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)


def write_sample(audit: list[dict], size: int = 20) -> None:
    """A deterministic spread through the audit for the seeded material-error review."""
    installed = [record for record in audit if record["Status"] == "installed"]
    if not installed:
        return
    step = max(1, len(installed) // size)
    picked = installed[::step][:size]
    with SAMPLE_OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=[*AUDIT_FIELDS, "Review", "Material_Error"]
        )
        writer.writeheader()
        for record in picked:
            writer.writerow({**record, "Review": "", "Material_Error": ""})


def write_manifest(entries: dict, rows: list[dict], stats: dict, status_counts: Counter) -> None:
    inventory = symbol_inventory(rows)
    manifest = {
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "site": {
            "url": "https://www.webonary.org/halbi/",
            "browse_url": BASE_URL,
            "page_size": PAGE_SIZE,
            "pages_read": stats["pages_read"],
            "missing_pages": stats["missing_pages"],
        },
        "publication": {
            "compiler": "Woods, Fran",
            "title": "Halbi - English Dictionary",
            "publisher": "SIL International",
            "year": 2019,
            "reported_entries": 7556,
            "sil_archive_entry": 84724,
            "copyright": "(c) 2019 SIL International",
        },
        "extraction": {
            "distinct_guids": len(entries),
            "installed_rows": len(rows),
            "status_counts": dict(status_counts),
            "entry_kinds": dict(Counter(e["kind"] for e in entries.values())),
            "rows_with_hindi_gloss": sum(
                1 for row in rows if "Hindi gloss:" in row["Notes"]
            ),
            "rows_with_derivation": sum(
                1 for row in rows if row["Derivation_Parent_Keys"]
            ),
            "phonemic_symbol_inventory": dict(sorted(inventory.items())),
        },
        "dialect": {
            "tag": DIALECT_TAG,
            "source_language_id": DIALECT_SOURCE_ID,
            "name": DIALECT_NAME,
            "basis": "Bhatpal village, Bastar; Raj Murea speakers; fieldwork 1967-1978",
        },
    }
    MANIFEST_OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def report(entries: dict, rows: list[dict], stats: dict, status_counts: Counter) -> None:
    print(f"pages read              {stats['pages_read']}")
    if stats["missing_pages"]:
        print(f"pages missing from cache {len(stats['missing_pages'])}")
    print(f"distinct entry GUIDs    {len(entries)}")
    print(f"installed rows          {len(rows)}")
    for status, count in sorted(status_counts.items()):
        print(f"  {status:<48} {count}")
    kinds = Counter(e["kind"] for e in entries.values())
    for kind, count in kinds.most_common():
        print(f"  kind {kind:<43} {count}")
    inventory = symbol_inventory(rows)
    print(f"phonemic symbols        {len(inventory)}")
    print("  " + " ".join(sorted(inventory)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="fetch uncached pages")
    parser.add_argument("--offline", action="store_true", help="use only the cache")
    parser.add_argument("--install", action="store_true", help="write canonical artifacts")
    args = parser.parse_args()

    offline = args.offline or not args.refresh
    entries, stats = crawl(offline=offline)
    rows, audit, status_counts = build_rows(entries)
    report(entries, rows, stats, status_counts)

    if args.install:
        write_forms(rows)
        write_audit(audit)
        write_sample(audit)
        write_manifest(entries, rows, stats, status_counts)
        print(f"\nwrote {FORM_OUTPUT.relative_to(ROOT)}")
        print(f"wrote {AUDIT_OUTPUT.relative_to(ROOT)}")
        print(f"wrote {SAMPLE_OUTPUT.relative_to(ROOT)}")
        print(f"wrote {MANIFEST_OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""Reproducible snapshot of the English Wiktionary Proto-Indo-Iranian reconstruction corpus.

The corpus is the pages in ``Category:Proto-Indo-Iranian lemmas`` (plus the
non-lemma reconstruction pages that hang off them).  Each page is fetched with
its exact revision id so a snapshot is reproducible and auditable, and the raw
wikitext is cached verbatim under ``tmp/wiktionary-piir-cache/``.

Wiktionary text is CC BY-SA 4.0 / GFDL.  What Jambu installs from it are the
reconstructions themselves together with the scholarly citations the article
carries (``{{R:...}}`` templates), which is what makes the corpus useful here:
every reconstruction is attributed to the paper or dictionary that made it.

Usage::

    python data/other/params/raw_data/wiktionary_piir.py fetch     # snapshot
    python data/other/params/raw_data/wiktionary_piir.py sources   # citation register
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://en.wiktionary.org/w/api.php"
UA = "jambu-research/1.0 (https://github.com/aryamanarora/jambu; aryaman.arora2020@gmail.com)"
CACHE = os.path.join("tmp", "wiktionary-piir-cache")
CATEGORIES = [
    "Category:Proto-Indo-Iranian lemmas",
    "Category:Proto-Indo-Iranian non-lemma forms",
    "Category:Proto-Indo-Iranian roots",
]


def _get(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2"}
    url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as handle:
                return json.load(handle)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def category_members(category: str) -> list[str]:
    titles: list[str] = []
    cont: dict = {}
    while True:
        data = _get({
            "action": "query", "list": "categorymembers",
            "cmtitle": category, "cmlimit": "500", **cont,
        })
        titles.extend(m["title"] for m in data.get("query", {}).get("categorymembers", []))
        if "continue" not in data:
            return titles
        cont = data["continue"]


def fetch_pages(titles: list[str]) -> list[dict]:
    """Fetch wikitext + revision id for every title, 50 at a time."""
    out: list[dict] = []
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        data = _get({
            "action": "query", "prop": "revisions",
            "rvprop": "content|ids|timestamp", "rvslots": "main",
            "titles": "|".join(batch),
        })
        for page in data.get("query", {}).get("pages", []):
            revs = page.get("revisions") or []
            if not revs:
                continue
            rev = revs[0]
            out.append({
                "title": page["title"],
                "pageid": page["pageid"],
                "revid": rev["revid"],
                "timestamp": rev["timestamp"],
                "wikitext": rev["slots"]["main"]["content"],
            })
        time.sleep(0.2)
    return out


def snapshot(path: str = None) -> str:
    titles: list[str] = []
    seen = set()
    for category in CATEGORIES:
        for title in category_members(category):
            if title.startswith("Reconstruction:Proto-Indo-Iranian/") and title not in seen:
                seen.add(title)
                titles.append(title)
    titles.sort()
    pages = fetch_pages(titles)
    pages.sort(key=lambda p: p["title"])
    os.makedirs(CACHE, exist_ok=True)
    path = path or os.path.join(CACHE, "pages.jsonl")
    with open(path, "w", encoding="utf-8") as handle:
        for page in pages:
            handle.write(json.dumps(page, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
    manifest = {
        "categories": CATEGORIES,
        "pages": len(pages),
        "titles_listed": len(titles),
        "snapshot_sha256": digest,
        "max_revid": max(p["revid"] for p in pages),
        "latest_timestamp": max(p["timestamp"] for p in pages),
    }
    with open(os.path.join(CACHE, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2))
    return path


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "fetch":
        snapshot()
    else:
        print(__doc__)


# --------------------------------------------------------------------------
# Citation register
# --------------------------------------------------------------------------

import re

REF_TEMPLATE = re.compile(r"\{\{(R:[^|}\n]+)")


def load_snapshot(path: str = None) -> list[dict]:
    path = path or os.path.join(CACHE, "pages.jsonl")
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def citation_counts(pages: list[dict]) -> dict:
    """template name -> {"uses": n, "pages": sorted headwords}"""
    out: dict = {}
    for page in pages:
        head = page["title"].split("/", 1)[1]
        for name in sorted(set(REF_TEMPLATE.findall(page["wikitext"]))):
            rec = out.setdefault(name.strip(), {"uses": 0, "pages": []})
            rec["pages"].append(head)
        for name in REF_TEMPLATE.findall(page["wikitext"]):
            out[name.strip()]["uses"] += 1
    for rec in out.values():
        rec["pages"] = sorted(set(rec["pages"]))
    return out


def fetch_template_texts(names: list[str]) -> dict:
    """Fetch each Template:R:... wikitext so the citation can be resolved."""
    titles = ["Template:" + n for n in names]
    pages = fetch_pages(titles)
    return {p["title"].removeprefix("Template:"): p["wikitext"] for p in pages}


CITE_FIELD = re.compile(r"\|\s*(last\d?|first\d?|editors?|title|trans-title|year|publisher|location|url|series|seriesvolume)\s*=\s*([^|\n}]*)")
REDIRECT = re.compile(r"^\s*#REDIRECT\s*\[?\[?Template:(R:[^\]\n|]+)", re.I)


def _strip_markup(value: str) -> str:
    value = re.sub(r"<!--.*?-->", "", value, flags=re.S)
    value = re.sub(r"<!--.*", "", value, flags=re.S)
    value = re.sub(r"\{\{#[^}]*\}?\}?.*", "", value, flags=re.S)
    value = re.sub(r"\{\{w\|[^|}]*\|([^}]*)\}\}", r"\1", value)
    value = re.sub(r"\{\{w\|([^}]*)\}\}", r"\1", value)
    value = re.sub(r"\{\{xlit\|[^|]*\|([^}]*)\}\}", r"\1", value)
    value = re.sub(r"\{\{[^}]*\}\}", "", value)
    value = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", value)
    value = re.sub(r"\[\[([^\]]*)\]\]", r"\1", value)
    value = value.replace("w:", "").replace("''", "")
    return re.sub(r"\s+", " ", value).strip(" .")


def resolve_citation(name: str, texts: dict, _seen=None) -> dict:
    """Follow #REDIRECTs and pull the cite-book fields out of a reference template."""
    _seen = _seen or set()
    if name in _seen or name not in texts:
        return {"canonical": name}
    _seen.add(name)
    text = texts[name]
    redirect = REDIRECT.match(text)
    if redirect:
        target = redirect.group(1).strip()
        resolved = resolve_citation(target, texts, _seen)
        resolved.setdefault("canonical", target)
        resolved["canonical"] = resolved.get("canonical", target)
        return resolved
    fields: dict = {"canonical": name}
    for key, value in CITE_FIELD.findall(text):
        value = _strip_markup(value)
        if value and key not in fields:
            fields[key] = value
    authors = [fields.get(k) for k in ("last", "last2", "last3") if fields.get(k)]
    fields["authors"] = "; ".join(authors) or fields.get("editors") or fields.get("editor", "")
    return fields


# Hand-checked bibliographic records for the templates whose MediaWiki source is
# built from parser functions the field scraper cannot flatten (multi-volume
# works whose author/title switch on the volume parameter, and titles wrapped in
# {{w}}).  Everything else is read straight off the template.
CITATION_OVERRIDES = {
    "R:inc:IAIL": dict(
        authors="Lubotsky, Alexander", year="2011",
        title="The Indo-Aryan Inherited Lexicon (in progress)",
        publisher="Indo-European Etymological Dictionary Project, Leiden University",
        access="Brill IEDO subscription; no open edition",
    ),
    "R:inc:EWAia": dict(
        authors="Mayrhofer, Manfred", year="1986–2001",
        title="Etymologisches Wörterbuch des Altindoarischen",
        publisher="Carl Winter, Heidelberg",
        access="In copyright; scans on archive.org (lending)",
    ),
    "R:ira:EDIV": dict(
        authors="Cheung, Johnny", year="2007",
        title="Etymological Dictionary of the Iranian Verb",
        publisher="Brill, Leiden/Boston",
        access="In copyright; scan on archive.org",
    ),
    "R:ira:ESIJa": dict(
        authors="Rastorgueva, V. S.; Edel'man, D. I.", year="2000–",
        title="Ėtimologičeskij slovar' iranskix jazykov",
        publisher="Vostočnaja literatura, Moscow",
        access="In copyright; volumes circulate as scans",
    ),
    "R:ine:HCHIEL": dict(
        authors="Klein, Jared; Joseph, Brian; Fritz, Matthias (eds.)", year="2017–2018",
        title="Handbook of Comparative and Historical Indo-European Linguistics",
        publisher="De Gruyter Mouton, Berlin/Boston",
        access="In copyright",
    ),
    "R:ine:LIV": dict(
        authors="Rix, Helmut (ed.)", year="2001",
        title="Lexikon der indogermanischen Verben, 2nd edn.",
        publisher="Reichert, Wiesbaden",
        access="In copyright",
    ),
    "R:ine:LIPP": dict(
        authors="Dunkel, George E.", year="2014",
        title="Lexikon der indogermanischen Partikeln und Pronominalstämme",
        publisher="Universitätsverlag Winter, Heidelberg",
        access="In copyright; scan on archive.org",
    ),
    "R:iir:Lubotsky:1999": dict(
        authors="Lubotsky, Alexander", year="1999",
        title="The Indo-Iranian substratum",
        publisher="in Carpelan, Parpola & Koskikallio (eds.), Early Contacts between "
                  "Uralic and Indo-European, Helsinki 2001, 301–317",
        url="https://hdl.handle.net/1887/2691",
        access="Open access (Leiden repository)",
    ),
    "R:os:Abaev": dict(
        authors="Abaev, V. I.", year="1958–1989",
        title="Istoriko-ėtimologičeskij slovar' osetinskogo jazyka",
        publisher="Nauka, Moscow/Leningrad",
        access="In copyright; scans circulate",
    ),
    "R:iir:Lipp:2009": dict(
        authors="Lipp, Reiner", year="2009",
        title="Die indogermanischen und einzelsprachlichen Palatale im Indoiranischen",
        publisher="Universitätsverlag Winter, Heidelberg",
        access="In copyright",
    ),
    "R:ine:IEW": dict(
        authors="Pokorny, Julius", year="1959",
        title="Indogermanisches etymologisches Wörterbuch",
        publisher="Francke, Bern/München",
        url="https://starlingdb.org/cgi-bin/response.cgi?basename=\\data\\ie\\pokorny",
        access="In copyright; searchable index at starlingdb.org",
    ),
    "R:CDIAL": dict(
        authors="Turner, Ralph Lilley", year="1962–1966",
        title="A Comparative Dictionary of the Indo-Aryan Languages",
        publisher="Oxford University Press, London",
        url="https://dsal.uchicago.edu/dictionaries/soas/",
        access="Already the backbone of Jambu (source key CDIAL)",
    ),
}

REGISTER_HEADER = [
    "Rank", "Template", "Aliases", "Uses", "Etyma", "Authors", "Year", "Title",
    "Publisher", "URL", "Access",
]


def citation_register(pages: list[dict] = None) -> list[dict]:
    """Rank every work the PIIr corpus cites by how many etyma it supports."""
    pages = pages or load_snapshot()
    counts = citation_counts(pages)
    with open(os.path.join(CACHE, "citations.json"), encoding="utf-8") as handle:
        texts = json.load(handle)["templates"]
    merged: dict = {}
    for name, rec in counts.items():
        info = resolve_citation(name, texts)
        canonical = info.get("canonical", name)
        entry = merged.setdefault(canonical, {
            "Template": canonical, "Aliases": set(), "Uses": 0, "_etyma": set(),
            "Authors": "", "Year": "", "Title": "", "Publisher": "", "URL": "", "Access": "",
        })
        entry["Aliases"].add(name)
        entry["Uses"] += rec["uses"]
        entry["_etyma"].update(rec["pages"])
        for key, field in (("authors", "Authors"), ("year", "Year"), ("title", "Title"),
                           ("publisher", "Publisher"), ("url", "URL")):
            if not entry[field] and info.get(key):
                entry[field] = info[key]
    for canonical, override in CITATION_OVERRIDES.items():
        entry = merged.get(canonical)
        if entry is None:
            continue
        for key, value in override.items():
            entry[key.capitalize() if key != "url" else "URL"] = value
    rows = sorted(merged.values(), key=lambda r: (-r["Uses"], r["Template"]))
    out = []
    for rank, entry in enumerate(rows, start=1):
        out.append({
            "Rank": rank, "Template": entry["Template"],
            "Aliases": " ".join(sorted(entry["Aliases"] - {entry["Template"]})),
            "Uses": entry["Uses"], "Etyma": len(entry["_etyma"]),
            "Authors": entry["Authors"], "Year": entry["Year"], "Title": entry["Title"],
            "Publisher": entry["Publisher"], "URL": entry["URL"], "Access": entry["Access"],
        })
    return out


def write_register(path: str) -> list[dict]:
    import csv as _csv
    rows = citation_register()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=REGISTER_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows


# --------------------------------------------------------------------------
# Wikitext parsing
# --------------------------------------------------------------------------

POS_HEADINGS = {
    "Noun", "Verb", "Adjective", "Adverb", "Numeral", "Pronoun", "Root", "Suffix",
    "Prefix", "Particle", "Proper noun", "Participle", "Conjunction", "Determiner",
    "Interjection", "Preposition", "Postposition",
}
SECTION = re.compile(r"^=+\s*(.+?)\s*=+\s*$", re.M)
DESC = re.compile(r"\{\{(?:desc|descendant)\|([^}]*)\}\}")
INH = re.compile(r"\{\{(?:inh|der|bor|inh\+|der\+)\|iir-pro\|(ine-pro|iir-pro)\|([^|}]+)")
UNCERTAIN = re.compile(r"\{\{(?:unk|unc|unknown)\|", re.I)


def _template_args(blob: str) -> tuple[list[str], dict]:
    positional, named = [], {}
    depth = 0
    current = ""
    parts = []
    for ch in blob:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += ch
    parts.append(current)
    for part in parts:
        if "=" in part and not part.split("=", 1)[0].strip().startswith("{"):
            key, value = part.split("=", 1)
            named[key.strip()] = value.strip()
        else:
            positional.append(part.strip())
    return positional, named


def clean_gloss(line: str) -> str:
    """A definition line minus templates, labels, links and qualifiers."""
    line = re.sub(r"\{\{(?:lb|label|tlb|topics?|C|cln|senseid|defdate|qualifier|q)\|[^}]*\}\}", "", line)
    line = re.sub(r"\{\{(?:l|m|w|link)\|[^|}]*\|([^|}]*)(?:\|[^}]*)?\}\}", r"\1", line)
    line = re.sub(r"\{\{[^}]*\}\}", "", line)
    line = re.sub(r"\[\[[^|\]]*\|([^\]]*)\]\]", r"\1", line)
    line = re.sub(r"\[\[([^\]]*)\]\]", r"\1", line)
    line = re.sub(r"<[^>]+>", "", line)
    line = line.replace("''", "").replace("'''", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip(" ,;:")


def parse_page(page: dict) -> dict:
    """One Proto-Indo-Iranian reconstruction page → a structured record."""
    text = page["wikitext"]
    head = page["title"].split("/", 1)[1]

    # section index: (heading, body)
    bounds = [(m.start(), m.end(), m.group(1)) for m in SECTION.finditer(text)]
    sections = []
    for i, (_, end, heading) in enumerate(bounds):
        stop = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        sections.append((heading, text[end:stop]))

    pos, glosses = "", []
    for heading, body in sections:
        base = re.sub(r"\s*\d+$", "", heading)
        if base in POS_HEADINGS:
            if not pos:
                pos = base.lower()
            for line in body.splitlines():
                if line.startswith("#") and not line.startswith(("#*", "#:")):
                    gloss = clean_gloss(line.lstrip("# ").strip())
                    if gloss:
                        glosses.append(gloss)

    pie = ""
    for lang, form in INH.findall(text):
        if lang == "ine-pro" and not pie:
            pie = form.strip()

    descendants: list[dict] = []
    for blob in DESC.findall(text):
        positional, named = _template_args(blob)
        if not positional:
            continue
        lang = positional[0].strip()
        word = positional[1].strip() if len(positional) > 1 else ""
        descendants.append({
            "lang": lang,
            "word": "" if word == "-" else word,
            "roman": named.get("tr", "") or named.get("ts", ""),
            "borrowed": named.get("bor") == "1",
        })

    citations = []
    for start in (m.start() for m in re.finditer(r"\{\{R:", text)):
        depth, i = 0, start
        while i < len(text):
            if text.startswith("{{", i):
                depth += 1
                i += 2
            elif text.startswith("}}", i):
                depth -= 1
                i += 2
                if depth == 0:
                    break
            else:
                i += 1
        blob = text[start + 2:i - 2]
        name, _, rest = blob.partition("|")
        positional, named = _template_args(rest)
        page_no = (named.get("page") or named.get("pages") or "").strip()
        entry = (named.get("head") or named.get("entry") or "").strip()
        # A bare first positional is a page number in some templates and a
        # head-word in others; numerals are pages, anything else is a head-word.
        if positional and positional[0]:
            first = positional[0].strip()
            if re.fullmatch(r"[\dIVXivx]+(\s*[-–]\s*\d+)?", first):
                page_no = page_no or first
            else:
                entry = entry or first
        # `;` separates citations in a CLDF Source cell and `[]` delimits the
        # locator, so neither may survive inside one.
        clean = lambda v: re.sub(r"\s+", " ", v.replace(";", ",").replace("[", "(").replace("]", ")")).strip()
        citations.append({
            "template": name.strip(), "page": clean(page_no), "entry": clean(entry),
        })

    return {
        "entry_key": f"enwikt:{page['pageid']}",
        "pageid": page["pageid"],
        "revid": page["revid"],
        "title": page["title"],
        "head": head,
        "pos": pos,
        "glosses": glosses,
        "pie": pie,
        "descendants": descendants,
        "citations": citations,
        "uncertain": bool(UNCERTAIN.search(text)),
    }


# --------------------------------------------------------------------------
# Matching Sanskrit descendants onto CDIAL head-words
# --------------------------------------------------------------------------

# Turner prints Sanskrit /e o/ as ē ō and vocalic r̥ l̥ with a ring/vertical line
# below; Wiktionary romanises the same phonemes as e o ṛ ḷ.  Folding those, the
# Vedic pitch accent and the anusvāra notation is what makes two spellings of one
# head-word comparable.  Vowel *length* is never folded — it is contrastive.
# ``ś`` is s + COMBINING ACUTE, so accent-stripping has to protect it or it
# collapses into plain s.
_SIBILANT_GUARD = "\ue000"


def match_key(form: str) -> str:
    """A comparison key that folds notation, never phonology.

    Turner prints Sanskrit /e o/ as ``ē ō`` and vocalic r̥ l̥ with a vertical line
    below; Wiktionary romanises the same phonemes as ``e o ṛ ḷ``.  Folding those,
    the Vedic pitch accent, the anusvāra/velar-nasal notation and the
    reconstruction asterisk is what makes two spellings of one head-word
    comparable.  Vowel *length* is never folded — it is contrastive.
    """
    import unicodedata

    text = unicodedata.normalize("NFC", form or "")
    text = text.replace("ś", _SIBILANT_GUARD).replace("Ś", _SIBILANT_GUARD)
    text = unicodedata.normalize("NFD", text)
    text = text.replace("\u0301", "").replace("\u0300", "")        # pitch accent
    text = re.sub(r"([rl])\u0329", "\\1\u0325", text)              # vertical line below
    text = re.sub(r"([rl])\u0323", "\\1\u0325", text)              # dot below (vocalic)
    text = unicodedata.normalize("NFC", text)
    text = text.replace(_SIBILANT_GUARD, "ś")
    text = text.replace("ē", "e").replace("ō", "o")                # Turner's e/o
    text = text.replace("ṃ", "ṁ").replace("ṅ", "ŋ")                # anusvāra / velar nasal
    return text.strip().lower().lstrip("*").strip("-").strip()


STOPWORDS = {
    "a", "an", "the", "of", "to", "or", "and", "in", "on", "at", "with", "for",
    "be", "is", "as", "by", "from", "one", "who", "that", "it", "its", "his",
    "her", "their", "something", "someone", "kind", "sort", "esp", "etc",
}


def gloss_tokens(gloss: str) -> set:
    return {
        w for w in re.findall(r"[a-z]+", (gloss or "").lower())
        if len(w) > 2 and w not in STOPWORDS
    }


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

SOURCE_KEY = "wiktionary-piir"
STAMP = "20260827"
PARAMS_OUT = f"data/other/params/{STAMP}-wiktionary-piir.csv"
TEXTS_OUT = f"data/other/entry_texts/{STAMP}-wiktionary-piir.csv"
AUDIT_OUT = f"data/other/params/raw_data/{STAMP}-wiktionary-piir-audit.csv"
ASSIGNMENTS_FILE = "data/etymology-assignments.csv"
FORMS = "cldf/forms.csv"
EDGES = "cldf/edges.csv"
MERGES = "cldf/merges.csv"
ALIASES = "cldf/form-id-aliases.csv"
PROFILE = "conversion/cdial.txt"

# A CDIAL head-word id is a Turner entry number, optionally with a promoted
# section-form suffix. The shape alone is not enough: `make_cldf.py` also mints
# pre-ID `<file>-<row>` ids such as `0-113243`, which match the same pattern and
# would be silently linked as if they were CDIAL entries. The entry numbers are
# therefore checked against `data/cdial/params.csv`, the authoritative list.
CDIAL_PARAMS = "data/cdial/params.csv"
CDIAL_ID = re.compile(r"^(\d+[a-z]?)(-\d+x*)?$")


def _cdial_entry_numbers(path: str = CDIAL_PARAMS) -> set:
    import csv as _csv
    _csv.field_size_limit(10 ** 9)
    with open(path, encoding="utf-8", newline="") as handle:
        return {row[0] for row in _csv.reader(handle) if row and row[0]}


def is_cdial_head(form_id: str, entry_numbers: set) -> bool:
    match = CDIAL_ID.match(form_id or "")
    return bool(match) and match.group(1) in entry_numbers
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

AUDIT_HEADER = [
    "Entry_Key", "Page_ID", "Revision", "Title", "PIIr_Form", "POS", "Gloss",
    "PIE_Parent", "Uncertain", "Sanskrit", "Match_Key", "Candidates",
    "Matched_CDIAL_Head", "Decision", "Rank", "Reason", "Existing_Parent",
    "Citations",
]


class IncompleteBuildError(RuntimeError):
    """Raised when cldf/ has not been through a full pipeline run."""


def _cdial_index():
    """head match-key → CDIAL entry ids, plus gloss/redirect/rank-1-parent lookups.

    Reads the *built* graph, so it must run against a complete build: an
    interrupted pipeline leaves cldf/ before ``assign_form_ids.py`` has applied
    the curated overlay, and every head-word then looks parentless.
    """
    import csv as _csv
    _csv.field_size_limit(10 ** 9)
    entry_numbers = _cdial_entry_numbers()
    heads: dict = {}
    gloss: dict = {}
    redirect: dict = {}
    language: dict = {}
    with open(FORMS, encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            language[row["ID"]] = row["Language_ID"]
            if row["Language_ID"] == "Indo-Aryan" and is_cdial_head(row["ID"], entry_numbers):
                heads.setdefault(match_key(row["Form"]), []).append(row["ID"])
                gloss[row["ID"]] = row["Gloss"]
                if row.get("Redirect"):
                    redirect[row["ID"]] = row["Redirect"]
    with open(MERGES, encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            redirect.setdefault(row["Addendum_ID"], row["Main_ID"])
    # A rebuild reads a cldf/ that already contains this source's own edges. Those
    # must not count as "the head-word already has an accepted etymology", or every
    # rank-1 link would demote itself to a ranked alternative on the next run.
    ours = set()
    if os.path.exists(ALIASES):
        with open(ALIASES, encoding="utf-8", newline="") as handle:
            ours = {
                row["Form_ID"] for row in _csv.DictReader(handle)
                if row["Legacy_ID"].startswith("wiir-")
            }
    if not any(i.startswith("f_") for i in language):
        raise IncompleteBuildError(
            f"{FORMS} carries no durable f_ ids, so the pipeline did not finish. "
            "Run `make all` before building this source (see the two-pass note in "
            "source_checklists/20260827-wiktionary-piir.md)."
        )
    rank1: dict = {}
    with open(EDGES, encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            if row["Rank"] != "1" or row["Kind"] not in {"reflex", "borrowed", "variant"}:
                continue
            if row["Parent_ID"] in ours or row["Parent_ID"].startswith("wiir-"):
                continue
            rank1[row["Child_ID"]] = row["Parent_ID"]
    # cldf/edges.csv is a build product: an interrupted or pre-overlay build would
    # under-report the accepted etymologies and this source would then claim rank-1
    # slots that belong to curated links. data/etymology-assignments.csv is the
    # durable record of those decisions, so it is unioned in and wins.
    if os.path.exists(ASSIGNMENTS_FILE):
        with open(ASSIGNMENTS_FILE, encoding="utf-8", newline="") as handle:
            for row in _csv.DictReader(handle):
                etymon = row.get("Etymon_ID", "")
                if row.get("Rank") != "1" or etymon.startswith("wiir-") or etymon in ours:
                    continue
                if (row.get("Status") or "accepted").strip().lower() != "accepted":
                    continue
                rank1.setdefault(row["Form_ID"], etymon)
    return heads, gloss, redirect, rank1, language


def resolve_match(record, heads, gloss, redirect, cited_cdial):
    """Conservative head resolution. Returns (etymon_id | None, candidates, reason)."""
    candidates: list = []
    keys: list = []
    for desc in record["descendants"]:
        if desc["lang"] != "sa" or desc["borrowed"]:
            continue
        raw = desc["roman"] or desc["word"]
        if raw and DEVANAGARI.search(raw):
            raw = devanagari_to_iast(raw)
        if not raw or DEVANAGARI.search(raw):
            continue
        key = match_key(_house(raw))
        if not key:
            continue
        keys.append(key)
        candidates.extend(heads.get(key, []))
    candidates = sorted(set(candidates))
    if not candidates:
        return None, candidates, keys, "no CDIAL head with this spelling"
    # An addendum that redirects onto another candidate is the same head-word twice.
    collapsed = [c for c in candidates if redirect.get(c) not in set(candidates)]
    if len(collapsed) == 1:
        candidates_left = collapsed
        reason = "unique after collapsing addendum onto its main entry" if len(candidates) > 1 else "unique spelling match"
    else:
        candidates_left = collapsed or candidates
        reason = ""
    if len(candidates_left) > 1:
        entry_tokens = set()
        for text in record["glosses"]:
            entry_tokens |= gloss_tokens(text)
        scored = [c for c in candidates_left if entry_tokens & gloss_tokens(gloss.get(c, ""))]
        if len(scored) == 1:
            return scored[0], candidates, keys, "disambiguated by gloss overlap"
        return None, candidates, keys, "ambiguous: several CDIAL heads share this spelling"
    etymon = candidates_left[0]
    entry_tokens = set()
    for text in record["glosses"]:
        entry_tokens |= gloss_tokens(text)
    target_tokens = gloss_tokens(gloss.get(etymon, ""))
    if entry_tokens and target_tokens and not (entry_tokens & target_tokens):
        if cited_cdial:
            return etymon, candidates, keys, "glosses disjoint but the article cites CDIAL"
        return etymon, candidates, keys, "review:semantics — glosses do not overlap"
    return etymon, candidates, keys, reason or "unique spelling match"


_TOKENIZER = None


def _house(form: str) -> str:
    """Run a source spelling through Jambu's CDIAL sound profile."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from segments import Tokenizer
        _TOKENIZER = Tokenizer(PROFILE)
    return _TOKENIZER(form, column="IPA").replace(" ", "").replace("#", " ")


def _piir_form(head: str) -> str:
    """The reconstruction as Jambu should display it (Wiktionary omits the asterisk
    in the page title; laryngeal/uncertainty notation is preserved verbatim)."""
    return head.strip()


def _citation_source(record) -> str:
    """CLDF citation string: the works the article credits, else Wiktionary itself."""
    parts = []
    seen = set()
    aliases = citation_aliases()
    for citation in record["citations"]:
        canonical = aliases.get(citation["template"], citation["template"])
        key = BIB_KEYS.get(canonical)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        locator = []
        if citation.get("entry"):
            locator.append(f"s.v. {citation['entry']}")
        if citation.get("page"):
            locator.append(f"p. {citation['page']}")
        parts.append(f"{key}[{', '.join(locator)}]" if locator else key)
    parts.append(f"{SOURCE_KEY}[{record['title']}, revision {record['revid']}]")
    # `;` with no space is the separator a CLDF Source cell uses elsewhere in the
    # repo, and tests/test_cldf.py splits on it without stripping.
    return ";".join(parts)


def build(dry_run=False):
    import csv as _csv

    pages = load_snapshot()
    records = [parse_page(page) for page in pages]
    heads, gloss, redirect, rank1, language = _cdial_index()

    params_rows, text_rows, audit_rows, assignments = [], [], [], []
    stats = collections.Counter()
    # Two articles can reach the same CDIAL head-word (Wiktionary splits the
    # privative and the augment *a-, both continued by CDIAL 1). Only one of them
    # can be the accepted etymology: a rank-1 assignment is an upsert, so a second
    # claimant would silently replace the first. The first in page-id order keeps
    # rank 1 and the rest become visible ranked alternatives.
    claimed_rank1: dict = {}
    for record in sorted(records, key=lambda r: r["pageid"]):
        param_id = f"wiir-{record['pageid']}"
        cited_cdial = any(c["template"] == "R:CDIAL" for c in record["citations"])
        etymon, candidates, keys, reason = resolve_match(
            record, heads, gloss, redirect, cited_cdial
        )
        decision, rank, existing = "unlinked", "", ""
        if etymon:
            existing = rank1.get(etymon, "")
            if not existing:
                decision, rank = "linked", "1"
            elif language.get(existing) in {"Indo-ir", "IE"}:
                decision, rank = "alternate", "2"
                reason = (reason + "; " if reason else "") + \
                    "review:duplicate — the head already has a Proto-Indo-Iranian parent"
            else:
                decision, rank = "alternate", "2"
                reason = (reason + "; " if reason else "") + \
                    "review:conflict — the head already has an accepted etymology"
            if reason.startswith("review:semantics") and rank == "1":
                decision, rank = "alternate", "2"
            if rank == "1":
                earlier = claimed_rank1.get(etymon)
                if earlier:
                    decision, rank = "alternate", "2"
                    reason = (reason + "; " if reason else "") + \
                        f"review:duplicate-claim — {earlier} already claims this head-word"
                else:
                    claimed_rank1[etymon] = param_id
        stats[decision] += 1

        source = _citation_source(record)
        params_rows.append([
            param_id, "Indo-ir", _piir_form(record["head"]),
            "; ".join(record["glosses"]), source,
        ])
        text_rows.append({
            "Form_ID": param_id, "Position": 100000 + record["pageid"] % 100000,
            "Kind": "etymology", "Format": "markdown",
            "Content": etymology_note(record), "Source": source,
        })
        if etymon and rank:
            assignments.append({
                "Form_ID": etymon, "Etymon_ID": param_id, "Kind": "reflex",
                "Rank": rank, "Status": "accepted", "Source": source,
                "Notes": reason,
            })
        audit_rows.append({
            "Entry_Key": record["entry_key"], "Page_ID": record["pageid"],
            "Revision": record["revid"], "Title": record["title"],
            "PIIr_Form": record["head"], "POS": record["pos"],
            "Gloss": "; ".join(record["glosses"]), "PIE_Parent": record["pie"],
            "Uncertain": "yes" if record["uncertain"] else "",
            "Sanskrit": "; ".join(
                d["roman"] or (devanagari_to_iast(d["word"]) or d["word"])
                if DEVANAGARI.search(d["word"] or "") and not d["roman"]
                else (d["roman"] or d["word"])
                for d in record["descendants"]
                if d["lang"] == "sa" and not d["borrowed"]
            ),
            "Match_Key": "; ".join(dict.fromkeys(keys)),
            "Candidates": " ".join(candidates), "Matched_CDIAL_Head": etymon or "",
            "Decision": decision, "Rank": rank, "Reason": reason,
            "Existing_Parent": existing,
            "Citations": "; ".join(
                c["template"] + "".join(
                    f"[{part}]" for part in (c.get("entry"), c.get("page")) if part
                )
                for c in record["citations"]
            ),
        })

    if dry_run:
        return stats, params_rows, assignments, audit_rows

    with open(PARAMS_OUT, "w", newline="", encoding="utf-8") as handle:
        _csv.writer(handle, lineterminator="\n").writerows(params_rows)
    with open(TEXTS_OUT, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(
            handle, fieldnames=["Form_ID", "Position", "Kind", "Format", "Content", "Source"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(text_rows)
    with open(AUDIT_OUT, "w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=AUDIT_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)
    return stats, params_rows, assignments, audit_rows


import collections  # noqa: E402  (used by build)


# --------------------------------------------------------------------------
# Template → Jambu bibliography key
# --------------------------------------------------------------------------
# Only works with a BibTeX record appear in an installed citation string; the
# audit keeps the full raw template list for every article either way.  Keys
# that already exist in cldf/sources.bib or the CDIAL abbreviation catalogue are
# reused rather than duplicated.
# Wiktionary reference templates are heavily redirected (R:sa:EWAia → R:inc:EWAia,
# R:iir-pro:Substrate → R:iir:Lubotsky:1999, …).  Citations are folded onto the
# canonical template before the bibliography key is looked up, so an alias never
# silently drops the citation.
ALIASES_PATH = f"data/other/params/raw_data/{{}}-indo-iranian-source-register.csv"


_ALIAS_CACHE = None


def citation_aliases() -> dict:
    import csv as _csv
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    path = ALIASES_PATH.format(STAMP)
    if not os.path.exists(path):
        return {}
    out: dict = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in _csv.DictReader(handle):
            out[row["Template"]] = row["Template"]
            for alias in row["Aliases"].split():
                out[alias] = row["Template"]
    _ALIAS_CACHE = out
    return out


HUMAN_NAMES = {
    "lubotsky2011iail": "Lubotsky, Indo-Aryan Inherited Lexicon",
    "mayrhofer1992ewaia": "Mayrhofer, EWAia",
    "cheung2007ediv": "Cheung, EDIV",
    "klein2017hchiel": "Klein–Joseph–Fritz, HCHIEL",
    "CDIAL": "Turner, CDIAL",
    "rastorgueva2000esija": "Rastorgueva–Edel'man, ESIJa",
    "rix2001liv": "Rix, LIV²",
    "dunkel2014lipp": "Dunkel, LIPP",
    "lubotsky1999substratum": "Lubotsky, The Indo-Iranian substratum",
    "martinez2014avestan": "Martínez García–de Vaan, Introduction to Avestan",
    "IEW": "Pokorny, IEW",
    "mallory2006oipiew": "Mallory–Adams, Oxford Introduction",
    "lipp2009palatale": "Lipp, Die Palatale im Indoiranischen",
    "mallory1997eiec": "Mallory–Adams, EIEC",
    "bailey1979khotan": "Bailey, Dictionary of Khotan Saka",
    "wodtko2008nil": "Wodtko–Irslinger–Schneider, NIL",
    "kuemmel2015suppletive": "Kümmel 2015",
    "lubotsky1988accentuation": "Lubotsky 1988",
    "goto2013oiamorphology": "Gotō 2013",
    "abaev1958osetinskogo": "Abaev, Osetinskij slovar'",
    "kuemmel2017agricultural": "Kümmel 2017",
    "mayrhofer1973persepolitana": "Mayrhofer, Onomastica Persepolitana",
    "witzel2003exchange": "Witzel 2003",
    "devaan2003avestanvowels": "de Vaan, The Avestan Vowels",
    "novak2013easterniranian": "Novák 2013",
    "strand": "Strand, Nuristani Etymological Lexicon",
    "sandell2014lengthening": "Sandell 2014",
    "cantera2017phonology": "Cantera 2017",
}

# Cited by head-word rather than by page.
HEADWORD_INDEXED = {"R:CDIAL", "R:kho:Bailey", "R:os:Abaev"}

# `mayrhofer-kewa` is deliberately absent. KEWA is installed as article scans on
# CDIAL entries and tests/test_kewa.py holds it to prose only: its key must never
# appear as a form-level citation. The 7 articles that credit KEWA still record it
# in the audit's Citations column and in the source register.
BIB_KEYS = {
    "R:inc:IAIL": "lubotsky2011iail",
    "R:inc:EWAia": "mayrhofer1992ewaia",
    "R:ira:EDIV": "cheung2007ediv",
    "R:ine:HCHIEL": "klein2017hchiel",
    "R:CDIAL": "CDIAL",
    "R:ira:ESIJa": "rastorgueva2000esija",
    "R:ine:LIV": "rix2001liv",
    "R:ine:LIPP": "dunkel2014lipp",
    "R:iir:Lubotsky:1999": "lubotsky1999substratum",
    "R:ae:Brill": "martinez2014avestan",
    "R:ine:IEW": "IEW",
    "R:ine:Mallory:2006": "mallory2006oipiew",
    "R:iir:Lipp:2009": "lipp2009palatale",
    "R:ine:EIEC": "mallory1997eiec",
    "R:kho:Bailey": "bailey1979khotan",
    "R:ine:NIL": "wodtko2008nil",
    "R:iir:Kummel:2015": "kuemmel2015suppletive",
    "R:iir:Lubotsky:1988": "lubotsky1988accentuation",
    "R:inc:Goto:2013": "goto2013oiamorphology",
    "R:os:Abaev": "abaev1958osetinskogo",
    "R:iir:Kummel:2017": "kuemmel2017agricultural",
    "R:ira:Mayrhofer:1973": "mayrhofer1973persepolitana",
    "R:Witzel:2003": "witzel2003exchange",
    "R:ae:deVaan:2003": "devaan2003avestanvowels",
    "R:ira:Novak:2013": "novak2013easterniranian",
    "R:iir-nur:NEL": "strand",
    "R:iir:Sandell": "sandell2014lengthening",
    "R:ira:Cantera:2017": "cantera2017phonology",
}


def etymology_note(record) -> str:
    """The article's etymological claim, as a short markdown sidecar.

    The graph edge carries the relation; this preserves the reasoning, the
    uncertainty marking and the literature the article credits.
    """
    lines = []
    head = f"**Proto-Indo-Iranian \\*{record['head']}**"
    if record["glosses"]:
        head += " ‘" + "; ".join(record["glosses"]) + "’"
    lines.append(head + ".")
    if record["pie"]:
        verb = "Possibly from" if record["uncertain"] else "From"
        lines.append(f"{verb} Proto-Indo-European {record['pie']}.")
    elif record["uncertain"]:
        lines.append("No accepted Proto-Indo-European etymology; the article marks the origin as unknown or disputed.")
    branches = {"ira-pro": "Proto-Iranian", "iir-nur-pro": "Proto-Nuristani",
                "inc-pro": "Proto-Indo-Aryan", "ae": "Avestan", "ae-old": "Old Avestan",
                "ae-yng": "Young Avestan", "peo": "Old Persian"}
    witnesses = []
    for code, label in branches.items():
        for desc in record["descendants"]:
            if desc["lang"] == code and desc["word"]:
                witnesses.append(f"{label} {desc['word']}")
                break
    if witnesses:
        lines.append("Witnesses: " + ", ".join(witnesses) + ".")
    aliases = citation_aliases()
    credited = []
    for citation in record["citations"]:
        key = BIB_KEYS.get(aliases.get(citation["template"], citation["template"]))
        if key:
            credited.append(HUMAN_NAMES.get(key, key))
    if credited:
        lines.append("Reconstruction credited to " + "; ".join(dict.fromkeys(credited)) + ".")
    return " ".join(lines)


# --------------------------------------------------------------------------
# Devanagari → IAST
# --------------------------------------------------------------------------
# 98 articles give their Sanskrit witness in Devanagari with no `tr=`, which
# would otherwise cost them a CDIAL match for a purely orthographic reason.
# Only the Sanskrit inventory is covered; anything outside it makes the
# transliteration fail loudly rather than guess.

_DEVA_VOWELS = {
    "अ": "a", "आ": "ā", "इ": "i", "ई": "ī", "उ": "u", "ऊ": "ū",
    "ऋ": "ṛ", "ॠ": "ṝ", "ऌ": "ḷ", "ॡ": "ḹ",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
}
_DEVA_MATRAS = {
    "ा": "ā", "ि": "i", "ी": "ī", "ु": "u", "ू": "ū",
    "ृ": "ṛ", "ॄ": "ṝ", "ॢ": "ḷ", "ॣ": "ḹ",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
}
_DEVA_CONSONANTS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ṅ",
    "च": "c", "छ": "ch", "ज": "j", "झ": "jh", "ञ": "ñ",
    "ट": "ṭ", "ठ": "ṭh", "ड": "ḍ", "ढ": "ḍh", "ण": "ṇ",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "ळ": "ḷ",
    "श": "ś", "ष": "ṣ", "स": "s", "ह": "h",
    "क़": "q", "ख़": "x", "ग़": "ġ", "ज़": "z", "ड़": "ṛ", "ढ़": "ṛh", "फ़": "f",
}
_DEVA_SIGNS = {"ं": "ṃ", "ँ": "m̐", "ः": "ḥ", "ऽ": "'"}
_VIRAMA = "्"
# Vedic accents and the avagraha carry no segmental value for head-word matching.
_DEVA_IGNORE = {"॑", "॒", "॓", "॔", "‌", "‍", "।", "॥"}


def devanagari_to_iast(text: str) -> str | None:
    """Transliterate Sanskrit Devanagari. Returns None if a character is out of scope."""
    import unicodedata

    text = unicodedata.normalize("NFC", text or "")
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        pair = text[i:i + 2]
        if ch in _DEVA_IGNORE:
            i += 1
            continue
        if pair in _DEVA_CONSONANTS:          # nukta digraphs first
            consonant, step = _DEVA_CONSONANTS[pair], 2
        elif ch in _DEVA_CONSONANTS:
            consonant, step = _DEVA_CONSONANTS[ch], 1
        else:
            consonant = None
        if consonant is not None:
            i += step
            if i < len(text) and text[i] == _VIRAMA:
                out.append(consonant)
                i += 1
            elif i < len(text) and text[i] in _DEVA_MATRAS:
                out.append(consonant + _DEVA_MATRAS[text[i]])
                i += 1
            else:
                out.append(consonant + "a")   # inherent vowel
            continue
        if ch in _DEVA_VOWELS:
            out.append(_DEVA_VOWELS[ch])
        elif ch in _DEVA_SIGNS:
            out.append(_DEVA_SIGNS[ch])
        elif ch in "-–—":
            out.append("-")
        elif ch.isspace():
            out.append(" ")
        elif "ऀ" <= ch <= "ॿ":
            return None                        # in-script but out of scope: refuse to guess
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def install():
    """Build the installed files and merge this source's rows into the overlay.

    Overlay rows are keyed on ``Etymon_ID``, so re-running replaces this source's
    own rows and leaves every other curated decision untouched.
    """
    import csv as _csv

    stats, params_rows, assignments, audit_rows = build()
    fields = ["Form_ID", "Etymon_ID", "Kind", "Rank", "Status", "Source", "Notes"]
    # `Etymon_ID` is rewritten from `wiir-…` to a durable `f_…` by the first
    # assign_form_ids.py run, so it cannot identify this source's own rows on a
    # rebuild. The citation string can: it survives the rewrite untouched.
    with open(ASSIGNMENTS_FILE, encoding="utf-8", newline="") as handle:
        kept = [
            r for r in _csv.DictReader(handle)
            if f"{SOURCE_KEY}[" not in (r.get("Source") or "")
            and not r["Etymon_ID"].startswith("wiir-")
        ]
    with open(ASSIGNMENTS_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kept + assignments)
    return stats, params_rows, assignments, audit_rows

#!/usr/bin/env python3
"""Install the Zargari lexical material in Rezai Baghbidi's 2003 Romani Studies article.

The article (Romani Studies 5th ser. 13/2: 123--148) is a grammatical sketch, not a
dictionary: its lexicon lives in glossed examples inside numbered sections plus a handful
of glossed list/table blocks.  The PDF carries a genuine Type 1 text layer, so nothing here
is OCR.  ``extract_spans`` decodes it by *glyph name* rather than by the publisher's lossy
``ToUnicode`` map -- the MinionPro subsets encode ``T_h``/``f_i``/``f_l``/``f_f``/``f_f_i``
ligatures, ``.sc`` small capitals, oldstyle figures, Greek, and the Indological
``tunderdot``/``sunderdot``/``macronacute``/``macrontilde`` glyphs, all of which the
embedded ``ToUnicode`` map either drops or mangles.

Every single-quoted gloss span in the numbered sections is a raw source record and appears
in the audit with an explicit status.  ``DECISIONS`` is the checked-in curation layer: one
line per span.  ``EXTRA`` carries the glossed list/table blocks that print their glosses in
a left-hand column instead of in quotes.

Editorial policy, applied uniformly and recorded per record in the audit:

* install every isolated Zargari word the article glosses;
* install a multi-word Zargari item only where the article presents it inside a lexical
  list (interrogatives, indefinites, quantitatives, adverbs, adpositions, conjunctions,
  interjections, and the section 5 lexicon), tagged ``multiword-expression``;
* leave clause and phrase examples, unglossed paradigm cells, and non-Zargari comparanda
  and donor forms out of the installed CSV; they stay in the audit with a reason;
* donor and comparative statements become ``Etymology`` prose, never graph edges: the
  article cites Azari Turkish, Persian, Arabic, Greek, Armenian, Early Romani, Hindi and
  Sanskrit by spelling only, without CDIAL or DEDR identifiers.

Three explicit normalizations are applied to the printed text, each visible in the audit,
which keeps the raw span alongside the curated result:

* the syllable dots of sections 2.4--2.5 and the bracketed non-phonemic glottal onset of
  section 2.4.2 are dropped from ``Form`` (printed stress acutes are kept);
* optional segments printed in parentheses -- ``bax(t)``, ``(ā)kātu``, ``ām(m)ā`` -- are
  expanded into an explicit head plus alternate rows rather than left as punctuation;
* glosses keep the source's wording but use ``;`` between senses, and a verb stem glossed
  with a bare English verb (``xā-/xā-l- 'eat'``) is written ``to eat`` so that stems and
  infinitives of one lexeme carry the same gloss.

Printed page numbers follow the page carrying the gloss span, which is the article's own
citation unit; a handful of examples begin on the preceding page.

Run from ``data/``.  ``--pdf`` points at the article scan (default
``../tmp/pdfs/zargari/zargari-rezai-baghbidi-2003.pdf``); the importer refuses to run
without it and verifies its SHA-256.  ``--install`` writes the canonical outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

SOURCE_ID = "rezaibaghbidi2003zargari"
SNAPSHOT_DATE = "2026-08-25"
COLLATION_DATE = "2026-08-25"
PDF_SHA256 = "d8103f0596fb79dfac0019b6728802f409a41ceb846e351ccfd0d01a9c4434ca"
PDF_PAGES = 26
FIRST_PRINTED_PAGE = 123
LAST_PRINTED_PAGE = 148

LANGUAGE_ID = "Zarg"
DIALECT_TAG = "dialect:Zarg:zargari:Zargari"

ROOT = Path(__file__).resolve().parents[4]
RAW_DIR = ROOT / "data/other/forms/raw_data"
DEFAULT_PDF = ROOT.parent / "tmp/pdfs/zargari/zargari-rezai-baghbidi-2003.pdf"
FORM_OUTPUT = ROOT / "data/other/forms/20260825-rezai-baghbidi-zargari.csv"
AUDIT_OUTPUT = RAW_DIR / "20260825-rezai-baghbidi-zargari-audit.csv"
SAMPLE_OUTPUT = RAW_DIR / "20260825-rezai-baghbidi-zargari-sample.csv"
MANIFEST_OUTPUT = RAW_DIR / "20260825-rezai-baghbidi-zargari-manifest.json"

FORM_FIELDS = [
    "Language_ID", "Parameter_ID", "Form", "Gloss", "Native", "Phonemic", "Notes",
    "Source", "Cognateset", "Etymology", "Entry_Key", "Variant_Of_Key",
    "Borrowed_From_Key", "Derivation_Parent_Keys", "Tags",
]
AUDIT_FIELDS = [
    "Snapshot_Date", "Collation_Date", "Unit_ID", "PDF_Page", "Printed_Page", "Section",
    "Span_Index", "Raw_Form", "Raw_Gloss", "Status", "Reason", "Final_Forms", "Final_Gloss",
    "Final_Tags", "Etymology", "Emitted_Keys", "Review", "Material_Error", "Source",
    "Record_SHA256",
]

# Exclusion codes used by DECISIONS. Every span not marked ``ingest`` or ``dup`` carries one.
EXCLUSION_REASONS = {
    "x-clause": "clause or sentence example illustrating syntax, not a lexical citation",
    "x-phrase": "multi-word phrase example outside a lexical list; the component lexemes are not separately glossed",
    "x-compare": "comparandum in another language (Hindi, Sanskrit, Qorbati, Seliyeri)",
    "x-donor": "donor-language form cited as an etymon, not a Zargari attestation",
    "x-meta": "metalinguistic prose, an English gloss of a preceding gloss, a reconstruction, or a quotation",
}


# --------------------------------------------------------------------------------------
# Text layer
# --------------------------------------------------------------------------------------

# Glyph names in the embedded MinionPro subsets that the Adobe Glyph List does not cover.
EXTRA_GLYPHS = {
    "tunderdot": "ṭ",
    "sunderdot": "ṣ",
    # Composite accents typeset as a single glyph over the preceding vowel.
    "macronacute": "̄́",
    "macrontilde": "̄̃",
}
SMALL_CAP_MARK = "‹"  # marks a small-capital glyph while the running heads are stripped

_HEAD_RE = re.compile(r"^(?:‹.*?\d+|\d+\s+‹.*)$")
_SECTION_RE = re.compile(r"(?:(?<=\s)|^)(\d+(?:\.\d+)+)\.\s")
_QUOTE_RE = re.compile(r"‘([^’]*)’")
_DELIM_RE = re.compile(r"(?:;|:|\be\.g\.|\bbut:|\bNote:|\bnamely,|\bExamples?:|^)\s*")
# Section 5.4 is the last numbered section; the unnumbered back matter must not be swept
# into it.
_BACK_MATTER = "List of Abbreviations"
RAW_FORM_CONTEXT = 120  # characters of preceding text kept as the audit's raw evidence


def _resolve_glyph(name, unresolved):
    if "_" in name:
        return "".join(_resolve_glyph(part, unresolved) for part in name.split("_"))
    if "." in name:
        base, suffix = name.split(".", 1)
    else:
        base, suffix = name, ""
    if suffix == "sc":
        return SMALL_CAP_MARK + _resolve_glyph(base, unresolved)
    if suffix:
        return _resolve_glyph(base, unresolved)
    matched = re.fullmatch(r"(zero|one|two|three|four|five|six|seven|eight|nine)oldstyle", base)
    if matched:
        return _resolve_glyph(matched.group(1), unresolved)
    matched = re.fullmatch(r"([A-Za-z]+)small", base)
    if matched:
        return SMALL_CAP_MARK + matched.group(1).lower()
    if base in EXTRA_GLYPHS:
        return EXTRA_GLYPHS[base]
    from pdfminer.encodingdb import name2unicode

    try:
        return name2unicode(base)
    except Exception:  # pragma: no cover - guarded by test_zargari_text_layer_is_complete
        unresolved[name] = unresolved.get(name, 0) + 1
        return "�"


def extract_pages(pdf_path):
    """Return {pdf_page: text} decoded through the fonts' /Differences glyph names."""
    import pdfminer.pdffont as pdffont
    from pdfminer.psparser import literal_name
    from pdfminer.pdftypes import list_value, resolve1

    unresolved = {}
    original_init = pdffont.PDFSimpleFont.__init__

    def patched_init(self, descriptor, widths, spec):
        original_init(self, descriptor, widths, spec)
        differences = {}
        encoding = resolve1(spec.get("Encoding")) if "Encoding" in spec else None
        if isinstance(encoding, dict):
            code = 0
            for item in list_value(encoding.get("Differences", [])):
                item = resolve1(item)
                if isinstance(item, (int, float)):
                    code = int(item)
                else:
                    differences[code] = literal_name(item)
                    code += 1
        merged = dict(self.unicode_map.cid2unichr) if self.unicode_map else dict(self.cid2unicode)
        for code, glyph in differences.items():
            merged[code] = _resolve_glyph(glyph, unresolved)
        self.cid2unicode = merged
        self.unicode_map = None

    pdffont.PDFSimpleFont.__init__ = patched_init
    try:
        import pdfplumber

        with pdfplumber.open(str(pdf_path)) as pdf:
            if len(pdf.pages) != PDF_PAGES:
                raise SystemExit(f"expected {PDF_PAGES} PDF pages, found {len(pdf.pages)}")
            pages = {i: (page.extract_text() or "") for i, page in enumerate(pdf.pages, 1)}
    finally:
        pdffont.PDFSimpleFont.__init__ = original_init
    if unresolved:
        raise SystemExit(f"unmapped glyph names in the text layer: {sorted(unresolved)}")
    return pages


def build_stream(pages):
    """Join the body lines into one page-indexed stream, de-hyphenating line breaks."""
    lines = []
    for pdf_page in sorted(pages):
        for line in pages[pdf_page].split("\n"):
            stripped = line.strip()
            if stripped and not _HEAD_RE.match(stripped):
                lines.append((pdf_page, stripped))
    chars, page_of = [], []
    for index, (pdf_page, line) in enumerate(lines):
        if line.endswith("-") and index + 1 < len(lines):
            piece, separator = line[:-1], ""
        else:
            piece, separator = line, " "
        for char in piece + separator:
            chars.append(char)
            page_of.append(pdf_page)
    return "".join(chars), page_of


def extract_spans(pdf_path):
    """Every single-quoted gloss span in a numbered section, in printed order."""
    text, page_of = build_stream(extract_pages(pdf_path))
    cut = text.find(_BACK_MATTER)
    if cut < 0:
        raise SystemExit("could not locate the unnumbered back matter")
    marks = [(m.start(1), m.group(1)) for m in _SECTION_RE.finditer(text)]
    marks = [(pos, num) for pos, num in marks if pos < cut]
    if not marks:
        raise SystemExit("no numbered sections found")
    # Top-level section numbers are set in a decorative symbol font, so sections 1--5 have no
    # machine-readable number. Everything before the first numbered subsection belongs to the
    # unnumbered abstract and introduction, cited here as section 1.
    marks = [(0, "1")] + marks
    spans = []
    for index, (start, section) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else cut
        segment = text[start:end]
        previous = 0
        for span_index, match in enumerate(_QUOTE_RE.finditer(segment), 1):
            left = segment[previous:match.start()]
            cuts = [d.end() for d in _DELIM_RE.finditer(left)]
            raw_form = (left[cuts[-1]:] if cuts else left).strip().strip(",").strip()
            # A gloss that is not preceded by a delimiter (a definition inside running prose)
            # would otherwise drag a whole paragraph into the audit's raw column.
            if len(raw_form) > RAW_FORM_CONTEXT:
                raw_form = "…" + raw_form[-RAW_FORM_CONTEXT:]
            position = start + match.start()
            spans.append({
                "section": section,
                "span_index": span_index,
                "pdf_page": page_of[position],
                "printed_page": page_of[position] + FIRST_PRINTED_PAGE - 1,
                "raw_form": raw_form,
                "raw_gloss": match.group(1).strip(),
            })
            previous = match.end()
    return spans


# --------------------------------------------------------------------------------------
# Curation layer
# --------------------------------------------------------------------------------------
#
# section | span | status | forms | gloss | tags | etymology | note
#
# ``forms`` lists one lexeme: the first spelling is the installed head and each further
# ``/``-separated spelling becomes an ``alternate`` row pointing at it with Variant_Of_Key.
# ``span`` values of the shape ``N+k`` are additional records attached to span ``N`` (a
# printed plural, a gendered counterpart, a perfect stem, a listed synonym); the audit
# reports them on the parent span's row.
# ``dup:<section>:<span>`` folds a repeated mention into the record already installed at
# that span, appending this span's citation (and any etymology given here) to it.

_DECISIONS_A = r"""
1|1|x-meta|||||the speakers' own name for the language, quoted as a name rather than glossed
1|2|ingest|kālo/qālo|black; dark|adj|The Gypsy word kālā or kāūlā, cf. Hindi kālā.|Zargari forms cited parenthetically inside the etymology of the Iranian ethnonym Kowli
1|3|x-donor|||||Persian zargar 'goldsmith', quoted inside the village elder's account
2.3.1|1|ingest|pani/pāni/phani/bani|water|noun|Hindi pānī, Sanskrit pānīya-, cf. Qorbati of Širāz punew, punu; Qorbati of Sabzevār, Qā'enāt and Neyšābur panew, punew, punow.|phani and bani are the printed outputs of the aspiration/voicing tendency; the spelling pāni and the comparanda are printed at 5.4
2.3.1|2|ingest|thān/tān|place|noun||
2.3.1|3|ingest|thimi/timi|price|noun||
2.3.1|4|ingest|kher/ker|house|noun||
2.3.1|5|ingest|čhal/čal|shadow|noun|cf. Early Romani učhal.|
2.3.2|1|x-phrase|||||
2.3.2|2|x-clause|||||
2.3.3|1|ingest|pašana|mosquito|noun||printed gloss 'mosquitto'; the aspirated realization [phašana] is phonetic
2.3.3|2|ingest|per/ber|belly; stomach|noun|Hindi peṭ 'basket; belly', Sanskrit peṭa-/peṭā- 'basket', cf. Qorbati of Širāz pitu 'belly'.|the aspirated realization [pher] is phonetic; the comparanda are printed at 5.4
2.3.3|3|ingest|opro|up; above|adv spatial||gloss 'above' added from the repeated mention at 3.20.1.3
2.3.3|4|ingest|kiri/giri|ant|noun||the aspirated realization [khiri] is phonetic
2.3.3|5|ingest|bakri|sheep|noun||the aspirated realization [bakhri] is phonetic
2.3.3|6|ingest|kakliqā|partridge|noun||the aspirated realization [kakhliqā] is phonetic
2.3.4|1|ingest|khitabi/kithabi|book|noun||
2.3.5|1|ingest|dikh|see!|verb impv 2sg||the deaspirated realization [dik] is phonetic
2.3.5|2|ingest|jākh|eye|noun||the deaspirated realization [jāk] is phonetic
2.3.5|3|ingest|jekh/jek|one|num||the deaspirated spelling jek is printed here and used throughout
2.3.6|1|ingest|baxt/bax|luck|noun||the source prints bax(t) for the optional final cluster
2.3.6|2|ingest|vāst/vās|arm; hand|noun|< Old Indo-Aryan *hasta-, cf. Sanskrit hasta-, Hindi hāth.|the source prints vās(t)
2.3.6|3|ingest|dānd/dān|tooth|noun||the source prints dān(d)
2.3.7|1|ingest|toppā|ball|noun||
2.3.7|2|ingest|ākkur|walnut|noun|cf. Early Romani akhor.|
2.3.7|3|ingest|pukko/puqqo|shoulder|noun|cf. Early Romani phiko.|
2.3.7|4|ingest|āmmā/āmā|but|conj||the variant āmā is printed as ām(m)ā at 3.20.3.1 and 4.4.1
2.3.7|5|ingest|āssajipej|to laugh|verb inf||
2.3.7|6|ingest|tāššā|tomorrow|adv temporal||
2.3.7|7|ingest|māččho|fish|noun|cf. Early Romani mačho.|
2.3.7|8|ingest|sudždžo|clean|adj||
2.3.7|9|ingest|bellu|testicle|noun|cf. Early Romani pelo.|
2.3.7|10|ingest|zorreti|maize|noun||
2.3.8|1|ingest|tatho/tatto|hot; warm|adj||
2.3.8|2|ingest|dākhār/dākkār|king|noun|< Early Romani thagar < Armenian thagavor.|etymology printed at 2.3.25
2.3.9|1|ingest|ānglāl/ānglār|from the front|adv spatial||the source prints ānglāl [ānglāl, ānglār]
2.3.9|2|ingest|somlāl/somlār|gold|noun||
2.3.10|1|ingest|vānvro/vāvro|egg|noun||the realizations [vanvro], [vavro], [vāvro] are phonetic; vā(n)vro is printed at 3.1.9
2.3.10|2|ingest|mān|me|pron personal 1sg obl||the nasalized realizations [man] and [ma] are phonetic
2.3.11|1|ingest|m-āč|do not stay!|verb impv neg 2sg||the source derives it from mā-āč
2.3.11|2|ingest|m-ān|do not bring!|verb impv neg 2sg||the source derives it from mā-ān
2.3.12|1|ingest|olungu|their; for them; to them|pron personal 3pl dat gen||the source derives it from olun-ke
2.3.12|2|ingest|olundār|from them; with them|pron personal 3pl abl instr||the source derives it from olun-tār
2.3.13|1|ingest|soj-|to sleep|verb pres stem|The source derives the stem-final -j from *-v.|
2.3.13|2|ingest|ruj-|to cry; to weep|verb pres stem|The source derives the stem-final -j from *-v.|
2.3.13|3|ingest|nāšaj-|to make run|verb caus pres stem|The source derives the stem-final -j from *-v.|
2.3.14|1|ingest|karas|butter|noun||the palatalized realization [kjaras] is phonetic
2.3.14|2|ingest|kirmo|worm|noun||the palatalized realization [kjirmo] is phonetic
2.3.14|3|ingest|gerās/gerāst|horse|noun|< Armenian grast.|the variant gerāst is printed as ge.rās(t) at 2.4.1
2.3.15|1|ingest|eftā|seven|num loanword|< Greek εφτά.|etymology printed at 5.3; the realization [efdā] is phonetic
2.3.15|2|ingest|oxto/oxtó|eight|num loanword|< Greek οχτώ.|etymology printed at 5.3; oxtó is the stress-marked citation at 2.5.1
2.3.16|1|ingest|kāšt/qāšt|wood|noun||
2.3.16|2|ingest|kāt/qāt|scissors|noun||
2.3.16|3|ingest|kon/qon|who?; which?|pron interr||gloss 'which?' added from the interrogative list at 3.3.5.1
2.3.16|4|ingest|olusku/olusqu|his; for him; to him|pron personal 3sg dat gen||
2.3.16|5|ingest|sākulā/sāqulā|bag|noun||
2.3.17|1|ingest|dastas|handle; group|noun||the geminate realization [dassas] is phonetic
2.3.17|2|ingest|pistas|pistachio|noun||the geminate realization [pissas] is phonetic
2.3.18|1|ingest|ānāv|name|noun|< Old Indo-Aryan *nāman-, cf. Sanskrit nāman-, Hindi nām.|
2.3.18|2|dup:2.3.6:2|||||
2.3.19|1|ingest|čhiv/čhip/čiph|language; tongue|noun|< čhib; the source also records the further developments čhip and čiph.|
2.3.20|1|ingest|čuv|throw!|verb impv 2sg||the realization [čuw] is phonetic
2.3.20|2|ingest|guruv|ox; bull|noun||the realization [guruw] is phonetic
2.3.20|3|ingest|džuv|louse|noun||the realization [džuw] is phonetic
2.3.20|4|ingest|usquv|hat|noun||the realization [usquw] is phonetic
2.3.21|1|ingest|idž/iž|yesterday|adv temporal||
2.3.21|2|ingest|mindž/minž|female organ|noun||
2.3.22|1|ingest|haftas|week|noun loanword|cf. Persian hafte.|
2.3.22|2|ingest|har|every|quantifier|cf. Persian har.|
2.3.22|3|ingest|hāvās|air; weather|noun loanword|cf. Persian havā.|
2.3.22|4|ingest|heš|nothing|pron indef|cf. Persian hič.|
2.3.22|5|ingest|hevidži|carrot|noun loanword|cf. Persian havij.|
2.3.23|1|ingest|pučh|ask!|verb impv 2sg||
2.3.23|2|ingest|pušlom|I asked|verb pret 1sg||
2.3.23|3|ingest|pušlān|you asked|verb pret 2sg||printed gloss 'you (sg.) asked'
2.3.24|1|ingest|āngunlo/āngulno|last; previous|adj||
2.3.24|2|ingest|gudlo/guldo|sugar; candy; sweet|noun||
2.3.25|1|dup:2.3.8:2|||||
2.3.26|1|x-meta|||||quotation from Windfuhr (1970: 274)
2.3.26|2|ingest|bu-lovu/bi-lovu|moneyless|adj||the source's hyphens mark the prefix
2.3.26|3|ingest|tu-gu/tu-ke|for you; to you; your|pron personal 2sg dat gen||
2.3.26|4|ingest|ti-phen-is/te-phen-es|you should say|verb subj 2sg||
2.3.27|1|ingest|čhub-ālo/čhib-ālo|cheeky|adj|Literally 'having a tongue'.|
2.3.27|2|x-meta|||||the literal rendering of the preceding form
2.3.27|3|ingest|čuk-ālo/čuq-ālo/čik-ālo|muddy|adj||
2.3.27|4|ingest|dukh-āv/dikh-āv|I should see|verb subj 1sg||
2.4.1|1|ingest|nāj|finger|noun||
2.4.1|2|ingest|lon|salt|noun||
2.4.1|3|dup:2.3.6:3|||||
2.4.1|4|ingest|durom|road|noun loanword|< Greek δρόμος; the source also compares Early Romani drom.|etymology printed at 5.3; the source's syllable dots are editorial
2.4.1|5|ingest|derāk|grape|noun|< Early Romani drakh.|the source's syllable dots are editorial
2.4.1|6|dup:2.3.14:3|||||
2.4.1|7|ingest|phorāl|brother|noun|< Early Romani phral.|the source's syllable dots are editorial
2.4.1|8|ingest|terin|three|num||the source's syllable dots are editorial
2.4.1|9|ingest|qurbāqās|frog|noun||the source's syllable dots are editorial
2.4.1|10|ingest|pārtlameki|to burst; to explode|verb inf loanword|An Azari Turkish infinitive in -meki, cf. section 3.17.3.3.|the source's syllable dots are editorial
2.4.2|1|ingest|āmāl|friend|noun||printed [']ā.māl; section 2.4.2 states the glottal onset is not phonemic and need not be written
2.4.2|2|ingest|enna|nine|num loanword|< Greek εννέα.|etymology printed at 5.3; printed [']en.na
2.4.2|3|dup:2.3.21:1|||||
2.4.2|4|dup:2.3.15:2|||||
2.5.1|1|ingest|áwsa|tear|noun||stress-marked citation; the source's syllable dots are editorial
2.5.1|2|ingest|bukhāló|hungry|adj||stress-marked citation
2.5.1|3|ingest|lāčhó|good|adj||stress-marked citation; the unstressed spelling lāčho occurs only inside phrase examples
2.5.1|4|ingest|mitér|urine|noun||stress-marked citation
2.5.1|5|ingest|pārnó|white|adj||stress-marked citation; the unstressed spelling pārno occurs only inside phrase examples
2.5.1|6|ingest|aja/ája/agar|if|conj|< Azari Turkish áya < Persian ágar.|agar is the variant printed at 3.20.3.1 and 4.6.5
2.5.1|7|ingest|bulúti|cloud|noun loanword|< Azari Turkish bulút.|the source also prints boluti at 5.1
2.5.1|8|dup:2.3.15:2|||||
2.5.1|9|ingest|qoqālā/qóqālā|bone|noun loanword|< Greek κόκκαλο.|qóqālā is the stress-marked citation; etymology printed at 5.3
2.5.2|1|ingest|dévlā|O God!|noun voc||stress-marked citation
2.5.3|1|ingest|méphen|do not say!|verb impv neg 2sg||
2.5.3|2|ingest|mā́khuv|do not weave!|verb impv neg 2sg||
2.5.3|3|ingest|nájilom|I did not come|verb neg 1sg||
2.5.3|4|ingest|nájilomās|I had not come|verb neg 1sg||
2.5.3|5|ingest|nā́džānāvās|I did not know|verb neg 1sg||
2.6.1|1|x-clause|||||
2.6.1|2|x-clause|||||
2.6.1|3|x-clause|||||
2.6.1|4|x-clause|||||
"""

_DECISIONS_B = r"""
3.1.3|1|x-clause|||||
3.1.3|2|x-clause|||||
3.1.3|3|x-clause|||||
3.1.3|4|x-clause|||||
3.1.3|5|x-clause|||||
3.1.3|6|x-clause|||||
3.1.3|7|x-clause|||||
3.1.3|8|x-clause|||||
3.1.3|9|x-clause|||||
3.1.3|10|x-clause|||||
3.1.3|11|x-clause|||||
3.1.4|1|x-clause|||||
3.1.4|2|x-clause|||||
3.1.4|3|x-clause|||||
3.1.4|4|x-clause|||||
3.1.4|5|x-clause|||||
3.1.4|6|x-clause|||||
3.1.5|1|x-clause|||||
3.1.5|2|x-clause|||||
3.1.5|3|x-clause|||||
3.1.5|4|x-clause|||||
3.1.6|1|x-clause|||||
3.1.6|2|x-clause|||||
3.1.6|3|x-clause|||||
3.1.6|4|x-clause|||||
3.1.6|5|x-clause|||||
3.1.7|1|x-clause|||||
3.1.7|2|x-clause|||||
3.1.7|3|x-clause|||||
3.1.7|4|x-clause|||||
3.1.8|1|ingest|bār|stone|noun||
3.1.8|2|ingest|berš|year|noun||
3.1.8|3|ingest|čhār|ash|noun||
3.1.8|4|ingest|čhon|moon|noun||
3.1.8|5|ingest|dād|father|noun||
3.1.8|6|dup:2.3.6:3|||||
3.1.8|7|ingest|dār|door|noun||
3.1.8|8|ingest|gād|shirt|noun||
3.1.8|9|ingest|jiv|snow|noun||
3.1.8|10|ingest|khām|sun|noun||
3.1.8|11|ingest|khān|smell|noun||
3.1.8|12|ingest|khās|grass|noun||
3.1.8|13|ingest|khil|oil|noun||
3.1.8|14|dup:2.4.1:2|||||
3.1.8|15|ingest|masik|month|noun||
3.1.8|16|ingest|murš|man|noun|Hindi māriṣ 'honourable man', Sanskrit māriṣa- 'honourable man', cf. Qorbati of Širāz mārez 'man'.|comparanda printed at 5.4
3.1.8|17|ingest|phus|straw|noun||
3.1.8|18|ingest|qāšt|tree|noun||section 2.3.16 gives kāšt/qāšt 'wood'; this is the source's separate 'tree' sense
3.1.8|19|ingest|rom|Gypsy man; husband|noun||
3.1.8|20|ingest|thāv|string|noun||
3.1.8|21|ingest|thuv|smoke|noun||
3.1.8|22|dup:2.3.6:2|||||
3.1.8|23|ingest|vešq|mountain|noun||
3.1.8|24|ingest|vušt|lip|noun||
3.1.8|25|x-phrase|||||
3.1.8|26|x-phrase|||||
3.1.8|27|ingest|dis|day|noun||
3.1.8|28|ingest|dise|days|noun pl||printed plural of dis
3.1.8|29|dup:3.1.8:19|||||
3.1.8|30|ingest|roma|Gypsy men; husbands|noun pl||printed plural of rom
3.1.8|31|ingest|čhib|language; tongue|noun f||section 2.3.19 derives čhiv from this form
3.1.8|32|ingest|čhiba|languages; tongues|noun f pl||printed plural of čhib
3.1.9|1|ingest|bārbāl/bārbājli|wind|noun f||
3.1.9|1+1|ingest|bārbājla|wind|noun f pl||printed irregular plural of bārbāl/bārbājli
3.1.9|2|ingest|bāšno|cock|noun||the printed plural bāšno is identical to the singular
3.1.9|3|ingest|bori|bride; daughter-in-law|noun f||
3.1.9|3+1|ingest|bojra|brides; daughters-in-law|noun f pl||printed irregular plural of bori
3.1.9|4|ingest|čhā|boy; son|noun||
3.1.9|4+1|ingest|čhāvu|boys; sons|noun pl||printed irregular plural of čhā
3.1.9|5|ingest|gi|heart|noun||the printed plural gi is identical to the singular
3.1.9|6|ingest|džukel|dog|noun||
3.1.9|6+1|ingest|džukle|dogs|noun pl||printed irregular plural of džukel
3.1.9|7|ingest|džuvel|woman|noun f||
3.1.9|7+1|ingest|džuvla|women|noun f pl||printed irregular plural of džuvel
3.1.9|8|ingest|muj|face; mouth|noun||the printed plural muj is identical to the singular
3.1.9|9|ingest|rizi|rice|noun||the printed plural rizi is identical to the singular
3.1.9|10|ingest|šoru|head|noun||the printed plural šoru is identical to the singular
3.1.9|11|dup:2.3.10:1|||||
3.1.9|12|ingest|zi|balance; scales|noun f||the printed plural zi is identical to the singular
3.1.9|13|ingest|zimi|broth; soup|noun f loanword|< Greek ζουμί.|the printed plural zimi is identical to the singular; etymology and the 'soup' sense printed at 5.3
3.1.9|14|dup:3.1.9:10|||||
3.1.9|15|x-meta|||||the reconstruction *šoro and its plural, cited to explain number neutralization
3.1.10|1|ingest|šir-i|lion|noun m loanword|< Persian šir.|the source's hyphen marks the masculine adaptation suffix
3.1.10|2|ingest|köraken-is|son-in-law|noun m loanword|< Azari Turkish körakan.|
3.1.10|3|ingest|qahreman-i|hero|noun m loanword|< Persian qahramān.|
3.1.10|3+1|ingest|qahreman-isa|heroine|noun f loanword|< Persian qahramān.|printed feminine counterpart of qahreman-i
3.1.10|4|ingest|faqir-is|beggar|noun m loanword|< Persian faqir.|
3.1.10|4+1|ingest|faqir-isa|beggar|noun f loanword|< Persian faqir.|printed feminine counterpart of faqir-is
3.1.10|5|ingest|pandžara-s|window|noun m loanword|< Persian panjere.|
3.1.10|6|ingest|džudža-s|chicken|noun m loanword|< Persian juje.|
3.1.10|6+1|ingest|džudža-na|chicken|noun f loanword|< Persian juje.|printed feminine counterpart of džudža-s
3.1.10|7|ingest|miva-s|fruit|noun m loanword|< Persian mive.|
3.1.10|7+1|ingest|miva-na|fruit|noun f loanword|< Persian mive.|printed feminine counterpart of miva-s
3.1.10|8|ingest|zandžir-a|chain|noun f loanword|< Persian zanjir.|
3.1.10|9|ingest|tulki-na/tulkina|fox|noun f loanword|< Azari Turkish tulki.|the unhyphenated spelling tulkina is printed at 5.1
3.1.11|1|x-clause|||||
3.1.11|2|x-clause|||||
3.1.11|3|x-phrase|||||
3.1.11|4|x-phrase|||||
3.1.11|5|ingest|tinčhā|child|noun|< *tikno čhavo.|glossed parenthetically inside the possessive example
3.1.11|6|x-phrase|||||
3.1.12|1|x-phrase|||||
3.1.12|2|x-phrase|||||
3.1.12|3|x-clause|||||
3.1.12|4|x-meta|||||the literal rendering of the preceding clause
3.1.12|5|x-clause|||||
3.1.13|1|x-clause|||||
3.1.13|2|x-clause|||||
3.1.13|3|x-meta|||||the literal rendering of the preceding clause
3.1.13|4|x-clause|||||
3.1.13|5|x-clause|||||
3.1.13|6|x-clause|||||
3.1.14|1|ingest|idž-dan/ižār-dān|since yesterday|adv temporal|Formed with the Azari Turkish ablative suffix -dan.|
3.1.14|2|ingest|qānāx-dān/qānāxdān|since when?|adv interr temporal|Formed with the Azari Turkish ablative suffix -dan.|the unhyphenated spelling is printed at 3.3.5.1
3.1.15|1|ingest|opr-āl|from above|adv spatial||
3.1.15|2|dup:2.3.9:1|||||
3.1.15|3|ingest|ter-āl|from the bottom|adv spatial|The source notes it stands for an expected *tel-ār.|
3.1.15|4|ingest|teli|bottom|noun||section 3.20.1.3 gives the adverb 'below' and 3.20.2.1 the postposition 'beneath, under'
3.1.16|1|x-clause|||||
3.1.16|2|ingest|baγi-sti|in the garden; to the garden|noun loc||locative of an otherwise unglossed baγi 'garden'
3.1.16|3|ingest|vešk-isti|in the mountain; to the mountain|noun loc||locative of vešq 'mountain'
3.1.16|4|x-meta|||||the English concept 'every', not a Zargari form
3.1.16|5|dup:3.20.1.2:7|||||
3.1.16|6|dup:3.20.1.2:8|||||
3.1.17|1|x-phrase|||||
3.1.18|1|x-clause|||||
3.1.19|1|x-clause|||||
3.2.1|1|ingest|ruv|wolf|noun||printed gloss 'wolf, the wolf' illustrates the absence of a definite article
3.2.1|2|dup:2.3.3:5|||||
3.2.1|3|x-clause|||||
3.2.2|1|dup:2.3.5:3|||||
3.2.2|2|x-phrase|||||
3.2.2|3|x-phrase|||||
3.2.2|4|x-phrase|||||
3.3.1.3|1|x-clause|||||
3.3.1.3|2|x-clause|||||
3.3.1.3|3|x-clause|||||
3.3.1.3|4|x-clause|||||
3.3.2.3|1|x-clause|||||
3.3.2.3|2|x-clause|||||
3.3.2.3|3|x-phrase|||||
3.3.2.3|4|x-phrase|||||
3.3.2.3|5|x-phrase|||||
3.3.2.3|6|x-phrase|||||
3.3.2.4|1|ingest|elakāvā|this same|pron demonstrative prox m||
3.3.2.4|2|ingest|elakaja|this same|pron demonstrative prox f||
3.3.2.4|3|ingest|elakālā|these same|pron demonstrative prox pl||
3.3.2.4|4|ingest|elakovā|that same|pron demonstrative dist m||
3.3.3.1|1|x-phrase|||||
3.3.3.1|2|x-phrase|||||
3.3.3.1|3|x-phrase|||||
3.3.3.1|4|x-phrase|||||
3.3.3.1|5|x-clause|||||
3.3.3.1|6|ingest|āmāros|ours|pron poss 1pl obl||
3.3.3.1|7|ingest|āmārostār|from ours; with ours|pron poss 1pl abl instr||
3.3.3.1|8|x-clause|||||
3.3.3.1|9|x-clause|||||
3.3.3.1|10|x-clause|||||
3.3.3.1|11|x-clause|||||
3.3.3.3|1|x-phrase|||||
3.3.3.3|2|x-phrase|||||
3.3.3.3|3|ingest|pu-dādus|his own father; her own father|noun obl poss||printed gloss marks the oblique
3.3.3.3|4|ingest|pu-dādustār|from his own father; with his own father|noun abl instr poss||
3.3.4.1|1|x-clause|||||
3.3.4.1|2|x-clause|||||
3.3.4.1|3|x-clause|||||
3.3.5.1|1|x-clause|||||
3.3.5.1|2|x-clause|||||
3.3.5.1|3|x-clause|||||
3.3.5.1|4|x-clause|||||
3.3.5.1|5|x-clause|||||
3.3.5.1|6|x-clause|||||
3.3.5.2|1|x-clause|||||
3.4.1|1|x-phrase|||||
3.4.1|2|x-phrase|||||
3.4.1|3|x-phrase|||||
3.4.1|4|x-phrase|||||
3.4.1|5|x-phrase|||||
3.4.1|6|x-phrase|||||
3.4.1|7|x-phrase|||||
3.4.1|8|x-clause|||||
3.4.1|9|x-clause|||||
3.4.1|10|x-clause|||||
3.4.3|1|x-clause|||||
3.4.3|2|x-clause|||||
3.4.3|3|x-clause|||||
3.4.3|4|x-clause|||||
3.5.1.1|1|dup:3.5.1:t29|||||
3.5.1.1|2|ingest|jokus-jek|twenty-one|num||printed gloss '21'
3.5.1.1|3|ingest|pejindā-tirāndā-duj|eighty-two|num||printed gloss '82'
3.5.1.1|4|ingest|šel-i-pāndž|one hundred and five|num||printed gloss '105'
3.5.1.1|5|ingest|šel-i-pejindā-sārāndā-šov|one hundred and ninety-six|num||printed gloss '196'
3.5.1.2|1|x-phrase|||||
3.5.1.2|2|x-phrase|||||
3.5.1.2|3|x-phrase|||||
3.5.1.2|4|x-phrase|||||
3.5.1.2|5|x-phrase|||||
3.5.1.2|6|x-phrase|||||
3.5.1.2|7|x-phrase|||||
3.5.1.3|1|ingest|dana|piece|noun||used as a counting classifier after numerals
3.5.1.3|2|x-phrase|||||
3.5.2.1|1|ingest|aval|first|num ord||
3.5.2.1|2|x-phrase|||||
3.5.2.1|3|x-phrase|||||
3.5.3.1|1|ingest|dujendār-jek/duj-jek/paš|half|num||the source gives paš as an equivalent
3.5.3.1|2|ingest|pāndžundār-ištār/pāndž-ištār|four fifths|num||printed gloss 'four fifth'
"""

_DECISIONS_C = r"""
3.6.2|1|ingest|biken-|to sell|verb pres stem||
3.6.2|1+1|ingest|bikend-|to sell|verb perfect stem||printed biken-d-
3.6.2|2|ingest|čhin-|to cut|verb pres stem||
3.6.2|2+1|ingest|čhind-|to cut|verb perfect stem||printed čhin-d-
3.6.2|3|ingest|nāngāv-|to hit; to strike|verb pres stem||printed with an en dash, nāngāv–
3.6.2|3+1|ingest|nāngāvd-|to hit; to strike|verb perfect stem||printed nāngāv-d-
3.6.2|4|ingest|beš-|to sit|verb pres stem||
3.6.2|4+1|ingest|bešd-|to sit|verb perfect stem||printed beš-d-
3.6.2|5|ingest|čor-|to steal|verb pres stem||
3.6.2|5+1|ingest|čord-|to steal|verb perfect stem||printed čor-d-
3.6.2|6|ingest|d-|to give|verb pres stem||the source gives an identical perfect stem d-
3.6.2|7|ingest|pāng-|to break|verb pres stem||
3.6.2|7+1|ingest|pāngl-|to break|verb perfect stem||printed pāng-l-
3.6.2|7+2|ingest|pučh-|to ask|verb pres stem||printed with an en dash, pučh–; the gloss span is shared with pāng- in the printed line
3.6.2|7+3|ingest|pušl-|to ask|verb perfect stem||printed puš-l-
3.6.2|8|ingest|dikh-|to see; to look|verb pres stem||
3.6.2|8+1|ingest|dikhl-|to see; to look|verb perfect stem||printed dikh-l-
3.6.2|9|ingest|l-|to take; to get; to buy|verb pres stem||the source gives an identical perfect stem l-
3.6.2|10|ingest|māng-|to want|verb pres stem||
3.6.2|10+1|ingest|māngl-|to want|verb perfect stem||printed māng-l-
3.6.2|11|ingest|xā-|to eat|verb pres stem||
3.6.2|11+1|ingest|xāl-|to eat|verb perfect stem||printed xā-l-
3.7.4|1|x-clause|||||
3.7.4|2|x-clause|||||
3.8.1|1|x-meta|||||the English term 'historic present'
3.8.1|2|x-clause|||||
3.8.1|3|x-clause|||||
3.8.1|4|x-clause|||||
3.8.1|5|x-clause|||||
3.9.1|1|x-clause|||||
3.9.1|2|x-clause|||||
3.9.1|3|x-clause|||||
3.9.1|4|x-clause|||||
3.9.1|5|x-clause|||||
3.9.1|6|x-meta|||||the literal rendering of the preceding clause
3.10.1|1|x-clause|||||
3.10.1|2|x-clause|||||
3.10.1|3|x-clause|||||
3.10.1|4|x-clause|||||
3.10.2|1|ingest|gölu|he went|verb pret 3sg m||irregular perfect of džejipej 'to go'
3.10.2|1+1|ingest|geli|she went|verb pret 3sg f||irregular perfect of džejipej 'to go'
3.10.2|2|ingest|gölum|I went|verb pret 1sg||
3.10.2|3|ingest|mulo|he died|verb pret 3sg m||
3.10.2|3+1|ingest|muli|she died|verb pret 3sg f||
3.10.2|4|ingest|pölu|he fell|verb pret 3sg m||
3.10.2|4+1|ingest|peli|she fell|verb pret 3sg f||
3.10.2|5|ingest|ajili|he came; she came|verb pret 3sg||the source notes it shows no gender distinction
3.11.1|1|x-clause|||||
3.11.1|2|x-clause|||||
3.11.1|3|x-clause|||||
3.11.1|4|x-clause|||||
3.11.1|5|x-clause|||||
3.12.1|1|x-clause|||||
3.12.1|2|x-clause|||||
3.12.1|3|x-clause|||||
3.12.1|4|x-clause|||||
3.12.2|1|x-clause|||||
3.13.1|1|dup:2.3.23:1|||||
3.13.1|2|dup:2.3.5:1|||||
3.13.2|1|dup:3.6.2:6|||||
3.13.2|2|ingest|čid-i|pull!|verb impv 2sg||the source's hyphen separates the stem from the harmonic ending
3.13.2|3|ingest|d-e|give!|verb impv 2sg||
3.13.2|4|ingest|γānd-o|comb!|verb impv 2sg||
3.13.2|5|ingest|jir-i|put on!|verb impv 2sg||
3.13.2|6|ingest|l-e|buy!; get!|verb impv 2sg||
3.13.3|1|ingest|sov|sleep!|verb impv 2sg||the source notes the imperative keeps the basic stem variant sov- beside present soj-
3.13.3|1+1|ingest|sovun|sleep!|verb impv 2pl||
3.13.3|2|dup:3.17.3.1:5|||||
3.13.3|3|ingest|ruv|cry!|verb impv 2sg||homonymous with ruv 'wolf'
3.13.3|3+1|ingest|ruvun|cry!|verb impv 2pl||
3.13.3|4|ingest|rujipej|to cry; to weep|verb inf||printed ruj-ipej
3.14.1|1|ingest|isipej|to be|verb inf copula||printed is-ipej at 3.18.2
3.14.1|2|ingest|bešipej|to sit|verb inf||
3.14.1|3|ingest|bešdo|sat|verb pp m||
3.14.1|4|ingest|bešdo-som|I am sitting|verb 1sg m||the source glosses it literally as 'I have sat'
3.14.1|4+1|ingest|bešdi-som|I am sitting|verb 1sg f||the source glosses it literally as 'I have sat'
3.15.1|1|ingest|nāšipej|to run|verb inf||
3.15.1|1+1|ingest|nāš-|to run|verb pres stem||
3.15.1|2|ingest|nāšajipej|to make run|verb inf caus||
3.15.1|2+1|ingest|nāšāvd-|to make run|verb caus perfect stem||
3.15.1|3|ingest|pijipej|to drink|verb inf||
3.15.1|3+1|ingest|pij-|to drink|verb pres stem||
3.15.1|4|ingest|pijajipej|to make drink|verb inf caus||
3.15.1|4+1|ingest|pijaj-|to make drink|verb caus pres stem||
3.15.1|4+2|ingest|pijāvd-|to make drink|verb caus perfect stem||
3.15.2|1|ingest|šukhar-|to cause to become dry; to dry|verb caus pres stem|A factitive in -ar to the adjective šukho 'dry'.|
3.15.2|1+1|ingest|šukhard-|to cause to become dry; to dry|verb caus perfect stem||
3.15.2|2|ingest|šukho|dry|adj||
3.15.2|3|ingest|čikar-|to cause to be muddy|verb caus pres stem|A factitive in -ar to the noun čik 'mud'.|
3.15.2|3+1|ingest|čikard-|to cause to be muddy|verb caus perfect stem||
3.15.2|4|ingest|čik|mud|noun||
3.16.1|1|ingest|čhindiv-|to be cut|verb pass pres stem||
3.16.1|1+1|ingest|čhindil-|to be cut|verb pass perfect stem||
3.16.1|2|ingest|xaliv-|to be eaten|verb pass pres stem||
3.16.1|2+1|ingest|xalil-|to be eaten|verb pass perfect stem||
3.16.1|3|ingest|phuriv-|to get old|verb pass pres stem||
3.16.1|3+1|ingest|phuril-|to get old|verb pass perfect stem||
3.16.1|4|ingest|phuro|old|adj||
3.16.1|5|ingest|khiniv-|to get tired|verb pass pres stem||
3.16.1|5+1|ingest|khinil-|to get tired|verb pass perfect stem||
3.16.1|6|ingest|khino|tired|adj||
3.17.1.1|1|x-clause|||||
3.17.1.1|2|x-clause|||||
3.17.2.1|1|ingest|xajeni/xajenis|eating|verb participle pres m||
3.17.2.1|1+1|ingest|xajenisa|eating|verb participle pres f||
3.17.2.2|1|ingest|čhindo|cut|verb pp m||
3.17.2.2|1+1|ingest|čhindi|cut|verb pp f||
3.17.2.2|1+2|ingest|čhinde|cut|verb pp pl||
3.17.3.1|1|ingest|čhinipej|to cut|verb inf||
3.17.3.1|2|dup:3.15.1:1|||||
3.17.3.1|3|dup:3.15.1:2|||||
3.17.3.1|4|ingest|phenipej|to say; to tell|verb inf||
3.17.3.1|5|ingest|sojipej|to sleep|verb inf||
3.17.3.2|1|ingest|dejipej|to give|verb inf||irregular infinitive to the stem d-
3.17.3.2|2|ingest|džejipej|to go|verb inf||irregular infinitive to the stem džā-
3.17.3.2|3|ingest|lejipej|to buy; to get|verb inf||irregular infinitive to the stem l-
3.17.3.3|1|ingest|akmeki|to plant|verb inf loanword|< Azari Turkish akmek.|
3.18.2|1|dup:3.14.1:1|||||
3.18.2|2|ingest|ojipey|to become|verb inf copula||printed oj-ipey, with -ipey for the -ipej of every other infinitive in the article
3.19.1.1|1|ingest|ašti|can|verb modal indecl||
3.19.1.1|2|ingest|n-ašti|cannot|verb modal indecl neg||
3.19.1.1|3|x-clause|||||
3.19.1.1|4|x-clause|||||
3.19.1.1|5|x-clause|||||
3.19.1.2|1|ingest|garak|must|verb modal indecl||
3.19.1.2|2|x-clause|||||
3.19.1.2|3|x-clause|||||
3.20.1.2|1|ingest|āngunlo dis|the day before yesterday|adv temporal multiword-expression||
3.20.1.2|2|ingest|avdis|today|adv temporal||
3.20.1.2|3|ingest|āvā berš|this year|adv temporal multiword-expression||
3.20.1.2|4|ingest|āvur dis|the day after tomorrow|adv temporal multiword-expression||
3.20.1.2|5|ingest|ājāndās berš|next year|adv temporal multiword-expression||
3.20.1.2|6|ingest|bersi|last year|adv temporal||
3.20.1.2|7|ingest|diseste/dis-este|every day|adv temporal||the hyphenated spelling is printed at 3.1.16
3.20.1.2|8|ingest|rājtādu/rājt-ādu|every night|adv temporal||the hyphenated spelling is printed at 3.1.16
3.20.1.2|9|dup:2.3.21:1|||||
3.20.1.2|10|ingest|nāklās rāt/idž bijavli|last night|adv temporal multiword-expression||the source prints two expressions separated by a slash
3.20.1.2|11|ingest|šambas|Saturday|noun temporal||
3.20.1.2|12|ingest|tāqānāsqu|up to now|adv temporal||
3.20.1.2|13|dup:2.3.7:6|||||
3.20.1.2|14|ingest|terin berš āngunlo/terin berš ānglo|three years ago|adv temporal multiword-expression||the source prints āng(un)lo
3.20.1.3|1|ingest|ākātu/kātu/ātu|here|adv spatial||the source prints (ā)kātu/ātu
3.20.1.3|2|ingest|ānglo|front; in front|adv spatial||
3.20.1.3|3|ingest|anvri|out; outside|adv spatial||
3.20.1.3|4|ingest|ānvro|inside|adv spatial||
3.20.1.3|5|ingest|bāšu|near|adv spatial||
3.20.1.3|6|ingest|khal|this side|adv spatial||
3.20.1.3|7|ingest|mošgār|in the middle|adv spatial||
3.20.1.3|8|ingest|okorik|that side|adv spatial||
3.20.1.3|9|ingest|okotu/kotu/otu|there|adv spatial||the source prints (o)kotu/otu
3.20.1.3|10|dup:2.3.3:3|||||
3.20.1.3|11|ingest|teli|below|adv spatial||
3.20.1.3|12|ingest|pālo|back|adv spatial||
3.20.1.4|1|ingest|doqri|well|adv manner||
3.20.1.4|2|ingest|khal|this way|adv manner||homonymous with khal 'this side'
3.20.1.4|3|ingest|reštār|happily|adv manner||
3.20.1.4|4|ingest|sige|quickly|adv manner||
3.20.1.5|1|ingest|but|very|adv degree||the quantitative pronoun list at 3.3.7.1 gives 'many, much, several'
3.20.1.5|2|ingest|ela|only|adv degree||
3.20.1.5|2+1|ingest|feqeti|only|adv degree||printed as a synonym of ela
3.20.1.5|3|ingest|soni|enough|adv degree||
3.20.1.5|4|ingest|xajnri/xajri|few; little|adv degree quantifier||the source prints xaj(n)ri
3.20.1.6|1|ingest|hatmi|undoubtedly|adv||
3.20.1.6|2|ingest|vājā|yes|part||
3.20.1.6|3|ingest|sori|yes|part||used as an affirmative reply to a negative question; homonymous with the interrogative sori 'how?'
3.20.1.7|1|ingest|nā|no|part neg||
3.20.1.8|1|ingest|avalan|first; firstly|adv||
3.20.1.8|2|ingest|āndāmā|together|adv loanword|< Greek αντάμα.|etymology printed at 5.3
3.20.1.8|3|ingest|dā|also; too|adv||
3.20.1.8|4|ingest|kāš|I wish|part||
3.20.1.8|5|ingest|pāndā|again|adv||
3.20.1.8|6|ingest|sar|like|adv||also listed as a postposition at 3.20.2.1
3.20.1.8|7|ingest|šājad|perhaps|adv||
3.20.1.8|7+1|ingest|balkam|perhaps|adv||printed as a synonym of šājad
3.20.1.8|8|ingest|jani|i.e.; that is to say|adv||
3.20.2.1|1|ingest|anvri/avri|out of|postp spatial||the source prints a(n)vri
3.20.2.1|2|ingest|ānglo|in front of|postp spatial||
3.20.2.1|3|ingest|ānvro/āvro|in; inside|postp spatial||the source prints ā(n)vro
3.20.2.1|4|ingest|bāšu|near|postp spatial||
3.20.2.1|5|ingest|opro|above|postp spatial||
3.20.2.1|6|ingest|pālo|behind|postp spatial||
3.20.2.1|7|ingest|sar|like|postp||
3.20.2.1|8|ingest|sarik|for the sake of; to; towards|postp||
3.20.2.1|9|ingest|tā|till; until; as far as|postp||
3.20.2.1|10|ingest|teli|beneath; under|postp spatial||
3.20.3.1|1|dup:2.5.1:6|||||
3.20.3.1|2|dup:2.3.7:4|||||
3.20.3.1|3|ingest|kālus güra|because; since; so that|conj multiword-expression||
3.20.3.1|4|ingest|ki|who; whom; which; that; when; where; so that|conj relative||
3.20.3.1|5|ingest|maja|unless|conj|< Persian magar.|section 3.3.5.2 describes the same word as a borrowed question particle
3.20.3.1|6|ingest|tā|in order that; so that|conj||homonymous with the postposition tā
3.20.3.1|7|ingest|vali|but; however|conj||
3.20.3.1|8|ingest|jā|or|conj||
3.20.3.1|9|ingest|jo/o|and|conj||the source prints ( j)o
3.20.4.1|1|ingest|ne-phendom|I did not say|verb neg 1sg||
3.20.4.1|2|ingest|nā-šundās|he did not hear|verb neg 3sg||
3.20.4.2|1|ingest|isi/si|is; are; there is; there are|verb copula 3sg 3pl||the source prints (i)si
3.20.4.2|2|ingest|nā-nāj|is not; are not; there is not; there are not|verb copula neg 3sg 3pl||the printed line breaks after nā- and resumes with -nāj
3.20.4.3|1|dup:2.3.11:1|||||
3.20.4.3|2|dup:2.3.11:2|||||
3.20.5.1|1|ingest|ah|ah!; ugh!|interj||
3.20.5.1|2|ingest|āx|alas!; ouch!|interj||
3.20.5.1|3|ingest|bah|how nice!; wow!|interj||
3.20.5.1|4|ingest|he|oh!|interj||
3.20.5.1|5|ingest|vāj|alas!; woe!|interj||
4.1.1|1|x-phrase|||||
4.1.1|2|x-phrase|||||
4.1.1|3|x-phrase|||||
4.2.1|1|x-clause|||||
4.2.1|2|x-clause|||||
4.2.2|1|x-clause|||||
4.2.2|2|x-clause|||||
4.2.3|1|x-clause|||||
4.2.4|1|x-clause|||||
4.2.4|2|x-clause|||||
4.2.4|3|x-clause|||||
4.2.4|4|x-clause|||||
4.2.5|1|x-clause|||||
4.3.1|1|x-phrase|||||
4.4.1|1|dup:2.3.7:4|||||
4.4.1|2|dup:3.20.3.1:8|||||
4.4.1|3|dup:3.20.3.1:9|||||
4.4.1|4|dup:3.20.3.1:7|||||
4.5.1|1|x-clause|||||
4.6.1|1|dup:3.20.3.1:4|||||
4.6.1|2|x-clause|||||
4.6.1|3|x-clause|||||
4.6.1|4|x-clause|||||
4.6.2|1|x-meta|||||the English verb 'want', naming the complement-taking predicate
4.6.2|2|x-clause|||||
4.6.2|3|x-clause|||||
4.6.2|4|x-clause|||||
4.6.3|1|dup:3.20.3.1:4|||||
4.6.3|2|x-clause|||||
4.6.3|3|x-clause|||||
4.6.4|1|dup:3.20.3.1:3|||||
4.6.4|2|dup:3.20.3.1:4|||||
4.6.4|3|dup:3.20.3.1:6|||||
4.6.4|4|x-clause|||||
4.6.5|1|dup:2.5.1:6|||||
4.6.5|2|dup:3.20.3.1:5|||||
4.6.5|3|x-clause|||||
4.6.5|4|x-clause|||||
5.1|1|ingest|bilakis|wrist|noun loanword|< Azari Turkish bilak.|
5.1|2|ingest|boluti|cloud|noun loanword|< Azari Turkish bulut.|the stress-marked citation bulúti is printed at 2.5.1
5.1|3|ingest|dirseki|elbow|noun loanword|< Azari Turkish dirsak.|
5.1|4|ingest|döbiki|knee|noun loanword|< Azari Turkish döbik.|
5.1|5|ingest|jārpaki|leaf|noun loanword|< Azari Turkish yārpak.|
5.1|6|ingest|kujruka|tail|noun loanword|< Azari Turkish kuyruk.|
5.1|7|ingest|naštaliki|breakfast|noun loanword|< Azari Turkish nāštāloq.|
5.1|8|ingest|pāmboqi|cotton|noun loanword|< Azari Turkish pāmboq.|
5.1|9|ingest|qaši|eyebrow|noun loanword|< Azari Turkish qāš.|
5.1|10|ingest|qatiki|yoghurt|noun loanword|< Azari Turkish qātoq.|
5.1|11|ingest|saremsaki|garlic|noun loanword|< Azari Turkish sarimsāq.|
5.1|12|ingest|süti|milk|noun loanword|< Azari Turkish süt.|
5.1|13|ingest|tosbāqās|tortoise|noun loanword|< Azari Turkish tosbāqā.|
5.1|14|dup:3.1.10:9|||||
5.2|1|ingest|āsemān|sky|noun loanword|< Persian ās(e)mān.|
5.2|2|ingest|diz|town|noun loanword|< Persian dez/dež 'fortress, fortified town'.|
5.2|3|ingest|resipej|to arrive|verb inf loanword|< Persian res-[idan].|printed res-[ipej]
5.2|4|ingest|xoš|good; pleasant|adj loanword|< Persian xoš.|
5.2|5|x-donor|||||the Persian donor forms listed together after the Zargari examples
5.2|6|ingest|ejbi|defect|noun loanword|< Arabic ‘aib, through Azari Turkish.|
5.3|1|dup:3.20.1.8:2|||||
5.3|2|dup:2.4.1:4|||||
5.3|3|dup:2.3.15:1|||||
5.3|4|dup:2.4.2:2|||||
5.3|5|ingest|luludi|flower|noun loanword|< Greek λουλούδι.|the article prints λoυλούλι
5.3|6|dup:2.5.1:9|||||
5.3|7|dup:2.3.15:2|||||
5.3|8|ingest|sārāndā|forty|num loanword|< Greek σαράντα.|
5.3|9|ingest|tipta|something|pron indef loanword|< Greek τίποτα.|the source cites it inside jek tipta 'something'
5.3|10|ingest|sir tipta|everything|pron indef multiword-expression loanword|< Greek τίποτα.|
5.3|11|ingest|tirāndā|thirty|num loanword|< Greek τριάντα.|
5.3|12|dup:3.1.9:13|||||
5.4|1|ingest|kān|ear|noun|Hindi kān, Sanskrit kárṇa-, cf. Seliyeri hal-kerne.|
5.4|2|ingest|kāšt|tree|noun|Hindi kāṣṭh, kāṣṭha 'wood', Sanskrit kāṣṭhá- 'wood', cf. Qorbati of Širāz kāštā 'tree; wood'.|section 2.3.16 gives kāšt/qāšt in the sense 'wood'
5.4|3|x-compare|||||
5.4|4|x-compare|||||
5.4|5|x-compare|||||
5.4|6|ingest|mās|meat|noun|Hindi mās, Sanskrit māṁsá-, Vedic mās-, cf. Qorbati of Sabzevār and Neyšābur masi, masil, masir; Qorbati of Qā'enāt masi, masil, masir, moñsi.|
5.4|7|dup:3.1.8:16|||||
5.4|8|x-compare|||||
5.4|9|x-compare|||||
5.4|10|x-compare|||||
5.4|11|ingest|nāk|nose|noun|Hindi nāk, Sanskrit nakra-/nakrā-, cf. Qorbati of Qā'enāt and Neyšābur bar-nōgi.|
5.4|12|dup:2.3.1:1|||||
5.4|13|dup:2.3.3:2|||||
5.4|14|x-compare|||||
5.4|15|x-compare|||||
5.4|16|x-compare|||||
5.4|17|ingest|jag|fire|noun|Hindi agan, agin, agini, āg, Sanskrit agní-, cf. Qorbati of Sabzevār and Neyšābur agi, agir, ōgi; Qorbati of Qā'enāt ogi.|
"""

DECISIONS = _DECISIONS_A + _DECISIONS_B + _DECISIONS_C


# Glossed list and table blocks whose glosses are printed in a left-hand column instead of
# in single quotes, so they are not gloss spans. Transcribed from the layout-preserving
# render of the same text layer.
#
# printed page | section | item | status | forms | gloss | tags | etymology | printed line | note

EXTRA = r"""
133|3.2.2|t01|ingest|jedana|one piece of|num indef||or by jedana (literally: one piece of)|the source's parenthetical literal rendering; jedana marks indefiniteness
136|3.3.5.1|t01|dup:2.3.16:3||||| who? which? nom. kon/qon|
136|3.3.5.1|t02|ingest|kos/qos|who?; which?|pron interr obl||obl. kos/qos|
136|3.3.5.1|t03|ingest|kosku/qosqu|who?; which?|pron interr dat gen||dat./gen. kosku/qosqu|
136|3.3.5.1|t04|ingest|kostār/qostār|who?; which?|pron interr abl instr||abl./ins. kostār/qostār|
136|3.3.5.1|t05|ingest|kostu/qostu|who?; which?|pron interr loc||loc. kostu/qostu|
136|3.3.5.1|t06|ingest|so|what?|pron interr nom obl||what? nom./obl. so|
136|3.3.5.1|t07|ingest|sosku|what?|pron interr dat gen||dat./gen. sosku|
136|3.3.5.1|t08|ingest|sostār|what?|pron interr abl instr||abl./ins. sostār|
136|3.3.5.1|t09|ingest|sostu|what?|pron interr loc||loc. sostu|
136|3.3.5.1|t10|ingest|qarik/qari|where?|pron interr||where? qari(k), qonari|
136|3.3.5.1|t11|ingest|qonari|where?|pron interr||where? qari(k), qonari|
136|3.3.5.1|t12|ingest|qātār|whence?|pron interr||whence? qātār|
136|3.3.5.1|t13|ingest|qojinek|which?|pron interr||which? qojinek, qojna|
136|3.3.5.1|t14|ingest|qojna|which?|pron interr||which? qojinek, qojna|
136|3.3.5.1|t15|ingest|qānāx|when?|pron interr temporal||when? qānāx; abl. qānāxdān|
136|3.3.5.1|t16|dup:3.1.14:2||||| abl. qānāxdān|
136|3.3.5.1|t17|ingest|sosku|why?|pron interr||why? sosku|homonymous with the dative/genitive of so 'what?'
136|3.3.5.1|t18|ingest|sodžiras|how?|pron interr manner||how? sodžiras, sori, sojrās|
136|3.3.5.1|t19|ingest|sori|how?|pron interr manner||how? sodžiras, sori, sojrās|homonymous with the affirmative particle sori 'yes'
136|3.3.5.1|t20|ingest|sojrās|how?|pron interr manner||how? sodžiras, sori, sojrās|
136|3.3.5.1|t21|ingest|so kiros|at what time?|pron interr temporal multiword-expression||at what time? so kiros, qojna sāhati, qojna vaxti|
136|3.3.5.1|t22|ingest|qojna sāhati|at what time?|pron interr temporal multiword-expression||at what time? so kiros, qojna sāhati, qojna vaxti|
136|3.3.5.1|t23|ingest|qojna vaxti|at what time?|pron interr temporal multiword-expression||at what time? so kiros, qojna sāhati, qojna vaxti|
136|3.3.5.1|t24|ingest|qojna sāhatdan|at what time?|pron interr temporal multiword-expression abl||abl. qojna sāhatdan, qojna vaxtistar|
136|3.3.5.1|t25|ingest|qojna vaxtistar|at what time?|pron interr temporal multiword-expression abl||abl. qojna sāhatdan, qojna vaxtistar|
136|3.3.5.1|t26|ingest|qozom|how many?; how much?|pron interr quantifier||how many? how much? qozom|
136|3.3.5.1|t27|ingest|qozom-un|how many?; how much?|pron interr quantifier pl||pl. qozom-un|
136|3.3.5.1|t28|ingest|qozomundār|how many?; how much?|pron interr quantifier abl instr||abl./ins. qozomundār|
136|3.3.6.1|t01|ingest|kon/qon|somebody|pron indef||somebody kon/qon, jek nāmāti|the same forms are listed as interrogatives at 3.3.5.1
136|3.3.6.1|t02|ingest|jek nāmāti|somebody|pron indef multiword-expression||somebody kon/qon, jek nāmāti|
136|3.3.6.1|t03|ingest|so|something|pron indef||something so, jek tipta|the same form is listed as an interrogative at 3.3.5.1
136|3.3.6.1|t04|ingest|jek tipta|something|pron indef multiword-expression loanword|tipta < Greek τίποτα.|something so, jek tipta|
136|3.3.6.1|t05|ingest|har kon/har qon|whoever|pron indef multiword-expression||whoever har kon/qon|
136|3.3.6.1|t06|ingest|har kos/har qos|whoever|pron indef multiword-expression obl||obl. har kos/qos|
136|3.3.6.1|t07|ingest|har so|whatever|pron indef multiword-expression||whatever har so|
136|3.3.6.1|t08|ingest|heč kon/heč qon|nobody|pron indef multiword-expression||nobody heč kon/qon|
136|3.3.6.1|t09|ingest|heč kos/heč qos|nobody|pron indef multiword-expression obl||obl. heč kos/qos|
136|3.3.6.1|t10|ingest|heč so|nothing|pron indef multiword-expression||nothing heč so, heš, hešta|
136|3.3.6.1|t11|dup:2.3.22:4||||| nothing heč so, heš, hešta|
136|3.3.6.1|t12|ingest|hešta|nothing|pron indef||nothing heč so, heš, hešta|
136|3.3.6.1|t13|ingest|āvur|another; another one|pron indef||another (one) (jek nāmāti, jek tipta) āvur|the source prints the optional heads jek nāmāti and jek tipta before it
136|3.3.6.1|t14|ingest|har duj|both|pron indef multiword-expression||both har duj|
136|3.3.6.1|t15|ingest|sir fen|everybody; everything|pron indef multiword-expression||everybody sir fen; everything sir fen, sir tipta|
136|3.3.6.1|t16|dup:5.3:10||||| everything sir fen, sir tipta|
136|3.3.6.1|t17|ingest|āpus|each one|pron indef||each one āpus|
136|3.3.6.1|t18|ingest|āvrek|each other|pron reciprocal||each other āvrek, jek āvrus; obl. āvrekis|
136|3.3.6.1|t19|ingest|jek āvrus|each other|pron reciprocal multiword-expression||each other āvrek, jek āvrus; obl. āvrekis|
136|3.3.6.1|t20|ingest|āvrekis|each other|pron reciprocal obl||each other āvrek, jek āvrus; obl. āvrekis|
137|3.3.7.1|t01|ingest|hāmi|all|quantifier||all hāmi, sir|
137|3.3.7.1|t02|ingest|sir|all|quantifier||all hāmi, sir|
137|3.3.7.1|t03|dup:2.3.22:2||||| every har|
137|3.3.7.1|t04|ingest|heč/heš|any; not any|quantifier||(not) any heč/heš|heš is also listed as the indefinite pronoun 'nothing' at 2.3.22 and 3.3.6.1
137|3.3.7.1|t05|ingest|but|many; much; several|quantifier||many, much, several but|the adverb list at 3.20.1.5 gives the sense 'very'
137|3.3.7.1|t06|dup:3.20.1.5:4||||| few, a few, little, a little ( je) xajri|the source prints the fuller gloss 'few, a few, little, a little' and the optional head je
137|3.5.1|t01|ingest|sefr|zero|num||0 sefr|
137|3.5.1|t02|dup:2.3.5:3|||||1 jek(h)|
137|3.5.1|t03|ingest|duj|two|num||2 duj|
137|3.5.1|t04|dup:2.4.1:8|||||3 terin|
137|3.5.1|t05|ingest|ištār|four|num||4 ištār|
137|3.5.1|t06|ingest|pāndž|five|num||5 pāndž|
137|3.5.1|t07|ingest|šov|six|num||6 šov|
137|3.5.1|t08|dup:2.3.15:1|||||7 eftā|
137|3.5.1|t09|dup:2.3.15:2|||||8 oxto|
137|3.5.1|t10|dup:2.4.2:2|||||9 enna|
137|3.5.1|t11|ingest|deš|ten|num||10 deš|
137|3.5.1|t12|ingest|deš-jek|eleven|num||11 deš-jek|
137|3.5.1|t13|ingest|deš-duj|twelve|num||12 deš-duj|
137|3.5.1|t14|ingest|deš-terin|thirteen|num||13 deš-terin|
137|3.5.1|t15|ingest|deš-ištār|fourteen|num||14 deš-ištār|
137|3.5.1|t16|ingest|deš-pāndž|fifteen|num||15 deš-pāndž|
137|3.5.1|t17|ingest|deš-šov|sixteen|num||16 deš-šov|
137|3.5.1|t18|ingest|deš-eftā|seventeen|num||17 deš-eftā|
137|3.5.1|t19|ingest|deš-oxto|eighteen|num||18 deš-oxto|
137|3.5.1|t20|ingest|deš-enna|nineteen|num||19 deš-enna|
137|3.5.1|t21|ingest|jokus|twenty|num||20 jokus|
137|3.5.1|t22|dup:5.3:11|||||30 tirāndā|
137|3.5.1|t23|dup:5.3:8|||||40 sārāndā|
137|3.5.1|t24|ingest|pejindā|fifty|num||50 pejindā|
137|3.5.1|t25|ingest|pejindā-deš|sixty|num||60 pejindā-deš|
137|3.5.1|t26|ingest|pejindā-jokus|seventy|num||70 pejindā-jokus|
137|3.5.1|t27|ingest|pejindā-tirāndā|eighty|num||80 pejindā-tirāndā|
137|3.5.1|t28|ingest|pejindā-sārāndā|ninety|num||90 pejindā-sārāndā|
137|3.5.1|t29|ingest|šel|one hundred|num||100 šel|
138|3.5.1|t30|ingest|duj-šel|two hundred|num||200 duj-šel|
138|3.5.1|t31|ingest|sila|one thousand|num||1,000 sila|
138|3.5.1|t32|ingest|deš-sila|ten thousand|num||10,000 deš-sila|
138|3.5.1|t33|ingest|šel-sila|one hundred thousand|num||100,000 šel-sila|
138|3.5.1|t34|ingest|jek milijāna/milijāna|one million|num multiword-expression||1,000,000 ( jek) milijāna|
"""


# --------------------------------------------------------------------------------------
# Record construction
# --------------------------------------------------------------------------------------

def _parse_block(block, width):
    rows = []
    for line in block.strip("\n").split("\n"):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) != width:
            raise SystemExit(f"expected {width} fields, found {len(parts)}: {line!r}")
        rows.append([part.strip() for part in parts])
    return rows


def parse_decisions():
    decisions = {}
    order = []
    for section, span, status, forms, gloss, tags, etymology, note in _parse_block(DECISIONS, 8):
        key = (section, span)
        if key in decisions:
            raise SystemExit(f"duplicate decision for {key}")
        decisions[key] = {
            "section": section, "span": span, "status": status, "forms": forms,
            "gloss": gloss, "tags": tags, "etymology": etymology, "note": note,
            "printed_line": "",
        }
        order.append(key)
    return decisions, order


def parse_extra():
    extra = {}
    order = []
    for page, section, item, status, forms, gloss, tags, etymology, printed, note in _parse_block(EXTRA, 10):
        key = (section, item)
        if key in extra:
            raise SystemExit(f"duplicate extra record for {key}")
        extra[key] = {
            "section": section, "span": item, "status": status, "forms": forms,
            "gloss": gloss, "tags": tags, "etymology": etymology, "note": note,
            "printed_page": int(page), "printed_line": printed,
        }
        order.append(key)
    return extra, order


def citation(printed_page, section):
    return f"{SOURCE_ID}[p. {printed_page}, section {section}]"


def entry_key(printed_page, section, span):
    return f"{SOURCE_ID}:p{printed_page}:s{section}:i{span}"


def build_records(spans):
    """Return (records, audit) — installed rows in printed order plus one audit row per unit."""
    decisions, _ = parse_decisions()
    extra, extra_order = parse_extra()
    span_index = {(s["section"], str(s["span_index"])): s for s in spans}

    missing = [k for k in span_index if k not in decisions]
    if missing:
        raise SystemExit(f"{len(missing)} gloss spans have no curation entry: {sorted(missing)[:10]}")
    stray = [k for k in decisions if "+" not in k[1] and k not in span_index]
    if stray:
        raise SystemExit(f"curation entries without a gloss span: {sorted(stray)}")

    units = []
    for key in sorted(decisions, key=lambda k: _sort_key(k, span_index)):
        decision = dict(decisions[key])
        parent = (key[0], key[1].split("+")[0])
        source_span = span_index[parent]
        decision["pdf_page"] = source_span["pdf_page"]
        decision["printed_page"] = source_span["printed_page"]
        decision["raw_form"] = source_span["raw_form"] if "+" not in key[1] else ""
        decision["raw_gloss"] = source_span["raw_gloss"] if "+" not in key[1] else ""
        decision["unit"] = "span" if "+" not in key[1] else "span-attached"
        units.append(decision)
    for key in extra_order:
        decision = dict(extra[key])
        decision["pdf_page"] = decision["printed_page"] - FIRST_PRINTED_PAGE + 1
        decision["raw_form"] = decision["printed_line"]
        decision["raw_gloss"] = decision["gloss"]
        decision["unit"] = "list"
        units.append(decision)

    records = {}
    emitted_by_unit = {}
    for unit in units:
        if unit["status"] == "ingest":
            key = entry_key(unit["printed_page"], unit["section"], unit["span"])
            emitted_by_unit[(unit["section"], unit["span"])] = _emit(records, unit, key)
        elif not unit["status"].startswith("dup:") and unit["status"] not in EXCLUSION_REASONS:
            raise SystemExit(f"unknown status {unit['status']!r} at {unit['section']}:{unit['span']}")
    # Repeated mentions are folded in only after every primary record exists, because the
    # article often glosses a word first and gives its etymology several sections later.
    for unit in units:
        if unit["status"].startswith("dup:"):
            _, target_section, target_span = unit["status"].split(":")
            emitted_by_unit[(unit["section"], unit["span"])] = [
                _apply_duplicate(records, unit, target_section, target_span)
            ]
    audit = [_audit_row(unit, emitted_by_unit.get((unit["section"], unit["span"]), [])) for unit in units]
    return list(records.values()), audit


def _sort_key(key, span_index):
    section, span = key
    base, _, attached = span.partition("+")
    entry = span_index[(section, base)]
    return (entry["pdf_page"], entry["printed_page"], int(base), int(attached or 0), section)


def _emit(records, unit, key):
    forms = [form for form in unit["forms"].split("/") if form]
    if not forms:
        raise SystemExit(f"ingest record without a form at {unit['section']}:{unit['span']}")
    if not unit["gloss"]:
        raise SystemExit(f"ingest record without a gloss at {unit['section']}:{unit['span']}")
    emitted = []
    for index, form in enumerate(forms):
        record_key = key if index == 0 else f"{key}:variant:{index + 1}"
        tags = unit["tags"].split()
        if index:
            tags = tags + ["alternate"]
        record = {
            "Language_ID": LANGUAGE_ID,
            "Parameter_ID": "",
            "Form": unicodedata.normalize("NFC", form),
            "Gloss": unit["gloss"],
            "Native": "",
            "Phonemic": "",
            "Notes": "",
            "Source": citation(unit["printed_page"], unit["section"]),
            "Cognateset": "",
            "Etymology": unit["etymology"],
            "Entry_Key": record_key,
            "Variant_Of_Key": key if index else "",
            "Borrowed_From_Key": "",
            "Derivation_Parent_Keys": "",
            "Tags": " ".join(tags + [DIALECT_TAG]),
        }
        records[record_key] = record
        emitted.append(record_key)
    return emitted


def _apply_duplicate(records, unit, target_section, target_span):
    target = None
    for record_key, record in records.items():
        if record_key.endswith(f":s{target_section}:i{target_span}"):
            target = record
            break
    if target is None:
        raise SystemExit(
            f"duplicate at {unit['section']}:{unit['span']} points at an uninstalled record "
            f"{target_section}:{target_span}"
        )
    extra_citation = citation(unit["printed_page"], unit["section"])
    # CLDF citations are joined with a bare semicolon; a separator space would leak into
    # every consumer that splits row["Source"] on ";".
    if extra_citation not in target["Source"].split(";"):
        target["Source"] = f"{target['Source']};{extra_citation}"
    if unit["etymology"] and unit["etymology"] not in target["Etymology"]:
        target["Etymology"] = " ".join(filter(None, [target["Etymology"], unit["etymology"]]))
    return target["Entry_Key"]


def _audit_row(unit, emitted):
    status = unit["status"]
    if status == "ingest":
        reason = "Zargari lexical item glossed by the source"
    elif status.startswith("dup:"):
        reason = f"repeated mention; citation folded into the record installed at {status[4:]}"
    else:
        reason = EXCLUSION_REASONS[status]
    payload = "|".join([
        unit["section"], unit["span"], unit["raw_form"], unit["raw_gloss"], status,
        unit["forms"], unit["gloss"], unit["tags"], unit["etymology"],
    ])
    return {
        "Snapshot_Date": SNAPSHOT_DATE,
        "Collation_Date": COLLATION_DATE,
        "Unit_ID": f"{unit['section']}:{unit['span']}",
        "PDF_Page": unit["pdf_page"],
        "Printed_Page": unit["printed_page"],
        "Section": unit["section"],
        "Span_Index": unit["span"],
        "Raw_Form": unit["raw_form"],
        "Raw_Gloss": unit["raw_gloss"],
        "Status": status.split(":")[0],
        "Reason": reason,
        "Final_Forms": unit["forms"],
        "Final_Gloss": unit["gloss"],
        "Final_Tags": unit["tags"],
        "Etymology": unit["etymology"],
        "Emitted_Keys": " ".join(emitted),
        "Review": unit["note"],
        "Material_Error": "no",
        "Source": citation(unit["printed_page"], unit["section"]),
        "Record_SHA256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


# --------------------------------------------------------------------------------------
# Validation and output
# --------------------------------------------------------------------------------------

def canonical_tags():
    import sys

    sys.path.insert(0, str(ROOT))
    from tags import GENDER_TAGS, GRAMMATICAL_TAGS

    return set(GENDER_TAGS) | set(GRAMMATICAL_TAGS)


def validate(records, audit):
    problems = []
    known = canonical_tags()
    keys = Counter(record["Entry_Key"] for record in records)
    problems += [f"duplicate Entry_Key {key}" for key, count in keys.items() if count > 1]
    for record in records:
        if not record["Form"] or record["Form"] != record["Form"].strip():
            problems.append(f"bad Form {record['Form']!r} in {record['Entry_Key']}")
        if "�" in record["Form"] or "�" in record["Gloss"]:
            problems.append(f"replacement character in {record['Entry_Key']}")
        if record["Form"] != unicodedata.normalize("NFC", record["Form"]):
            problems.append(f"non-NFC Form in {record['Entry_Key']}")
        tags = record["Tags"].split()
        if DIALECT_TAG not in tags:
            problems.append(f"missing dialect tag in {record['Entry_Key']}")
        unknown = [tag for tag in tags if tag != DIALECT_TAG and tag not in known]
        if unknown:
            problems.append(f"non-canonical tags {unknown} in {record['Entry_Key']}")
        if record["Variant_Of_Key"] and record["Variant_Of_Key"] not in keys:
            problems.append(f"dangling Variant_Of_Key in {record['Entry_Key']}")
    pages = {row["Printed_Page"] for row in audit}
    if not pages <= set(range(FIRST_PRINTED_PAGE, LAST_PRINTED_PAGE + 1)):
        problems.append(f"printed pages outside {FIRST_PRINTED_PAGE}-{LAST_PRINTED_PAGE}")
    return problems


def summarize(records, audit):
    statuses = Counter(row["Status"] for row in audit)
    return {
        "audit_rows": len(audit),
        "gloss_spans": sum(1 for row in audit if not str(row["Span_Index"]).startswith("t")
                           and "+" not in str(row["Span_Index"])),
        "installed_rows": len(records),
        "statuses": dict(sorted(statuses.items())),
        "variant_rows": sum(1 for record in records if record["Variant_Of_Key"]),
        "rows_with_etymology": sum(1 for record in records if record["Etymology"]),
        "loanword_rows": sum(1 for record in records if "loanword" in record["Tags"].split()),
    }


def write_outputs(records, audit, install):
    if not install:
        return
    FORM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with FORM_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for record in records:
            writer.writerow([record[field] for field in FORM_FIELDS])
    with AUDIT_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(audit)
    ingested = [row for row in audit if row["Status"] == "ingest"]
    step = max(1, len(ingested) // 25)
    sample = ingested[::step][:25]
    with SAMPLE_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        writer.writerows(sample)
    manifest = {
        "source_id": SOURCE_ID,
        "snapshot_date": SNAPSHOT_DATE,
        "bibliography": (
            "Rezai Baghbidi, Hassan. 2003. The Zargari language: An endangered European "
            "Romani in Iran. Romani Studies 5th ser. 13(2): 123–148."
        ),
        "acquisition": "Author's PDF of the published article, downloaded from ResearchGate on 2026-08-25",
        "pdf_sha256": PDF_SHA256,
        "pdf_pages": PDF_PAGES,
        "pdf_redistributed": False,
        "article_printed_pages": [FIRST_PRINTED_PAGE, LAST_PRINTED_PAGE],
        "extraction": {
            "method": "publisher text layer decoded by Type 1 /Differences glyph names",
            "why": (
                "the embedded ToUnicode map silently drops the first element of the T_h, f_i, "
                "f_l, f_f and f_f_i ligatures and mangles oldstyle figures and small capitals"
            ),
            "ocr": False,
            "checked_in_layer": "the DECISIONS and EXTRA tables in "
                                "data/other/forms/raw_data/rezai_baghbidi_zargari_2003.py",
        },
        "language_model": {
            "base_language": LANGUAGE_ID,
            "dialect_tag": DIALECT_TAG,
            "note": (
                "Zargari is a Balkan Romani variety spoken in Zargar village, Ābyek district, "
                "Qazvin Province, Iran; it is its own Glottolog language (zarg1238) and is "
                "modelled as a base language, with the village retained as a dialect tag"
            ),
        },
        "scope": {
            "included": (
                "every isolated Zargari word the article glosses, plus multi-word Zargari items "
                "printed inside its lexical lists"
            ),
            "excluded": (
                "clause and phrase examples, unglossed paradigm tables (case suffixes, personal "
                "endings, personal pronouns, demonstratives, possessives, reflexives, the copula, "
                "and the mediopassive and Turkish-loan conjugations), and Hindi, Sanskrit, "
                "Qorbati, Seliyeri, Persian, Arabic, Greek, Armenian and Early Romani comparanda"
            ),
            "etymology_policy": (
                "donor and comparative statements are kept as Etymology prose; the article prints "
                "no CDIAL, DEDR or other etymon identifiers, so every row stays unlinked"
            ),
        },
        "outputs": {
            "forms": str(FORM_OUTPUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUTPUT.relative_to(ROOT)),
            "sample": str(SAMPLE_OUTPUT.relative_to(ROOT)),
        },
    }
    manifest.update(summarize(records, audit))
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--install", action="store_true", help="write the canonical outputs")
    parser.add_argument("--skip-hash", action="store_true", help="accept a differently-hashed scan")
    parser.add_argument("--audit-sample", type=int, metavar="N",
                        help="print N seeded raw-vs-output audit units and exit")
    parser.add_argument("--seed", type=int, default=20260825, help="seed for --audit-sample")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(
            f"the article scan is required but missing: {args.pdf}\n"
            "It is not redistributed; place the published PDF there and rerun."
        )
    digest = hashlib.sha256(args.pdf.read_bytes()).hexdigest()
    if digest != PDF_SHA256 and not args.skip_hash:
        raise SystemExit(f"unexpected PDF SHA-256 {digest}, expected {PDF_SHA256}")

    spans = extract_spans(args.pdf)
    records, audit = build_records(spans)
    problems = validate(records, audit)
    for problem in problems:
        print("PROBLEM:", problem)
    if args.audit_sample:
        import random

        rng = random.Random(args.seed)
        for row in sorted(rng.sample(audit, min(args.audit_sample, len(audit))),
                          key=lambda row: (row["Printed_Page"], row["Section"], row["Span_Index"])):
            print(f"p{row['Printed_Page']} section {row['Section']} unit {row['Span_Index']} "
                  f"[{row['Status']}]")
            print(f"   raw   : {row['Raw_Form']!r} -> {row['Raw_Gloss']!r}")
            print(f"   parsed: {row['Final_Forms']!r} | {row['Final_Gloss']!r} | "
                  f"{row['Final_Tags']!r}")
            if row["Review"]:
                print(f"   note  : {row['Review']}")
        return
    summary = summarize(records, audit)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if problems:
        raise SystemExit(f"{len(problems)} validation problems")
    write_outputs(records, audit, args.install)
    if args.install:
        print(f"wrote {FORM_OUTPUT} and {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()

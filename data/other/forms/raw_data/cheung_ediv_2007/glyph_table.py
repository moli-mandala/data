#!/usr/bin/env python3
"""Build ``glyphs.csv``: the audited glyph-name -> Unicode table for Cheung (2007).

Run from the git root (needs the PDF): ``.venv/bin/python -I
data/other/forms/raw_data/cheung_ediv_2007/glyph_table.py``.

Evidence used, in order of preference (recorded per row in ``Method``):

``agl``      the glyph name is a standard Adobe Glyph List name (``scaron``, ``imacron``).
``name``     a descriptive Dutch glyph name, decoded compositionally
             (``onderstip`` dot below, ``keepje`` caron, ``onderboogje`` inverted breve
             below, ``ring`` ring below on consonants, ``golfje`` tilde, ``stijgertje``
             acute, ``daler`` grave, ``tweedaler`` double grave, ``sleufje`` breve,
             ``streepje`` macron, ``haakje`` ogonek, ``trema`` diaeresis, ``stip``/``overstip``
             dot below/above, ``dakje`` circumflex, ``sjwa`` schwa), each confirmed against the
             rasterised outline.
``outline``  numeric/PUA names (``tt260``, ``gg6``, ``uniF16B``) identified from the
             embedded CFF outline (nearest named glyph in the same PDF plus a text-art
             review of the outline) and checked against printed context
             (``Gāθās``; ``(+ *niž-)``; Sogdian ``prβ’γtch``; Lithuanian ``drį̃sti``).
``mark``     a zero/near-zero-advance glyph drawn entirely left of its origin, i.e. a
             non-spacing diacritic over/under the *preceding* letter; decoded to the
             Unicode combining character (Unicode order = postfix, as printed). The
             ``uniF3xx`` names of the Star fonts are TITUS-style PUA copies of U+03xx.

Every non-ASCII glyph used outside the Greek/Arabic-script fonts must appear here;
``decode.py`` refuses unknown names. ``Postfix_Mark`` comes from outline geometry and is
cross-checked against the decision (a combining output iff the outline lies left of 0).
"""

from __future__ import annotations

import csv
import io
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pdf_layer  # noqa: E402

ROOT = HERE.parents[4]  # git root (data/)
PDF = Path("/Users/aryamanarora/Documents/Linguistics/Indo-European/Iranic/Cheung Iranian.pdf")
CACHE = ROOT.parent / "tmp/iranic/cheung/extract.json.gz"
OUT = HERE / "glyphs.csv"

SKIP_FONTS = ("PersianNaskhStar",)  # Arabic-script words: placeholder, see decode.py

A, G, M, B, DOT, RING, DIA = "́", "̀", "̄", "̆", "̇", "̊", "̈"
CAR, DGR, IBR, DB, RB, OG = "̌", "̏", "̑", "̣", "̥", "̨"
BRB, IBRB, TB, MB, TIL = "̮", "̯", "̰", "̱", "̃"

# glyph name -> (output, method, evidence); applies to every transliteration font
# unless a (font, glyph) override exists below.
NAMED = {
    # numeric / opaque names: outline + context
    "tt260": ("ā", "outline", "= amacron outline (IoU .92); (+ *ā-), Gāθās, Skt. bhā"),
    "tt259": ("Ā", "outline", "identical to Amacron outline (IoU 1.00)"),
    "tt302": ("ā" + B, "outline", "a + macron + breve; Sh. wā̆rv-, /ā̆bām/"),
    "tt330": ("ā" + A, "outline", "a + macron + acute; Skt. vā́ja-, kā́yamāna-"),
    "tt328": ("ā" + G, "outline", "a + macron + grave"),
    "tt276": ("ą", "outline", "a + ogonek; Av. hąm, dąn, bąn-"),
    "tt2232": ("ā" + RING, "outline", "a + macron + ring above; Av. auuā̊ṇtəm"),
    "tt928": ("ž", "outline", "identical to zcaron outline (IoU 1.00); (+ *niž-)"),
    "tt927": ("Ž", "outline", "Z + caron outline"),
    "tt1264": ("β", "outline", "beta; Sogd. prβ’γtch, MSogd. ptwyβ"),
    "gg6": ("γ", "outline", "gamma; Pash. zγam-, Sh. γāv-/γēvd"),
    "ee2": ("ɛ", "outline", "open e; Meim. bɛrma, Sariq. γɛyv-"),
    "oo4": ("ɔ", "outline", "open o (reversed c); Jow. bɔrma"),
    "ss2": ("ʃ", "outline", "esh; Jow. bam-tʃi/a-tʃin-"),
    "tt531": ("ī" + A, "outline", "dotless i + macron + acute; bī́-, iznī́"),
    "tt547": ("ī" + B, "outline", "i + macron + breve; Sh. wī̆rv-"),
    "tt894": ("ǟ", "outline", "a + diaeresis + macron; cǟp-"),
    "tt472": ("ǖ", "outline", "u + diaeresis + macron"),
    "tt8": ("ä" + A, "outline", "a + diaeresis + acute"),
    "tt1204": ("œ" + A, "outline", "oe ligature + acute (IoU .84 to oe)"),
    "tt769": ("ŋᵛ", "outline", "eng with superscript v; Av. -aŋᵛha (labialised ŋ)"),
    "tt311": ("ẽ", "outline", "e + tilde"),
    "tt1017": ("x" + DIA, "outline", "x + diaeresis"),
    "tt537": ("i" + B + A, "outline", "dotless i + breve + acute"),
    "tt1210": ("ǣ", "outline", "ae + macron"),
    "tt1200": ("ǿ", "outline", "o slash + acute"),
    "tt545": ("į" + TIL, "outline", "i + ogonek + tilde; Lith. drį̃sti, lį̃sti"),
    "tt470": ("ǘ", "outline", "u + diaeresis + acute"),
    "tt264": ("ȃ", "outline", "a + inverted breve; Skt. svȃr"),
    "tt1208": ("ǽ", "outline", "ae + acute"),
    "tt437": ("ẹ" + A, "outline", "e + dot below + acute"),
    "tt932": ("ǧ", "outline", "g + caron"),
    "tt266": ("ă", "outline", "a + breve"),
    "tt438": ("ę" + TIL, "outline", "e + ogonek + tilde"),
    # descriptive Dutch names
    "sjwa": ("ə", "name", "sjwa = schwa; YAv. dərəta-"),
    "tonderstip": ("ṭ", "name", "t onder stip (dot below)"),
    "monderstip": ("ṃ", "name", "m + dot below"),
    "nonderstip": ("ṇ", "name", "n + dot below"),
    "sonderstip": ("ṣ", "name", "s + dot below"),
    "ronderstip": ("ṛ", "name", "r + dot below"),
    "dstip": ("ḍ", "name", "d + stip; outline: dot below"),
    "jkeepje": ("ǰ", "name", "j + keepje (caron)"),
    "Jkeepje": ("J" + CAR, "name", "J + caron"),
    "gkeepje": ("ǧ", "name", "g + caron"),
    "xkeepje": ("x" + CAR, "name", "x + caron"),
    "ekeepje": ("ě", "name", "e + caron"),
    "skeepstipje": ("ṣ" + CAR, "name", "s + caron + dot below (Av. ṣ̌)"),
    "ckeepstipje": ("c" + DB + CAR, "name", "c + caron + dot below"),
    "skeepstijgertje": ("š" + A, "name", "s + caron + acute (Av. š́)"),
    "uonderboogje": ("u" + IBRB, "name", "u + inverted breve below (u̯)"),
    "Uonderboogje": ("U" + IBRB, "name", "U + inverted breve below"),
    "ionderboogje": ("i" + IBRB, "name", "i + inverted breve below (i̯)"),
    "Ionderboogje": ("I" + IBRB, "name", "I + inverted breve below"),
    "dboogje": ("d" + IBRB, "name", "d + inverted breve below (outline)"),
    "eboogje": ("ȇ", "name", "e + inverted breve above (outline)"),
    "rring": ("r" + RB, "name", "r + ring below (r̥)"),
    "Rring": ("R" + RB, "name", "R + ring below"),
    "Hring": ("H" + RB, "name", "H + ring below (outline)"),
    "nring": ("n" + RB, "name", "n + ring below"),
    "rringstijger": ("r" + RB + A, "name", "r + ring below + acute"),
    "rstreepring": ("r" + RB + M, "name", "r + ring below + macron (outline)"),
    "tondergolfje": ("t" + TB, "name", "t + tilde below (Av. t̰)"),
    "lmidgolfje": ("ɫ", "name", "l + middle tilde"),
    "lgolfje": ("l" + TIL, "name", "l + tilde above (outline)"),
    "mgolfje": ("m" + TIL, "name", "m + tilde"),
    "rgolfje": ("r" + TIL, "name", "r + tilde (outline)"),
    "ustreepgolfje": ("ū" + TIL, "name", "u + macron + tilde"),
    "kstijgertje": ("ḱ", "name", "k + acute"),
    "gstijgertje": ("ǵ", "name", "g + acute"),
    "jstijger": ("j" + A, "name", "j + acute"),
    "xstijger": ("x" + A, "name", "x + acute"),
    "ngstijger": ("ŋ" + A, "name", "eng + acute (outline)"),
    "honderstreepje": ("ẖ", "name", "h + macron below"),
    "tonderstreepje": ("ṯ", "name", "t + macron below"),
    "ronderstreepje": ("ṟ", "name", "r + macron below (outline)"),
    "rstreepje": ("r" + M, "name", "r + macron (outline)"),
    "pstreepje": ("p" + M, "name", "p + macron (outline)"),
    "estreepsleufje": ("ē" + B, "name", "e + macron + breve"),
    "ustreepsleufje": ("ū" + B, "name", "u + macron + breve"),
    "estreepstijger": ("ḗ", "name", "e + macron + acute"),
    "ostreepstijgertje": ("ṓ", "name", "o + macron + acute"),
    "ustreepstijgertje": ("ū" + A, "name", "u + macron + acute"),
    "estreepdaler": ("ḕ", "name", "e + macron + grave"),
    "ostreepstipstijger": ("ọ" + M + A, "name", "o + dot below + macron + acute (outline)"),
    "astreephaakje": ("ą" + M, "name", "a + ogonek + macron"),
    "isleufje": ("ĭ", "name", "i + breve"),
    "esleufje": ("ĕ", "name", "e + breve (outline)"),
    "usleufje": ("ŭ", "name", "u + breve"),
    "xsleufje": ("x" + B, "name", "x + breve (outline: cup above)"),
    "rsleufje": ("r" + B, "name", "r + breve (outline)"),
    "ysleufje": ("y" + B, "name", "y + breve (outline)"),
    "hsleufje": ("ḫ", "name", "h + breve below (outline)"),
    "usleufstijgertje": ("ŭ" + A, "name", "u + breve + acute"),
    "esleufstijgertje": ("ĕ" + A, "name", "e + breve + acute"),
    "ohaakje": ("ǫ", "name", "o + ogonek"),
    "ehaakje": ("ę", "name", "e + ogonek"),
    "otremastreepje": ("ȫ", "name", "o + diaeresis + macron"),
    "otremastijgertje": ("ö" + A, "name", "o + diaeresis + acute"),
    "estipstijgertje": ("ė" + A, "name", "e + dot above + acute (Lith. ė́)"),
    "eoverstip": ("ė", "name", "e + dot above"),
    "roverstip": ("ṙ", "name", "r + dot above"),
    "etweedaler": ("ȅ", "name", "e + double grave"),
    "itweedaler": ("ȉ", "name", "i + double grave"),
    "utweedaler": ("ȕ", "name", "u + double grave"),
    "atweedaler": ("ȁ", "name", "a + double grave"),
    "rtweedaler": ("ȑ", "name", "r + double grave"),
    "rdaler": ("r" + G, "name", "r + grave"),
    "goortje": ("ģ", "name", "g + comma above; Latv. ģìbstu"),
    "ldakje": ("l̂", "name", "l + circumflex"),
    "yogh": ("ʒ", "outline", "flat-topped ezh (dz), not yogh ȝ"),
    "stijgertje": (A, "mark", "acute over preceding letter"),
    "stijgertjehoog": (A, "mark", "acute over preceding capital/ascender"),
    "f008": (A, "mark", "acute over preceding letter (Wakhi stress)"),
    "f029": (DB, "mark", "dot below preceding letter"),
    "f02c": (DB, "mark", "dot below preceding letter"),
    "uniF029": (DB, "mark", "dot below preceding letter"),
    "ordmasculine": ("°", "outline", "raised circle = abbreviation/truncation mark (OP °bigna-)"),
    # Star / TITUS-style PUA names
    "uniF16B": ("ū", "outline", "u + macron (TITUS PUA for U+016B)"),
    "uniF131": ("ı", "outline", "dotless i (PUA U+0131)"),
    "uniF259": ("ə" + M, "outline", "schwa + macron"),
    "uniF254": ("ɔ" + M, "outline", "open o + macron"),
    "uniF267": ("ī", "outline", "dotless i + macron (IoU 1.0 to Star imacron)"),
    "uniF4D3": ("ǟ", "outline", "a + diaeresis + macron"),
    "uniF19A": ("š", "outline", "s + caron (followed by U+0301 for Av. š́)"),
    "uniF1E7": ("ç", "outline", "c + cedilla (OP ç)"),
    "uniF115": ("§", "outline", "section sign"),
    "uniF17E": ("~", "outline", "swung dash (repetition sign)"),
    "uniE006": ("ƕ", "outline", "Gothic hwair; Goth. leiƕan"),
    "uniF3D1": ("ϑ", "outline", "theta symbol"),
    "theta1": ("ϑ", "agl", "theta symbol"),
    "afii10105": ("ž", "outline", "z + caron (misnamed afii glyph)"),
    "afii10055": (RING, "mark", "ring above preceding letter"),
    "vturn": ("ʌ", "agl", "turned v"),
    "wsuper": ("ʷ", "agl", "superscript w"),
    "alefpatah": ("ˋ", "outline", "spacing grave-shaped mark after Ormuri o (do ˋrnók); review"),
    "breveinvsubnosp": (IBRB, "mark", "inverted breve below"),
    "brevesubnosp": (BRB, "mark", "breve below"),
    "afii10093": ("ы", "outline", "Cyrillic yeru shape; Wakhi transcription ы"),
    "afii10090": ("ш", "outline", "Cyrillic sha shape; Wa. pətшn (review)"),
    ";#23#23": ("→", "outline", "rightwards arrow introducing the reference column"),
    ".notdef.g0002": ("ǰ" + DB, "outline", "j + caron + dot below (TITUS; prose only)"),
    "FL196": ("", "outline", "Greek-word adjunct glyph (dropped with Greek)"),
    "FL113": ("", "outline", "Greek-word adjunct glyph (dropped with Greek)"),
}
for code in range(0x300, 0x370):
    NAMED.setdefault(f"uniF{code:03X}", (chr(code), "mark", f"TITUS PUA copy of U+{code:04X}"))

FONT_NAMED = {
    ("TimesNewRomanStar,Italic", "asciitilde"): (TIL, "mark", "tilde above preceding letter"),
    ("TimesNewRomanStar", "underscore"): (MB, "mark", "macron below preceding letter"),
    ("TimesNewRomanStar", "section"): (OG, "mark", "ogonek under preceding letter (misnamed)"),
    ("TimesNewRomanStar", "asciitilde"): (TIL, "mark", "tilde above preceding letter"),
    ("TimesNewRomanStar,Italic", "underscore"): (MB, "mark", "macron below preceding letter"),
    ("TimesNewRomanStar,Italic", "periodcentered"): (DB, "mark", "dot below preceding letter"),
    ("TimesNewRomanStar", "periodcentered"): (DB, "mark", "dot below preceding letter"),
    ("TimesNewRomanStar,Italic", "mu"): ("ü", "outline", "u + diaeresis (misnamed); Festschrift für"),
    ("TimesNewRomanStar,Italic", "grave"): ("‘", "outline", "turned comma (ayin) in MP/Sogd. forms"),
    ("TimesNewRomanStar,Italic", "quotesingle"): ("’", "outline", "apostrophe (aleph) in MP forms"),
    ("TimesNewRomanStar", "uni0235"): ("ˮ", "outline", "two stacked apostrophes (CSogd.); review"),
    ("TITUSCyberbitBasic,Italic", ".notdef.g0003"): ("ẹ" + CAR, "outline", "e + dot below + caron (Yaghnobi)"),
    ("TITUSCyberbitBasic,Italic", ".notdef.g0004"): ("ẹ" + A, "outline", "e + dot below + acute (Yaghnobi)"),
    ("TITUSCyberbitBasic", ".notdef.g0003"): ("u" + IBRB, "outline", "u + inverted breve below"),
    ("TITUSCyberbitBasic", ".notdef.g0004"): ("i" + IBRB, "outline", "i + inverted breve below"),
}

# GreekMousaStar is an 8-bit "SGreek"-layout font: uniF0xx = the ASCII key position.
# Verified on outlines (b=β, d=δ, g=γ, q=θ, c=χ, x=ξ, j=ς, w=ω, y=ψ) and words
# (σῆμα, χορηγία, ϝεχέτω). Its letters also spell Sogdian/Choresmian β δ γ θ.
GREEK_KEYS = dict(zip("abcdefghijklmnopqrstuwxyz", "αβχδεφγηιςκλμνοπθρστυωξψζ"))
GREEK_KEYS.update(zip("ABCDEFGHIKLMNOPQRSTUWXYZ", "ΑΒΧΔΕΦΓΗΙΚΛΜΝΟΠΘΡΣΤΥΩΞΨΖ"))
PSILI, DASIA, PERI = "̓", "̔", "͂"
GREEK_OTHER = {
    "uniF020": (" ", "outline", "space"),
    "uniF028": ("(", "outline", "parenthesis"),
    "uniF029": (")", "outline", "parenthesis"),
    "uniF02C": (",", "outline", "comma"),
    "uniF027": (PSILI, "mark", "smooth breathing"),
    "uniF060": (DASIA, "mark", "rough breathing"),
    "uniF02F": (A, "mark", "acute"),
    "uniF05C": (G, "mark", "grave"),
    "uniF05E": (PERI, "mark", "circumflex (perispomeni)"),
    "uniF091": (PSILI + PERI, "mark", "smooth breathing + circumflex"),
    "uniF092": (DASIA + PERI, "mark", "rough breathing + circumflex"),
    "uniF093": (DASIA + A, "mark", "rough breathing + acute"),
    "uniF094": (PSILI + A, "mark", "smooth breathing + acute"),
    "uniF096": (PSILI + G, "mark", "breathing + grave (low-confidence)"),
    "uniF097": (DIA + A, "mark", "diaeresis + acute"),
    "uniF08B": ("ῖ", "outline", "iota + circumflex"),
    "uniF08C": ("ἶ", "outline", "iota + smooth breathing + circumflex"),
    "uniF08D": ("ἷ", "outline", "iota + rough breathing + circumflex"),
    "uniF0C1": ("ῆ", "outline", "eta + circumflex; σῆμα, γῆρας"),
    "uniF0E0": ("ῦ", "outline", "upsilon + circumflex; θαῦμα, μῦθος, κῦμα, πνεῦμα"),
    "uniF0E1": ("ὖ", "outline", "upsilon + smooth breathing + circumflex; αὖλαξ"),
    "uniF0DA": ("ύ", "outline", "upsilon + acute; νεύω, κύρειν"),
    "uniF0B7": ("ῥ", "outline", "rho + rough breathing"),
    "uniF09B": ("ἕ", "outline", "epsilon + rough breathing + acute; ἕννυμι"),
    "uniF0A9": ("ᾶ", "outline", "alpha + circumflex; Doric σᾶμα, ζᾶλος"),
    "uniF0A3": ("ά", "outline", "alpha + acute; σάρξ"),
    "uniF0EE": ("ῶ", "outline", "omega + circumflex; κῶμος, κτῶμαι"),
    "uniF023": ("ϝ", "outline", "digamma; ϝεχέτω"),
    "uniF056": ("η", "outline", "eta-shaped outline at key 'V' (low-confidence)"),
}


def greek(glyph: str):
    if glyph in GREEK_OTHER:
        return GREEK_OTHER[glyph]
    if glyph.startswith("uniF0") and chr(int(glyph[4:], 16)) in GREEK_KEYS:
        key = chr(int(glyph[4:], 16))
        return GREEK_KEYS[key], "outline", f"SGreek key position '{key}'"
    return None


FIELDS = ["Font", "Glyph", "Output", "Codepoints", "Method", "Evidence", "Postfix_Mark",
          "Bbox", "Count"]


def agl_char(name: str) -> str | None:
    from fontTools import agl
    char = agl.toUnicode(name)
    if len(char) == 1 and not (0xE000 <= ord(char) < 0xF900):
        return char
    return None


def font_geometry() -> dict[tuple[str, str], tuple[int, int, int, int] | None]:
    import pypdf
    from fontTools.cffLib import CFFFontSet
    from fontTools.pens.boundsPen import BoundsPen
    from fontTools.ttLib import TTFont

    out = {}
    seen = set()
    for page in pypdf.PdfReader(str(PDF)).pages:
        res = page.get("/Resources")
        fonts = res.get_object().get("/Font") if res else None
        if not fonts:
            continue
        for ref in fonts.get_object().values():
            font = ref.get_object()
            name = str(font["/BaseFont"]).split("+", 1)[-1]
            desc = font.get("/FontDescriptor")
            if name in seen or not desc or "/FontFile3" not in desc.get_object():
                continue
            seen.add(name)
            data = desc.get_object()["/FontFile3"].get_object().get_data()
            cff = CFFFontSet()
            cff.decompile(io.BytesIO(data), TTFont())
            top = cff.topDictIndex[0]
            # fontTools needs the charset before CharStrings for these bare CFF programs
            top.charset = top.converters["charset"].read(top, top.rawDict.get("charset", 0))
            for glyph in top.CharStrings.keys():
                pen = BoundsPen(top.CharStrings)
                charstring = top.CharStrings[glyph]
                charstring.draw(pen)
                bounds = tuple(int(v) for v in pen.bounds) if pen.bounds else None
                out[(name, glyph)] = (bounds, getattr(charstring, "width", None))
        if len(seen) >= 22:
            break
    return out


def main() -> None:
    pages = pdf_layer.extract(PDF, CACHE)
    token = re.compile("\x01([^:\x02]+):([^\x02]+)\x02")
    counts: Counter = Counter()
    for lines in pages:
        for line in lines:
            for span in line["spans"]:
                for m in token.finditer(span["t"]):
                    counts[(m.group(1), m.group(2))] += 1
    geometry = font_geometry()
    rows = []
    unresolved = []
    for (font, glyph), count in sorted(counts.items()):
        if font.startswith(SKIP_FONTS):
            continue
        bbox, advance = geometry.get((font, glyph), (None, None))
        # non-spacing: drawn left of the origin, or a (near-)zero advance straddling it
        postfix = bool(bbox and (bbox[2] <= 0 or (bbox[0] < 0 and (advance or 0) <= 62)))
        if font.startswith("GreekMousaStar"):
            decided = greek(glyph)
            if decided is None:
                unresolved.append((font, glyph, count))
                continue
            output, method, evidence = decided
        elif (font, glyph) in FONT_NAMED:
            output, method, evidence = FONT_NAMED[(font, glyph)]
        elif glyph in NAMED:
            output, method, evidence = NAMED[glyph]
        else:
            char = agl_char(glyph)
            if char is None:
                unresolved.append((font, glyph, count))
                continue
            if char.isascii() and not postfix:
                continue  # plain ASCII: decoded by name in decode.py, no table row needed
            output, method, evidence = char, "agl", "standard glyph name"
        output = unicodedata.normalize("NFC", output) if not postfix else output
        is_mark = bool(output) and all(unicodedata.combining(c) for c in output)
        if postfix != is_mark and output:
            unresolved.append((font, glyph, count, "mismatch", bbox, output))
        rows.append({
            "Font": font, "Glyph": glyph, "Output": output,
            "Codepoints": " ".join(f"U+{ord(c):04X}" for c in output),
            "Method": method, "Evidence": evidence, "Postfix_Mark": "yes" if postfix else "no",
            "Bbox": " ".join(map(str, bbox)) if bbox else "", "Count": count,
        })
    if unresolved:
        raise SystemExit(f"unresolved glyphs: {unresolved}")
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} glyph rows; methods: "
          f"{dict(Counter(r['Method'] for r in rows))}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""House-transcription policy for the sound profiles in ``conversion/*.txt``.

Every profile converts a source's transcription into the same house transcription that
``conversion/cdial.txt`` produces for Turner, so the profiles must agree on what the output
alphabet is.  The policy, distilled from ``cdial.txt`` / ``cdial-post.txt``:

* **Vowels follow Turner's phonemic system.**  The house short/long pairs are a/ā, i/ī, u/ū,
  so a source's IPA ``ʌ`` and (for Indo-Aryan and Dravidian sources) ``ə`` are ``a``, ``ɪ`` is
  ``i``, ``ʊ`` is ``u``, and the vowel the source writes as *long* — plain ``a``/``i``/``u`` in
  the SIL-style transcriptions that reserve ``ʌ``/``ɪ``/``ʊ`` for the short series, ``aː``
  or ``ɑ`` elsewhere — is ``ā``/``ī``/``ū``.  Which convention a source follows is a
  per-profile decision recorded in ``VOWELS`` below, derived from each source's inventory.
  ``ə`` stays ``ə`` where Turner himself writes it (Dardic, Nuristani, Kashmiri, Burushaski,
  Romani) and in non-Indo-Aryan families where it is a phoneme.  ``e ɛ o ɔ æ ɨ`` keep their
  quality (Turner uses all of them); a profile must not flatten ``ɛ`` to ``e`` or ``ɔ`` to ``o``.
* **Length is a macron** (``aː`` → ``ā``, ``eː`` → ``ē``), never a colon or a doubled vowel; a
  long consonant is written double (``kː`` → ``kk``).  A long lax vowel is the long tense
  series (``ʌː`` → ``ā``, ``ɪː`` → ``ī``, ``ʊː`` → ``ū``).
* **Consonants use the Indological letters**: ``c j ś ṣ ź ẓ ṭ ḍ ṇ ṛ ḷ ñ ŋ ʦ ʣ``, aspiration
  as ``ʰ``, the palatal glide as ``y``, ``v`` for ``w`` / ``ʋ``, plain ``g`` / ``r`` / ``h``.

``python profile_policy.py check`` lists every rule that breaks the policy;
``python profile_policy.py fix`` rewrites the profiles (adding ``Vː`` / ``Cː`` rules for every
length bigram that actually occurs in the sources routed through a profile).  Both read the
sources through ``source_meta`` to know which files use which profile.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
import unicodedata as ud
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILE_DIR = ROOT / "conversion"
REFERENCE = {"cdial", "cdial-post"}

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def nfc(s: str) -> str:
    return ud.normalize("NFC", s)


# ----------------------------------------------------------------------------- output alphabet

# Output-side substitutions: any of these in a rule's house output is a policy violation with
# an unambiguous house spelling.  Longest keys first so ``tʃ`` wins over ``ʃ``.
CONSONANTS = {
    "tʃ": "c", "dʒ": "j", "ʧ": "c", "ʤ": "j", "č": "c", "ǰ": "j",
    "ʃ": "ś", "š": "ś", "ɕ": "ś", "ʒ": "ź", "ž": "ź", "ʑ": "ź",
    "ʂ": "ṣ", "ʐ": "ẓ", "ʈ": "ṭ", "ɖ": "ḍ", "ɳ": "ṇ", "ɽ": "ṛ", "ɭ": "ḷ",
    "ɲ": "ñ", "ṅ": "ŋ", "ɡ": "g", "ɾ": "r", "ʋ": "v", "w": "v",
    "ts": "ʦ", "dz": "ʣ", "ʦ": "ʦ", "ʣ": "ʣ",
    "ʱ": "ʰ", "ɦ": "h", "ṃ": "ṁ", "ʲ": "ʸ", "ɟ": "j", "ǝ": "ə",
}
_SUBST = sorted(CONSONANTS.items(), key=lambda kv: -len(kv[0]))

# House vowels (with any combining marks they carry).
HOUSE_VOWELS = set("aāəʌæɪiīʊuūeēɛoōɔɨ")

# Per-profile vowel decisions (Turner-phonemic).  Defaults: ʌ → a, ɪ → i, ʊ → u, ə → a,
# a/i/u unchanged, length from ː.  A profile listed in LONG_A / LONG_I / LONG_U writes its
# plain a/i/u for the long vowel (it reserves ʌ or ə / ɪ / ʊ for the short one, and marks no ː);
# one in SCHWA keeps ə as the distinct vowel Turner writes ə.  Derived from each source's
# vowel inventory (counts of ʌ ə a aː ɑ ɪ i iː ʊ u uː) on 2026-09-19 and reviewed by hand.
LONG_A = {
    # SIL / survey IPA that writes the short vowel ʌ or ə and the long one a
    "chattisgarhi", "rajasthani", "kannauji", "toulmin", "halbi-woods", "magahi-survey",
    "nirmaan-mewari", "wadiyara", "bajjika", "census-danuwar", "selected-angika",
    "selected-majhi", "kochila-tharu", "kullui", "sdml", "ghatage", "ghatage-western",
    "sil-pahari-pothwari", "sil-western-tharu", "sil-northern-dhule-bhils", "sil-noira",
    "sil-haryanvi", "pahari", "sil-kurumba-2012", "sil-dhurwa-2021",
    # Census / LSI state-volume ASCII: plain a i u are the long vowels, A I U the short
    "census-ascii", "more-ascii",
    # Nepal surveys in the LinSuN convention (ʌ = अ, a = आ)
    "dewas-rai", "eastern-magar", "magar-2024", "maikoti-kham", "mewahang", "sampang",
    "thakali", "western-tamang", "yamphu", "majhi-bote", "north-gorkha",
}
LONG_I = {
    "chattisgarhi", "rajasthani", "kannauji", "nirmaan-mewari", "wadiyara", "sil-pahari-pothwari",
    "sil-western-tharu", "sil-bagheli", "sil-bareli-pauri", "sil-eastern-gujari", "sil-malvi",
    "sil-desia", "sil-bishnupriya", "sil-dogri", "sil-korwa-kodaku", "sil-bonda-didayi",
    "sil-bonda-further", "sil-gadaba", "sil-kurumba-2012", "sil-irula", "sil-dhurwa-2021",
    "pahari", "northern", "ssnp", "kurux-nepal", "census-ascii", "more-ascii",
    # Woods' Halbi: the Devanagari column shows his plain i/u/a are ī/ū/ā throughout
    "halbi-woods",
}
LONG_U = {
    "chattisgarhi", "rajasthani", "kannauji", "nirmaan-mewari", "wadiyara", "sil-pahari-pothwari",
    "sil-western-tharu", "sil-noira", "sil-bareli-pauri", "sil-eastern-gujari", "sil-malvi",
    "sil-nimadi", "sil-desia", "sil-korwa-kodaku", "sil-bonda-didayi",
    "sil-bonda-further", "northern", "ssnp", "halbi-woods", "census-ascii", "more-ascii",
}
# ə is a distinct vowel: Turner writes it for Dardic, Nuristani, Kashmiri, Burushaski and
# Romani; it is a phoneme in the Tibeto-Burman, Munda, Kusunda and Nihali sources.
SCHWA = {
    "dhakal-darai",  # Dhakal 2011 pp.43–49: ə/a quality contrast, no distinctive length.
    "kumari-gaddi-ipa",  # Kumari et al. 2026 p.12: /ə/ contrasts with /ɑ/ in Gaddi.
    "peterson-turi",  # 2024 pp.266–267: preserve uncertain schwa status in source IPA.
    "berger", "yoshioka", "buddruss-grangali", "buddruss-wama", "buddruss-waigali",
    "buddruss-shina", "degener-shina", "drasi", "liljegren", "liljegren-hindukush", "kalasha",
    "khowar", "kalkoti", "strand", "nured", "perder-dameli", "dameli-donors",
    "schmidt-kashmiri", "torwali-student", "ssnp", "northern", "weinreich-domaaki",
    "boretzky-romani", "zargari", "domari-aleppo", "house",
    "dewas-rai", "eastern-magar", "gurung", "humla", "magar-2024", "maikoti-kham", "majhi-bote",
    "mewahang", "mustang-loke", "naaba", "north-gorkha", "pyangaun-newar", "rabha", "sampang",
    "thakali", "western-tamang", "yamphu", "chhulung", "sil-adi", "sil-amri-karbi",
    "sil-bangladesh", "sil-meitei", "sil-lahul", "tagin-puroik", "kusunda-aaley-bodt",
    "kusunda-gipan", "kusunda-watters", "sil-kullu",
    "cfel-koda-api", "cfel-koda-print",  # Publisher IPA: Munda schwa retained as a distinct quality.
    "pinnow-munda", "pinnow-juang", "munda-proto-kherwarian", "zide-sora-juray",
    "bhattacharya-bonda", "bahl-korwa", "kharia-living", "sil-korku", "sil-ho", "sil-bhumij",
    "santali-cluster", "sil-korwa-kodaku", "sil-bonda-didayi", "sil-bonda-further", "nihali", "nihali-konow",
    "zoller-2023", "merriam-reconstruction", "muduga", "keed",
    # scholarly Dravidian transcriptions: Kodagu, Toda, Kota, Badaga ə is a phoneme
    "ia-dravidian-ipa", "census-dravidian", "dedr", "toda", "badaga-hockings", "kudiya",
}
# Scholarly Munda transcriptions where ʌ is a distinct phoneme (Zide's Sora), kept as is.
KEEP_WEDGE = {"zide-sora-juray", "pinnow-munda"}


def vowel_map(profile: str) -> dict[str, str]:
    """Short-vowel letter → house letter for one profile (marks are re-attached by the caller)."""
    m = {"ʌ": "a", "ɪ": "i", "ʊ": "u", "ə": "ə" if profile in SCHWA else "a",
         "ɨ": "ɨ",
         "a": "ā" if profile in LONG_A else "a",
         "i": "ī" if profile in LONG_I else "i",
         "u": "ū" if profile in LONG_U else "u"}
    if profile in KEEP_WEDGE:
        m["ʌ"] = "ʌ"
    return m
# Marks that keep their meaning in house transcription; a vowel carrying any other mark
# (diaeresis, circumflex, caron …) is source orthography and is left to the profile author.
HOUSE_MARKS = {"\u0304", "\u0303", "\u0301", "\u0300"}

# Aspirate letters: a rule output ``Ch`` where the source grapheme marks aspiration is ``Cʰ``.
ASPIRABLE = set("kgcjtdṭḍpbmnlrvṛḷʦʣ")
_ASPIRATE_OUT = re.compile(r"^([kgcjtdṭḍpbmnlrvṛḷʦʣ])h$")

# Rules that look like reinterpretations but are the source's own notation, reviewed when the
# profile was written: DEDR's ``è`` is Burrow–Emeneau's open e, Emeneau's Toda ``ü`` is the
# front rounded vowel that the Toda profile writes ``y``.
# Bailey Padari and LSI Simla Siraji retain literal historical Roman notation, not reconstructed IPA.
# Rangri original distinguishes literal v/w, including table-initial W; preserve both.
KEEP = {("grierson-malvi-rangri-1908", "w"), ("grierson-malvi-rangri-1908", "W"), ("grierson-simla-siraji-1916", "w"), ("bailey-padari-1908", "w"), ("dedr", "è"), ("dedr", "ḕ"), ("keed", "è"), ("toda", "üː"), ("toda", "ü"),
        # Bailey Bhalesi p75 sentence22 prints a marked c in kaṇčā, alongside
        # plain c elsewhere. Preserve the reviewed historical glyph without
        # assigning an unsupported IPA value or erasing its visible mark.
        ("bailey-bhalesi-1908", "č"),
        # Chhattisgarhi / Magahi survey ɔ and æ are Hindi au and ai (Turner writes ai/au)
        ("chattisgarhi", "ɔ"), ("chattisgarhi", "æ"), ("magahi-survey", "ɔ"), ("magahi-survey", "æ"),
        # the Chhattisgarhi and Malvi surveys use ɨ for a centralised short i
        ("chattisgarhi", "ɨ"), ("sil-malvi", "ɨ"),
        # Mathew & Chamberlain 2022 Bonda/Didayi survey (Gutob/Gorum response cells),
        # data/other/forms/raw_data/sil_gutob_gorum_2022/profile-review.json:
        # these eight IPA graphemes carry contrasts in the actual source inventory.
        # Retain them rather than merge ɪ/i, ʊ/u, ʌ/a, ə/a, or ɕ/ʃ.
        ("sil-gutob-gorum", "ɪː"), ("sil-gutob-gorum", "ʊː"),
        ("sil-gutob-gorum", "əː"), ("sil-gutob-gorum", "ʌ"),
        ("sil-gutob-gorum", "ə"), ("sil-gutob-gorum", "ɪ"),
        ("sil-gutob-gorum", "ʊ"), ("sil-gutob-gorum", "ɕ")}

# Marathi sources that write the dental affricates as c/j against palatal č/ǰ (SDML's stated
# convention; Ghatage's survey vocabularies): the dental pair is the house ʦ/ʣ, so the contrast
# survives in house letters instead of collapsing onto c/j.
MARATHI_AFFRICATES = {"c": "ʦ", "j": "ʣ", "č": "c", "ǰ": "j"}
OVERRIDES = {name: MARATHI_AFFRICATES for name in ("sdml", "ghatage-western", "ghatage")}
# ɑ is the open long vowel in the census / More survey IPA, as in every other survey profile
# Dhakal 2011 p.51 table 3.7: c/dz are alveolar affricates; j is a glide.
OVERRIDES["dhakal-darai"] = {"c": "ʦ", "dz": "ʣ", "j": "y"}
OVERRIDES["census-ipa"] = {"ɑ": "ā"}
OVERRIDES["more-ipa"] = {"ɑ": "ā"}
# The Census / LSI state-volume ASCII scheme (key printed in each volume): capitals are the
# short vowels and retroflexes — A = short a, E ɛ, O ɔ, I i, U u; T ṭ, D ḍ, N ṇ, L ḷ, R ṛ
# (retroflex trill/flap), M ŋ, M' ñ, S' ś — while plain a/i/u are the long vowels (LONG_*).
_CENSUS_ASCII = {"A": "a", "E": "ɛ", "O": "ɔ", "I": "i", "U": "u", "T": "ṭ", "D": "ḍ",
                 "N": "ṇ", "L": "ḷ", "R": "ṛ", "M": "ŋ", "M'": "ñ", "M’": "ñ", "S'": "ś",
                 "S’": "ś", "S": "s", "J": "j", "K": "k", "Th": "ṭʰ", "Dh": "ḍʰ"}
for _name in ("census-ascii", "more-ascii", "more-himachal"):
    OVERRIDES[_name] = _CENSUS_ASCII
# Mahato's Kisan lists capitalise the first letter of ordinary words
OVERRIDES["more-ipa"] = {"ɑ": "ā", **{c: c.lower() for c in "ABDFIKMNPSTU"}}
# Varenkamp's Ho lists print a vertical bar between the morphemes of an inflected response
# (``dʒom|ida|dʒomʌme`` under 'eat'); the house morpheme boundary is the hyphen.
OVERRIDES["sil-ho"] = {"|": "-"}
OVERRIDES["selected-orissa"] = {"R": "ṛ"}   # the 2002 Orissa key: R is the retroflex flap
OVERRIDES["dadra-varli"] = {"L": "ḷ"}

LENGTH_MARKS = ("ː", ":")  # half-length ˑ is a source distinction and is carried over
MACRON = "̄"


def base_and_marks(s: str) -> tuple[str, str]:
    """Split an NFC grapheme into its base letter(s) and trailing combining marks."""
    d = ud.normalize("NFD", s)
    i = len(d)
    while i > 0 and ud.combining(d[i - 1]):
        i -= 1
    return d[:i], d[i:]


# A long lax vowel is the long tense series: Turner's a/ā, i/ī, u/ū are [ʌ]/[aː], [ɪ]/[iː],
# [ʊ]/[uː], and no house form has ʌ̄ / ɪ̄ / ʊ̄.
LAX_LONG = {"ʌ": "a", "ɪ": "i", "ʊ": "u"}


def lengthen(out: str) -> str:
    """House long form of an output: macron on a vowel, doubling on a consonant."""
    out = nfc(out)
    if not out:
        return out
    base, marks = base_and_marks(out)
    if base and base[-1] in HOUSE_VOWELS | {"ɐ", "ɑ", "ɤ", "ɵ", "ɯ", "ø", "œ", "ʉ", "ɘ", "ɜ"}:
        if MACRON in marks:
            return out
        base = base[:-1] + LAX_LONG.get(base[-1], base[-1])
        return nfc(base + MACRON + marks)
    # kʰ → kkʰ: the aspiration modifier stays on the second consonant
    tail = ""
    while base and ud.category(base[-1]) == "Lm":
        tail, base = base[-1] + tail, base[:-1]
    if base and base[-1].isalpha():
        # ṭ → ṭṭ (marks such as the dot below belong to the letter)
        letter = base[-1] + "".join(m for m in marks if m != MACRON)
        return nfc(base[:-1] + letter + letter + tail)
    return out


def house_output(grapheme: str, out: str, rules: dict[str, str] | None = None,
                 profile: str | None = None) -> str:
    """The policy-conformant output for one rule, given its source grapheme.

    ``rules`` maps the profile's other graphemes to their current outputs, so a length bigram
    (``aː``) can be derived from its base rule (``a``); ``profile`` selects the vowel decisions."""
    g, o = nfc(grapheme), nfc(out)
    vowels = vowel_map(profile) if profile else {}
    if not o.strip() or o == "#":
        return o
    # markup / punctuation pass through
    if o.startswith("<") or all(not ch.isalpha() for ch in o):
        return o
    # length bigrams: house long form of the base's house output
    if len(g) > 1 and g[-1] in LENGTH_MARKS:
        base = g[:-1]
        if rules is not None and base in rules:
            want = lengthen(house_output(base, rules[base], rules, profile))
            # the same letters and marks in another canonical order (ǖ / ṻ) is not a change
            # unless the house order (macron first) is what the inventory expects
            wb, wm = base_and_marks(want)
            ob_, om_ = base_and_marks(o)
            if wb == ob_ and sorted(wm) == sorted(om_) and wb[-1:] not in HOUSE_VOWELS:
                return o
            return want
        # no base rule: accept the current output once its letters are house letters
        o = o.replace("ː", MACRON).replace(":", MACRON)
        for k, v in _SUBST:
            if k in o and k != v:
                o = o.replace(k, v)
        return nfc(o)
    # aspiration is the modifier letter, never a plain h
    if g in ("ʰ", "ʱ"):
        return "ʰ"
    gb, gm = base_and_marks(g)
    ob, om = base_and_marks(o)
    # vowels: Turner's short/long system, per the profile's decisions; the source's own marks
    # (nasal, accent) are kept and a macron is added when the letter is the long one
    if len(gb) == 1 and gb in HOUSE_VOWELS and set(gm) <= HOUSE_MARKS:
        letter = vowels.get(gb, gb)
        if MACRON in gm:
            letter = base_and_marks(letter)[0]  # ā is already long
        if len(ob) != 1 or ob not in HOUSE_VOWELS or ob != letter or (MACRON in om) != (MACRON in gm):
            o = nfc(letter + gm)
    # consonant spelling
    for k, v in _SUBST:
        if k in o and k != v:
            o = o.replace(k, v)
    m = _ASPIRATE_OUT.match(o)
    if m and ("ʰ" in g or "ʱ" in g or g.endswith("h")):
        o = m.group(1) + "ʰ"
    return nfc(o)


# ------------------------------------------------------------------------------- source scan


def length_marks(profile: str) -> str:
    """The graphemes a profile reads as a length mark: ``ː``, ``:`` and any single character it
    maps to ``ː`` (Haryanvi's ``·``, Pothwari's ``ˑ``)."""
    marks = set(LENGTH_MARKS)
    # Hahn p172 song uses colons as clause punctuation, not vowel length.
    # Its profile removes punctuation in display while Original remains exact.
    if profile == "hahn-asur-1900":
        marks.discard(":")
    path = PROFILE_DIR / f"{profile}.txt"
    if path.exists():
        header, body = read_profile(path)
        col = header.index("IPA") if "IPA" in header else 1
        for row in body:
            # A literal colon can be source-script punctuation, not IPA length.
            # Respect an explicit preservation rule instead of combining it
            # with every long vowel found elsewhere in the source.
            if len(row) > col and row[0] == ':' and row[col] == ':':
                marks.discard(':')
            if len(row) > col and row[col] == "ː" and len(nfc(row[0])) == 1:
                marks.add(nfc(row[0]))
    return "".join(sorted(marks))


def source_inventory() -> dict[str, dict]:
    """Per profile: the length bigrams and glide letters of the sources routed through it."""
    sys.path.insert(0, str(ROOT))
    import source_meta

    meta = source_meta.load()
    inv: dict[str, dict] = defaultdict(lambda: {"long": Counter(), "marks": set(), "affricate": 0, "j": 0, "y": 0, "forms": 0})
    patterns: dict[str, re.Pattern] = {}
    for path in sorted(glob.glob(str(ROOT / "data/other/forms/*.csv"))):
        rel = str(Path(path).relative_to(ROOT))
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 8:
                    continue
                key = row[7].split("[", 1)[0].strip()
                profile, convert = meta.transcription(key, rel, row[0])
                if not convert or profile is None:
                    continue
                form = row[5] if profile in ("lsi", "ia-dravidian-ipa") and len(row) > 5 and row[5] else row[2]
                form = nfc(form)
                d = inv[profile]
                d["forms"] += 1
                if profile not in patterns:
                    patterns[profile] = re.compile(
                        r"((?:tʃ|dʒ|ts|dz|ʧ|ʤ|.)[\u0300-\u036f]*[ʰʱ]?)([" + re.escape(length_marks(profile)) + "])"
                    )
                for m in patterns[profile].finditer(form):
                    d["long"][m.group(1)] += 1
                    d["marks"].add(m.group(2))
                if "dʒ" in form or "ʤ" in form or "tʃ" in form:
                    d["affricate"] += 1
                d["j"] += "j" in form
                d["y"] += "y" in form
    return inv


# ---------------------------------------------------------------------------------- profiles


def read_profile(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [line.rstrip("\r\n").replace("\r", "").split("\t") for line in handle]
    header, body = rows[0], [r for r in rows[1:] if r and r[0] != ""]
    return header, body


def write_profile(path: Path, header: list[str], body: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(header) + "\n")
        for row in body:
            handle.write("\t".join(row) + "\n")


def profile_paths() -> list[Path]:
    return sorted(p for p in PROFILE_DIR.glob("*.txt") if p.stem not in REFERENCE)


def glide_rule(profile: str, inv: dict) -> bool:
    """``j`` is the IPA glide (→ ``y``) in sources that write the affricate as ``dʒ`` and do
    not themselves use ``y`` for it."""
    d = inv.get(profile, {})
    return d.get("affricate", 0) > 0 and d.get("y", 0) < d.get("j", 0)


def audit(inv: dict) -> dict[str, list[tuple[str, str, str]]]:
    """Every (grapheme, current output, policy output) that differs, per profile."""
    findings: dict[str, list[tuple[str, str, str]]] = {}
    for path in profile_paths():
        header, body = read_profile(path)
        col = header.index("IPA") if "IPA" in header else 1
        rules = {nfc(r[0]): (r[col] if len(r) > col else "") for r in body}
        out = []
        for row in body:
            g = row[0]
            o = row[col] if len(row) > col else ""
            want = house_output(g, o, rules, path.stem)
            if nfc(g) == "j" and glide_rule(path.stem, inv):
                want = "y"
            if (path.stem, nfc(g)) in KEEP:
                want = nfc(o)
            want = OVERRIDES.get(path.stem, {}).get(nfc(g), want)
            if want != nfc(o):
                out.append((g, o, want))
        # length bigrams present in the sources but not covered by an explicit rule
        have = {nfc(r[0]) for r in body}
        for base in inv.get(path.stem, {}).get("long", {}):
            letter = base_and_marks(base)[0].rstrip("ʰʱ")
            for mark in sorted(inv[path.stem]["marks"]):
                if nfc(base + mark) not in have and base.strip() and ud.category(base[0]) == "Ll" and nfc(letter) in have:
                    out.append((base + mark, "(missing)", ""))
        if out:
            findings[path.stem] = out
    return findings


def fix(inv: dict) -> dict[str, int]:
    changed: dict[str, int] = {}
    for path in profile_paths():
        header, body = read_profile(path)
        col = header.index("IPA") if "IPA" in header else 1
        rules = {nfc(r[0]): r for r in body}
        outputs = {nfc(r[0]): (r[col] if len(r) > col else "") for r in body}
        n = 0
        for row in body:
            g = row[0]
            o = row[col] if len(row) > col else ""
            want = house_output(g, o, outputs, path.stem)
            if nfc(g) == "j" and glide_rule(path.stem, inv):
                want = "y"
            if (path.stem, nfc(g)) in KEEP:
                want = nfc(o)
            want = OVERRIDES.get(path.stem, {}).get(nfc(g), want)
            if want != nfc(o):
                while len(row) <= col:
                    row.append("")
                row[col] = want
                n += 1
        # overrides for graphemes the profile does not list yet (M' beside M)
        for g_, out_ in OVERRIDES.get(path.stem, {}).items():
            if nfc(g_) not in rules and all(nfc(part) in rules for part in g_):
                new = [g_] + [""] * (len(header) - 1)
                new[col] = out_
                body.append(new)
                rules[nfc(g_)] = new
                n += 1
        # IPA affricate digraphs must not fall apart into t + ś
        if inv.get(path.stem, {}).get("affricate", 0):
            for digraph, out in (("tʃ", "c"), ("dʒ", "j"), ("ʧ", "c"), ("ʤ", "j")):
                if nfc(digraph) not in rules and any(
                    nfc(part) in rules for part in (digraph[-1],)
                ):
                    new = [digraph] + [""] * (len(header) - 1)
                    new[col] = out
                    body.append(new)
                    rules[nfc(digraph)] = new
                    n += 1
        # explicit rules for every length bigram the routed sources actually contain
        for base, count in sorted(inv.get(path.stem, {}).get("long", {}).items(), key=lambda kv: -kv[1]):
            if not base.strip() or ud.category(base[0]) != "Ll":
                continue
            gb = nfc(base)
            for mark in sorted(inv[path.stem]["marks"]):
                key = nfc(gb + mark)
                if key in rules:
                    continue
                base_rule = rules.get(gb)
                letter, marks = base_and_marks(gb)
                aspirate = ""
                if base_rule is None and letter[-1:] in ("ʰ", "ʱ") and letter[:-1]:
                    aspirate, letter = "ʰ", letter[:-1]
                    if not marks and letter in rules:
                        base_rule = rules[letter]
                        base_out = house_output(letter, base_rule[col] if len(base_rule) > col else letter, outputs, path.stem) + aspirate
                        new = [key] + [""] * (len(header) - 1)
                        new[col] = lengthen(base_out)
                        body.append(new)
                        rules[key] = new
                        n += 1
                        continue
                if base_rule is None and marks and letter in rules:
                    # ``ɛ̃ː``: the profile tokenizes the letter and its marks separately, so
                    # derive the long form from the letter's rule and carry the marks
                    letter_out = rules[letter][col] if len(rules[letter]) > col else letter
                    marks_out = "".join(
                        (rules[m][col] if m in rules and len(rules[m]) > col else m) for m in marks
                    )
                    base_rule = rules[letter]
                    base_out = house_output(letter, letter_out, outputs, path.stem) + marks_out
                elif base_rule is None:
                    continue
                else:
                    base_out = base_rule[col] if len(base_rule) > col else gb
                new = [key] + [""] * (len(header) - 1)
                new[col] = lengthen(house_output(gb, base_out, outputs, path.stem))
                if len(header) > 2 and len(base_rule) > 2:
                    new[2:] = base_rule[2:]
                body.append(new)
                rules[key] = new
                n += 1
        if n:
            write_profile(path, header, body)
            changed[path.stem] = n
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["check", "fix"])
    parser.add_argument("profiles", nargs="*", help="limit the report to these profiles")
    args = parser.parse_args()
    inv = source_inventory()
    if args.command == "check":
        findings = audit(inv)
        for name, rows in findings.items():
            if args.profiles and name not in args.profiles:
                continue
            print(f"== {name}: {len(rows)} rule(s)")
            for g, o, want in rows:
                print(f"   {g!r:14} {o!r:14} -> {want!r}")
        total = sum(len(v) for v in findings.values())
        print(f"{total} policy violations in {len(findings)} profiles")
        sys.exit(1 if total else 0)
    changed = fix(inv)
    for name, n in changed.items():
        print(f"{name}: {n} rule(s) rewritten or added")
    print(f"{sum(changed.values())} rules in {len(changed)} profiles")


if __name__ == "__main__":
    main()

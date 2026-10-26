"""
sca.py — a small ordered sound-change applier over Jambu's segment inventory.

Words are segmented with the curated CDIAL grapheme profile (align.py's segmenter), so units such
as `ḍʰ`, `r̩`, `ā́` or `ⁿd` are single segments. Rules are applied in file order; each rule applies
simultaneously, left to right, non-overlapping, with environments read off that rule's input.

Rule file syntax (one statement per line; `;` starts a comment, as does `#` at line start):

    class Stop = k kʰ g gʰ c cʰ j jʰ ṭ ṭʰ ḍ ḍʰ t tʰ d dʰ p pʰ b bʰ
    class Unasp = Stop - kʰ gʰ cʰ jʰ ṭʰ ḍʰ tʰ dʰ pʰ bʰ      (`-` removes members; also inside {…})
    [repeat:] [optional:] [if:FLAG[,FLAG…]] TARGET > REPLACEMENT [/ ENV [| ENV …]] [// EXC [| EXC …]]  [; @tag — note]
    %exclude-source KEY          (file directive: drop that source's pairs from evaluation)
    %source KEY …  …rules…  %end (rules applied after the chain only for pairs from those sources)
    %gold-drop tone|caron|circumflex|<mark> … [@SOURCE …]   (drop these marks from this lect's attested forms
                                                 — only SOURCE's forms when @SOURCE is given —
                                                 in the accent and tone scores too: e.g. Strand's
                                                 glide ǐ, which is not a tone)
    %accent chain|vedic          (accent score: the chain's `+stress` accents, or also the Vedic one)

`repeat:` reapplies the rule until the word stops changing (e.g. three-consonant clusters).
`optional:` marks a sporadic change: the extension-aware and verb scores try the chain with and without it
(best per reflex).
`if:` restricts a rule to words with (or, with `!`, without) a word-level flag. evaluate.py sets
the flags per etymon: `verb` (glossed "to …", or an unglossed -ti/-tē headword), `numeral`,
`starred` (Turner's *reconstructions), `root` (consonant-final citation such as *kʰēll),
`accented` (the headword carries a Vedic accent). `if:verb,!numeral` requires all conditions.
Tags after `;` are free comments to the engine; CATALOG.md lists the shared tag vocabulary.

Tokens are whitespace separated:
    k            a literal segment
    Stop, V, C   a class (V and C are built in, derived from the segment's articulatory parse)
    Acc Long Short  built-in vowel classes: accented (acute/grave), long (macron, ai, au), not long
    Stress       a vowel carrying a latent accent (for accent-placement rules only: using it in a
                 segmental rule makes the plain NED depend on the accent tier)
    {ś ṣ}        an inline set
    !V           any segment not in V (never matches the word boundary; `!#` = any segment)
    (C)          optional token (environments and targets; one token each, e.g. `(a) (y) a t i`;
                 in a target, `$n` of a skipped position emits nothing)
    C*           zero or more (environments only): `# C* _` first syllable, `_ C* #` final
                 syllable, `_ C* a` next vowel a, `_ !#* Acc` pretonic, `Acc !#* _` post-tonic
    #            word boundary (environments only)
    _            the focus position (environments only)
    ∅ or 0       empty target (insertion) or empty replacement (deletion)
    $1 (target, environment)  the same segment as target position 1, aspiration aside
                 (geminates: `C $1`; not the first half of a geminate: `C > … // _ $1`)
    $2           copy of the 2nd matched target segment, optionally modified:
                 $1-asp $1+asp $1-long $1+long $1-nas $1+nas $1-acc $1+stress $1-stress
Latent accent tier: `$1-acc` removes the visible accent but keeps it latent on the vowel, and
`$1+stress` places a latent accent. A latent accent never affects matching (the vowel is its plain
value) and moves with the vowel through later rules (onto the result of a merger or quality
change; lost when the vowel is deleted). It is shown only for the accent score (`realize`), so
accent rules cannot change the accent-blind NED.
A set/class in REPLACEMENT maps index-wise onto the set/class at the same position of TARGET,
or at position n when suffixed `@n`:
    {ṭ ṭʰ} > {ḍ ḍʰ} / V _ V
    {t tʰ d dʰ} y > {c c j j} {c cʰ j jʰ}@1
"""
from __future__ import annotations

import os
import functools
import re
import sys
import unicodedata
import dataclasses
from dataclasses import dataclass, field
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
if DATA not in sys.path:
    sys.path.insert(0, DATA)

from align import Seg, load_graphemes, segmenter  # noqa: E402

PROFILE = os.path.join(DATA, "conversion", "cdial-post.txt")
BOUNDARY = "#"
EMPTY = {"∅", "0"}


@lru_cache(maxsize=1)
def tokenizer():
    return segmenter(load_graphemes(PROFILE))


DIPH_FIRST = {"a", "á", "à"}
DIPH_SECOND = {"i", "í", "ì", "u", "ú", "ù"}


def tokenize(form: str) -> tuple[str, ...]:
    """Profile segmentation, plus two repairs: editorial braces (`*ga{l}lati`) are dropped, and an
    accented diphthong split by the profile (`saínya` → a + í) is re-joined as one segment."""
    toks = tokenizer()(form.replace("{", "").replace("}", ""))
    out: list[str] = []
    for t in toks:
        if out and out[-1] in DIPH_FIRST and t in DIPH_SECOND and (out[-1] != "a" or t not in "iu"):
            out[-1] = nfc(out[-1] + t)
        else:
            out.append(t)
    return tuple(out)


def gold_tokens(form: str, drop: frozenset = frozenset()) -> tuple[str, ...]:
    """Canonical segmentation for scoring (applied to both prediction and attested form).

    Drops parenthesised optional letters (`agaru(a)`), homonym digits and every accent/tone mark on
    vowels (the accent score keeps them), unifies spelling variants of one vowel letter (ε/ɛ, ǝ/ə,
    υ/ʊ) but never merges vowel qualities, reads the hiatus diaeresis `ü` as `u`,
    and splits the profile's `ai`/`au` tokens: in MIA/NIA transcription these are vowel
    sequences (`tarai` < tárati), so they must not count as one segment.
    """
    form = re.sub(r"\([^)]*\)", "", form).replace("ü", "u")
    form = re.sub(r"[0-9¹²³⁴⁵ᵘⁱᵃʸʷᵒˈˌ]", "", form)  # homonym numbers, transitional glides, IPA stress marks (Blair ˈ)
    form = nfc("".join(GOLD_VOWELS.get(ch, ch) for ch in unicodedata.normalize("NFD", form)))  # NFD first: έ ὲ are ε + mark
    toks = []
    for t in tokenize(form):
        if all(unicodedata.combining(c) for c in t):
            if toks and kind(toks[-1]) == "V":
                # a mark the profile split off (ə̃, ɨ́): keep nasal/length on its vowel, drop accent/tone
                keep = "".join(c for c in t if c not in GOLD_DROP_MARKS and c not in ACCENT_MARKS and c not in drop)
                if keep:
                    toks[-1] = nfc(toks[-1] + keep)
            continue  # otherwise a stray mark left by a source typo
        if kind(t) == "V":  # every accent/tone mark (Kho. guniyá, Mai. ō̂); never the acute of ś ź
            t = nfc("".join(c for c in unicodedata.normalize("NFD", t)
                            if c not in GOLD_DROP_MARKS and c not in ACCENT_MARKS and c not in drop))
        toks.append(t)
    return canon(tuple(toks))


# Attested-side only: stress marks carry no etymological signal in the reflex transcriptions, and a
# few sources keep lax-vowel letters that the house transcription folds (profile_policy.py).
GOLD_DROP_MARKS = {"\u0301", "\u0300", "\u030d"}
# Spelling variants of one vowel letter only: vowel qualities (ə ʌ ɛ ɔ ɪ ʊ vs a e o i u) are NOT
# merged in any score (they were folded to house vowels before 2026-09-29).
GOLD_VOWELS = {"ǝ": "ə", "ε": "ɛ", "υ": "ʊ", "É": "E"}
# names usable in `%gold-drop`
DROP_NAMES = {"caron": {"\u030c"}, "circumflex": {"\u0302"}, "grave": {"\u0300"},
              "tone": {"\u030c", "\u0302", "\u1dc4", "\u1dc5", "\u1dc6", "\u1dc7", "\u1dc8", "\u1dc9"}}


# Accent scoring (a second metric, next to the accent-blind NED): every mark that locates the
# accent on a vowel — stress acute/grave, the vertical line, and the contour-tone marks — reads as
# "this vowel is accented" (normalised to the acute); tone *type* is not scored.
ACCENT_MARKS = {"́", "̀", "̍", "̂", "̌", "᷄", "᷅", "᷆",
                "᷇", "᷈", "᷉"}
ACUTE = "́"


def _acc_vowel(t: str) -> str:
    """A vowel token with its accent-location mark(s) replaced, in place, by one acute (so an
    accented diphthong keeps the accent on the letter that carried it: ái, not aí)."""
    d = unicodedata.normalize("NFD", t)
    if not any(c in ACCENT_MARKS for c in d):
        return t
    out, placed = [], False
    for c in d:
        if c in ACCENT_MARKS:
            if not placed:
                out.append(ACUTE)
                placed = True
        else:
            out.append(c)
    return nfc(_order_marks("".join(out)))


def accent_tokens(form: str, drop: frozenset = frozenset(), _norm=None) -> tuple[str, ...] | None:
    """Canonical tokens for the accent score, or None when the attested form marks no accent
    (then it is not accent-scored: its source does not write accent). `_norm` is the per-vowel
    mark normaliser (default: one acute; the tone score passes _tone_vowel)."""
    form = re.sub(r"\([^)]*\)", "", form).replace("ü", "u")
    form = re.sub(r"[0-9¹²³⁴⁵ᵘⁱᵃʸʷᵒˈˌ]", "", form)
    form = nfc("".join(GOLD_VOWELS.get(ch, ch) for ch in unicodedata.normalize("NFD", form)))  # NFD first: έ ὲ are ε + mark
    # ɔ̌ is the house spelling of Turner's long centralised ɔ̈̄ (CDIAL only), not a rising tone
    form = nfc(unicodedata.normalize("NFD", form).replace("ɔ\u030c", "ɔ"))
    toks, marked = [], False
    for t in tokenize(form):
        if all(unicodedata.combining(c) for c in t):
            if toks and kind(toks[-1]) == "V":   # a mark the profile split off (ɨ́, ə̃): re-read the vowel
                marked |= any(c in ACCENT_MARKS and c not in drop for c in t)
                d = unicodedata.normalize("NFD", toks[-1] + t)
                toks[-1] = (_norm or _acc_vowel)(nfc("".join(c for c in d if c not in drop)))
            continue
        if kind(t) == "V":
            d = unicodedata.normalize("NFD", t)
            marked |= any(c in ACCENT_MARKS and c not in drop for c in d)
            t = (_norm or _acc_vowel)(nfc("".join(c for c in d if c not in drop)))
        toks.append(t)
    return canon(tuple(toks)) if marked else None


# Tone score (a third metric): accent position as in the accent score, plus the tone TYPE. Marks are
# read as four classes, each normalised to one representative mark: high/level (acute, vertical
# line) → acute; rising (caron, macron-acute, grave-acute) → caron; falling (circumflex,
# macron-grave, acute-grave) → circumflex; low (grave, grave-macron) → grave.
TONE_CLASS = {"\u0301": "\u0301", "\u030d": "\u0301",
              "\u030c": "\u030c", "\u1dc4": "\u030c", "\u1dc8": "\u030c",
              "\u0302": "\u0302", "\u1dc6": "\u0302", "\u1dc9": "\u0302",
              "\u0300": "\u0300", "\u1dc5": "\u0300", "\u1dc7": "\u0301"}
TONE_COST = 0.25


def _tone_vowel(t: str) -> str:
    """A vowel with its accent marks replaced by one class mark. An explicit rising/falling/low mark
    wins over a plain acute (a tone rule's output over the generic realised accent)."""
    d = unicodedata.normalize("NFD", t)
    marks = [TONE_CLASS[c] for c in d if c in TONE_CLASS]
    if not marks:
        return t
    cls = next((m for m in marks if m != "\u0301"), marks[0])
    base = "".join(c for c in d if c not in TONE_CLASS)
    return nfc(_order_marks(base + cls))


def tone_tokens(form: str, drop: frozenset = frozenset()) -> tuple[str, ...] | None:
    """Canonical tokens for the tone score (None when the form marks no accent)."""
    return accent_tokens(form, drop, _tone_vowel)


def tone_pred(toks: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_tone_vowel(t) if kind(t) == "V" else t for t in toks)


def tone_edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Segmental edit distance ignoring all tone/accent marks, plus ACCENT_COST once when the
    accented vowel(s) are not aligned to each other, else plus TONE_COST when an aligned accented
    vowel carries a different tone class."""
    if a == b:
        return 0.0
    strip = lambda x: nfc("".join(c for c in unicodedata.normalize("NFD", x) if c not in TONE_CLASS))
    cls = lambda x: next((TONE_CLASS[c] for c in unicodedata.normalize("NFD", x) if c in TONE_CLASS), None)
    pa, pb = tuple(strip(x) for x in a), tuple(strip(y) for y in b)
    n, m = len(pa), len(pb)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (pa[i - 1] != pb[j - 1]))
    g2p, i, j = {}, n, m
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (pa[i - 1] != pb[j - 1]):
            g2p[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    acc_a = {i for i, x in enumerate(a) if cls(x)}
    acc_b = {g2p.get(j, -1 - j) for j, y in enumerate(b) if cls(y)}
    if acc_a != acc_b:
        return D[n][m] + ACCENT_COST
    wrong = any(cls(b[j]) != cls(a[g2p[j]]) for j, y in enumerate(b) if cls(y) and j in g2p)
    return D[n][m] + (TONE_COST if wrong else 0.0)


def acc_pred(toks: tuple[str, ...]) -> tuple[str, ...]:
    """A prediction read for the accent score: accent marks on vowels normalised to the acute."""
    return tuple(_acc_vowel(t) if kind(t) == "V" else t for t in toks)


def deacc_pred(toks: tuple[str, ...]) -> tuple[str, ...]:
    """A prediction read for the accent-blind score: stress marks dropped from vowels, exactly as
    they are dropped from the attested form, so accent-placement rules cannot change NED."""
    return tuple(nfc("".join(c for c in unicodedata.normalize("NFD", t) if c not in GOLD_DROP_MARKS and c not in ACCENT_MARKS))
                 if kind(t) == "V" else t for t in toks)


ACCENT_COST = 0.5


def accent_edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Segmental edit distance ignoring accents, plus ACCENT_COST once when the accent is wrong:
    the gold accented vowel(s), carried through the alignment, are not exactly the predicted
    accented vowel(s). A missing and a misplaced accent cost the same, so an unaccented
    prediction is never cheaper than a guess."""
    if a == b:
        return 0.0
    pa, pb = tuple(_deacc(x) for x in a), tuple(_deacc(y) for y in b)
    n, m = len(pa), len(pb)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (pa[i - 1] != pb[j - 1]))
    # backtrace: gold index -> predicted index for aligned (matched or substituted) pairs
    g2p, i, j = {}, n, m
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j - 1] + (pa[i - 1] != pb[j - 1]):
            g2p[j - 1] = i - 1
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    acc_a = {i for i, x in enumerate(a) if x != pa[i]}
    acc_b = {g2p.get(j, -1 - j) for j, y in enumerate(b) if y != pb[j]}
    return D[n][m] + (ACCENT_COST if acc_a != acc_b else 0.0)


def canon(toks: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for t in toks:
        # an a+i / a+u diphthong token splits in two, whatever marks its halves carry
        # (ai, ái, āu, and the profiles' nasal ãĩ ãũ), each half keeping its own marks
        d = unicodedata.normalize("NFD", t)
        bases = [j for j in range(len(d)) if not unicodedata.combining(d[j])]
        if len(bases) == 2 and d[bases[0]] == "a" and d[bases[1]] in "iu":
            k = bases[1]
            out.extend((nfc(d[:k]), nfc(d[k:])))
        else:
            out.append(t)
    return tuple(out)


def _deacc(t: str) -> str:
    return nfc("".join(c for c in unicodedata.normalize("NFD", t) if c not in "\u0301\u0300"))


class Word(tuple):
    """A segment tuple carrying word-level flags (verb, numeral, …) for `if:` rules. Rule
    application preserves the flags; plain tuples behave as words with no flags."""

    def __new__(cls, segs, flags=frozenset()):
        w = super().__new__(cls, segs)
        w.flags = frozenset(flags)
        return w


def flags_of(word) -> frozenset:
    return getattr(word, "flags", frozenset())


@lru_cache(maxsize=None)
def feat(seg: str, name: str) -> bool:
    """Built-in vowel classes Acc / Long / Short."""
    if kind(seg) != "V":
        return False
    d = unicodedata.normalize("NFD", seg)
    if name == "Acc":
        return "\u0301" in d or "\u0300" in d
    long_ = Seg(seg).long or "\u0304" in d
    return long_ if name == "Long" else not long_


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@lru_cache(maxsize=None)
def kind(seg: str) -> str:
    return Seg(seg).kind


# ─────────────────────────────────────────────────────────────────────────────
# Segment modifiers ($1+asp, $1-long, …)
# ─────────────────────────────────────────────────────────────────────────────

MARKS = {"long": "̄", "nas": "̃", "acc": "́"}


@lru_cache(maxsize=None)
def modify(seg: str, mods: tuple[tuple[str, str], ...]) -> str:
    for sign, feat in mods:
        if feat == "asp":
            seg = seg.replace("ʰ", "")
            if sign == "+":
                seg += "ʰ"
            continue
        d = unicodedata.normalize("NFD", seg)
        drop = {MARKS[feat]} | ({"̀", "̂"} if feat == "acc" else set()) \
            | ({"ː"} if feat == "long" else set())
        d = "".join(ch for ch in d if ch not in drop)
        if sign == "+":
            # insert the mark right after the first base letter (NFC then reorders canonically)
            for k, ch in enumerate(d):
                if not unicodedata.combining(ch):
                    d = d[:k + 1] + MARKS[feat] + d[k + 1:]
                    break
        seg = nfc(_order_marks(d))
    return str(seg)  # never a Stressed: this cache is shared by all words


# ─────────────────────────────────────────────────────────────────────────────
# Latent accent tier
# ─────────────────────────────────────────────────────────────────────────────

class Stressed(str):
    """A vowel carrying a latent accent. It *is* its plain value (compares, hashes and matches as
    the unaccented vowel), so no rule sees it; it only rides along with the vowel. `src` is
    "vedic" (the accent `$1-acc` hid) or "rule" (placed by `$1+stress`)."""


def stressed(seg: str, src: str) -> Stressed:
    s = Stressed(seg)
    s.src = src
    return s


def latent(seg) -> str | None:
    return getattr(seg, "src", None)


def realize(toks, model: str = "chain") -> tuple[str, ...]:
    """Make latent accents visible for the accent score. Rule-placed accents win; with
    model="vedic" the carried Vedic accent is shown when no rule placed one; "vedic-only" shows
    only the carried Vedic accent (the chain's placement rules ignored), for comparisons."""
    rule = model != "vedic-only" and any(latent(t) == "rule" for t in toks)
    out = []
    for t in toks:
        src = latent(t)
        show = (src == "rule" and model != "vedic-only") or \
            (src == "vedic" and model in ("vedic", "vedic-only") and not rule)
        # A vowel whose tone a rule already wrote (â ǎ à …) keeps it: adding the acute would drop it.
        written = any(c in TONE_CLASS for c in unicodedata.normalize("NFD", str(t)))
        out.append(modify(str(t), (("+", "acc"),)) if show and kind(t) == "V" and not written else str(t))
    return tuple(out)


MARK_ORDER = {"\u0304": 0, "\u0303": 1, "\u0301": 2, "\u0300": 2}


def _order_marks(d: str) -> str:
    """Put combining marks in the profile's order (macron, tilde, accent: `ī̃`, not `ĩ̄`);
    all three share combining class 230, so NFC would keep whatever order they were added in."""
    out, i = [], 0
    while i < len(d):
        j = i + 1
        while j < len(d) and unicodedata.combining(d[j]):
            j += 1
        out.append(d[i] + "".join(sorted(d[i + 1:j], key=lambda m: MARK_ORDER.get(m, 3))))
        i = j
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Token predicates
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Tok:
    kind: str                      # lit | set | kind | any | boundary | ref | focus
    value: object = None           # literal str, tuple of members, V/C, (index, mods)
    negated: bool = False
    optional: bool = False
    star: bool = False

    def matches(self, seg: str) -> bool:
        if seg == BOUNDARY:
            return self.kind == "boundary" and not self.negated
        if self.kind == "boundary":
            return self.negated  # `!#` = any real segment
        if self.kind == "lit":
            hit = seg == self.value
        elif self.kind == "set":
            hit = seg in self.value
        elif self.kind == "kind":
            hit = kind(seg) == self.value
        elif self.kind == "feat":
            # `Stress` reads the latent tier (uncached: equal strings may differ in latent accent)
            hit = (latent(seg) is not None if self.value == "Stress"
                   else latent(seg) == "vedic" if self.value == "Vedic"   # the carried Vedic accent only
                   else feat(seg, self.value))
        else:
            raise ValueError(f"token {self} cannot match")
        return hit != self.negated

    def members(self) -> tuple[str, ...] | None:
        return self.value if self.kind == "set" else None


def split_tokens(text: str) -> list[str]:
    """Whitespace split that keeps `{a b c}` (optionally prefixed by ! or wrapped in ()) whole."""
    out, cur, depth = [], "", 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch.isspace() and depth == 0:
            if cur:
                out.append(cur)
            cur = ""
        else:
            cur += ch
    if depth:
        raise ValueError(f"unbalanced braces in {text!r}")
    if cur:
        out.append(cur)
    return out


def expand_members(items: list[str], classes) -> tuple[str, ...]:
    """Class/set members: names expand to their class; `-` switches to removing members."""
    out: list[str] = []
    removing = False
    for m in items:
        if m == "-":
            removing = True
            continue
        vals = classes.get(m, (nfc(m),))
        if removing:
            out = [x for x in out if x not in vals]
        else:
            out.extend(vals)
    return tuple(out)  # duplicates kept: parallel mapping sets need them ({c c j j})


def parse_tok(raw: str, classes: dict[str, tuple[str, ...]], *, env: bool, repl: bool = False) -> Tok:
    optional = star = False
    if raw.startswith("(") and raw.endswith(")") and len(raw) > 2:
        if repl:
            raise ValueError(f"optional token {raw!r} not allowed in a replacement")
        optional, raw = True, raw[1:-1]
    if raw.endswith("*") and len(raw) > 1:
        if not env:
            raise ValueError(f"{raw!r}: `*` is only allowed in environments")
        star, raw = True, raw[:-1]
    negated = False
    if raw.startswith("!") and len(raw) > 1:
        negated, raw = True, raw[1:]
    if raw == BOUNDARY:
        if not env:
            raise ValueError("# only allowed in environments")
        return Tok("boundary", negated=negated, optional=optional, star=star)
    if raw.startswith("$") and raw[1:].isdigit():
        return Tok("same", int(raw[1:]) - 1, negated, optional, star)
    if raw.startswith("{") and raw.endswith("}"):
        return Tok("set", expand_members(raw[1:-1].split(), classes), negated, optional, star)
    if raw in ("V", "C") and raw not in classes:
        return Tok("kind", raw, negated, optional, star)
    if raw in ("Acc", "Long", "Short", "Stress", "Vedic") and raw not in classes:
        return Tok("feat", raw, negated, optional, star)
    if raw in classes:
        return Tok("set", classes[raw], negated, optional, star)
    return Tok("lit", nfc(raw), negated, optional, star)


@dataclass
class Rule:
    text: str
    line: int
    target: list[Tok]
    repl: list[object]             # str literal | ("ref", idx, mods) | ("map", members, tgt_idx)
    envs: list[tuple[list[Tok], list[Tok]]]
    excs: list[tuple[list[Tok], list[Tok]]]
    anchors: frozenset[str] | None = field(default=None)  # literal segments one of which must occur
    repeat: bool = False           # `repeat:` prefix — reapply until the word stops changing
    cond: tuple = ()               # `if:` conditions: ((flag, wanted), …)
    has_optional: bool = False     # target contains optional tokens
    name: str = ""                 # set when the rule comes from the central changes.sc (`use NAME`)
    origin: str = ""               # "BLOCK:LINE" when the rule comes from `include BLOCK` (blocks/BLOCK.sc)
    tags: tuple = ()               # @tags from the rule's comment (and its central definition)
    optional: bool = False         # `optional:` prefix — sporadic: forks the derivation (see apply_rules_multi)
    alts: tuple = ()               # `X > A | B`: further outcomes (Rules for B, …); this rule is outcome A

    def __str__(self):
        return self.text


def parse_env(text: str, classes) -> tuple[list[Tok], list[Tok]]:
    toks = split_tokens(text)
    if toks.count("_") != 1:
        raise ValueError(f"environment {text!r} needs exactly one _")
    k = toks.index("_")
    left = [parse_tok(t, classes, env=True) for t in toks[:k]]
    right = [parse_tok(t, classes, env=True) for t in toks[k + 1:]]
    return left, right


def parse_rule(text: str, line: int, classes) -> Rule:
    repeat = optional = False
    cond: list[tuple[str, bool]] = []
    prefix = ""
    while True:
        t = text.lstrip()
        if t.startswith("repeat:"):
            repeat, text = True, t[len("repeat:"):]
            prefix += "repeat: "
        elif t.startswith("optional:"):
            optional, text = True, t[len("optional:"):]
            prefix += "optional: "
        elif t.startswith("if:"):
            spec, _, text = t[3:].partition(" ")
            for f in spec.split(","):
                if f:
                    cond.append((f.lstrip("!"), not f.startswith("!")))
            prefix += f"if:{spec} "
        else:
            break
    body, _, exc = text.partition("//")
    body, _, env = body.partition("/")
    if ">" not in body:
        raise ValueError(f"line {line}: rule needs '>': {text!r}")
    lhs, rhs = body.split(">", 1)
    if "|" in rhs:
        # a change with several outcomes, `X > A | B / env`: one rule per outcome; the derivation forks
        # into all of them (apply_rules_multi), a single derivation takes the first
        tail = (f"/{env}" if env.strip() else "") + (f"//{exc}" if exc.strip() else "")
        outs = [parse_rule(f"{prefix}{lhs}> {a.strip()} {tail}", line, classes) for a in rhs.split("|")]
        return dataclasses.replace(outs[0], text=prefix + text.strip(), alts=tuple(outs[1:]))
    lhs_t, rhs_t = split_tokens(lhs), split_tokens(rhs)
    target = [] if lhs_t in (["∅"], ["0"]) else [parse_tok(t, classes, env=False) for t in lhs_t]
    for k, t in enumerate(target):
        if t.kind == "same" and not 0 <= t.value < k:
            raise ValueError(f"line {line}: target ${t.value + 1} must refer to an earlier position")
    if target and all(t.optional for t in target):
        raise ValueError(f"line {line}: a target needs at least one non-optional token")
    repl: list[object] = []
    if rhs_t not in (["∅"], ["0"], []):
        for j, t in enumerate(rhs_t):
            if t.startswith("$"):
                head = t[1:]
                num = ""
                while head and head[0].isdigit():
                    num += head[0]
                    head = head[1:]
                mods = []
                while head:
                    sign = head[0]
                    rest = head[1:]
                    nxt = min([p for p in (rest.find("+"), rest.find("-")) if p >= 0] or [len(rest)])
                    feat, head = rest[:nxt], rest[nxt:]
                    if sign not in "+-" or feat not in ("asp", "long", "nas", "acc", "stress"):
                        raise ValueError(f"line {line}: bad modifier in {t!r}")
                    mods.append((sign, feat))
                idx = int(num) - 1
                if not 0 <= idx < len(target):
                    raise ValueError(f"line {line}: {t} refers past the target")
                repl.append(("ref", idx, tuple(mods)))
                continue
            src = j
            if "@" in t and t.startswith("{"):
                t, _, at = t.rpartition("@")
                src = int(at) - 1
            tok = parse_tok(t, classes, env=False, repl=True)
            if tok.kind == "lit":
                repl.append(tok.value)
            elif tok.kind == "set":
                if src >= len(target) or target[src].members() is None \
                        or len(target[src].members()) != len(tok.value):
                    raise ValueError(f"line {line}: replacement set {t} must align with an "
                                     f"equal-sized set at target position {src + 1}")
                repl.append(("map", tok.value, src))
            else:
                raise ValueError(f"line {line}: {t!r} not allowed in replacement")
    envs = [parse_env(e, classes) for e in env.split("|")] if env.strip() else [([], [])]
    excs = [parse_env(e, classes) for e in exc.split("|")] if exc.strip() else []
    for side in [x for e in envs + excs for x in e]:
        for t in side:
            if t.kind == "same" and not 0 <= t.value < len(target):
                raise ValueError(f"line {line}: ${t.value + 1} in an environment must refer to the target")
    # cheap prefilter: if the first target token is a positive literal/set, the word must contain it
    anchors = None
    if target and target[0].kind in ("lit", "set") and not target[0].negated and not target[0].optional:
        t0 = target[0]
        anchors = frozenset([t0.value] if t0.kind == "lit" else t0.value)
    return Rule(prefix + text.strip(), line, target, repl, envs, excs, anchors, repeat,
                tuple(cond), any(t.optional for t in target), optional=optional)


def strip_comment(line: str) -> str:
    """`;` starts a comment anywhere; a line starting with `#` is a comment (rules never do)."""
    if line.lstrip().startswith("#"):
        return ""
    return line.split(";", 1)[0]


# ─────────────────────────────────────────────────────────────────────────────
# Family files: several lects' chains in one file
# ─────────────────────────────────────────────────────────────────────────────
#
# A family file (rules/<family>.sc) opens with `%lects STEM …` and may name subsets with
# `%set NAME = STEM|SET …`. Every other line is an ordinary chain line, optionally prefixed with a
# lect selector: `[K Sh] use x` (only those lects), `[-Kho] include y` (all but Kho), `[pahari -Garh]`
# (a set minus a lect). An unprefixed line belongs to every lect. A lect's chain is the file read
# with only its own lines, so line numbers (Rule.line) are lines of the family file.
#
# A lect is still addressed by the path rules/OIA-<stem>.sc: when no such file exists, the loader
# finds the family file in the same directory that lists the stem. A real OIA-<stem>.sc (a candidate,
# a snapshot, a lect outside any family) is read as a one-lect file, as before.

LECT_SELECTOR = re.compile(r"^\s*\[([^\]]*)\]\s?(.*)$")
_LECT_PATH = re.compile(r"OIA-(.+)\.sc$")


def _family_header(lines) -> tuple[tuple[str, ...], dict[str, frozenset[str]]] | None:
    """(members, sets) of a family file, or None when the file has no `%lects` line."""
    members: tuple[str, ...] | None = None
    sets: dict[str, frozenset[str]] = {}
    for raw in lines:
        line = strip_comment(raw).strip()
        if line.startswith("%lects"):
            members = tuple(line.split()[1:])
        elif line.startswith("%set") and members is not None:
            name, _, spec = line[4:].partition("=")
            sets[name.strip()] = frozenset(_expand_selector(spec.split(), members, sets, "%set"))
    return None if members is None else (members, sets)


def _expand_selector(toks, members, sets, where: str) -> set[str]:
    pos, neg = set(), set()
    for t in toks:
        bag, name = (neg, t[1:]) if t.startswith("-") else (pos, t)
        if name in sets:
            bag |= sets[name]
        elif name in members:
            bag.add(name)
        else:
            raise ValueError(f"{where}: unknown lect or set {name!r} in selector [{' '.join(toks)}]")
    return (pos or set(members)) - neg


@functools.lru_cache(maxsize=64)
def _family_index(directory: str, stamp: tuple) -> dict[str, str]:
    index = {}
    for name, _ in stamp:
        p = os.path.join(directory, name)
        with open(p, encoding="utf-8") as f:
            head = _family_header(f.read().split("\n"))
        if head is None:
            continue
        for stem in head[0]:
            if stem in index:
                raise ValueError(f"lect {stem!r} is listed in both {index[stem]} and {p}")
            index[stem] = p
    return index


def family_index(directory: str) -> dict[str, str]:
    """stem → family file, for the family files in `directory`."""
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        return {}
    stamp = tuple(sorted((n, os.path.getmtime(os.path.join(directory, n))) for n in os.listdir(directory)
                         if n.endswith(".sc") and not n.startswith("OIA-")))
    return _family_index(directory, stamp)


def resolve_lect(path: str) -> tuple[str, str | None]:
    """(source file, stem or None): where the chain addressed by `path` lives. A real file is a one-lect
    file (stem None); rules/OIA-<stem>.sc that does not exist resolves to its family file."""
    if os.path.isfile(path):
        return path, None
    m = _LECT_PATH.search(os.path.basename(path))
    fam = family_index(os.path.dirname(path) or ".").get(m.group(1)) if m else None
    if fam is None:
        raise FileNotFoundError(f"no chain at {path} (no such file, and no family file lists the lect)")
    return fam, m.group(1)


def has_chain(path: str) -> bool:
    try:
        resolve_lect(path)
        return True
    except FileNotFoundError:
        return False


def lect_paths(directory: str) -> list[str]:
    """Every lect chain in a rules directory, as rules/OIA-<stem>.sc paths (one-lect files and family
    members), sorted by path."""
    out = {os.path.join(directory, n) for n in os.listdir(directory) if _LECT_PATH.fullmatch(n)}
    out |= {os.path.join(directory, f"OIA-{s}.sc") for s in family_index(directory)}
    return sorted(out)


def source_lines(path: str) -> list[str]:
    """All lines of the file that holds the chain (Rule.line indexes these; see raw_line)."""
    src, _ = resolve_lect(path)
    with open(src, encoding="utf-8") as f:
        return f.read().split("\n")


def lect_lines(path: str) -> list[tuple[int, str]]:
    """(line number, text) of the chain's own lines: the whole one-lect file, or the family file's
    lines selected for the lect, selectors removed."""
    src, stem = resolve_lect(path)
    lines = source_lines(path)
    if stem is None:
        return list(enumerate(lines, 1))
    members, sets = _family_header(lines)
    out = []
    for n, raw in enumerate(lines, 1):
        m = LECT_SELECTOR.match(raw)
        if m:
            if stem not in _expand_selector(m.group(1).split(), members, sets, f"{src}:{n}"):
                continue
            raw = m.group(2)
        out.append((n, raw))
    return out


def require_one_lect_file(path: str) -> None:
    """For tools that rewrite a chain file in place: a family lect has no file of its own."""
    src, stem = resolve_lect(path)
    if stem is not None:
        raise ValueError(f"{path} is lect {stem} of the family file {src}: edit that file (or explode it with "
                         f"families.py, edit the one-lect files, and merge them back)")


def lect_text(path: str) -> str:
    """The chain as a one-lect file (selectors removed, other lects' lines dropped)."""
    return "\n".join(t for _, t in lect_lines(path)) + "\n"


def load_meta(path: str) -> dict:
    """File-level directives:
    `%exclude-source KEY …` drops those sources' pairs from evaluation (malformed notation);
    `%source KEY …` … `%end` wraps rules applied, after the main chain, only when scoring against
    pairs from those sources — per-source transcription (`@notation`) that would otherwise be
    noise for the lect's other sources. Block rules do not count as chain rules.
    `%verb-menu +ITEM …` / `%ext-menu +ITEM …` add lect-specific verb formations / noun extensions
    to the menus (each placebo-tested in that lect).
    `%accent chain|vedic` picks how the accent score gets a predicted accent: the marks the chain
    itself writes (default), or the Vedic accent carried to the vowel that continues it."""
    meta = {"exclude_sources": set(), "source_tails": load_rules_full(path)[1], "gold_drop": set(), "gold_drop_src": {},
            "accent": "chain", "verb_menu": (), "ext_menu": ()}
    for _, raw in lect_lines(path):
        line = raw.split(";", 1)[0].strip()
        if line.startswith("%exclude-source"):
            meta["exclude_sources"].update(line.split()[1:])
        elif line.startswith("%ext-menu"):
            meta["ext_menu"] += tuple(t.lstrip("+") for t in line.split()[1:])   # noun extensions, same rule
        elif line.startswith("%verb-menu"):
            # lect-specific verb formations beyond evaluate.VERB_MENU (`%verb-menu +alla`), each
            # admitted only with a placebo test in the lect's notes
            meta["verb_menu"] += tuple(t.lstrip("+") for t in line.split()[1:])
        elif line.startswith("%accent"):
            meta["accent"] = line.split()[1]  # chain | vedic: source of the accent-score prediction
        elif line.startswith("%gold-drop"):
            # `%gold-drop caron @strand`: only that source's forms (Strand's glide ǐ in a lect
            # whose other sources write a real rising tone)
            args = line.split()[1:]
            srcs = [a[1:] for a in args if a.startswith("@")]
            marks = set().union(*(DROP_NAMES.get(m, set(unicodedata.normalize("NFD", m)))
                                  for m in args if not m.startswith("@")))
            if srcs:
                for src in srcs:
                    meta["gold_drop_src"][src] = meta["gold_drop_src"].get(src, frozenset()) | marks
            else:
                meta["gold_drop"] |= marks
    meta["gold_drop"] = frozenset(meta["gold_drop"])
    return meta


BLOCKS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blocks")


def block_path(name: str) -> str:
    return os.path.join(BLOCKS, f"{name}.sc")


@functools.lru_cache(maxsize=None)
def block_lines(name: str) -> tuple[str, ...]:
    with open(block_path(name), encoding="utf-8") as f:
        return tuple(f.read().split("\n"))


def load_block(name: str, at: int, classes: dict, path: str) -> list[Rule]:
    """`include NAME`: the rules of blocks/NAME.sc, in order, as if written at line `at` of the lect
    file. A block holds `use` lines and plain rules (and may define classes of its own, which do not
    leak into the lect); each rule keeps `origin` = "NAME:LINE" so tools can find its comment."""
    if not os.path.exists(block_path(name)):
        raise ValueError(f"{path}:{at}: unknown block {name!r} (no {block_path(name)})")
    local = dict(classes)
    out = []
    for k, raw in enumerate(block_lines(name), 1):
        line = strip_comment(raw).strip()
        if not line or line.startswith("%"):
            continue
        if line.startswith("class "):
            cname, _, members = line[6:].partition("=")
            local[cname.strip()] = expand_members(members.split(), local)
            continue
        if line.startswith("include "):
            out.extend(load_block(line[8:].strip(), at, local, path))
            continue
        opt_use = line.startswith("optional: use ")
        if opt_use:
            line = line.split(":", 1)[1].strip()
        if line.startswith("use "):
            cname = line[4:].strip()
            defs = central()
            if cname not in defs:
                raise ValueError(f"{block_path(name)}:{k}: unknown central change {cname!r}")
            rule = dataclasses.replace(defs[cname][0], name=cname, tags=comment_tags(defs[cname][1]), optional=opt_use)
        else:
            rule = parse_rule(line, k, local)
        if ";" in raw:
            rule = dataclasses.replace(rule, tags=rule.tags + comment_tags(raw.split(";", 1)[1]))
        out.append(dataclasses.replace(rule, line=at, origin=f"{name}:{k}"))
    return out


def apply_rules_multi(rules: list, word: tuple, cap: int = 16) -> list:
    """Derivation with sporadic (`optional:`) rules: a set of forms goes through the chain. An optional
    rule forks a form only where it changes it (applied / skipped); identical forms merge. Returns
    [(form, skipped)] — skipped = indices (into `rules`) of the optional rules that form did not undergo;
    a rule with several outcomes (`X > A | B`, `rule.alts`) forks into each, and a form that took
    outcome j ≥ 1 of rule k carries ("alt", k, j) in `skipped`. The first entry is the all-on, first-outcome
    derivation. At most `cap` forms are kept per word."""
    cands = [(word, frozenset())]
    for k, r in enumerate(rules):
        new, seen = [], set()
        for w, sk in cands:
            out = apply_rule_full(r, w)
            opts = [(out, sk)]
            for j, a in enumerate(r.alts, 1):
                o = apply_rule_full(a, w)
                if o != out:
                    opts.append((o, sk | {("alt", k, j)}))
            if r.optional and out != w:
                opts.append((w, sk | {k}))
            for f, s2 in opts:
                key = tuple(map(str, f))
                if key not in seen and len(new) < cap:
                    seen.add(key); new.append((f, s2))
        cands = new
    return cands


def rules_taken(rules: list, skipped) -> list:
    """The rules a fork of apply_rules_multi underwent: optional rules it skipped dropped, and the chosen
    outcome of each several-outcome rule."""
    alt = {k: j for k, j in ((x[1], x[2]) for x in skipped if isinstance(x, tuple))}
    return [r.alts[alt[k] - 1] if k in alt else r for k, r in enumerate(rules) if k not in skipped]


def raw_line(rule: "Rule", lines) -> str:
    """The source line of a rule: from its block file when included, else from the chain's source
    lines (`source_lines(path)`; a family file's lect selector is dropped)."""
    if rule.origin:
        name, k = rule.origin.rsplit(":", 1)
        return block_lines(name)[int(k) - 1]
    raw = lines[rule.line - 1]
    m = LECT_SELECTOR.match(raw)
    return m.group(2) if m else raw


def load_rules(path: str) -> list[Rule]:
    return load_rules_full(path)[0]


CENTRAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "changes.sc")


@lru_cache(maxsize=4)
def load_central(path: str = CENTRAL, mtime: float = 0.0) -> dict[str, tuple[Rule, str]]:
    """The central Indo-Aryan sound-change definitions: `change NAME = RULE ; @tags — note`, parsed
    with the central file's own classes. Returns {name: (rule, comment)}."""
    classes: dict[str, tuple[str, ...]] = {}
    out: dict[str, tuple[Rule, str]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for n, raw in enumerate(lines, 1):
        line = strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("class "):
            cname, _, members = line[6:].partition("=")
            classes[cname.strip()] = expand_members(members.split(), classes)
        elif line.startswith("change "):
            cname, _, text = line[7:].partition("=")
            cname = cname.strip()
            if cname in out:
                raise ValueError(f"changes.sc line {n}: duplicate change {cname}")
            comment = raw.split(";", 1)[1].strip() if ";" in raw else ""
            out[cname] = (parse_rule(text.strip(), n, classes), comment)
    return out


def central():
    return load_central(CENTRAL, os.path.getmtime(CENTRAL) if os.path.exists(CENTRAL) else 0.0)


def load_rules_full(path: str) -> tuple[list[Rule], dict[str, list[Rule]]]:
    classes: dict[str, tuple[str, ...]] = {}
    rules: list[Rule] = []
    tails: dict[str, list[Rule]] = {}
    block: list[str] | None = None
    for n, raw in lect_lines(path):
        line = strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("%source"):
            block = line.split()[1:]
            for key in block:
                tails.setdefault(key, [])
            continue
        if line.startswith("%end"):
            block = None
            continue
        if line.startswith("%"):
            continue  # other file-level directive (see load_meta)
        if line.startswith("class "):
            name, _, members = line[6:].partition("=")
            classes[name.strip()] = expand_members(members.split(), classes)
            continue
        if line.startswith("include "):
            for rule in load_block(line[8:].strip(), n, classes, path):
                if block is None:
                    rules.append(rule)
                else:
                    for key in block:
                        tails[key].append(rule)
            continue
        opt_use = line.startswith("optional: use ") or line.startswith("optional:use ")
        if opt_use:
            line = line.split(":", 1)[1].strip()
        if line.startswith("use "):
            cname = line[4:].strip()
            defs = central()
            if cname not in defs:
                raise ValueError(f"{path}:{n}: unknown central change {cname!r}")
            rule = dataclasses.replace(defs[cname][0], line=n, name=cname,
                                       tags=comment_tags(defs[cname][1]), optional=opt_use)
        else:
            rule = parse_rule(line, n, classes)
        if ";" in raw:
            rule = dataclasses.replace(rule, tags=rule.tags + comment_tags(raw.split(";", 1)[1]))
        if block is None:
            rules.append(rule)
        else:
            for key in block:
                tails[key].append(rule)
    return rules, tails


def comment_tags(comment: str) -> tuple[str, ...]:
    """A rule's @tags: those before the note's em dash (`; @hiatus @x — note`). An @name inside the
    note is a cross-reference or history ("the @morph rule was removed"), not a tag."""
    return tuple(re.findall(r"@[\w\-.]+", comment.split("—", 1)[0]))


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

def _tok_ok(t: Tok, seg: str, matched) -> bool:
    if t.kind == "same":
        ref = matched[t.value] if matched and t.value < len(matched) else None
        if ref is None or seg == BOUNDARY:
            return t.negated and seg != BOUNDARY
        hit = modify(seg, (("-", "asp"),)) == modify(ref, (("-", "asp"),))
        return hit != t.negated
    return t.matches(seg)


def _match_seq(toks: list[Tok], word: tuple[str, ...], pos: int, step: int, matched=()) -> bool:
    """Match env tokens starting at `pos`, walking by `step` (+1 rightwards, -1 leftwards)."""
    if not toks:
        return True
    t, rest = toks[0], toks[1:]
    if t.star:
        if _match_seq(rest, word, pos, step, matched):
            return True
        p = pos
        while 0 <= p < len(word) and _tok_ok(t, word[p], matched):
            p += step
            if _match_seq(rest, word, p, step, matched):
                return True
        return False
    if t.optional and _match_seq(rest, word, pos, step, matched):
        return True
    if 0 <= pos < len(word) and _tok_ok(t, word[pos], matched):
        return _match_seq(rest, word, pos + step, step, matched)
    return False


def _match_target(target: list[Tok], w: tuple[str, ...], i: int) -> bool:
    for k, t in enumerate(target):
        seg = w[i + k]
        if t.kind == "same":
            if seg == BOUNDARY or modify(seg, (("-", "asp"),)) != modify(w[i + t.value], (("-", "asp"),)):
                return False
        elif not t.matches(seg):
            return False
    return True


def _target_matches(target: list[Tok], w: tuple[str, ...], i: int, n: int):
    """Matches of a target with optional tokens at w[i:], longest first: (end, per-token segs)."""
    def rec(k, pos, got):
        if k == len(target):
            yield pos, tuple(got)
            return
        t = target[k]
        if pos < n:
            seg = w[pos]
            ok = _tok_ok(t, seg, got) if t.kind == "same" else t.matches(seg)
            if ok and seg != BOUNDARY:
                yield from rec(k + 1, pos + 1, got + [seg])
        if t.optional:
            yield from rec(k + 1, pos, got + [None])
    for end, segs in rec(0, i, []):
        if end > i:
            yield end, segs


def _env_ok(envs, word, start, end, matched=()) -> bool:
    """word is boundary-padded; the focus spans [start, end)."""
    for left, right in envs:
        if _match_seq(list(reversed(left)), word, start - 1, -1, matched) \
                and _match_seq(right, word, end, 1, matched):
            return True
    return False


def _emit(rule: Rule, matched: tuple) -> list[str]:
    out = []
    set_stress = drop_stress = False
    for r in rule.repl:
        if isinstance(r, str):
            out.append(r)
        elif r[0] == "ref":
            seg = matched[r[1]]
            if seg is not None:
                mods = tuple(m for m in r[2] if m[1] != "stress")
                new = modify(seg, mods) if mods else str(seg)
                if ("+", "stress") in r[2] and kind(new) == "V":
                    new, set_stress = stressed(new, "rule"), True
                elif ("-", "stress") in r[2]:
                    drop_stress = True
                elif ("-", "acc") in mods and feat(seg, "Acc"):
                    new = stressed(new, "vedic")      # the accent goes latent, not away
                out.append(new)
        else:  # map
            _, members, j = r
            if matched[j] is not None:
                src = rule.target[j].members()
                out.append(members[src.index(matched[j])])
    # a latent accent on a rewritten vowel moves to the vowel that continues it
    carried = [m for m in matched if m is not None and latent(m)]
    if carried and not set_stress and not drop_stress and not any(latent(o) for o in out):
        vowels = [k for k, o in enumerate(out) if kind(o) == "V"]
        if vowels:
            pos = [k for k, m in enumerate(matched) if m is not None and latent(m)]
            k = pos[0] if len(out) == len(matched) and pos[0] in vowels else vowels[0]
            out[k] = stressed(out[k], latent(carried[0]))
    return out


def _conds_ok(rule: Rule, word) -> bool:
    if not rule.cond:
        return True
    fl = flags_of(word)
    return all((f in fl) == want for f, want in rule.cond)


def _rewrap(word, out: list[str]):
    return Word(out, word.flags) if isinstance(word, Word) else tuple(out)


def apply_rule(rule: Rule, word: tuple[str, ...]) -> tuple[str, ...]:
    if rule.anchors is not None and rule.anchors.isdisjoint(word):
        return word
    if not _conds_ok(rule, word):
        return word
    w = (BOUNDARY,) + tuple(word) + (BOUNDARY,)
    n = len(w) - 1          # real segments occupy 1..n-1
    out: list[str] = []
    L = len(rule.target)
    changed = False
    if L == 0:
        for g in range(1, n + 1):
            if _env_ok(rule.envs, w, g, g) and not _env_ok(rule.excs, w, g, g):
                out.extend(_emit(rule, ()))
                changed = True
            if g < n:
                out.append(w[g])
    elif rule.has_optional:
        i = 1
        while i < n:
            for end, segs in _target_matches(rule.target, w, i, n):
                if _env_ok(rule.envs, w, i, end, segs) \
                        and not (rule.excs and _env_ok(rule.excs, w, i, end, segs)):
                    out.extend(_emit(rule, segs))
                    i = end
                    changed = True
                    break
            else:
                out.append(w[i])
                i += 1
    else:
        i = 1
        while i < n:
            if i + L <= n and _match_target(rule.target, w, i):
                segs = w[i:i + L]
                if _env_ok(rule.envs, w, i, i + L, segs) \
                        and not (rule.excs and _env_ok(rule.excs, w, i, i + L, segs)):
                    out.extend(_emit(rule, segs))
                    i += L
                    changed = True
                    continue
            out.append(w[i])
            i += 1
    return _rewrap(word, out) if changed else word


def apply_rule_full(rule: Rule, word: tuple[str, ...]) -> tuple[str, ...]:
    if not rule.repeat:
        return apply_rule(rule, word)
    for _ in range(8):
        new = apply_rule(rule, word)
        if new == word:
            break
        word = new
    return word


def apply_rules(rules: list[Rule], word: tuple[str, ...], trace: bool = False):
    steps = []
    for r in rules:
        new = apply_rule_full(r, word)
        if trace and new != word:
            steps.append((r, new))
        word = new
    return (word, steps) if trace else word


# ─────────────────────────────────────────────────────────────────────────────
# Distance
# ─────────────────────────────────────────────────────────────────────────────

def edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def ned(a, b) -> float:
    m = max(len(a), len(b))
    return edit_distance(a, b) / m if m else 0.0

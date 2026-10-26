#!/usr/bin/env python3
"""
evaluate.py — score an ordered sound-change file against attested reflexes.

Subcommands (run from data/):
    score     RULES              train/test metrics, per-rule leave-one-out value, worst misses
    apply     RULES FORM…        derivation trace for individual etyma
    frontier  RULES              greedy backward elimination → (#rules, NED) curve
    suggest   RULES              hill-climbing aid: best next rules on train + best insertion point
    probe     RULES              vowel correspondence table by syllable position, Vedic accent,
                                 next vowel and weight, against the chain's output (train)
    induce                       greedy automatic rule induction baseline → .sc file + curve

The pair file defaults to pairs/<stem>.tsv where <stem> is the rules filename up to its first dot
(rules/OIA-Pk.sc and rules/OIA-Pk.auto.sc both read pairs/OIA-Pk.tsv).

Metrics are over segments (the CDIAL grapheme profile), comparing canonicalised forms
(sca.gold_tokens / sca.canon):
    NED    mean of edit_distance / max(len(pred), len(gold))   — the frontier's y-axis
    ED     mean raw edit distance
    exact  share of reflexes predicted exactly
Every reflex counts once; an etymon with three Pk reflexes contributes three pairs. The zero-rule
point of any curve is the no-change baseline (the etymon itself).
"""
from __future__ import annotations

import argparse
import unicodedata
import csv
import os
import functools
import re
import sys
import time
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sca  # noqa: E402
from align import Seg, describe  # noqa: E402

ACCENT_MODEL = None  # --accent-model override of the rule file's `%accent` directive


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

_NUM = (r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
        r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
        r"eighty|ninety|hundred|thousand)")
# the whole gloss is a number: "eighty", "84", "forty-eight", "one hundred", "the number five"
NUMERAL_GLOSS = re.compile(rf"^(the number )?{_NUM}([\s\-]+(and[\s\-]+)?{_NUM})*$", re.I)
VERB_ENDING = re.compile(r"(ati|āti|ēti|ōti|īti|yatē|atē|ayati|nāti|nōti|ṇāti|ṇōti)$")
NUMERAL_ENDINGS = ("daśa", "daśan", "viṁśati", "triṁśat", "catvāriṁśat", "pañcāśat", "ṣaṣṭi",
                   "saptati", "aśīti", "navati", "śata", "sahasra")


@functools.lru_cache(maxsize=1)
def english_verbs() -> frozenset[str]:
    """English words glossed "to X" at least 3 times in the forms (english_verbs.txt): a bare-verb
    gloss on a verbal headword ("bring" for ā́nayati) marks the reflex as a verb."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "english_verbs.txt")
    with open(path, encoding="utf-8") as f:
        return frozenset(w.strip() for w in f if w.strip() and not w.startswith("#"))


def word_flags(etymon: str, toks, glosses) -> set[str]:
    """Word-level flags for `if:` rules, from the headword and the glosses of its reflexes."""
    flags = set()
    plain = sca._deacc(etymon.strip().lstrip("*√").rstrip("-")).replace("-", "")
    gl = [g.strip().strip("'\"‘’") for g in glosses if g and g.strip()]
    if plain.endswith(NUMERAL_ENDINGS) or (gl and sum(bool(NUMERAL_GLOSS.match(g)) for g in gl) * 2 >= len(gl)):
        flags.add("numeral")
    if gl and sum(g.lower().startswith("to ") for g in gl) * 2 >= len(gl):
        flags.add("verb")
    elif (VERB_ENDING.search(plain) or (toks and sca.kind(toks[-1]) == "C")) and "numeral" not in flags:
        # a verbal headword: -ati/-atē/… or a bare (starred) root (*cāh "desires", *ḍāl "throws")
        # glossed in the 3rd person ("falls down", "releases, lets go"), as a bare English verb
        # ("bring", "climb up": Kullui), as a survey clause ("he killed", "come!, he came"), or not at all
        g0 = re.sub(r"^\s*(?:\d+[).]|[a-z][).])\s*", "", gl[0]) if gl else ""   # "1) eat; 2) bite"
        first = re.sub(r"[^a-z]", "", g0.split()[0].lower()) if g0.split() else ""
        # a pronoun followed by a word ("he killed"); a bare pronoun gloss ("we": asmad) is not a clause
        clause = any(re.match(r"(he|she|it|they|i|we|you)\s+\w", part.strip().lower()) or part.strip().endswith("!")
                     for g in gl for part in g.split(","))
        root = not VERB_ENDING.search(plain)   # a bare root needs a 3rd-person or clause gloss
        third = first.endswith("s") and not first.endswith("ss")
        if clause or third or (not root and (not gl or first in english_verbs() or first.endswith("s"))):
            flags.add("verb")
    if etymon.strip().startswith("*"):
        flags.add("starred")
    if toks and sca.kind(toks[-1]) == "C":
        flags.add("root")
    if any(sca.feat(t, "Acc") for t in toks):
        flags.add("accented")
    return flags


class Dataset:
    """Unique etyma (inputs) plus the attested reflexes each must be scored against. Etyma are
    sca.Word tuples carrying word-level flags (verb, numeral, starred, root, accented)."""

    def _drop(self, src: str) -> frozenset:
        """Marks dropped from this source's attested forms (`%gold-drop`, lect-wide or `@SOURCE`)."""
        return self.gold_drop | self.gold_drop_src.get(src, frozenset())

    def __init__(self, path: str, split: str | None = None, exclude_sources=(), gold_drop=frozenset(),
                 gold_drop_src=None):
        self.etyma: list[tuple[str, ...]] = []   # segmented etymon per unique etymon
        self.raw: list[str] = []
        self.golds: list[list[tuple[tuple[str, ...], str]]] = []  # per etymon: (tokens, raw)
        self.gold_src: list[list[str]] = []                          # parallel: source key
        self.agolds: list[list[tuple[str, ...] | None]] = []        # parallel: accent-score tokens
        self.gold_gloss: list[list[str]] = []                        # parallel: the reflex's gloss
        self.tgolds: list[list[tuple[str, ...] | None]] = []        # parallel: tone-score tokens (tonal sources only)
        self.verb_extra: tuple = ()     # lect-specific verb formations (%verb-menu)
        self.ext_extra: tuple = ()      # lect-specific noun extensions (%ext-menu)
        self.gold_drop, self.gold_drop_src = gold_drop, dict(gold_drop_src or {})
        self.accent_model = "chain"  # how predictions get an accent for the accent score (set by datasets)
        self.tails: dict[str, list] = {}  # %source blocks: extra rules for one source's pairs
        idx: dict[tuple[str, str], int] = {}
        glosses: list[list[str]] = []
        with open(path, encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if split and r["Split"] != split:
                    continue
                if r.get("Source") in exclude_sources:
                    continue
                key = (r["Etymon_ID"], r["Etymon"])
                if key not in idx:
                    toks = sca.tokenize(r["Etymon"])
                    if not toks:
                        continue
                    idx[key] = len(self.etyma)
                    self.etyma.append(toks)
                    self.raw.append(r["Etymon"])
                    self.golds.append([])
                    self.gold_src.append([])
                    self.agolds.append([])
                    self.gold_gloss.append([])
                    glosses.append([])
                glosses[idx[key]].append(r.get("Gloss", ""))
                drop = self._drop(r.get("Source", ""))
                gold = sca.gold_tokens(r["Reflex"], drop)
                if gold:
                    self.golds[idx[key]].append((gold, r["Reflex"]))
                    self.gold_src[idx[key]].append(r.get("Source", ""))
                    self.agolds[idx[key]].append(sca.accent_tokens(r["Reflex"], drop))
                    self.gold_gloss[idx[key]].append(r.get("Gloss", "") or "")
        keep = [i for i, g in enumerate(self.golds) if g]
        self.etyma = [sca.Word(self.etyma[i], word_flags(self.raw[i], self.etyma[i], glosses[i]))
                      for i in keep]
        self.raw = [self.raw[i] for i in keep]
        self.golds = [self.golds[i] for i in keep]
        self.gold_src = [self.gold_src[i] for i in keep]
        self.agolds = [self.agolds[i] for i in keep]
        self.gold_gloss = [self.gold_gloss[i] for i in keep]
        # tone score: only sources that write a tone TYPE (rising/falling/low marks) somewhere here
        RISE_FALL = {"\u030c", "\u0302", "\u0300", "\u1dc4", "\u1dc5", "\u1dc6", "\u1dc8", "\u1dc9"}
        tonal = {src for gs, ss in zip(self.golds, self.gold_src) for (_, raw), src in zip(gs, ss)
                 if any(c in RISE_FALL and c not in self._drop(src)
                        for c in unicodedata.normalize("NFD", raw).replace("ɔ\u030c", "ɔ"))}  # ɔ̌: see accent_tokens
        self.tgolds = [[sca.tone_tokens(raw, self._drop(src)) if src in tonal else None for (_, raw), src in zip(gs, ss)]
                       for gs, ss in zip(self.golds, self.gold_src)]
        self.n_tpairs = sum(g is not None for gs in self.tgolds for g in gs)
        self.n_pairs = sum(len(g) for g in self.golds)
        self.n_apairs = sum(g is not None for gs in self.agolds for g in gs)

    def cost(self, i: int, pred: tuple[str, ...]) -> tuple[float, int, int]:
        """(sum NED, sum ED, #exact) of prediction `pred` for etymon i over its reflexes."""
        p0 = sca.canon(sca.deacc_pred(pred))
        n = e = x = 0
        for (gold, _), src in zip(self.golds[i], self.gold_src[i]):
            tail = self.tails.get(src)
            p = sca.canon(sca.deacc_pred(sca.apply_rules(tail, pred))) if tail else p0
            d = sca.edit_distance(p, gold)
            m = max(len(p), len(gold)) or 1
            n += d / m
            e += d
            x += d == 0
        return n, e, x

    def tcost(self, i: int, pred: tuple[str, ...]) -> tuple[float, int]:
        """(sum tone-NED, #exact) over etymon i's reflexes from tonal sources."""
        n = x = 0
        for g, src in zip(self.tgolds[i], self.gold_src[i]):
            if g is None:
                continue
            tail = self.tails.get(src)
            q = sca.apply_rules(tail, pred) if tail else pred
            p = sca.canon(sca.tone_pred(sca.realize(q, self.accent_model)))
            d = sca.tone_edit_distance(p, g)
            n += d / (max(len(p), len(g)) or 1)
            x += d == 0
        return n, x

    def acost(self, i: int, pred: tuple[str, ...]) -> tuple[float, int]:
        """(sum accent-NED, #exact with accent) over etymon i's accent-marked reflexes."""
        p0 = sca.canon(sca.acc_pred(sca.realize(pred, self.accent_model)))
        n = x = 0
        for g, src in zip(self.agolds[i], self.gold_src[i]):
            if g is None:
                continue
            tail = self.tails.get(src)
            p = sca.canon(sca.acc_pred(sca.realize(sca.apply_rules(tail, pred), self.accent_model))) if tail else p0
            d = sca.accent_edit_distance(p, g)
            n += d / (max(len(p), len(g)) or 1)
            x += d == 0
        return n, x


# ─────────────────────────────────────────────────────────────────────────────
# Extension-aware score
# ─────────────────────────────────────────────────────────────────────────────

# Fixed menu of MIA stem extensions, added to the OIA headword (its final vowel replaced) before
# the chain runs (the variant carries the word flag `ext`, so citation rules can say `if:!ext`), so the chain's own sound changes must derive -ā/-o/-ī/-ṛā/-lā/-rā. Verbs are
# never extended. Each attested reflex is scored against its closest variant (one choice per
# reflex: several endings often coexist). No cost is charged for an extension, so this is an
# optimistic score; its placebo-tested menu is the bound on that optimism. (The plain bare-headword
# NED it used to be reported beside is deprecated, 2026-09-30: it applies every sporadic rule.)
EXT_MENU = ("aka", "akka", "ikā", "ikka", "iyaka", "uka", "aḍaka", "allaka", "araka", "akam")


def extend(word, suffix: str):
    """The headword's stem (final vowels dropped) + a noun extension; a nominal form (no `verb` flag)."""
    toks = list(word)
    while toks and sca.kind(toks[-1]) == "V":
        toks.pop()
    if not toks:
        return None
    return sca.Word(tuple(toks) + sca.tokenize(suffix), (sca.flags_of(word) - {"verb"}) | {"ext"})


# The verb menu: OIA verb formations built on the root (the headword minus a present ending
# -ati/-atē/-ayati/-ōti/-nāti/-ti, else minus its final vowel), each run through the lect's chain
# like the extensions, so the chain has to derive the lect's citation form (infinitive, verbal noun,
# participle …). Offered for every headword (a noun's stem gives the denominative's forms).
VERB_MENU = ("", "ati", "anaka", "anakam", "aṇa", "anīya", "tavya", "tavyakam", "itavya", "itavyakam", "ita", "itum", "ikka", "ami", "ayāmi", "āpanaka", "āpanakam", "āpitavyakam")
_VERB_END = re.compile(r"^(.+?)(?:ay|āy)?(?:a|ā|ē|ō|i)?t(?:i|ē)(?:aya)?$")


def verb_form(word, suffix: str):
    """The OIA verb form root + suffix (a verb form: flags `verb`, `vform`, `ext`)."""
    # strip the accent from vowels only: in NFD the acute of ś ź is the same combining mark
    raw = "".join(sca._deacc(x) if sca.kind(x) == "V" else x for x in word)
    m = _VERB_END.match(raw) if "verb" in sca.flags_of(word) else None
    if m:
        toks = list(sca.tokenize(m.group(1)))
    else:
        # not a present in -ti/-tē (a noun, a participle jita, a starred root *cammakka): drop its
        # final vowel like the noun menu does, so jita + -anaka is jitanaka, not jitaanaka (> *jitenā)
        toks = list(sca.tokenize(raw))
        while toks and sca.kind(toks[-1]) == "V":
            toks.pop()
    if len(toks) < 2:
        return None
    return sca.Word(tuple(toks) + sca.tokenize(suffix), sca.flags_of(word) | {"ext", "vform", "verb"})


def verbal_gloss(gloss: str) -> bool:
    """The reflex itself is glossed as a verb ("to burn", "to be hot")."""
    return gloss.strip().strip("'\"‘’").lower().startswith("to ")


def verb_ok(word, gloss: str) -> bool:
    """Verb formations are scored for verb etyma, and for other etyma only against reflexes glossed as
    verbs (the denominative: ārta 'hot' + -anakam > S āṭaṇu 'to grow hot')."""
    return "verb" in sca.flags_of(word) or verbal_gloss(gloss)


def menu(word, ext_extra=(), verb_extra=()):
    """[(variant name, input word)]: the bare headword, every noun extension and every verb formation."""
    vs = [("bare", word)]
    vs += [(suf, x) for suf in EXT_MENU + tuple(ext_extra) for x in [extend(word, suf)] if x is not None]
    vs += [("v:" + (suf or "root"), x) for suf in VERB_MENU + tuple(verb_extra) for x in [verb_form(word, suf)] if x is not None]
    return vs


def ext_metrics(ds: "Dataset", rules, keep: list | None = None) -> dict:
    """The score: per reflex, the minimum NED over the variants of the headword (`menu`: bare, every
    noun extension, and the verb formations, which count only for verb etyma or reflexes glossed as
    verbs, `verb_ok`), each run through the full chain, sporadic (`optional:`) rules forking.
    Also reported on the verb etyma alone ("verb_*").
    `keep`, if given, collects (etymon index, chosen variant, scored prediction, gold) per reflex."""
    names, inputs, owner = [], [], []
    for i, w in enumerate(ds.etyma):
        for name, x in menu(w, getattr(ds, "ext_extra", ()), getattr(ds, "verb_extra", ())):
            names.append(name); inputs.append(x); owner.append(i)
    outs = run_multi(rules, inputs)
    cands = [[] for _ in ds.etyma]
    for name, i, o in zip(names, owner, outs):
        cands[i].extend((name, p) for p in o)
    tot = exact = atot = ttot = 0.0
    vtot = vexact = 0.0; vn = 0
    chosen, vchosen = Counter(), Counter()
    for i, w in enumerate(ds.etyma):
        isverb = "verb" in sca.flags_of(w)
        for (gold, _), src, agold, tgold, gl in zip(ds.golds[i], ds.gold_src[i], ds.agolds[i], ds.tgolds[i], ds.gold_gloss[i]):
            tail = ds.tails.get(src)
            vok = isverb or verbal_gloss(gl)
            best = None
            for name, p in cands[i]:
                if not vok and name.startswith("v:"):
                    continue
                q = sca.canon(sca.deacc_pred(sca.apply_rules(tail, p) if tail else p))
                d = sca.edit_distance(q, gold)
                v = d / (max(len(q), len(gold)) or 1)
                if best is None or v < best[0]:
                    best = (v, name, d == 0, p, q)
            if keep is not None:
                keep.append((i, best[1], best[4], gold))
            tot += best[0]; exact += best[2]; chosen[best[1]] += 1
            if isverb:
                vtot += best[0]; vexact += best[2]; vn += 1; vchosen[best[1]] += 1
            if agold is not None:
                pa = sca.apply_rules(tail, best[3]) if tail else best[3]
                qa = sca.canon(sca.acc_pred(sca.realize(pa, ds.accent_model)))
                atot += sca.accent_edit_distance(qa, agold) / (max(len(qa), len(agold)) or 1)
            if tgold is not None:
                pt = sca.apply_rules(tail, best[3]) if tail else best[3]
                qt = sca.canon(sca.tone_pred(sca.realize(pt, ds.accent_model)))
                ttot += sca.tone_edit_distance(qt, tgold) / (max(len(qt), len(tgold)) or 1)
    N = ds.n_pairs or 1
    return {"ext_ned": tot / N, "ext_exact": exact / N, "chosen": chosen,
            "ext_aned": atot / ds.n_apairs if ds.n_apairs else 0.0,
            "ext_tned": ttot / ds.n_tpairs if getattr(ds, "n_tpairs", 0) else 0.0,
            "verb_pairs": vn, "verb_ned": vtot / vn if vn else 0.0, "verb_exact": vexact / vn if vn else 0.0,
            "verb_chosen": vchosen}


def verb_metrics(ds: "Dataset", rules) -> dict:
    """The score on the verb etyma alone (kept for old callers; ext_metrics reports it directly)."""
    x = ext_metrics(ds, rules)
    return {"pairs": x["verb_pairs"], "menu": x["verb_ned"], "menu_exact": x["verb_exact"], "chosen": x["verb_chosen"]}

def fmt_ext(m: dict) -> str:
    n = sum(m["chosen"].values()) or 1
    ch = "  ".join(f"{k} {100 * v / n:.0f}%" for k, v in m["chosen"].most_common())
    return f"extension-aware NED {m['ext_ned']:.4f}  exact {100 * m['ext_exact']:.1f}%   chosen: {ch}"


def datasets(rules_path: str, pairs: str | None = None):
    pairs = pairs or default_pairs(rules_path)
    meta = sca.load_meta(rules_path)
    tr = Dataset(pairs, "train", meta["exclude_sources"], meta["gold_drop"], meta["gold_drop_src"])
    te = Dataset(pairs, "test", meta["exclude_sources"], meta["gold_drop"], meta["gold_drop_src"])
    tr.tails = te.tails = meta["source_tails"]
    tr.accent_model = te.accent_model = ACCENT_MODEL or meta["accent"]
    tr.verb_extra = te.verb_extra = meta.get("verb_menu", ())
    tr.ext_extra = te.ext_extra = meta.get("ext_menu", ())
    return pairs, tr, te


def default_pairs(rules_path: str) -> str:
    stem = os.path.basename(rules_path).split(".")[0]
    return os.path.join(HERE, "pairs", f"{stem}.tsv")


def run(rules, etyma):
    return [sca.apply_rules(rules, w) for w in etyma]


def run_multi(rules, etyma):
    """Per etymon, every distinct output of the chain under its sporadic (`optional:`) rules."""
    if not any(r.optional for r in rules):
        return [[p] for p in run(rules, etyma)]
    return [[f for f, _ in sca.apply_rules_multi(rules, w)] for w in etyma]


def metrics(ds: Dataset, preds, accent: bool = False) -> dict:
    tn = te = tx = an = ax = on = ox = 0.0
    for i, p in enumerate(preds):
        n, e, x = ds.cost(i, p)
        tn += n; te += e; tx += x
        if accent and ds.n_apairs:
            a, y = ds.acost(i, p)
            an += a; ax += y
        if accent and getattr(ds, "n_tpairs", 0):
            a, y = ds.tcost(i, p)
            on += a; ox += y
    N, A = ds.n_pairs or 1, ds.n_apairs
    out = {"ned": tn / N, "ed": te / N, "exact": tx / N, "pairs": ds.n_pairs, "apairs": A}
    if accent and A:
        out.update(aned=an / A, aexact=ax / A)
    T = getattr(ds, "n_tpairs", 0)
    if accent and T:
        out.update(tned=on / T, texact=ox / T, tpairs=T)
    return out


def fmt(m: dict) -> str:
    s = f"NED {m['ned']:.4f}  ED {m['ed']:.3f}  exact {100 * m['exact']:.1f}%  (n={m['pairs']})"
    if "aned" in m:
        s += f"  ·  accent NED {m['aned']:.4f}  exact {100 * m['aexact']:.1f}% (n={m['apairs']})"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# score / apply
# ─────────────────────────────────────────────────────────────────────────────

def cmd_score(args):
    rules = sca.load_rules(args.rules)
    pairs, train, test = datasets(args.rules, args.pairs)
    print(f"{len(rules)} rules  ·  {pairs}")
    for name, ds in (("train", train), ("test", test)):
        print(f"  {name:5s} null   {fmt_ext(ext_metrics(ds, []))}")
        print(f"  {name:5s} rules  {fmt_ext(ext_metrics(ds, rules))}")
        if args.verbs:
            vm = verb_metrics(ds, rules)
            ch = "  ".join(f"{k} {v}" for k, v in vm["chosen"].most_common(6))
            print(f"  {name:5s} verbs  n={vm['pairs']}  NED on verb etyma {vm['menu']:.4f}   chosen: {ch}")

    if args.ext_loo or args.loo:
        base = ext_metrics(train, rules)["ext_ned"]
        print("\nper-rule value on train, extension-aware NED (increase when the rule is removed)")
        for k, r in enumerate(rules):
            loo = ext_metrics(train, rules[:k] + rules[k + 1:])["ext_ned"]
            print(f"  {loo - base:+.4f}  L{r.line:<4d} {r.text}", flush=True)

    if args.errors:
        ds = test if args.errors_split == "test" else train
        keep = []
        ext_metrics(ds, rules, keep=keep)   # the prediction the extension-aware score picks, per reflex
        raws = {(i, g): raw for i in range(len(ds.etyma)) for g, raw in ds.golds[i]}
        rows = []
        for i, _, q, gold in keep:
            d = sca.edit_distance(q, gold)
            if d:
                rows.append((d, ds.raw[i], "".join(q), raws.get((i, gold), "".join(gold))))
        rows.sort(key=lambda r: -r[0])
        print(f"\nworst {args.errors} {args.errors_split} misses (ED · etymon → predicted | attested)")
        for d, e, p, g in rows[:args.errors]:
            print(f"  {d}  {e} → {p} | {g}")
        # most frequent residual segment confusions, to point at the next rule to write
        conf = Counter()
        for i, _, q, gold in keep:
            for a, b in ops(q, gold):
                conf[(a or "∅", b or "∅")] += 1
        print(f"\ntop residual confusions ({args.errors_split}; predicted → attested)")
        print("  " + "  ".join(f"{a}→{b}:{n}" for (a, b), n in conf.most_common(args.errors)))


def cmd_apply(args):
    rules, tails = sca.load_rules_full(args.rules)
    for form in args.forms:
        toks = sca.tokenize(form)
        fl = set(args.flags.split(",")) if args.flags else word_flags(form, toks, [])
        w = sca.Word(toks, fl)
        out, steps = sca.apply_rules(rules, w, trace=True)
        print(f"{form}  [{' '.join(w)}]  flags: {','.join(sorted(fl)) or '-'}")
        for r, new in steps:
            print(f"  → {''.join(new):20s} L{r.line:<4d} {r.text}")
        print(f"  = {''.join(out)}")
        for src, tail in tails.items():
            print(f"    [{src}] {''.join(sca.apply_rules(tail, out))}")


# ─────────────────────────────────────────────────────────────────────────────
# frontier: greedy backward elimination
# ─────────────────────────────────────────────────────────────────────────────

def cmd_frontier(args):
    rules = sca.load_rules(args.rules)
    pairs, train, test = datasets(args.rules, args.pairs)
    out_rows = []

    def record(cur, removed, fires=""):
        mtr = metrics(train, run(cur, train.etyma), accent=True)
        mte = metrics(test, run(cur, test.etyma), accent=True)
        out_rows.append({"n_rules": len(cur), "train_ned": mtr["ned"], "test_ned": mte["ned"],
                         "train_exact": mtr["exact"], "test_exact": mte["exact"],
                         "test_ed": mte["ed"], "removed": removed, "removed_fires": fires,
                         "train_aned": mtr.get("aned", ""), "test_aned": mte.get("aned", ""),
                         "test_apairs": mte["apairs"]})
        print(f"  {len(cur):3d} rules  train {mtr['ned']:.4f}  test {mte['ned']:.4f}  "
              f"exact {100 * mte['exact']:.1f}%  {('− ' + removed) if removed else ''}", flush=True)

    cur = list(rules)
    record(cur, "")
    t0 = time.time()
    while cur:
        # prefix cache: outputs after the first k rules, so dropping rule k reruns only the tail
        prefix = [list(train.etyma)]
        for r in cur:
            prefix.append([sca.apply_rule_full(r, w) for w in prefix[-1]])
        # dropping rule k can only change the words rule k changed: rerun the tail for those alone
        # (per-word costs summed in index order, exactly as metrics() does, so ties resolve alike)
        base_cost = [train.cost(i, w)[0] for i, w in enumerate(prefix[-1])]
        N = train.n_pairs or 1
        best = None
        for k in range(len(cur)):
            costs = base_cost
            touched = [i for i, (a, b) in enumerate(zip(prefix[k], prefix[k + 1])) if a != b]
            if touched:
                costs = list(base_cost)
                for i in touched:
                    w = prefix[k][i]
                    for r in cur[k + 1:]:
                        w = sca.apply_rule_full(r, w)
                    costs[i] = train.cost(i, w)[0]
            tn = 0.0
            for c in costs:
                tn += c
            ned = tn / N
            # ties: drop the later rule (keeps earlier, usually more general ones)
            if best is None or ned <= best[0]:
                best = (ned, k)
        fires = sum(a != b for a, b in zip(prefix[best[1]], prefix[best[1] + 1]))
        removed = cur.pop(best[1])
        record(cur, f"L{removed.line} {removed.text}", fires)
    print(f"[frontier] {time.time() - t0:.0f}s", file=sys.stderr)
    write_csv(args.out, out_rows)


def write_csv(path, rows):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}", file=sys.stderr)


# ─────────────────────────────────────────────────────────────────────────────
# induce: greedy automatic baseline
# ─────────────────────────────────────────────────────────────────────────────

def ops_with_pos(p: tuple[str, ...], g: tuple[str, ...]):
    """Levenshtein edit script from p to g: (kind, p_index, p_seg, g_seg); insertions carry the
    gap index (insert before p[i])."""
    n, m = len(p), len(g)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (p[i - 1] != g[j - 1]))
    out = []
    i, j = n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (p[i - 1] != g[j - 1]):
            if p[i - 1] != g[j - 1]:
                out.append(("sub", i - 1, p[i - 1], g[j - 1]))
            i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            out.append(("del", i - 1, p[i - 1], None)); i -= 1
        else:
            out.append(("ins", i, None, g[j - 1])); j -= 1
    return out


def ops(p, g):
    return [(a, b) for _, _, a, b in ops_with_pos(p, g)]


def ctx_specs(seg: str):
    """Ways to describe a neighbouring segment in an environment (None = unconstrained)."""
    if seg == "#":
        return [None, "#"]
    return [None, seg, sca.kind(seg)]


def rule_text(tgt, rep, left, right) -> str:
    lhs = tgt if tgt else "∅"
    rhs = rep if rep else "∅"
    env = ""
    if left or right:
        env = f" / {left or ''} _ {right or ''}".replace("  ", " ")
    return f"{lhs} > {rhs}{env}"


def propose(train: Dataset, cur, try_rule, n_cands: int, nonlocal_ctx: bool = True):
    """Candidate single-change rules for the residual between `cur` (current outputs) and the
    attested forms, scored by `try_rule(text) -> (gain, …)`; returns [(gain, text)] best first.

    Candidates are the `n_cands` most frequent Levenshtein edits, each optionally conditioned on
    a neighbouring segment, V/C, or `#`. Same-environment winners are also offered merged into
    one set rule, but only when the group is phonologically coherent — one shared output over
    segments of one kind (V or C), or one align.describe change category — so a "rule" never
    smuggles unrelated changes past the rule-count axis."""
    support = Counter()
    for i, w in enumerate(cur):
        wp = ("#",) + tuple(w) + ("#",)
        vpos = [k for k, x in enumerate(w) if sca.kind(x) == "V"]
        for gold, _ in train.golds[i]:
            # positions index the uncanonicalised prediction so rules can target `ai`/`au`
            for kind_, pi, a, b in ops_with_pos(w, gold):
                left, right = wp[pi], wp[pi + 2] if kind_ != "ins" else wp[pi + 1]
                for L in ctx_specs(left):
                    for R in ctx_specs(right):
                        if kind_ == "ins" and not (L or R):
                            continue
                        support[(a or "", b or "", L, R)] += 1
                if kind_ != "ins" and sca.kind(a) == "V" and nonlocal_ctx:
                    # syllable-level contexts the adjacent-segment specs cannot express
                    first, last = pi == vpos[0], pi == vpos[-1]
                    nxt = next((w[k] for k in vpos if k > pi), None)
                    if first:
                        support[(a, b or "", "# C*", None)] += 1
                    if last:
                        support[(a, b or "", None, "C* #")] += 1
                    if nxt:
                        support[(a, b or "", None, f"C* {nxt}")] += 1
                        if first:
                            support[(a, b or "", "# C*", f"C* {nxt}")] += 1
    scored = []
    for tgt, rep, L, R in (c for c, _ in support.most_common(n_cands)):
        text = rule_text(tgt, rep, L, R)
        scored.append((try_rule(text)[0], text, (tgt, rep, L, R)))
    groups = defaultdict(list)
    for g, text, (tgt, rep, L, R) in scored:
        if g > 0 and tgt:
            groups[(L, R, "out", rep, sca.kind(tgt))].append((g, tgt, rep))
            if rep:
                cat = describe(Seg(tgt), Seg(rep))
                if cat not in ("vowel", "cons", "place"):
                    groups[(L, R, "cat", sca.kind(tgt) + cat, "")].append((g, tgt, rep))
    for (L, R, *_), members in groups.items():
        seen, uniq = set(), []
        for g, tgt, rep in sorted(members, reverse=True):
            if tgt not in seen:
                seen.add(tgt); uniq.append((tgt, rep))
        if len(uniq) < 2:
            continue
        lhs = "{" + " ".join(t for t, _ in uniq) + "}"
        reps = {r for _, r in uniq}
        rhs = (reps.pop() or "∅") if len(reps) == 1 else "{" + " ".join(r for _, r in uniq) + "}"
        text = " ".join((f"{lhs} > {rhs}" + (f" / {L or ''} _ {R or ''}" if (L or R) else "")).split())
        scored.append((try_rule(text)[0], text, None))
    scored.sort(key=lambda s: -s[0])
    return [(g, t) for g, t, _ in scored]


def cmd_induce(args):
    pairs = args.pairs
    train, test = Dataset(pairs, "train"), Dataset(pairs, "test")
    rules: list[sca.Rule] = []
    cur = list(train.etyma)
    cur_test = list(test.etyma)
    cost = [train.cost(i, w)[0] for i, w in enumerate(cur)]
    curve = []

    def record(text):
        mtr = metrics(train, cur)
        mte = metrics(test, cur_test)
        curve.append({"n_rules": len(rules), "train_ned": mtr["ned"], "test_ned": mte["ned"],
                      "train_exact": mtr["exact"], "test_exact": mte["exact"],
                      "test_ed": mte["ed"], "added": text})
        print(f"  {len(rules):3d} rules  train {mtr['ned']:.4f}  test {mte['ned']:.4f}  "
              f"exact {100 * mte['exact']:.1f}%  {('+ ' + text) if text else ''}", flush=True)

    def try_rule(text):
        rule = sca.parse_rule(text, 0, {})
        gain, changed = 0.0, []
        for i, w in enumerate(cur):
            new = sca.apply_rule_full(rule, w)
            if new is not w and new != w:
                c = train.cost(i, new)[0]
                gain += cost[i] - c
                changed.append((i, new, c))
        return gain, rule, changed

    record("")
    t0 = time.time()
    while len(rules) < args.max_rules:
        scored = propose(train, cur, try_rule, args.candidates)
        if not scored:
            break
        best_gain, best_text = scored[0]
        if best_gain < max(args.min_gain, args.min_gain_frac * train.n_pairs):
            break
        gain, rule, changed = try_rule(best_text)
        for i, new, c in changed:
            cur[i] = new
            cost[i] = c
        rules.append(rule)
        cur_test = [sca.apply_rule_full(rule, w) for w in cur_test]
        record(best_text)
    print(f"[induce] {len(rules)} rules in {time.time() - t0:.0f}s", file=sys.stderr)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# Greedy automatic baseline induced by evaluate.py induce from {os.path.basename(pairs)}\n"
                f"# (train split only; each rule is the single edit that most reduced train NED)\n\n")
        for r in rules:
            f.write(r.text + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    write_csv(args.curve, curve)


def cmd_suggest(args):
    """Hill-climbing aid: the residual edits that would most reduce TRAIN NED if appended as one
    more rule, and for the best few, the chain position where inserting them helps most."""
    rules = sca.load_rules(args.rules)
    pairs, train, _ = datasets(args.rules, args.pairs)
    cur = run(rules, train.etyma)
    cost = [train.cost(i, w)[0] for i, w in enumerate(cur)]
    N = train.n_pairs or 1

    def try_rule(text):
        rule = sca.parse_rule(text, 0, {})
        gain = 0.0
        for i, w in enumerate(cur):
            new = sca.apply_rule_full(rule, w)
            if new != w:
                gain += cost[i] - train.cost(i, new)[0]
        return gain, rule

    base = sum(cost) / N
    print(f"train NED {base:.4f} with {len(rules)} rules; candidates (ΔNED if appended at the end):")
    scored = [x for x in propose(train, cur, try_rule, args.candidates) if x[0] > 0][:args.top]
    prefix = [list(train.etyma)]
    for r in rules:
        prefix.append([sca.apply_rule_full(r, w) for w in prefix[-1]])
    for rank, (g, text) in enumerate(scored):
        line = f"  {-g / N:+.4f}  {text}"
        if rank < args.positions:
            cand = sca.parse_rule(text, 0, {})
            best = (base - g / N, len(rules))
            for k in range(len(rules)):
                state = [sca.apply_rule_full(cand, w) for w in prefix[k]]
                for r in rules[k:]:
                    state = [sca.apply_rule_full(r, w) for w in state]
                ned = metrics(train, state)["ned"]
                if ned < best[0] - 1e-9:
                    best = (ned, k)
            if best[1] < len(rules):
                line += f"    [best before L{rules[best[1]].line}: {best[0] - base:+.4f}]"
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# probe: vowel outcomes by syllable position, Vedic accent, next vowel and weight
# ─────────────────────────────────────────────────────────────────────────────

PROBE_FEATS = ("pos", "acc", "next", "syl", "nextσ")


def vowel_contexts(w) -> dict[int, dict[str, str]]:
    """Context features of each vowel of an etymon (read with its Vedic accent)."""
    vpos = [k for k, x in enumerate(w) if sca.kind(x) == "V"]
    acc = next((k for k in vpos if sca.feat(w[k], "Acc")), None)
    info = {}
    for n_, k in enumerate(vpos):
        nxt = vpos[n_ + 1] if n_ + 1 < len(vpos) else None
        if len(vpos) == 1:
            pos = "mono"
        elif n_ == 0:
            pos = "first"
        elif nxt is None:
            pos = "last"
        else:
            pos = "medial"
        if acc is None:
            accf = "unaccented"
        elif k == acc:
            accf = "tonic"
        else:
            accf = "pretonic" if k < acc else "posttonic"
        after = (nxt - k - 1) if nxt is not None else (len(w) - k - 1)
        closed = after >= 2 or (nxt is None and after >= 1)
        if nxt is None:
            nw = "-"
        else:
            nn = vpos[n_ + 2] if n_ + 2 < len(vpos) else None
            c2 = (nn - nxt - 1) if nn is not None else (len(w) - nxt - 1)
            heavy = sca.feat(w[nxt], "Long") or c2 >= 2 or (nn is None and c2 >= 1)
            nw = "heavy" if heavy else "light"
        info[k] = {"vowel": sca._deacc(w[k]), "pos": pos, "acc": accf,
                   "next": sca._deacc(w[nxt]) if nxt is not None else "#",
                   "syl": "closed" if closed else "open", "nextσ": nw}
    return info


def _aligned(w, target):
    """etymon index → aligned target segment (deaccented) or ∅, via align.py's weighted NW."""
    from align import align as nw_align
    es = [Seg(x) for x in w]
    for k, s_ in enumerate(es):
        s_.idx = k
    out = {}
    for a, b in nw_align(es, [Seg(x) for x in target]):
        if a is not None:
            out[a.idx] = sca._deacc(b.raw) if b is not None else "∅"
    return out


def _env_for(feats: dict[str, str]) -> tuple[str, str] | None:
    """(if-prefix + target note, environment) for a feature cell, or None if not expressible."""
    left, right, prefix, tonic = [], [], "", False
    for f, v in feats.items():
        if f == "pos":
            if v == "first":
                left.append("# C*")
            elif v == "last":
                right.append("C* #")
            elif v == "mono":
                left.append("# C*"); right.append("C* #")
            else:
                return None
        elif f == "acc":
            if v == "tonic":
                tonic = True
            elif v == "pretonic":
                right.append("!#* Acc")
            elif v == "posttonic":
                left.append("Acc !#*")
            else:
                prefix = "if:!accented "
        elif f == "next":
            if v == "#":
                right.append("C* #")
            else:
                right.append("C* {" + " ".join(dict.fromkeys([v, sca.modify(v, (("+", "acc"),))])) + "}")
        elif f == "syl":
            right.append("C C" if v == "closed" else None)
        else:  # nextσ: not expressible as a single environment
            return None
    if None in right or len(left) > 1 or len(right) > 1:
        return None
    env = f"{left[0] if left else ''} _ {right[0] if right else ''}".strip()
    return prefix, env if env != "_" else "", tonic


def cmd_probe(args):
    rules = sca.load_rules(args.rules)
    _, train, _ = datasets(args.rules, args.pairs)
    preds = run(rules, train.etyma)
    cells = defaultdict(lambda: {"gold": Counter(), "pred": Counter(), "joint": Counter()})
    n_v = Counter()
    ok_v = Counter()
    for i, w in enumerate(train.etyma):
        ctx = vowel_contexts(w)
        if not ctx:
            continue
        p_out = _aligned(w, sca.canon(preds[i]))
        for gold, _ in train.golds[i]:
            g_out = _aligned(w, gold)
            for k, fe in ctx.items():
                g, p = g_out.get(k, "∅"), p_out.get(k, "∅")
                v = fe["vowel"]
                if args.vowel and v != args.vowel:
                    continue
                n_v[v] += 1
                ok_v[v] += g == p
                keys = [((f, fe[f]),) for f in PROBE_FEATS]
                keys += [((f1, fe[f1]), (f2, fe[f2])) for a_, f1 in enumerate(PROBE_FEATS)
                         for f2 in PROBE_FEATS[a_ + 1:]]
                for key in keys:
                    c = cells[(v, key)]
                    c["gold"][g] += 1
                    c["pred"][p] += 1
                    c["joint"][(g, p)] += 1
    print(f"vowel outcomes on train ({args.rules}); correct = the chain's aligned vowel equals the "
          f"attested one")
    print("  " + "  ".join(f"{v}: {100 * ok_v[v] / n_v[v]:.0f}% of {n_v[v]}"
                           for v, _ in n_v.most_common(10)))
    rows = []
    for (v, key), c in cells.items():
        n = sum(c["gold"].values())
        if n < args.min:
            continue
        m, mc = c["gold"].most_common(1)[0]
        gains = sum(cnt for (g, p), cnt in c["joint"].items() if g == m and p != m)
        losses = sum(cnt for (g, p), cnt in c["joint"].items() if g == p and g != m)
        net = gains - losses
        if net <= 0:
            continue
        rows.append((net, gains, losses, n, v, key, m, c))
    rows.sort(key=lambda r: -r[0])
    print(f"\ncells where one outcome dominates the attested forms but the chain predicts something "
          f"else (net = items fixed − items broken if every vowel in the cell became the modal "
          f"outcome; single features and pairs):")
    shown = 0
    seen_rules, seen_sig = set(), set()
    for net, gains, losses, n, v, key, m, c in rows:
        if (v, n, gains, losses) in seen_sig:
            continue  # a pair cell identical to a simpler cell already shown
        seen_sig.add((v, n, gains, losses))
        feats = dict(key)
        spec = _env_for(feats)
        rule = ""
        if spec:
            prefix, env, tonic = spec
            tgt_set = [sca.modify(v, (("+", "acc"),))] if tonic else \
                list(dict.fromkeys([v, sca.modify(v, (("+", "acc"),))]))
            tgt = tgt_set[0] if len(tgt_set) == 1 else "{" + " ".join(tgt_set) + "}"
            rep_ = "∅" if m == "∅" else m
            rule = f"{prefix}{tgt} > {rep_}" + (f" / {env}" if env else "")
            if rule in seen_rules:
                continue
            seen_rules.add(rule)
        gold_s = " ".join(f"{o}:{100 * k / n:.0f}%" for o, k in c["gold"].most_common(3))
        pred_s = " ".join(f"{o}:{100 * k / n:.0f}%" for o, k in c["pred"].most_common(2))
        cell = ", ".join(f"{f}={val}" for f, val in key)
        print(f"  +{net:<4d} ({gains}/-{losses}) {v:3s} {cell:36s} n={n:<4d} attested {gold_s:24s} "
              f"chain {pred_s:16s} {('→ ' + rule) if rule else ''}")
        shown += 1
        if shown >= args.top:
            break
    print("\nContexts are read on the OIA headword, so a suggested rule belongs early in the chain "
          "— before @acc-loss for acc= cells, and before rules that change the conditioning "
          "vowels or clusters. Test each with `score --loo` / `suggest` before keeping it.")


# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("rules")
    s.add_argument("--pairs")
    s.add_argument("--loo", action="store_true", help="per-rule leave-one-out on train extension-aware NED (slow)")
    s.add_argument("--errors", type=int, default=25)
    s.add_argument("--errors-split", default="train", choices=["train", "test"])
    s.add_argument("--ext", action="store_true", help="no-op (the extension-aware NED is the score)")
    s.add_argument("--verbs", action="store_true", help="the score on verb etyma alone, with chosen forms")
    s.add_argument("--ext-loo", action="store_true", help="same as --loo")
    s.add_argument("--accent-model", choices=["chain", "vedic", "vedic-only"],
                   help="accent score: marks the chain writes (default) or the Vedic accent carried over")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("apply")
    s.add_argument("rules")
    s.add_argument("forms", nargs="+")
    s.add_argument("--flags", help="comma-separated word flags (default: derived from the form)")
    s.set_defaults(fn=cmd_apply)

    s = sub.add_parser("frontier")
    s.add_argument("rules")
    s.add_argument("--pairs")
    s.add_argument("--out")
    s.add_argument("--accent-model", choices=["chain", "vedic", "vedic-only"])
    s.set_defaults(fn=cmd_frontier)

    s = sub.add_parser("suggest")
    s.add_argument("rules")
    s.add_argument("--pairs")
    s.add_argument("--top", type=int, default=15)
    s.add_argument("--positions", type=int, default=5, help="search insertion point for the best N")
    s.add_argument("--candidates", type=int, default=150)
    s.set_defaults(fn=cmd_suggest)

    s = sub.add_parser("probe")
    s.add_argument("rules")
    s.add_argument("--pairs")
    s.add_argument("--min", type=int, default=12, help="minimum cell size")
    s.add_argument("--top", type=int, default=25)
    s.add_argument("--vowel", help="only this OIA vowel (deaccented), e.g. a")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("induce")
    s.add_argument("--pairs", default=os.path.join(HERE, "pairs", "OIA-Pk.tsv"))
    s.add_argument("--out", required=True)
    s.add_argument("--curve")
    s.add_argument("--max-rules", type=int, default=60)
    s.add_argument("--candidates", type=int, default=150)
    s.add_argument("--min-gain", type=float, default=0.5, help="stop below this train NED-sum gain…")
    s.add_argument("--min-gain-frac", type=float, default=0.0005,
                   help="…or below this fraction of #train pairs, whichever is larger")
    s.set_defaults(fn=cmd_induce)

    args = ap.parse_args()
    global ACCENT_MODEL
    ACCENT_MODEL = getattr(args, "accent_model", None)
    args.fn(args)


if __name__ == "__main__":
    main()

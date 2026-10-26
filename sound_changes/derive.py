#!/usr/bin/env python3
"""derive.py — rule-based derivations for every Indo-Aryan reflex (replaces align.py's naive
Needleman–Wunsch alignment and change codes for forms whose parent is a CDIAL headword).

    uv run python sound_changes/derive.py [--workers 3] [--only LANG …]     (from data/, after align.py)

For each form whose rank-1 chain reaches an Indo-Aryan headword through a `reflex` edge (loans are
skipped, as in extract.py), the lect's rule chain (rules/OIA-<File>.sc, a lect of a family file rules/<group>.sc; dialects without one borrow
a sister chain from chain_borrow.tsv) is run on the headword. The variant is the one the
extension-aware score picks for that reflex: the bare headword, a noun extension (EXT_MENU, full
rules) or, for verbs, a verb formation (VERB_MENU, every @morph rule and @morph source tail off).

Outputs (cldf/):
  alignments.csv    the Indo-Aryan rows are replaced: one column per segment, composed from the
                    derivation (each rule's before → after diff), so every column knows which rules
                    changed it. Change = "kept", or the rule refs that changed the column
                    ("<Chain>:<line>", "+aka"/"v:anaka" for the variant step), plus "?" when the
                    chain's prediction and the attested form still differ there (unexplained).
  derivations.csv   per form: chain, variant ("d:anakam" = a verb formation on a noun headword,
                    offered only when the reflex is glossed as a verb), etymon, the trace (rule ref → form after it),
                    prediction, attested (scoring tokens) and NED.
  sound_rules.csv   per chain rule: order, text, central name, change identity (synth.change_id),
                    tags, note, and how many derivations it fires in (and how many of those end exact).
  sound_chains.csv  per lect: the chain it uses (own or borrowed) and its derivation scores.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, DATA)
import sca  # noqa: E402
import evaluate as ev  # noqa: E402
from synth import change_id, TAG_RE  # noqa: E402
from align import Seg, align as nw_align  # noqa: E402

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
PARENT = "Indo-Aryan"
CACHE = os.path.join(os.path.dirname(DATA), "tmp", "sound_changes", "derive_cache")
ALIGN_HEADER = ["Form_ID", "Origin_ID", "Pos", "Etymon_Idx", "Etymon_Seg", "Reflex_Seg", "Change",
                "Prev_Seg", "Next_Seg"]


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────

def chain_map():
    """Lang → (chain Lang, File, borrowed?, note)."""
    own = {r["Lang"]: r["File"] for r in csv.DictReader(open(os.path.join(HERE, "languages.tsv"), encoding="utf-8"), delimiter="\t")}
    out = {l: (l, f, False, "") for l, f in own.items() if sca.has_chain(rules_path(f))}
    for r in csv.DictReader(open(os.path.join(HERE, "chain_borrow.tsv"), encoding="utf-8"), delimiter="\t"):
        c = r["Chain"]
        if c in out and r["Lang"] not in out:
            out[r["Lang"]] = (c, out[c][1], True, r["Note"])
    return out


def rules_path(stem):
    return os.path.join(HERE, "rules", f"OIA-{stem}.sc")


def load_inputs(forms_path, edges_path):
    """Forms whose rank-1 chain reaches an Indo-Aryan headword through a reflex edge."""
    info, lang_of = {}, {}
    with open(forms_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang_of[row["ID"]] = row["Language_ID"]
            info[row["ID"]] = (row["Form"], row["Gloss"], row["Source"].split(";")[0].split("[")[0])
    rank1 = {}
    with open(edges_path, encoding="utf-8") as f:
        for e in csv.DictReader(f):
            if e["Rank"] == "1" and e["Kind"] in ("reflex", "borrowed", "variant"):
                rank1[e["Child_ID"]] = (e["Parent_ID"], e["Kind"])

    def etymon_of(node):
        seen, cur = set(), node
        while cur not in seen:
            seen.add(cur)
            e = rank1.get(cur)
            if e is None:
                return None
            parent, kind = e
            if kind == "borrowed":
                return None
            if kind == "reflex":
                return parent if lang_of.get(parent) == PARENT else None
            cur = parent
        return None

    out = []
    for fid, (form, gloss, src) in info.items():
        lang = lang_of[fid]
        if lang == PARENT or not form.strip() or "°" in form:
            continue
        eid = etymon_of(fid)
        if eid and info[eid][0].strip():
            out.append((fid, lang, eid, info[eid][0], gloss or info[eid][1], form, src))
    return out, lang_of


# ─────────────────────────────────────────────────────────────────────────────
# Derivation of one chain's forms
# ─────────────────────────────────────────────────────────────────────────────

def variants_for(word, rules, vextra=(), eextra=()):
    """[(variant name, input word, rules)] as ext_metrics builds them (evaluate.menu; the full chain)."""
    return [("" if n == "bare" else (n if n.startswith("v:") else "+" + n), x, rules)
            for n, x in ev.menu(word, eextra, vextra)]


def score_tokens(toks):
    return sca.canon(sca.deacc_pred(sca.realize(toks, "chain")))


def ned(p, g):
    return sca.edit_distance(p, g) / (max(len(p), len(g)) or 1)


class Columns:
    """Segment columns composed across a derivation: each column is one segment's history."""

    def __init__(self, toks):
        self.cols = [{"ety": t, "cur": t, "by": []} for t in toks]

    def live(self):
        return [c for c in self.cols if c["cur"] is not None]

    @staticmethod
    def opcodes(before, after):
        """difflib opcodes, with a lost copy of a doubled segment taken from the right of the run
        (tt > t loses the second t, so the first keeps its history)."""
        ops = difflib.SequenceMatcher(a=before, b=after, autojunk=False).get_opcodes()
        out = []
        for tag, i1, i2, j1, j2 in ops:
            if tag == "delete" and i2 - i1 == 1:
                k = i1
                while k + 1 < len(before) and before[k + 1] == before[i1]:
                    k += 1
                if k != i1:   # delete the last of the run; the segments in between shift left
                    out.append(("delete", k, k + 1, j1 + (k - i1), j1 + (k - i1)))
                    continue
            out.append((tag, i1, i2, j1, j2))
        # re-sort (a shifted delete may now follow an equal block that covers it: harmless, since
        # step() only reads non-equal ops and indexes `live`/`after` directly)
        return out

    def prefix_step(self, after, ref):
        """The variant step: keep the longest common prefix, replace the rest (ghōṭa → ghōṭ+aka)."""
        live = self.live()
        before = [c["cur"] for c in live]
        k = 0
        while k < min(len(before), len(after)) and before[k] == after[k]:
            k += 1
        for c in live[k:]:
            c["cur"] = None
            c["by"].append(ref)
        new = [{"ety": None, "cur": v, "by": [ref]} for v in after[k:]]
        if live[k:]:
            at = self.cols.index(live[-1]) + 1   # after the replaced tail (ghōṭ[a] + aka)
            self.cols[at:at] = new
        else:
            self.cols.extend(new)

    def step(self, after, ref):
        live = self.live()
        before = [c["cur"] for c in live]
        after = list(after)
        if before == after:
            return
        ins = defaultdict(list)
        for tag, i1, i2, j1, j2 in self.opcodes(before, after):
            if tag == "equal":
                continue
            n = min(i2 - i1, j2 - j1)
            for t in range(n):
                c = live[i1 + t]
                if c["cur"] != after[j1 + t]:
                    c["cur"] = after[j1 + t]
                    c["by"].append(ref)
            for t in range(i1 + n, i2):
                live[t]["cur"] = None
                live[t]["by"].append(ref)
            for v in after[j1 + n:j2]:
                ins[i2].append({"ety": None, "cur": v, "by": [ref]})
        ids = {id(c): k for k, c in enumerate(live)}
        out = []
        for c in self.cols:
            k = ids.get(id(c))
            if k is not None:
                out.extend(ins.pop(k, []))
            out.append(c)
        out.extend(ins.pop(len(live), []))
        self.cols = out


def derive_chain(task):
    """All forms of one chain → (rows per form, rule table, per-rule counts)."""
    chain, stem, items = task
    path = rules_path(stem)
    rules, tails = sca.load_rules_full(path)
    meta = sca.load_meta(path)
    rid = lambda r: f"{chain}:{r.line}" + (f".{r.origin}" if r.origin else "")   # included rules share the include line
    ref = {id(r): rid(r) for r in rules}
    ref.update({id(a): rid(r) for r in rules for a in r.alts})   # an outcome of `X > A | B` traces as its rule
    for t in tails.values():
        for r in t:
            ref[id(r)] = rid(r)

    glosses = defaultdict(list)
    for fid, lang, eid, ety, gloss, form, src in items:
        glosses[(eid, ety)].append(gloss)
    words = {}
    for key, gl in glosses.items():
        toks = sca.tokenize(key[1])
        if toks:
            words[key] = sca.Word(toks, ev.word_flags(key[1], toks, gl))
    preds = {}   # (eid, ety) -> [(name, input word, rules, prediction)]
    for key, w in words.items():
        vextra = meta.get("verb_menu", ())
        # sporadic (`optional:`) rules fork a derivation; each fork keeps the rules it underwent, so the
        # trace below follows exactly the branch that matched the attested form
        forks = lambda rs, x: [(sca.rules_taken(rs, sk), f) for f, sk in sca.apply_rules_multi(rs, x)]
        preds[key] = [(n, x, used, f) for n, x, rs in variants_for(w, rules, vextra, meta.get("ext_menu", ()))
                      for used, f in forks(rs, x)]

    out = []
    fired = Counter()
    fired_ok = Counter()
    for fid, lang, eid, ety, gloss, form, src in items:
        key = (eid, ety)
        if key not in words:
            continue
        gold = sca.gold_tokens(form, meta["gold_drop"] | meta["gold_drop_src"].get(src, frozenset()))
        if not gold:
            continue
        tail = tails.get(src) or []
        vok = ev.verb_ok(words[key], gloss)
        best = None
        for n, x, rs, p in preds[key]:
            if not vok and n.startswith("v:"):
                continue
            q = score_tokens(sca.apply_rules(tail, p) if tail else p)
            v = ned(q, gold)
            if best is None or v < best[0] - 1e-12:
                best = (v, n, x, rs)
        v, name, x, rs = best
        # the trace: the variant step, the chain's rules, the source tail
        head = score_tokens(words[key])
        cols = Columns(head)
        steps = []
        if name:
            cols.prefix_step(score_tokens(x), name)
            steps.append([name, list(score_tokens(x))])
        cur = x
        for r in list(rs) + list(tail):
            new = sca.apply_rule_full(r, cur)
            if new != cur:
                a, b = score_tokens(cur), score_tokens(new)
                if a != b:
                    cols.step(b, ref[id(r)])
                    steps.append([ref[id(r)], list(b)])
                    fired[ref[id(r)]] += 1
                    if v == 0:
                        fired_ok[ref[id(r)]] += 1
            cur = new
        pred = score_tokens(cur)
        # residual: the prediction against the attested form
        live = cols.live()
        pairs = nw_align([Seg(t) for t in pred], [Seg(t) for t in gold]) if pred and gold else []
        k = 0
        resid = []   # per aligned pair: (live column or None, gold seg or None)
        for a, b in pairs:
            if a is not None:
                resid.append((live[k], b.raw if b else None))
                k += 1
            else:
                resid.append((None, b.raw))
        final = {}
        extra = defaultdict(list)   # attested-only segments inserted after a column (id) / at start
        last = None
        for c, g in resid:
            if c is None:
                extra[id(last) if last else None].append(g)
            else:
                final[id(c)] = g
                last = c
        rows = []

        def emit(ety_seg, ety_idx, g, by, unexplained):
            ch = " ".join(dict.fromkeys(by))
            if unexplained:
                ch = (ch + " ?").strip()
            rows.append((ety_seg, ety_idx, g, ch or "kept"))

        for g in extra.pop(None, []):
            emit(None, -1, g, [], True)
        ei = 0
        for c in cols.cols:
            idx = -1
            if c["ety"] is not None:
                idx = ei
                ei += 1
            if c["cur"] is None:
                if c["ety"] is not None:   # added by a step and lost again: nothing to show
                    emit(c["ety"], idx, None, c["by"], False)
            else:
                g = final.get(id(c))
                if c["ety"] is not None or g is not None:   # an added segment the reflex lacks: no cell
                    emit(c["ety"], idx, g, c["by"], g != c["cur"])
            for g2 in extra.pop(id(c), []):
                emit(None, -1, g2, [], True)
        out.append({"form": fid, "lang": lang, "origin": eid, "variant": name, "etymon": list(head),
                    "steps": steps, "pred": list(pred), "gold": list(gold), "ned": round(v, 4),
                    "cols": rows})

    # rule table (chain order), notes from the file's comments
    lines = sca.source_lines(path)
    defs = sca.central()
    table = []
    tail_src = {id(r): s for s, t in tails.items() for r in t}
    seen = set()
    order = 0
    for r in list(rules) + [r for t in tails.values() for r in t]:
        rf = ref[id(r)]
        if rf in seen:
            continue
        seen.add(rf)
        order += 1
        raw = sca.raw_line(r, lines)
        comment = raw.split(";", 1)[1] if ";" in raw else ""
        if r.name and r.name in defs:
            comment = defs[r.name][1] + " " + comment
        note = re.sub(TAG_RE, "", comment).strip(" —-:")
        note = re.sub(r"^\d+ lects?\s*(—\s*)?", "", note).strip(" —-:")   # central file bookkeeping
        table.append({"Rule_Ref": rf, "Chain": chain, "Order": order, "Line": r.line,
                      "Source": tail_src.get(id(r), ""), "Text": r.text, "Name": r.name,
                      "Change_ID": change_id(r), "Tags": " ".join(sorted(set(r.tags))), "Note": note,
                      "Forms": fired[rf], "Forms_Exact": fired_ok[rf]})
    return chain, out, table


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def encode_steps(start, steps):
    """The trace as one edit per step against the previous form: `ref pos ndel seg…` joined by
    ` ; ` (segments are space-separated tokens; the first form is the etymon)."""
    out, cur = [], list(start)
    for ref, after in steps:
        i = 0
        while i < min(len(cur), len(after)) and cur[i] == after[i]:
            i += 1
        j = 0
        while j < min(len(cur), len(after)) - i and cur[-1 - j] == after[-1 - j]:
            j += 1
        ins = after[i:len(after) - j]
        out.append(" ".join([ref, str(i), str(len(cur) - j - i)] + ins))
        cur = list(after)
    return " ; ".join(out)


def decode_steps(start, text):
    forms, cur = [], list(start)
    for part in filter(None, text.split(" ; ")):
        ref, pos, ndel, *ins = part.split(" ")
        pos, ndel = int(pos), int(ndel)
        cur = cur[:pos] + ins + cur[pos + ndel:]
        forms.append((ref, list(cur)))
    return forms


def fingerprint(stem, items):
    h = hashlib.sha1()
    for p in (sca.resolve_lect(rules_path(stem))[0], sca.CENTRAL, os.path.join(HERE, "sca.py"), os.path.join(HERE, "evaluate.py"), __file__):
        if os.path.exists(p):              # changes.sc is optional (the live chains inline everything)
            h.update(open(p, "rb").read())
    h.update(json.dumps(items, ensure_ascii=False).encode())
    return h.hexdigest()


def run_task(task):
    chain, stem, items = task
    os.makedirs(CACHE, exist_ok=True)
    key = fingerprint(stem, items)
    cp = os.path.join(CACHE, f"{stem}.pkl")
    if os.path.exists(cp):
        try:
            k, res = pickle.load(open(cp, "rb"))
            if k == key:
                return res
        except Exception:
            pass
    res = derive_chain(task)
    pickle.dump((key, res), open(cp, "wb"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", default=os.path.join(DATA, "cldf", "forms.csv"))
    ap.add_argument("--edges", default=os.path.join(DATA, "cldf", "edges.csv"))
    ap.add_argument("--cldf", default=os.path.join(DATA, "cldf"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="derive these chains only (no cldf writes; prints a sample)")
    args = ap.parse_args()

    items, lang_of = load_inputs(args.forms, args.edges)
    cmap = chain_map()
    by_chain = defaultdict(list)
    for it in items:
        c = cmap.get(it[1])
        if c:
            by_chain[(c[0], c[1])].append(it)
    tasks = sorted(((c, s, sorted(v)) for (c, s), v in by_chain.items()), key=lambda t: -len(t[2]))
    if args.only:
        tasks = [t for t in tasks if t[0] in args.only]
    print(f"[derive] {sum(len(t[2]) for t in tasks)} forms over {len(tasks)} chains", file=sys.stderr)
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            for r in ex.map(run_task, tasks):
                results.append(r)
                print(f"  {r[0]}: {len(r[1])}", file=sys.stderr, flush=True)
    else:
        for t in tasks:
            results.append(run_task(t))
            print(f"  {t[0]}: {len(results[-1][1])}", file=sys.stderr, flush=True)
    if args.only:
        for _, out, _ in results:
            for d in out[:5]:
                print(json.dumps({k: d[k] for k in ("lang", "variant", "etymon", "steps", "pred", "gold", "ned")}, ensure_ascii=False))
                for c in d["cols"]:
                    print("   ", c)
        return

    ia_forms = {d["form"] for _, out, _ in results for d in out}
    # alignments.csv: keep every non-Indo-Aryan row, replace the Indo-Aryan ones
    ap_path = os.path.join(args.cldf, "alignments.csv")
    tmp = ap_path + ".tmp"
    kept = dropped = 0
    with open(ap_path, encoding="utf-8") as f, open(tmp, "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(ALIGN_HEADER)
        for row in csv.DictReader(f):
            if lang_of.get(row["Origin_ID"]) == PARENT:
                dropped += 1
                continue
            w.writerow([row[h] for h in ALIGN_HEADER])
            kept += 1
        n = 0
        for _, out, _ in results:
            for d in out:
                ety = d["etymon"]
                for pos, (es, ei, rs, ch) in enumerate(d["cols"]):
                    prev = (ety[ei - 1] if ei > 0 else "#") if ei >= 0 else ""
                    nxt = (ety[ei + 1] if ei + 1 < len(ety) else "#") if ei >= 0 else ""
                    w.writerow([d["form"], d["origin"], pos, ei, es or "", rs or "", ch, prev, nxt])
                    n += 1
    os.replace(tmp, ap_path)
    print(f"[derive] alignments.csv: kept {kept} non-IA rows, replaced {dropped} IA rows with {n}", file=sys.stderr)

    with open(os.path.join(args.cldf, "derivations.csv"), "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["Form_ID", "Origin_ID", "Chain", "Variant", "Etymon", "Steps", "Prediction", "Attested", "NED"])
        for chain, out, _ in results:
            for d in out:
                w.writerow([d["form"], d["origin"], chain, d["variant"], " ".join(d["etymon"]),
                            encode_steps(d["etymon"], d["steps"]), " ".join(d["pred"]), " ".join(d["gold"]), d["ned"]])
    fields = ["Rule_Ref", "Chain", "Order", "Line", "Source", "Text", "Name", "Change_ID", "Tags", "Note", "Forms", "Forms_Exact"]
    with open(os.path.join(args.cldf, "sound_rules.csv"), "w", newline="", encoding="utf-8") as g:
        w = csv.DictWriter(g, fields)
        w.writeheader()
        for _, _, table in sorted(results, key=lambda r: r[0]):
            w.writerows(table)
    per_lang = defaultdict(list)
    for _, out, _ in results:
        for d in out:
            per_lang[d["lang"]].append(d["ned"])
    with open(os.path.join(args.cldf, "sound_chains.csv"), "w", newline="", encoding="utf-8") as g:
        w = csv.writer(g)
        w.writerow(["Language_ID", "Chain", "Borrowed", "Note", "Forms", "Mean_NED", "Exact"])
        for lang in sorted(per_lang):
            c = cmap[lang]
            v = per_lang[lang]
            w.writerow([lang, c[0], int(c[2]), c[3], len(v), round(sum(v) / len(v), 4), round(sum(x == 0 for x in v) / len(v), 4)])
    print(f"[derive] {len(ia_forms)} derivations; wrote derivations.csv, sound_rules.csv, sound_chains.csv", file=sys.stderr)


if __name__ == "__main__":
    main()

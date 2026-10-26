#!/usr/bin/env python3
"""
synth.py — cross-lect synthesis of the per-lect rule files (run from data/).

    python sound_changes/synth.py table            → results/summary.csv   (scores per lect)
    python sound_changes/synth.py tags             → results/tags.csv      (one row per tagged rule)
    python sound_changes/synth.py orders [LECT…]   → results/orders/<stem>.csv (ordering evidence)
    python sound_changes/synth.py orderagg         → results/order_consensus.csv (tag-pair precedence)
    python sound_changes/synth.py accent           → results/accent.csv    (accent score per lect)
    python sound_changes/synth.py ext              → results/extension.csv (extension-aware NED per lect)

`accent` scores every chain on the attested forms that mark accent (any lect, any source that
writes it), under four predictions: no accent; the chain's own accent (placed by `$1+stress` rules; equal
to "none" for chains without them); chain + Vedic (the Vedic accent carried through the
derivation on the latent accent tier, wherever no rule placed one); and Vedic only.

`orders` tests, for every pair of phonological rules i < j in a lect's chain, the two single-rule
moves that reverse them: j moved to just before i (Δ1) and i moved to just after j (Δ2), in TRAIN
NED. If both hurt (reported delta = min > 0) the data require i before j — neither rule can be
relocated past the other for free; if both help (reported delta = max < 0) the reverse order fits
better; mixed results are not reported. Only words on which the moved rule fires in either
position can change, so each test touches a small subset.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate as ev  # noqa: E402
import sca  # noqa: E402

RESULTS = os.path.join(HERE, "results")
NONPHON = {"@notation", "@morph", "@acc-loss"}
# tags may contain non-ASCII letters and combining marks (@sibilant-ś, @r-vocalic-ri)
TAG_RE = r"@[\w\-:.\u0300-\u036f\u02b0-\u02ff]+"


SET_RE = re.compile(r"\{[^}]*\}(?:@\d+)?|\S+")


def change_id(rule) -> str:
    """Identity of a rule as a sound change: its central name (`use NAME`) or its canonical text."""
    return "=" + rule.name if getattr(rule, "name", "") else canon_rule(rule.text)


def canon_rule(text: str) -> str:
    """A rule's identity for cross-lect comparison: whitespace normalised and set members sorted.
    A replacement set that maps onto a target set (same position, or `@n`) is permuted with it,
    so parallel mappings survive; environment sets are unordered and simply sorted. Nothing else
    is merged: rules that differ in conditioning, members or output are different changes."""
    text = re.sub(r"\s+", " ", text.strip())
    m = re.match(r"^((?:repeat:\s*|if:\S+\s+)*)(.*)$", text)
    prefix, body = m.group(1).strip(), m.group(2)
    if " > " not in body:
        return text
    lhs, rest = body.split(" > ", 1)
    rhs, sep, env = rest.partition(" / ") if " / " in rest else rest.partition(" // ")
    lt, rt = SET_RE.findall(lhs), SET_RE.findall(rhs)
    perms = {}
    for i, t in enumerate(lt):
        if t.startswith("{"):
            mem = t[1:-1].split()
            order = sorted(range(len(mem)), key=lambda k: mem[k])
            perms[i] = order
            lt[i] = "{" + " ".join(mem[k] for k in order) + "}"
    for i, t in enumerate(rt):
        if not t.startswith("{"):
            continue
        at = re.search(r"@(\d+)$", t)
        src = int(at.group(1)) - 1 if at else i
        mem = t[:t.index("}")][1:].split()
        if src in perms and len(perms[src]) == len(mem):
            mem = [mem[k] for k in perms[src]]
        else:
            mem = sorted(mem)
        rt[i] = "{" + " ".join(mem) + "}" + (at.group(0) if at else "")
    env = re.sub(r"\{([^}]*)\}", lambda mm: "{" + " ".join(sorted(mm.group(1).split())) + "}", env)
    out = " ".join(lt) + " > " + " ".join(rt) + (f" {sep.strip()} {env}" if sep else "")
    return (prefix + " " + out).strip()


def is_phon(tags, text):
    return bool(tags) and not (set(tags) & (NONPHON | {"@accent-placement"})) and "stress" not in text


def languages():
    with open(os.path.join(HERE, "languages.tsv"), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rules_path(stem):
    return os.path.join(HERE, "rules", f"OIA-{stem}.sc")


def rule_tags(path):
    """[(Rule, [tags], note)] in chain order, tags read from the rule line's comment."""
    rules = sca.load_rules(path)
    lines = sca.source_lines(path)
    out = []
    defs = sca.central()
    for r in rules:
        raw = sca.raw_line(r, lines)
        comment = raw.split(";", 1)[1] if ";" in raw else ""
        if r.name:   # `use NAME`: the central definition's tags + the lect's own comment
            comment = defs[r.name][1] + " " + comment
        tags = re.findall(TAG_RE, comment)
        note = re.sub(TAG_RE, "", comment).strip(" —-:")
        out.append((r, tags, note))
    return out


def read_curve(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ─────────────────────────────────────────────────────────────────────────────

def cmd_table(args):
    """Summary scores per lect, computed directly from each chain (no rules-vs-score frontier):
    rule count, train/test NED and exact match, and the no-rule baseline."""
    rows = []
    for L in languages():
        stem = L["File"]
        rp = rules_path(stem)
        row = {k: L[k] for k in ("Lang", "File", "Group", "Name", "Clade", "Etyma", "Pairs")}
        if sca.has_chain(rp) and L["Lang"] != "MIA":
            rules = sca.load_rules(rp)
            _, train, test = ev.datasets(rp)
            mtr, mte = ev.metrics(train, ev.run(rules, train.etyma)), ev.metrics(test, ev.run(rules, test.etyma))
            nte = ev.metrics(test, test.etyma)
            row.update(hand_rules=len(rules), hand_train_ned=mtr["ned"], hand_test_ned=mte["ned"],
                       hand_train_exact=mtr["exact"], hand_test_exact=mte["exact"],
                       null_test_ned=nte["ned"], null_test_exact=nte["exact"], test_pairs=test.n_pairs,
                       excluded_sources=" ".join(sorted(sca.load_meta(rp)["exclude_sources"])))
        rows.append(row)
        print(f"{L['Lang']:22s} {row.get('hand_test_ned', '-')!s:.6s}", file=sys.stderr)
    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    ev.write_csv(os.path.join(RESULTS, "summary.csv"), [{c: r.get(c, "") for c in cols} for r in rows])


def cmd_tags(args):
    out = []
    for L in languages():
        rp = rules_path(L["File"])
        if not sca.has_chain(rp):
            continue
        chain = rule_tags(rp)
        n = len(chain)
        for pos, (r, tags, note) in enumerate(chain):
            for t in tags or ["@untagged"]:
                out.append({"Lang": L["Lang"], "Group": L["Group"], "pos": pos, "n": n,
                            "rel_pos": round(pos / max(n - 1, 1), 4), "tag": t, "line": r.line,
                            "rule": r.text, "change": change_id(r) if is_phon(tags, r.text) else "",
                            "note": note})
    ev.write_csv(os.path.join(RESULTS, "tags.csv"), out)


def orders_one(stem):
    rp = rules_path(stem)
    if not sca.has_chain(rp):
        return stem, 0
    chain = rule_tags(rp)
    rules = [c[0] for c in chain]
    _, train, _ = ev.datasets(rp)
    W = train.etyma
    P = [list(W)]
    for r in rules:
        P.append([sca.apply_rule_full(r, w) for w in P[-1]])
    final = P[-1]
    base_cost = [train.cost(k, w)[0] for k, w in enumerate(final)]
    N = train.n_pairs or 1
    rows = []
    phon = [i for i, c in enumerate(chain) if not (set(c[1]) & NONPHON) and c[1]]
    fired = [None] + [{k for k in range(len(W)) if P[q + 1][k] != P[q][k]} for q in range(len(rules))]

    def total(order, touched, start):
        d = 0.0
        for k in touched:
            w = P[start][k]
            for q in order:
                w = sca.apply_rule_full(rules[q], w)
            d += train.cost(k, w)[0] - base_cost[k]
        return d

    for jj, j in enumerate(phon):
        for i in phon[:jj]:
            # Δ1: rule j moved to just before rule i
            t1 = set(fired[j + 1]) | {k for k in range(len(W))
                                      if sca.apply_rule_full(rules[j], P[i][k]) != P[i][k]}
            d1 = total([j] + list(range(i, j)) + list(range(j + 1, len(rules))), t1, i)
            # Δ2: rule i moved to just after rule j
            t2 = set(fired[i + 1]) | {k for k in range(len(W))
                                      if sca.apply_rule_full(rules[i], P[j + 1][k]) != P[j + 1][k]}
            d2 = total(list(range(i + 1, j + 1)) + [i] + list(range(j + 1, len(rules))), t2, i)
            lo, hi = min(d1, d2) / N, max(d1, d2) / N
            if lo >= 2e-4 or hi <= -2e-4:
                rows.append({"earlier": " ".join(chain[i][1]), "later": " ".join(chain[j][1]),
                             "earlier_rule": rules[i].text, "later_rule": rules[j].text,
                             "i": i, "j": j,
                             # >0: both single-rule moves that reverse the pair hurt (i must precede j);
                             # <0: both help (the reverse order fits better)
                             "delta_ned": round(lo if lo > 0 else hi, 5),
                             "words": len(t1 | t2)})
    os.makedirs(os.path.join(RESULTS, "orders"), exist_ok=True)
    path = os.path.join(RESULTS, "orders", f"{stem}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["earlier", "later", "earlier_rule", "later_rule", "i", "j",
                                          "delta_ned", "words"])
        w.writeheader()
        w.writerows(rows)
    return stem, len(rows)


def cmd_orders(args):
    langs = languages()
    stems = [L["File"] for L in langs if not args.lects or L["Lang"] in args.lects or L["File"] in args.lects]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for stem, n in pool.map(orders_one, stems):
            print(f"{stem}: {n} order-sensitive pairs", file=sys.stderr, flush=True)


def cmd_orderagg(args):
    """Family-wide precedence evidence: for each ordered pair of changes (A before B), the lects
    where moving B across A hurts (supports A < B) and where it helps (supports B < A). A change is
    a rule identity (canon_rule), never a tag: rules with different conditioning are not pooled."""
    from collections import defaultdict
    grp = {L["File"]: L["Group"] for L in languages()}
    lang = {L["File"]: L["Lang"] for L in languages()}
    sup = defaultdict(lambda: {"for": [], "against": [], "delta": 0.0})
    tagsof = defaultdict(set)
    ident = {}      # (stem, rule text) -> change identity (central name or canonical text)
    for L in languages():
        rp = rules_path(L["File"])
        if sca.has_chain(rp):
            for r, _, _ in rule_tags(rp):
                ident[(L["File"], r.text)] = change_id(r)
    odir = os.path.join(RESULTS, "orders")
    for fn in sorted(os.listdir(odir)) if os.path.isdir(odir) else []:
        stem = fn[:-4]
        if stem not in lang:
            continue
        with open(os.path.join(odir, fn), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                ta, tb = r["earlier"].split(), r["later"].split()
                if not is_phon(ta, r["earlier_rule"]) or not is_phon(tb, r["later_rule"]):
                    continue
                a = ident.get((stem, r["earlier_rule"]), canon_rule(r["earlier_rule"]))
                b = ident.get((stem, r["later_rule"]), canon_rule(r["later_rule"]))
                if a == b:
                    continue
                tagsof[a].update(ta); tagsof[b].update(tb)
                d = float(r["delta_ned"])
                key, sign = ((a, b), 1) if a < b else ((b, a), -1)
                e = sup[key]
                (e["for"] if sign * d > 0 else e["against"]).append(f"{lang[stem]}({grp[stem]})")
                e["delta"] += sign * d
    out = []
    for (a, b), e in sup.items():
        first, second = (a, b) if e["delta"] >= 0 else (b, a)
        support, contra = (e["for"], e["against"]) if e["delta"] >= 0 else (e["against"], e["for"])
        out.append({"first": first, "then": second, "first_tags": " ".join(sorted(tagsof[first])),
                    "then_tags": " ".join(sorted(tagsof[second])), "lects_supporting": len(set(support)),
                    "lects_contradicting": len(set(contra)), "sum_delta_ned": round(abs(e["delta"]), 4),
                    "supporting": " ".join(sorted(set(support))), "contradicting": " ".join(sorted(set(contra)))})
    out.sort(key=lambda r: (-r["lects_supporting"], -r["sum_delta_ned"]))
    ev.write_csv(os.path.join(RESULTS, "order_consensus.csv"), out)


def cmd_accent(args):
    cols = ["Lang", "Group", "train_apairs", "test_apairs"] + [f"{sp}_aned_{m}" for sp in ("train", "test")
                                                             for m in ("none", "chain", "vedic", "vedic-only")]
    rows = []
    for L in languages():
        if L["Lang"] == "MIA":
            continue
        path = rules_path(L["File"])
        rules = sca.load_rules(path)
        _, tr, te = ev.datasets(path)
        row = {"Lang": L["Lang"], "Group": L["Group"], "train_apairs": tr.n_apairs, "test_apairs": te.n_apairs}
        if tr.n_apairs or te.n_apairs:
            for sp, ds in (("train", tr), ("test", te)):
                preds = ev.run(rules, ds.etyma)
                plain = [sca.deacc_pred(tuple(p)) for p in preds]
                for model, pr in (("none", plain), ("chain", preds), ("vedic", preds), ("vedic-only", preds)):
                    ds.accent_model = "chain" if model == "none" else model
                    row[f"{sp}_aned_{model}"] = ev.metrics(ds, pr, accent=True).get("aned", "")
        rows.append(row)
        print(L["Lang"], row["test_apairs"], file=sys.stderr, flush=True)
    out = os.path.join(RESULTS, "accent.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


def cmd_ext(args):
    menu = ("bare",) + ev.EXT_MENU + tuple("v:" + (v or "root") for v in ev.VERB_MENU)
    cols = ["Lang", "Group"] + [f"{sp}_{k}" for sp in ("train", "test") for k in ("ned", "ext_ned", "ext_exact")] \
        + [f"test_chosen_{m}" for m in menu]
    rows = []
    for L in languages():
        if L["Lang"] == "MIA":
            continue
        path = rules_path(L["File"])
        rules = sca.load_rules(path)
        _, tr, te = ev.datasets(path)
        row = {"Lang": L["Lang"], "Group": L["Group"]}
        for sp, ds in (("train", tr), ("test", te)):
            row[f"{sp}_ned"] = ev.metrics(ds, ev.run(rules, ds.etyma))["ned"]
            m = ev.ext_metrics(ds, rules)
            row[f"{sp}_ext_ned"], row[f"{sp}_ext_exact"] = m["ext_ned"], m["ext_exact"]
            if sp == "test":
                n = sum(m["chosen"].values()) or 1
                for k in menu:
                    row[f"test_chosen_{k}"] = m["chosen"].get(k, 0) / n
        rows.append(row)
        print(L["Lang"], round(row["test_ned"], 4), round(row["test_ext_ned"], 4), file=sys.stderr, flush=True)
    out = os.path.join(RESULTS, "extension.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    sub.add_parser("tags").set_defaults(fn=cmd_tags)
    sub.add_parser("orderagg").set_defaults(fn=cmd_orderagg)
    sub.add_parser("accent").set_defaults(fn=cmd_accent)
    sub.add_parser("ext").set_defaults(fn=cmd_ext)
    s = sub.add_parser("orders")
    s.add_argument("lects", nargs="*")
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_orders)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

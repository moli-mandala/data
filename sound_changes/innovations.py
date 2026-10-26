#!/usr/bin/env python3
"""innovations.py — agent-independent, innovation-based subgrouping with bootstrap support.

    python sound_changes/innovations.py count      # per-etymon correspondence counts → results/innov_counts.json
    uv run --group plots python sound_changes/innovations.py tree [--reps 200]   → results/innov_clades.json

`count` aligns every OIA etymon with each attested reflex (align.py) and, for ~35 diagnostic
contexts on the headword, records how often the reflex shows the innovative outcome, per etymon.
`tree` turns rates into present (≥ 0.5) / absent (≤ 0.2) / unknown characters (≥ 6 cases),
clusters lects by shared presences weighted by rarity (absences carry no weight), and reports every
clade backed by ≥ 2 diagnostic innovations (present in ≥ 60% of members, in ≤ OUT of outsiders).
Support = share of bootstrap replicates (etyma resampled with replacement within each lect) whose
tree contains a clade with Jaccard ≥ 0.75 to it. Nuristani (correspondences, not descent) and the
unusable "MIA" lect are excluded.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results")


def languages():
    with open(os.path.join(HERE, "languages.tsv"), encoding="utf-8") as f:
        return [L for L in csv.DictReader(f, delimiter="\t") if L["Group"] != "nuristani" and L["Lang"] != "MIA"]


def cmd_count(args):
    import evaluate as ev
    import sca
    from charspec import char_defs, aligned  # the diagnostic definitions
    defs = char_defs()
    out = {}
    for L in languages():
        ds = ev.Dataset(os.path.join(HERE, "pairs", f"OIA-{L['File']}.tsv"))
        per = []
        for i, w0 in enumerate(ds.etyma):
            w = tuple(sca._deacc(x) for x in w0)
            cnt = defaultdict(lambda: [0, 0])
            for gold, _ in ds.golds[i]:
                O = aligned(w, tuple(sca._deacc(x) for x in gold))
                for k in range(len(w)):
                    o = O.get(k, "∅")
                    for name, fn in defs:
                        try:
                            r = fn(w, k, o, O)
                        except Exception:
                            r = None
                        if r is not None:
                            cnt[name][0] += 1
                            cnt[name][1] += bool(r)
            if cnt:
                per.append({n: c for n, c in cnt.items()})
        out[L["Lang"]] = per
        print(L["Lang"], len(per), file=sys.stderr, flush=True)
    json.dump(out, open(os.path.join(RES, "innov_counts.json"), "w", encoding="utf-8"), ensure_ascii=False)


def states(per, chars, idx=None, MIN=6):
    tot = defaultdict(lambda: [0, 0])
    for e in (per if idx is None else (per[i] for i in idx)):
        for c, (n, k) in e.items():
            tot[c][0] += n
            tot[c][1] += k
    st = {}
    for c in chars:
        n, k = tot.get(c, (0, 0))
        if n < MIN:
            st[c] = None
        else:
            r = k / n
            st[c] = 1 if r >= 0.5 else 0 if r <= 0.2 else None
    return st


def build(S, lects, chars, OUT):
    import numpy as np
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    pres = {c: {l for l in lects if S[l][c] == 1} for c in chars}
    known = {c: {l for l in lects if S[l][c] is not None} for c in chars}
    info = [c for c in chars if len(pres[c]) >= 2 and len(known[c] - pres[c]) >= 0.1 * max(len(known[c]), 1)]
    w = {c: math.log(max(len(known[c]), 1) / len(pres[c])) for c in info}
    N = len(lects)
    D = np.zeros((N, N))
    for i, a in enumerate(lects):
        for j in range(i + 1, N):
            b = lects[j]
            inter = uni = 0.0
            for c in info:
                sa, sb = S[a][c], S[b][c]
                if sa is None or sb is None:
                    continue
                if sa == 1 and sb == 1:
                    inter += w[c]
                if sa == 1 or sb == 1:
                    uni += w[c]
            D[i, j] = D[j, i] = 1 - inter / uni if uni else 1
    root = to_tree(linkage(squareform(D), "average"))
    clades = []

    def walk(node):
        if node.is_leaf():
            return
        m = frozenset(lects[i] for i in node.pre_order())
        if 2 <= len(m) <= N - 2:
            diag = []
            for c in info:
                ink = [l for l in m if S[l][c] is not None]
                outk = [l for l in known[c] if l not in m]
                if len(ink) < max(2, 0.5 * len(m)) or not outk:
                    continue
                inr = sum(S[l][c] for l in ink) / len(ink)
                outr = sum(S[l][c] for l in outk) / len(outk)
                if inr >= 0.6 and outr <= OUT:
                    diag.append((c, round(inr, 3), round(outr, 3), len(ink)))
            clades.append((m, diag))
        walk(node.left)
        walk(node.right)
    walk(root)
    return clades


def cmd_tree(args):
    import random
    counts = json.load(open(os.path.join(RES, "innov_counts.json"), encoding="utf-8"))
    meta = {L["Lang"]: L for L in languages()}
    lects = sorted(l for l in counts if l in meta)
    chars = sorted({c for l in lects for e in counts[l] for c in e})
    S = {l: states(counts[l], chars) for l in lects}
    result = {}
    for label, OUT in (("strict", 0.12), ("relaxed", 0.25)):
        full = [(m, d) for m, d in build(S, lects, chars, OUT) if len(d) >= 2]
        support = [0] * len(full)
        rng = random.Random(1)
        for rep in range(args.reps):
            Sb = {l: states(counts[l], chars, [rng.randrange(len(counts[l])) for _ in counts[l]]) for l in lects}
            rep_clades = [m for m, _ in build(Sb, lects, chars, OUT)]
            for i, (m, _) in enumerate(full):
                if any(len(m & r) / len(m | r) >= 0.75 for r in rep_clades):
                    support[i] += 1
            print(f"{label} rep {rep + 1}", file=sys.stderr, flush=True)
        result[label] = [{
            "members": sorted(m), "names": sorted(meta[x]["Name"] for x in m),
            "groups": sorted({meta[x]["Group"] for x in m}),
            "diagnostic": [{"char": c, "in": i, "out": o, "n": n} for c, i, o, n in sorted(d, key=lambda x: -x[1])],
            "support": round(support[k] / max(args.reps, 1), 3),
        } for k, (m, d) in enumerate(full)]
    rates = {l: {c: counts_rate(counts[l], c) for c in chars} for l in lects}
    json.dump({"clades": result, "chars": chars, "rates": rates, "reps": args.reps},
              open(os.path.join(RES, "innov_clades.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for label in result:
        print(f"== {label}")
        for c in sorted(result[label], key=lambda c: (-c["support"], -len(c["members"]))):
            print(f"  support {c['support']:.2f}  n={len(c['members']):3d}  {', '.join(c['names'])[:140]}")
            print(f"       {'; '.join(d['char'] + f' [{d[chr(105)+chr(110)]:.0%}/{d[chr(111)+chr(117)+chr(116)]:.0%}]' for d in c['diagnostic'][:5])}")


def counts_rate(per, c):
    n = sum(e[c][0] for e in per if c in e)
    k = sum(e[c][1] for e in per if c in e)
    return [k, n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("count").set_defaults(fn=cmd_count)
    s = sub.add_parser("tree")
    s.add_argument("--reps", type=int, default=200)
    s.set_defaults(fn=cmd_tree)
    a = ap.parse_args()
    a.fn(a)

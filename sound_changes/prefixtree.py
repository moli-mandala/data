#!/usr/bin/env python3
"""prefixtree.py — subgrouping from shared rule-list prefixes (innovations only).

    python sound_changes/prefixtree.py [--tol 2]  → results/prefix_tree.json, results/prefix_tree.txt

Every rule in a chain is an innovation (a change away from OIA); a retention is simply the absence
of a rule, so nothing here can group lects by what they kept. Each lect's phonological rules
(notation, morphology, accent-loss and accent-placement rules excluded) are the innovations,
each identified by the rule itself (synth.canon_rule): no pooling by tag, so s > h everywhere and
s > h / V _ V are different innovations.

The lists are put in one canonical order: innovations shared by more lects first (ties broken by
the consensus chronology). For compatible innovations this is the perfect-phylogeny order
(Gusfield 1991): the prefix tree (trie) of the lists is the tree, and the number of shared
innovations at a node is the length of the prefix its lects share. Innovations found in one lect
only are leaf edges and never affect the branching. An innovation that sits on more than one
branch is homoplastic (parallel or areal); the tree reports how often that happens.

The strict trie breaks a lect off at the first widespread innovation it lacks, however the rest of
its list looks (one missing or differently written rule is enough). The tolerant tree (`--tol t`)
keeps the prefix idea but lets it survive up to t disagreements: for two lects, walk the shared
innovations in canonical order and count those both have until the (t+1)-th position where only
one has it; average-linkage clustering on that shared-prefix length gives the tree. Nodes are
labelled with the innovations present in ≥ 70% of members and ≤ 20% of other lects.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sca  # noqa: E402
from synth import rule_tags, languages, RESULTS, change_id, is_phon  # noqa: E402

def lect_innovations(path):
    """A lect's innovations: its phonological rules, each identified by the rule itself
    (canon_rule). Rules are never pooled by tag, so differently conditioned changes stay apart."""
    ids = []
    for r, tags, _ in rule_tags(path):
        if not is_phon(tags, r.text):
            continue
        i = change_id(r)
        if i not in ids:
            ids.append(i)
    return ids


def tolerant_tree(seqs, count, key, meta, tol):
    import numpy as np
    from scipy.cluster.hierarchy import linkage, to_tree
    from scipy.spatial.distance import squareform
    order = sorted((i for i, c in count.items() if c >= 2), key=key)
    lects = sorted(seqs)
    has = {l: set(seqs[l]) for l in lects}
    N = len(lects)
    P = np.zeros((N, N))
    for a in range(N):
        for b in range(a + 1, N):
            A, B = has[lects[a]], has[lects[b]]
            shared = miss = 0
            for i in order:
                x, y = i in A, i in B
                if x and y:
                    shared += 1
                elif x or y:
                    miss += 1
                    if miss > tol:
                        break
            P[a, b] = P[b, a] = shared
    D = P.max() - P
    np.fill_diagonal(D, 0)
    root = to_tree(linkage(squareform(D), "average"))
    top = P.max()

    def build(node):
        m = [lects[i] for i in node.pre_order()]
        if node.is_leaf():
            return {"members": m, "depth": len(seqs[m[0]]), "kids": [], "diag": []}
        ms = set(m)
        diag = []
        for i in order:
            inn = sum(i in has[l] for l in ms) / len(ms)
            out = sum(i in has[l] for l in lects if l not in ms) / max(1, N - len(ms))
            if inn >= 0.7 and out <= 0.2:
                diag.append([i, round(inn, 2), round(out, 2)])
        return {"members": sorted(m), "depth": round(float(top - node.dist), 1),
                "kids": [build(node.left), build(node.right)], "diag": diag}
    return build(root)


def main():
    tol = int(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 2
    chrono = {}
    p = os.path.join(RESULTS, "chronology.csv")
    if os.path.exists(p):
        chrono = {r["change"]: float(r["med"]) for r in csv.DictReader(open(p, encoding="utf-8")) if r.get("change")}
    langs = [L for L in languages() if L["Lang"] != "MIA" and L["Group"] != "nuristani"]
    inn = {L["Lang"]: lect_innovations(os.path.join(HERE, "rules", f"OIA-{L['File']}.sc")) for L in langs}
    count = Counter(i for v in inn.values() for i in v)
    key = lambda i: (-count[i], chrono.get(i, 0.5), i)
    seqs = {l: sorted(v, key=key) for l, v in inn.items()}

    # trie, then compress unary paths; edge labels are the innovations shared below that point
    root = {"kids": {}, "lects": []}
    for l, s in seqs.items():
        node = root
        for i in s:
            if count[i] < 2:
                break          # autapomorphies: leaf material only
            node = node["kids"].setdefault(i, {"kids": {}, "lects": []})
        node["lects"].append(l)

    def build(node, label, depth):
        path = [label] if label else []
        while len(node["kids"]) == 1 and not node["lects"]:
            (lab, nxt), = node["kids"].items()
            path.append(lab)
            node = nxt
        d = depth + len(path)
        kids = [build(n, lab, d) for lab, n in sorted(node["kids"].items(), key=lambda kv: -count[kv[0]])]
        members = sorted(node["lects"] + [m for k in kids for m in k["members"]])
        return {"edge": path, "depth": d, "here": sorted(node["lects"]), "kids": kids, "members": members}

    tree = build(root, None, 0)
    # homoplasy: how many separate edges each shared innovation labels
    edges = Counter()

    def walk(n):
        for e in n["edge"]:
            edges[e] += 1
        for k in n["kids"]:
            walk(k)
    walk(tree)
    homo = {i: n for i, n in edges.items() if n > 1}
    meta = {L["Lang"]: {"name": L["Name"], "group": L["Group"]} for L in langs}
    tolerant = tolerant_tree(seqs, count, key, meta, tol)
    json.dump({"tree": tree, "tolerant": tolerant, "tol": tol, "meta": meta, "count": dict(count), "homoplasy": homo,
               "n_shared": sum(1 for c in count.values() if c >= 2)},
              open(os.path.join(RESULTS, "prefix_tree.json"), "w", encoding="utf-8"), ensure_ascii=False)
    lines = []

    def show(n, ind=0):
        if n["edge"] or n["here"]:
            lab = "; ".join(n["edge"][:4]) + (f" (+{len(n['edge']) - 4})" if len(n["edge"]) > 4 else "")
            names = ", ".join(meta[m]["name"] for m in n["members"])[:110]
            lines.append(f"{'  ' * ind}[{n['depth']:2d}] n={len(n['members']):3d}  {lab}  ::  {names}")
        for k in n["kids"]:
            show(k, ind + 1)
    show(tree)
    open(os.path.join(RESULTS, "prefix_tree.txt"), "w", encoding="utf-8").write("\n".join(lines))
    tl = []

    def tshow(n, ind=0):
        if len(n["members"]) > 1:
            names = ", ".join(meta[m]["name"] for m in n["members"])[:100]
            d = "; ".join(f"{i} {int(100*a)}/{int(100*b)}" for i, a, b in n["diag"][:3])
            tl.append(f"{'  ' * ind}[{n['depth']:5.1f}] n={len(n['members']):3d}  {d}  ::  {names}")
        for k in n["kids"]:
            if len(k["members"]) >= 3 or ind < 3:
                tshow(k, ind + 1)
    tshow(tolerant)
    open(os.path.join(RESULTS, "prefix_tree_tolerant.txt"), "w", encoding="utf-8").write("\n".join(tl))
    print("\n".join(tl[:70]))
    print(f"\n{len(homo)} of {sum(1 for c in count.values() if c >= 2)} shared innovations sit on >1 branch (homoplasy)")


if __name__ == "__main__":
    main()

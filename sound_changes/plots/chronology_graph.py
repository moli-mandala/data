#!/usr/bin/env python3
"""Layer the family-wide precedence evidence (results/order_consensus.csv) into stages.

An edge A → B is kept when ≥ MIN_SUPPORT lects require A before B and the contradicting lects are
≤ MAX_CONTRA of the supporting ones; tags that are notation/morphology or occur in < MIN_LECTS lects
are ignored; a node is a rule identity (synth.canon_rule), not a tag. Cycles (rare) are broken by dropping the weakest edge. Each tag's stage is the length
of the longest supported chain leading to it. Writes results/chronology_stages.csv.
"""
import csv
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
MIN_SUPPORT, MAX_CONTRA, MIN_LECTS = 20, 0.15, 15
NONPHON = {"@notation", "@morph", "@acc-loss", "@untagged", "@stress"}

# the unit is a rule identity ("change" column of tags.csv), never a tag
lects = defaultdict(set)
with open(os.path.join(RES, "tags.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["change"]:
            lects[r["change"]].add(r["Lang"])
ok = {t for t, ls in lects.items() if len(ls) >= MIN_LECTS}

edges = {}
with open(os.path.join(RES, "order_consensus.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        a, b, s, c = r["first"], r["then"], int(r["lects_supporting"]), int(r["lects_contradicting"])
        if a in ok and b in ok and s >= MIN_SUPPORT and c <= MAX_CONTRA * s:
            edges[(a, b)] = s - c

def find_cycle():
    adj = defaultdict(list)
    for (a, b) in edges:
        adj[a].append(b)
    color, stack = {}, []
    def dfs(u):
        color[u] = 1; stack.append(u)
        for v in adj[u]:
            if color.get(v) == 1:
                return stack[stack.index(v):] + [v]
            if not color.get(v):
                cyc = dfs(v)
                if cyc:
                    return cyc
        color[u] = 2; stack.pop()
        return None
    for u in list(adj):
        if not color.get(u):
            cyc = dfs(u)
            if cyc:
                return cyc
    return None

dropped = []
while (cyc := find_cycle()):
    weakest = min(zip(cyc, cyc[1:]), key=lambda e: edges[e])
    dropped.append((weakest, edges.pop(weakest)))

nodes = {x for e in edges for x in e}
preds = defaultdict(list)
for (a, b) in edges:
    preds[b].append(a)
stage = {}
def depth(n):
    if n not in stage:
        stage[n] = 0 if not preds[n] else 1 + max(depth(p) for p in preds[n])
    return stage[n]
for n in nodes:
    depth(n)

with open(os.path.join(RES, "chronology_stages.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["stage", "change", "lects", "must_follow"])
    for n in sorted(nodes, key=lambda n: (stage[n], -len(lects[n]))):
        ps = sorted(preds[n], key=lambda p: -edges[(p, n)])
        w.writerow([stage[n], n, len(lects[n]), " | ".join(f"{p} ({edges[(p, n)]})" for p in ps[:4])])
print(f"{len(nodes)} tags, {len(edges)} edges, {len(dropped)} cycle edges dropped: {dropped}")

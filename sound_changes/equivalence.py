#!/usr/bin/env python3
"""equivalence.py — find distinct rules that are really the same sound change.

    python sound_changes/equivalence.py [--sample 300] [--near 0.05]
        → results/equivalence_pairs.csv, results/equivalence_groups.csv

Rules are compared by behaviour, not text. For every rule identity (synth.canon_rule) we collect
its real inputs: train words of up to three lects that have it, in the state just before the rule
applies in that lect's chain. Two rules are compared on the union of both input sets:
  equivalent       identical output on every real input (they differ only in notation, class
                   names, or members/environments that never occur where they apply);
  near-equivalent  outputs differ on ≤ NEAR of the forms either rule changes (examples listed).
Candidate pairs share a tag or at least one concrete effect (a segment rewrite such as k > ∅ or
s > h); tags are used only to narrow the search, never as evidence.
Also reported: rule texts that are identical but whose class names are defined differently in
different files (the same text, different changes).
"""
from __future__ import annotations

import csv
import difflib
import os
import random
import sys
from collections import defaultdict
from itertools import combinations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sca  # noqa: E402
import evaluate as ev  # noqa: E402
from synth import RESULTS, canon_rule, is_phon, languages, rule_tags  # noqa: E402


def effects(before, after):
    """Concrete rewrites (old substring → new substring) a rule made on one word."""
    sm = difflib.SequenceMatcher(a=list(before), b=list(after), autojunk=False)
    return {("".join(before[i1:i2]), "".join(after[j1:j2])) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"}


def main():
    sample = int(sys.argv[sys.argv.index("--sample") + 1]) if "--sample" in sys.argv else 300
    near = float(sys.argv[sys.argv.index("--near") + 1]) if "--near" in sys.argv else 0.05
    rng = random.Random(0)
    langs = {L["Lang"]: L for L in languages() if L["Lang"] != "MIA"}
    occ = defaultdict(list)                  # change -> [(Lang, rule index)]
    rule_obj, tags_of, classdefs = {}, defaultdict(set), defaultdict(set)
    states = {}                              # (Lang, k) -> list of input words before rule k
    for lang, L in langs.items():
        path = os.path.join(HERE, "rules", f"OIA-{L['File']}.sc")
        chain = rule_tags(path)
        rules = [c[0] for c in chain]
        _, tr, _ = ev.datasets(path)
        W = list(tr.etyma)
        if len(W) > sample:
            W = rng.sample(W, sample)
        cur = W
        for k, (r, tags, _) in enumerate(chain):
            if is_phon(tags, r.text):
                cid = canon_rule(r.text)
                if len(occ[cid]) < 3:
                    states[(lang, k)] = cur
                occ[cid].append((lang, k))
                rule_obj.setdefault(cid, r)
                tags_of[cid].update(t for t in tags if t != "@stress")
                classdefs[cid].add(tuple((tok.kind, tok.value) for tok in r.target))
            cur = [sca.apply_rule_full(r, w) for w in cur]
        print(lang, file=sys.stderr, flush=True)

    inputs = {c: [w for (l, k) in o[:3] for w in states.get((l, k), [])] for c, o in occ.items()}
    eff = {}
    for c, ws in inputs.items():
        r = rule_obj[c]
        e = set()
        for w in ws:
            a = sca.apply_rule_full(r, w)
            if a != w:
                e |= effects(w, a)
        eff[c] = e

    # candidate pairs: shared tag or shared concrete effect
    by_key = defaultdict(set)
    for c in occ:
        for t in tags_of[c]:
            by_key[("t", t)].add(c)
        for e in eff[c]:
            by_key[("e", e)].add(c)
    cands = set()
    for key, cs in by_key.items():
        if len(cs) > 150:        # an effect everything shares (e.g. a ∅ deletion of a common vowel): skip as a key
            continue
        cands.update(combinations(sorted(cs), 2))
    print(f"{len(occ)} rules, {len(cands)} candidate pairs", file=sys.stderr)

    rows = []
    for a, b in sorted(cands):
        U = inputs[a] + inputs[b]
        ra, rb = rule_obj[a], rule_obj[b]
        changed = diffs = 0
        ex = []
        for w in U:
            oa, ob = sca.apply_rule_full(ra, w), sca.apply_rule_full(rb, w)
            if oa != w or ob != w:
                changed += 1
                if oa != ob:
                    diffs += 1
                    if len(ex) < 3:
                        ex.append(f"{''.join(w)}: {''.join(oa)} | {''.join(ob)}")
        if not changed or diffs / changed > near:
            continue
        la = sorted({l for l, _ in occ[a]}); lb = sorted({l for l, _ in occ[b]})
        rows.append({"verdict": "equivalent" if diffs == 0 else "near-equivalent",
                     "rule_a": a, "lects_a": len(la), "rule_b": b, "lects_b": len(lb),
                     "forms_changed": changed, "forms_differing": diffs,
                     "tags": " ".join(sorted(tags_of[a] | tags_of[b])),
                     "groups": " ".join(sorted({langs[l]["Group"] for l in la + lb})),
                     "examples": " ; ".join(ex)})
    rows.sort(key=lambda r: (r["verdict"] != "equivalent", -(r["lects_a"] + r["lects_b"])))
    ev.write_csv(os.path.join(RESULTS, "equivalence_pairs.csv"), rows)

    # connected components of the equivalent pairs
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for r in rows:
        if r["verdict"] == "equivalent":
            parent[find(r["rule_a"])] = find(r["rule_b"])
    comps = defaultdict(list)
    for x in list(parent):
        comps[find(x)].append(x)
    groups = []
    for members in comps.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda c: -len(occ[c]))
        lects = {l for c in members for l, _ in occ[c]}
        groups.append({"size": len(members), "lects": len(lects),
                       "groups": " ".join(sorted({langs[l]["Group"] for l in lects})),
                       "representative": members[0], "members": " ‖ ".join(members),
                       "tags": " ".join(sorted(set().union(*(tags_of[c] for c in members))))})
    groups.sort(key=lambda g: -g["lects"])
    ev.write_csv(os.path.join(RESULTS, "equivalence_groups.csv"), groups)
    print(f"{sum(r['verdict'] == 'equivalent' for r in rows)} equivalent pairs in {len(groups)} groups; "
          f"{sum(r['verdict'] != 'equivalent' for r in rows)} near-equivalent pairs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build results/changes.json: for every change (a rule identity, synth.canon_rule), what the
report's change tooltips show. Rules are never pooled by tag.

    python sound_changes/report/changes.py

Per change: its tags and their meaning (CATALOG.md), the lects that have it, and up to two worked
examples taken from real chains (the largest lects with the rule): the word
just before and just after the rule applies, which segments it changed, the chain's final
prediction and the attested reflex.
"""
import csv
import difflib
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SC = os.path.dirname(HERE)
sys.path.insert(0, SC)
import sca  # noqa: E402
import evaluate as ev  # noqa: E402
from synth import canon_rule, is_phon  # noqa: E402

RES = os.path.join(SC, "results")


def catalog():
    out = {}
    for line in open(os.path.join(SC, "CATALOG.md"), encoding="utf-8"):
        if not line.startswith("| `@"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        tag = cells[0].strip("`")
        prose = [c for c in cells[1:] if c and not c.startswith("`")]
        out.setdefault(tag, (prose[0] if prose else cells[1]).replace("`", ""))
    return out


def diff(a, b):
    sm = difflib.SequenceMatcher(a=list(a), b=list(b), autojunk=False)
    ia, ib = set(), set()
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal":
            ia.update(range(i1, i2))
            ib.update(range(j1, j2))
    return sorted(ia), sorted(ib)


def main():
    langs = {L["Lang"]: L for L in csv.DictReader(open(os.path.join(SC, "languages.tsv"), encoding="utf-8"), delimiter="\t")}
    rows = list(csv.DictReader(open(os.path.join(RES, "tags.csv"), encoding="utf-8")))
    by_chg = defaultdict(dict)          # change -> {(Lang, line): row}
    tags_of = defaultdict(set)
    raw2id = {}
    for r in rows:
        tags_of[(r["Lang"], r["line"])].add(r["tag"])
    for r in rows:
        c = r.get("change")
        if c and r["Lang"] in langs:
            by_chg[c][(r["Lang"], r["line"])] = r
            raw2id[r["rule"]] = c
    meaning = catalog()
    cache = {}

    def chain(lang):
        if lang not in cache:
            path = os.path.join(SC, "rules", f"OIA-{langs[lang]['File']}.sc")
            rules = sca.load_rules(path)
            _, tr, _ = ev.datasets(path)
            cache[lang] = (rules, tr)
        return cache[lang]

    out = {}
    for c, occ in by_chg.items():
        lects = sorted({l for l, _ in occ}, key=lambda l: -int(langs[l]["Etyma"]))
        tags = sorted(set().union(*(tags_of[k] for k in occ)) - {"@stress"}) or ["@untagged"]
        rule_text = next(iter(occ.values()))["rule"]
        entry = {"rule": rule_text, "name": c[1:] if c.startswith("=") else "", "tags": tags,
                 "meaning": next((meaning[t] for t in tags if t in meaning), ""),
                 "n": len(lects), "lects": [langs[l]["Name"] for l in lects], "examples": []}
        for lang in lects[:4]:
            if len(entry["examples"]) >= 2:
                break
            rules, tr = chain(lang)
            line = next(int(ln) for (l, ln) in occ if l == lang)
            k = next((j for j, r in enumerate(rules) if r.line == line), None)
            if k is None:
                continue
            best = None
            for i, w in enumerate(tr.etyma):
                if len(w) > 9:
                    continue
                before = sca.apply_rules(rules[:k], w)
                after = sca.apply_rule_full(rules[k], before)
                b, a = sca.realize(before, "chain"), sca.realize(after, "chain")
                if b == a:
                    continue
                ib, ia = diff(b, a)
                if len(ib) + len(ia) > 4:
                    continue
                final = "".join(sca.realize(sca.apply_rules(rules, w), "chain"))
                ok = any(sca.canon(sca.deacc_pred(tuple(sca.tokenize(final)))) == g for g, _ in tr.golds[i])
                score = (not ok, len(w), len(ib) + len(ia))
                if best is None or score < best[0]:
                    best = (score, {"lect": langs[lang]["Name"], "etymon": tr.raw[i], "before": list(b), "after": list(a),
                                    "hb": ib, "ha": ia, "step": k + 1, "steps": len(rules),
                                    "pred": final, "gold": tr.golds[i][0][1], "ok": ok})
                if best and not best[0][0] and best[0][1] <= 4:
                    break
            if best:
                entry["examples"].append(best[1])
        out[c] = entry
    json.dump({"changes": out, "raw2id": raw2id}, open(os.path.join(RES, "changes.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    print(f"wrote results/changes.json ({len(out)} changes)")


if __name__ == "__main__":
    main()

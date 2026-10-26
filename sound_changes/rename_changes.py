#!/usr/bin/env python3
"""rename_changes.py — give the central changes (changes.sc) descriptive names and update every
`use` line. Run from data/.

A name is the rule's main tag, then only what distinguishes it from other changes with that tag:
the affected segments (int-stop-loss.kg vs int-stop-loss.td) and/or the environment
(s-h vs s-h.intervocalic); a number is the last resort. Environment words already implied by the
tag (final-vowel-loss + final) are dropped.
"""
import glob
import os
import re
import sys
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CENTRAL = os.path.join(HERE, "changes.sc")
ENV = {"": "", "V _ V": "intervocalic", "# _": "initial", "_ #": "final", "V _": "postvocalic",
       "_ V": "prevocalic", "_ C": "preconsonantal", "C _": "postconsonantal", "V C* _ #": "final-polysyllabic",
       "# C* _": "first-syllable", "_ C* #": "last-syllable"}


def ascii_ish(s):
    s = unicodedata.normalize("NFD", s)
    s = s.replace("ʰ", "h").replace("ʦ", "ts").replace("ʣ", "dz").replace("∅", "0").replace("ŋ", "ng").replace("ñ", "ny")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9]+", "", s)


def parts(text):
    body = re.sub(r"^((repeat:|if:\S+)\s+)+", "", text)
    lhs = body.split(" > ")[0]
    env = body.split(" / ", 1)[1].split(" // ")[0].strip() if " / " in body else ""
    exc = " // " in body
    seg = ascii_ish(lhs) if not re.search(r"[A-Z]", lhs) else ascii_ish(re.sub(r"\$\d\S*", "", lhs)).lower()
    return seg[:10] or "x", ENV.get(env, ascii_ish(env).lower()[:14] or "cond"), exc, text.startswith("repeat:")


def main():
    lines = open(CENTRAL, encoding="utf-8").read().split("\n")
    entries = []
    for i, l in enumerate(lines):
        m = re.match(r"^change (\S+) = (.*?)\s*(;.*)?$", l)
        if m:
            tags = re.findall(r"@[\w\-.]+", (m.group(3) or "").split("—", 1)[0])  # as sca.comment_tags
            main_tag = next((t for t in tags if t not in ("@stress", "@cluster-simplify")), tags[0] if tags else "@change")
            base = re.sub(r"^@(x-[a-z-]+?-)?", "", main_tag)
            entries.append((i, m.group(1), m.group(2).strip(), base))
    by_base = defaultdict(list)
    for e in entries:
        by_base[e[3]].append(e)
    new = {}
    for base, es in by_base.items():
        if len(es) == 1:
            new[es[0][1]] = base
            continue
        info = {e[1]: parts(e[2]) for e in es}
        segs = {info[e[1]][0] for e in es}
        envs = {info[e[1]][1] for e in es}
        for e in es:
            seg, env, exc, rep = info[e[1]]
            bits = []
            if len(segs) > 1:
                bits.append(seg)
            if len(envs) > 1 and env and env not in base:
                bits.append(env)
            if exc:
                bits.append("except")
            if rep:
                bits.append("repeat")
            new[e[1]] = base + ("." + ".".join(bits) if bits else "")
        # resolve remaining clashes with numbers, most-used first (keep file order)
        seen = defaultdict(int)
        for e in es:
            n = new[e[1]]
            seen[n] += 1
            if seen[n] > 1:
                new[e[1]] = f"{n}.{seen[n]}"
    assert len(set(new.values())) == len(new), "name clash"
    for i, old, text, base in entries:
        lines[i] = re.sub(rf"^change {re.escape(old)} =", f"change {new[old]} =", lines[i])
    open(CENTRAL, "w", encoding="utf-8").write("\n".join(lines))
    n = 0
    for p in glob.glob(os.path.join(HERE, "rules", "*.sc")) + glob.glob(os.path.join(HERE, "blocks", "*.sc")):
        ls = open(p, encoding="utf-8").read().split("\n")
        for k, l in enumerate(ls):
            m = re.match(r"^use (\S+)(\s*)(;.*)?$", l)
            if m and m.group(1) in new:
                ls[k] = f"use {new[m.group(1)]:<40s} {m.group(3) or ''}".rstrip()
                n += 1
        open(p, "w", encoding="utf-8").write("\n".join(ls))
    renamed = sum(1 for o, v in new.items() if o != v)
    print(f"renamed {renamed} of {len(new)} central changes; updated {n} use lines")
    for o, v in list(new.items())[:0]:
        print(o, "→", v)


if __name__ == "__main__":
    sys.exit(main())

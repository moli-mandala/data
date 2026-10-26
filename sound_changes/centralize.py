#!/usr/bin/env python3
"""centralize.py — move every sound change shared by two or more lects into sound_changes/changes.sc
and replace it in the lect files by `use NAME`.

    python sound_changes/centralize.py            (run from data/; rewrites rule files + changes.sc)

1. Every phonological rule of every lect is identified by its canonical text (synth.canon_rule).
   Behaviour-equivalent formulations (results/equivalence_groups.csv, from equivalence.py) are
   mapped onto one representative; three reviewed near-equivalents (ARTIFACT) are mapped too.
2. Every identity used by ≥ 2 lects becomes a named central change, written with its most common
   formulation; the classes it uses are copied into changes.sc with explicit members (a class name
   whose members differ between changes gets a numbered variant, so no change silently alters).
3. In each lect the rule's line becomes `use NAME ; <the lect's own comment>`. A line is kept only
   if every train and test prediction (plain and with latent accents) is unchanged; ARTIFACT
   merges are accepted if train NED does not get worse. Anything else stays local. Central
   changes left with fewer than two users are dropped again.
"""
from __future__ import annotations

import csv
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sca  # noqa: E402
import evaluate as ev  # noqa: E402
from synth import RESULTS, canon_rule, is_phon, languages, rule_tags  # noqa: E402

ARTIFACT = {   # reviewed near-equivalents: differ only through upstream artifacts / notation
    "repeat: G $1 > $2": "G $1 > $2",                                # tripled geminates only
    "{d m n s t ḥ} > ∅ / _ #": "{d m n s t ḣ} > ∅ / _ #",            # two glyphs for the visarga
    "{ī ū} > {i u}": "{ī ī̃ ū ū̃} > {i ĩ u ũ}",                          # nasal long vowels listed or not
}
GENERIC = {"@stress", "@cluster-simplify", "@untagged"}
ENV_NAMES = {"": "", "V _ V": "intervocalic", "# _": "initial", "_ #": "final", "V _": "postvocalic",
             "_ V": "prevocalic", "_ C": "preconsonantal", "C _": "postconsonantal", "V C* _ #": "final-polysyllabic"}


def file_classes(path):
    classes = {}
    for _, raw in sca.lect_lines(path):
        line = sca.strip_comment(raw).strip()
        if line.startswith("class "):
            name, _, members = line[6:].partition("=")
            classes[name.strip()] = sca.expand_members(members.split(), classes)
    return classes


def slug(tags, text):
    main = next((t for t in tags if t not in GENERIC), tags[0] if tags else "@change")
    main = re.sub(r"^@(x-[a-z-]+?-)?", "", main)
    body = re.sub(r"^((repeat:|if:\S+)\s+)+", "", text)
    env = body.split(" / ", 1)[1].split(" // ")[0].strip() if " / " in body else ""
    suf = ENV_NAMES.get(env)
    return main + ("-" + suf if suf else "") if suf is not None else main


def predictions(path, datasets):
    rules = sca.load_rules(path)
    out = []
    for ds in datasets:
        for w in ds.etyma:
            p = sca.apply_rules(rules, w)
            out.append((sca.realize(p, "chain"), sca.realize(p, "vedic")))
    return out


def main():
    langs = [L for L in languages() if L["Lang"] != "MIA"]
    second = "--exact" in sys.argv   # later passes: group remaining local rules by exact text only
    # keep what changes.sc already defines (a later pass only adds)
    prev_lines, prev_names, prev_classes = [], set(), {}
    if os.path.exists(sca.CENTRAL):
        for raw in open(sca.CENTRAL, encoding="utf-8"):
            st = sca.strip_comment(raw).strip()
            if st.startswith("change "):
                prev_names.add(st[7:].split("=")[0].strip())
                prev_lines.append(raw.rstrip("\n"))
            elif st.startswith("class "):
                cn, _, mem = st[6:].partition("=")
                prev_classes[cn.strip()] = tuple(mem.split())
    groups = {}
    for g in ([] if second else csv.DictReader(open(os.path.join(RESULTS, "equivalence_groups.csv"), encoding="utf-8"))):
        members = g["members"].split(" ‖ ")
        for m in members:
            groups[m] = members[0]
    for a, b in ({} if second else ARTIFACT).items():
        groups[a] = groups.get(b, b)

    occ = defaultdict(list)          # key -> [(Lang, File, line no, raw text, canon, tags)]
    for L in langs:
        path = os.path.join(HERE, "rules", f"OIA-{L['File']}.sc")
        lines = sca.lect_text(path).split("\n")
        in_block = set()
        blk = False
        for i, raw in enumerate(lines, 1):
            s = sca.strip_comment(raw).strip()
            if s.startswith("%source"):
                blk = True
            elif s.startswith("%end"):
                blk = False
            elif blk:
                in_block.add(i)
        for r, tags, _ in rule_tags(path):
            if r.line in in_block or r.name or r.origin or not is_phon(tags, r.text):
                continue
            c = canon_rule(r.text)
            occ[groups.get(c, c)].append((L["Lang"], L["File"], r.line, r.text, c, tags))

    shared = {k: v for k, v in occ.items() if len({o[0] for o in v}) >= 2}
    print(f"{len(occ)} identities; {len(shared)} shared by ≥ 2 lects", file=sys.stderr)

    # central definitions: most common formulation; its classes copied with explicit members
    cls_members = dict(prev_classes)  # central class name -> members
    changes = {}                     # name -> dict
    used_names = Counter()
    for n in prev_names:
        base = re.sub(r"-\d+$", "", n)
        used_names[base] = max(used_names[base], int(n.rsplit("-", 1)[1]) if re.search(r"-\d+$", n) else 1)
    fcls = {}
    for key, os_ in sorted(shared.items(), key=lambda kv: -len({o[0] for o in kv[1]})):
        rep_text = Counter(o[3] for o in os_ if o[4] == key).most_common(1)
        rep_text = rep_text[0][0] if rep_text else Counter(o[3] for o in os_).most_common(1)[0][0]
        src = next(o for o in os_ if o[3] == rep_text)
        fcls.setdefault(src[1], file_classes(os.path.join(HERE, "rules", f"OIA-{src[1]}.sc")))
        text = rep_text
        for cname in sorted(set(re.findall(r"(?<![\w$])([A-Z][A-Za-z0-9]*)(?![\w])", sca.strip_comment(rep_text)))):
            if cname in ("V", "C") or cname not in fcls[src[1]]:
                continue       # built-ins (V C Acc Long Short Stress) are not file classes
            mem = fcls[src[1]][cname]
            name = cname
            k = 1
            while name in cls_members and cls_members[name] != mem:
                k += 1
                name = f"{cname}{k}"
            cls_members[name] = mem
            if name != cname:
                text = re.sub(rf"(?<![\w$]){cname}(?![\w])", name, text)
        tags = sorted({t for o in os_ for t in o[5]} - {"@stress"})
        base = slug(tags, rep_text)
        used_names[base] += 1
        name = base if used_names[base] == 1 else f"{base}-{used_names[base]}"
        while name in prev_names:
            used_names[base] += 1
            name = f"{base}-{used_names[base]}"
        changes[name] = {"key": key, "text": text, "tags": tags, "occ": os_}

    def write_central(active):
        out = ["; changes.sc — central definitions of the Indo-Aryan sound changes shared by two or more lects.",
               "; Lect rule files reference them with `use NAME`; a change defined here is the same change in",
               "; every lect that uses it (rules are never pooled by tag). Generated by centralize.py and then",
               "; maintained by hand. Classes are explicit; a numbered class (G2) is a variant with other members.",
               ""]
        need = set()
        for n in active:
            need |= set(re.findall(r"(?<![\w$])([A-Z][A-Za-z0-9]*)(?![\w])", changes[n]["text"]))
        for pl in prev_lines:
            need |= set(re.findall(r"(?<![\w$])([A-Z][A-Za-z0-9]*)(?![\w])", sca.strip_comment(pl).split("=", 1)[1]))
        for c in sorted(cls_members):
            if c in need:
                out.append(f"class {c} = {' '.join(cls_members[c])}")
        out.append("")
        out.extend(prev_lines)
        for n in sorted(active, key=lambda n: (changes[n]["tags"][0] if changes[n]["tags"] else "", n)):
            ch = changes[n]
            ls = sorted({o[0] for o in ch["occ"] if o[0] in active[n]})
            out.append(f"change {n} = {ch['text']:<44s} ; {' '.join(ch['tags'])} — {len(ls)} lects")
        open(sca.CENTRAL, "w", encoding="utf-8").write("\n".join(out) + "\n")

    active = {n: {o[0] for o in ch["occ"]} for n, ch in changes.items()}
    write_central(active)

    # rewrite each lect, verifying every line
    kept = defaultdict(set)
    reverted = []
    by_file = defaultdict(list)
    for n, ch in changes.items():
        for o in ch["occ"]:
            by_file[o[1]].append((o[2], n, o))
    lang_of = {L["File"]: L["Lang"] for L in langs}
    for fn, items in by_file.items():
        path = os.path.join(HERE, "rules", f"OIA-{fn}.sc")
        sca.require_one_lect_file(path)
        orig = open(path, encoding="utf-8").read().split("\n")
        _, tr, te = ev.datasets(path)
        base = predictions(path, (tr, te))
        base_ned = ev.metrics(tr, ev.run(sca.load_rules(path), tr.etyma))["ned"]
        lines = list(orig)

        def swap(lno, name):
            raw = lines[lno - 1]
            comment = raw.split(";", 1)[1] if ";" in raw else ""
            return f"use {name:<40s} ;{comment}" if comment else f"use {name}"
        def try_lines(cands, strict=True):
            nonlocal lines, base, base_ned
            trial = list(lines)
            for lno, name, o in cands:
                trial[lno - 1] = swap(lno, name)
            open(path, "w", encoding="utf-8").write("\n".join(trial))
            try:
                if strict:
                    ok = predictions(path, (tr, te)) == base
                else:
                    ok = ev.metrics(tr, ev.run(sca.load_rules(path), tr.etyma))["ned"] <= base_ned + 1e-9
            except Exception as e:
                ok = False
                print("error", fn, e, file=sys.stderr)
            if ok:
                lines = trial
                for lno, name, o in cands:
                    kept[name].add(lang_of[fn])
                if not strict:
                    base = predictions(path, (tr, te))
                    base_ned = ev.metrics(tr, ev.run(sca.load_rules(path), tr.etyma))["ned"]
            open(path, "w", encoding="utf-8").write("\n".join(lines))
            return ok

        exact = [x for x in sorted(items) if x[2][4] not in ARTIFACT]
        if exact and not try_lines(exact):
            for x in exact:            # find the ones that change behaviour
                if not try_lines([x]):
                    reverted.append((lang_of[fn], x[1], x[2][3]))
        for x in [x for x in sorted(items) if x[2][4] in ARTIFACT]:
            if not try_lines([x], strict=False):
                reverted.append((lang_of[fn], x[1], x[2][3]))
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        print(fn, file=sys.stderr, flush=True)

    # drop central changes left with < 2 users: restore their local text
    lonely = {n for n in changes if len(kept[n]) < 2}
    for n in lonely:
        for o in changes[n]["occ"]:
            if o[0] not in kept[n]:
                continue
            path = os.path.join(HERE, "rules", f"OIA-{o[1]}.sc")
            sca.require_one_lect_file(path)
            ls = open(path, encoding="utf-8").read().split("\n")
            raw = ls[o[2] - 1]
            comment = raw.split(";", 1)[1] if ";" in raw else ""
            ls[o[2] - 1] = f"{o[3]:<44s} ;{comment}" if comment else o[3]
            open(path, "w", encoding="utf-8").write("\n".join(ls))
    final = {n: kept[n] for n in changes if n not in lonely}
    write_central(final)
    with open(os.path.join(RESULTS, "centralize_log.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["status", "lect", "change", "local_rule"])
        for l, n, t in reverted:
            w.writerow(["kept local (behaviour differs)", l, n, t])
        for n in lonely:
            w.writerow(["dropped (fewer than 2 users)", "", n, changes[n]["text"]])
    uses = sum(len(v) for v in final.values())
    print(f"{len(final)} central changes, {uses} uses across lects; {len(reverted)} lines kept local; "
          f"{len(lonely)} candidates dropped")


if __name__ == "__main__":
    main()

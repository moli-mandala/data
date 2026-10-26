#!/usr/bin/env python3
"""families.py — family rule files (rules/<group>.sc): several lects' chains in one file.

    python3 sound_changes/families.py merge [--rules DIR] [--keep]   one-lect files → one file per group
    python3 sound_changes/families.py unite [--rules DIR] [--out chains.sc]  family files → one file, families as %sets
    python3 sound_changes/families.py explode OUTDIR [STEM …]        family lects → one-lect files (candidates)
    python3 sound_changes/families.py check [--rules DIR]            every lect's chain loads; selectors valid
    python3 sound_changes/families.py diff OLD_DIR [--rules DIR]     compiled chains equal across two dirs?
    python3 sound_changes/families.py shared [GROUP …]               lines most of a family has: candidates
                                                                     for the lects that lack them

Format (see sca.py, "Family files"): the file opens with `%lects STEM …`; `%set NAME = STEM …` names a
subset; a line prefixed `[A B]` belongs only to those lects, `[-C]` to all but C, unprefixed to all.

`merge` groups lects by languages.tsv (Group column; File = stem). Within a group it aligns the lects'
lines (shortest common supersequence, built by progressive LCS merging) so that a line every lect has
in the same place is written once without a selector, and each lect's chain is exactly its own lines
in its own order. Lines match on code + @tags; a matched line keeps the first lect's comment.
"""
import argparse
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sca  # noqa: E402

RULES = os.path.join(HERE, "rules")


def groups(rules_dir: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    with open(os.path.join(HERE, "languages.tsv"), encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            stem, g = r["File"], r["Group"]
            if g and os.path.isfile(os.path.join(rules_dir, f"OIA-{stem}.sc")) and stem not in out.get(g, []):
                out.setdefault(g, []).append(stem)
    return out


def key(raw: str):
    code = sca.strip_comment(raw).strip()
    if not code:
        return ("#", raw.strip())
    tags = sca.comment_tags(raw.split(";", 1)[1]) if ";" in raw else ()
    return ("c", " ".join(code.split()), tags)


def lcs_pairs(a: list, b: list) -> list[tuple[int, int]]:
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j)); i += 1; j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def merge_group(seqs: dict[str, list[str]]) -> list[tuple[set[str], str]]:
    """[(lects, raw line)] in an order that is a supersequence of every lect's line order."""
    return merge_rows({s: [({s}, r) for r in rows] for s, rows in seqs.items()})


def merge_rows(seqs: dict[str, list[tuple[set[str], str]]]) -> list[tuple[set[str], str]]:
    """Merge selected-line sequences (a family file's rows, or one lect's lines) with disjoint lect
    sets: lines that align (same code + @tags) are written once for the union of their lects, and
    each input's rows keep their order, so every lect's own lines stay exactly as they were."""
    keyed = {s: [key(r) for _, r in rows] for s, rows in seqs.items()}
    names = list(seqs)
    # start from the input sharing most with the others, then add the closest input each time
    score = {s: sum(len(lcs_pairs(keyed[s], keyed[t])) for t in names if t != s) for s in names}
    first = max(names, key=lambda s: (score[s], len(keyed[s])))
    merged = [(set(l), k, r) for k, (l, r) in zip(keyed[first], seqs[first])]
    todo = [s for s in names if s != first]
    while todo:
        mk = [k for _, k, _ in merged]
        best = max(todo, key=lambda s: len(lcs_pairs(mk, keyed[s])))
        todo.remove(best)
        pairs = lcs_pairs(mk, keyed[best])
        rows = seqs[best]
        out, i, j = [], 0, 0
        for pi, pj in pairs + [(len(merged), len(keyed[best]))]:
            out.extend(merged[i:pi])
            out.extend((set(rows[q][0]), keyed[best][q], rows[q][1]) for q in range(j, pj))
            if pi < len(merged):
                lects, k, r = merged[pi]
                out.append((lects | rows[pj][0], k, r))
            i, j = pi + 1, pj + 1
        merged = out
    return [(lects, r) for lects, _, r in merged]


def selector(lects: set[str], members: list[str]) -> str:
    if len(lects) == len(members):
        return ""
    rest = [s for s in members if s not in lects]
    if len(rest) < len(lects):
        return "[" + " ".join("-" + s for s in rest) + "] "
    return "[" + " ".join(s for s in members if s in lects) + "] "


def selector_sets(lects: set[str], members: list[str], sets: dict[str, list[str]]) -> str:
    """Selector for a file with named family sets: whole families by name, a family minus a few lects
    as `fam -x`, else the lects themselves; or `[-x …]` when that is shorter."""
    if len(lects) == len(members):
        return ""
    excl = ["-" + m for m in members if m not in lects]
    toks: list[str] = []
    covered: set[str] = set()
    for fam, fm in sets.items():
        inter = [m for m in fm if m in lects]
        if not inter:
            continue
        covered |= set(inter)
        if len(inter) == len(fm):
            toks.append(fam)
        elif len(fm) - len(inter) < len(inter):
            toks += [fam] + ["-" + m for m in fm if m not in lects]
        else:
            toks += inter
    toks += [m for m in members if m in lects and m not in covered]
    return "[" + " ".join(excl if len(excl) < len(toks) else toks) + "] "


def write_family(path: str, group: str, members: list[str], rows: list[tuple[set[str], str]]):
    head = [f"# Sound-change chains, family {group}: OIA (CDIAL headwords) → each lect below.",
            "# One file for the family (sound_changes/families.py; format in sca.py, \"Family files\"): a line",
            "# prefixed [A B] belongs only to those lects, [-C] to all but C; an unprefixed line to every lect.",
            "# A lect's chain is this file read with only its own lines, in order. Prefer widening a selector",
            "# (or dropping it) to adding a near-duplicate line, so smaller lects do not miss shared changes.",
            "%lects " + " ".join(members), ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(head + [selector(l, members) + r.strip() for l, r in rows]) + "\n")


# ── editing helpers for tools that install or revert lines (probe/installer scripts) ──────────────

def anchor_line(path: str, idx: int) -> tuple[str, int]:
    """(source file, line number) of the idx-th line of the lect's own text (sca.lect_text(path)
    split on newlines)."""
    src, _ = sca.resolve_lect(path)
    return src, sca.lect_lines(path)[idx][0]


def insert_lines(items) -> None:
    """items: (lect path, idx, text) — put `text` before the idx-th line of that lect's own text.
    Lines with the same source position and text are written once, selected for all their lects."""
    by_src: dict[str, dict[tuple[int, str], list[str]]] = {}
    for path, idx, text in items:
        src, stem = sca.resolve_lect(path)
        n = sca.lect_lines(path)[idx][0] if idx < len(sca.lect_lines(path)) else None
        by_src.setdefault(src, {}).setdefault((n, text), []).append(stem)
    for src, groups_ in by_src.items():
        with open(src, encoding="utf-8") as f:
            lines = f.read().rstrip("\n").split("\n")
        head = sca._family_header(lines)
        for (n, text), stems in sorted(groups_.items(), key=lambda kv: -(kv[0][0] or len(lines) + 1)):
            sel = selector(set(stems), list(head[0])) if head and None not in stems else ""
            pos = (n - 1) if n is not None else len(lines)
            lines.insert(pos, sel + text)
        with open(src, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def drop_lects(src: str, pattern: str, stems) -> int:
    """Take `stems` off every line of a family file whose text matches `pattern` (regex); a line
    left with no lect is deleted. Returns the number of lines changed."""
    import re
    rx = re.compile(pattern)
    with open(src, encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    members, sets = sca._family_header(lines)
    out, changed = [], 0
    for n, raw in enumerate(lines, 1):
        m = sca.LECT_SELECTOR.match(raw)
        body = m.group(2) if m else raw
        if not sca.strip_comment(body).strip() or not rx.search(body) or body.startswith("%lects"):
            out.append(raw); continue
        have = sca._expand_selector(m.group(1).split(), members, sets, f"{src}:{n}") if m else set(members)
        left = have - set(stems)
        if left == have:
            out.append(raw); continue
        changed += 1
        if left:
            out.append(selector(left, list(members)) + body)
    with open(src, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return changed


def signature(path: str):
    rules, tails = sca.load_rules_full(path)
    meta = sca.load_meta(path)
    sig = lambda r: (r.text, r.optional, r.name, r.origin, r.tags, r.cond, r.repeat,
                     tuple(a.text for a in r.alts))
    return ([sig(r) for r in rules], {k: [sig(r) for r in v] for k, v in tails.items()},
            {k: v for k, v in meta.items() if k != "source_tails"})


def cmd_merge(a):
    g = groups(a.rules)
    for group, members in g.items():
        seqs = {}
        for s in members:
            with open(os.path.join(a.rules, f"OIA-{s}.sc"), encoding="utf-8") as f:
                seqs[s] = [l for l in f.read().split("\n") if l.strip()]
        before = {s: signature(os.path.join(a.rules, f"OIA-{s}.sc")) for s in members}
        rows = merge_group(seqs)
        fam = os.path.join(a.rules, f"{group}.sc")
        write_family(fam, group, members, rows)
        # the family must reproduce every lect exactly before the one-lect files go
        tmp = {s: os.path.join(a.rules, f"OIA-{s}.sc") for s in members}
        for p in tmp.values():
            os.rename(p, p + ".premerge")
        try:
            bad = [s for s in members if signature(tmp[s]) != before[s]]
        except Exception:
            for p in tmp.values():
                os.rename(p + ".premerge", p)
            os.remove(fam)
            raise
        if bad:
            for p in tmp.values():
                os.rename(p + ".premerge", p)
            os.remove(fam)
            sys.exit(f"{group}: merged chains differ for {bad}; family file not written")
        for p in tmp.values():
            if a.keep:
                os.rename(p + ".premerge", p + ".bak")
            else:
                os.remove(p + ".premerge")
        shared = sum(1 for l, _ in rows if len(l) == len(members))
        print(f"{group:30} {len(members):2} lects  {sum(len(v) for v in seqs.values()):5} lines → {len(rows):4} "
              f"({shared} shared by all)")


def read_family(path: str) -> tuple[list[str], dict[str, frozenset[str]], list[tuple[set[str], str]]]:
    """(members, sets, rows) of a family file; rows exclude the header comment and %lects/%set lines."""
    with open(path, encoding="utf-8") as f:
        lines = f.read().rstrip("\n").split("\n")
    members, sets = sca._family_header(lines)
    rows, started = [], False
    for n, raw in enumerate(lines, 1):
        m = sca.LECT_SELECTOR.match(raw)
        body = m.group(2) if m else raw
        code = sca.strip_comment(body).strip()
        if code.startswith("%lects"):
            started = True; continue
        if not started or code.startswith("%set") or not body.strip():
            continue
        lects = sca._expand_selector(m.group(1).split(), members, sets, f"{path}:{n}") if m else set(members)
        rows.append((set(lects), body.strip()))
    return list(members), sets, rows


def cmd_unite(a):
    """All family files of a rules directory → one file whose %set NAME lines are the families."""
    fams = sorted(set(sca.family_index(a.rules).values()))
    if len(fams) < 2:
        sys.exit("fewer than two family files: nothing to unite")
    before = {os.path.basename(p): signature(p) for p in sca.lect_paths(a.rules)}
    members, sets, seqs = [], {}, {}
    for p in fams:
        name = os.path.basename(p)[:-3]
        m, fsets, rows = read_family(p)
        if fsets:
            sys.exit(f"{p}: has its own %set lines; unite would need to rename them")
        members += m; sets[name] = m; seqs[name] = rows
    rows = merge_rows(seqs)
    head = ["# Sound-change chains: OIA (CDIAL headwords) → every lect below, in one file.",
            "# Format (sca.py, \"Family files\"): a line prefixed [A B] belongs only to those lects, [-C] to all",
            "# but C, [fam] to a family (the %set lines; languages.tsv Group), [fam -C] to a family but C; an",
            "# unprefixed line to every lect. A lect's chain is this file read with only its own lines, in order.",
            "# Prefer widening a selector (or dropping it) to adding a near-duplicate line, so smaller lects and",
            "# families do not miss shared changes.",
            "%lects " + " ".join(members)]
    head += [f"%set {n} = " + " ".join(m) for n, m in sets.items()] + [""]
    text = "\n".join(head + [selector_sets(l, members, sets) + r for l, r in rows]) + "\n"
    out = os.path.join(a.rules, a.out)
    tmpdir = out + ".unite"
    os.makedirs(tmpdir, exist_ok=True)
    with open(os.path.join(tmpdir, a.out), "w", encoding="utf-8") as f:
        f.write(text)
    after = {os.path.basename(p): signature(p) for p in sca.lect_paths(tmpdir)}
    bad = sorted(n for n in before if after.get(n) != before[n])
    if bad:
        sys.exit(f"united chains differ for {bad}; left the trial file in {tmpdir}")
    for p in fams:
        os.remove(p)
    os.replace(os.path.join(tmpdir, a.out), out)
    os.rmdir(tmpdir)
    shared = sum(1 for l, _ in rows if len(l) == len(members))
    print(f"{len(fams)} family files ({sum(len(r) for r in seqs.values())} lines) → {out}: {len(rows)} lines, "
          f"{shared} shared by all {len(members)} lects; every lect's chain unchanged")


def cmd_explode(a):
    os.makedirs(a.outdir, exist_ok=True)
    paths = [os.path.join(a.rules, f"OIA-{s}.sc") for s in a.stems] if a.stems else sca.lect_paths(a.rules)
    for p in paths:
        with open(os.path.join(a.outdir, os.path.basename(p)), "w", encoding="utf-8") as f:
            f.write(sca.lect_text(p))
    print(f"{len(paths)} one-lect files → {a.outdir}")


def cmd_check(a):
    n = 0
    for p in sca.lect_paths(a.rules):
        sca.load_rules_full(p); sca.load_meta(p); n += 1
    print(f"{n} lect chains load")


def cmd_diff(a):
    old = {os.path.basename(p): p for p in sca.lect_paths(a.old)}
    new = {os.path.basename(p): p for p in sca.lect_paths(a.rules)}
    diff = [n for n in sorted(old.keys() & new.keys()) if signature(old[n]) != signature(new[n])]
    print(f"{len(old.keys() & new.keys())} common lects; only old: {sorted(old.keys() - new.keys())}; "
          f"only new: {sorted(new.keys() - old.keys())}; differing: {diff}")
    return 1 if diff else 0


def cmd_shared(a):
    """Per family (a family file, or each %set of a united file): lines most of it has, with the lects lacking them."""
    for fam in sorted(set(sca.family_index(a.rules).values())):
        with open(fam, encoding="utf-8") as f:
            lines = f.read().split("\n")
        members, sets = sca._family_header(lines)
        groups_ = {n: list(m) for n, m in sets.items()} if sets else {os.path.basename(fam)[:-3]: list(members)}
        for group, gm in groups_.items():
            if a.groups and group not in a.groups:
                continue
            print(f"\n== {group} ({len(gm)} lects)")
            for n, raw in enumerate(lines, 1):
                m = sca.LECT_SELECTOR.match(raw)
                if not m or not sca.strip_comment(m.group(2)).strip():
                    continue
                have = sca._expand_selector(m.group(1).split(), members, sets, f"{fam}:{n}") & set(gm)
                miss = [s for s in gm if s not in have]
                if len(have) >= 2 and 0 < len(miss) <= max(1, len(gm) // 4):
                    print(f"  {n:4}  missing {' '.join(miss):28} {m.group(2).strip()[:90]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("merge"); p.add_argument("--rules", default=RULES); p.add_argument("--keep", action="store_true")
    p = sub.add_parser("explode"); p.add_argument("outdir"); p.add_argument("stems", nargs="*"); p.add_argument("--rules", default=RULES)
    p = sub.add_parser("check"); p.add_argument("--rules", default=RULES)
    p = sub.add_parser("diff"); p.add_argument("old"); p.add_argument("--rules", default=RULES)
    p = sub.add_parser("unite"); p.add_argument("--rules", default=RULES); p.add_argument("--out", default="chains.sc")
    p = sub.add_parser("shared"); p.add_argument("groups", nargs="*"); p.add_argument("--rules", default=RULES)
    a = ap.parse_args()
    sys.exit({"merge": cmd_merge, "unite": cmd_unite, "explode": cmd_explode, "check": cmd_check, "diff": cmd_diff,
              "shared": cmd_shared}[a.cmd](a) or 0)


if __name__ == "__main__":
    main()

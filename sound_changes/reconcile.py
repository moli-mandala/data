#!/usr/bin/env python3
"""reconcile.py — harmonise the conventions the group agents used in the rule files.

    python sound_changes/reconcile.py audit     # report inconsistencies, change nothing
    python sound_changes/reconcile.py apply     # rewrite tags (comments only) and verify scores
    python sound_changes/reconcile.py notation  # move mid-chain @notation rules to the tail when inert

What `apply` changes:
  * group-specific `@x-<group>-…` tags that name the same process in several groups are replaced
    by one shared tag (TAG_MAP; the new tags are added to CATALOG.md). Tags live in `;` comments,
    so rule behaviour cannot change; every file is re-scored to confirm train NED is identical.
What `audit` reports (for manual follow-up):
  * class names defined with different members in different files;
  * `@notation` rules placed before phonological rules (notation should be a chain tail or a
    `%source` block);
  * remaining `@x-` tags used by only one group.
"""
import csv
import os
import re
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sca  # noqa: E402

TAG_MAP = {
    # one shared name for the same process across groups
    "@anusvara-loss": ["@x-nw-anusvara-loss", "@x-cep-anusvara-loss", "@x-mia-insular-anusvara-loss",
                       "@x-dardic-south-anusvara-loss", "@x-eastern-i-anusvara-loss"],
    "@aphaeresis": ["@x-rg-aphaeresis", "@x-wpah-aphaeresis"],
    "@asp-devoicing": ["@x-nw-aspirate-devoicing", "@x-cep-devoicing-asp"],
    "@aspiration-shift": ["@x-cep-aspiration-shift", "@x-dardic-north-aspiration-shift"],
    "@final-a-short": ["@x-marathi-bhil-final-shortening", "@x-eastern-final-a-short", "@x-nw-final-a-short",
                       "@x-bihari-tharu-final-long-short"],
    "@deaffrication": ["@x-marathi-bhil-ch-s", "@x-marathi-bhil-c-s", "@x-eastern-deaffrication",
                       "@x-rg-deaffrication", "@x-mia-insular-j-d"],
    "@deretroflexion": ["@x-eastern-deretroflex", "@x-nw-deretroflexion", "@x-cep-retroflex-loss",
                        "@x-dardic-south-deretroflex"],
    "@dv-b": ["@x-rg-dv-b", "@x-nw-dv-b"],
    "@final-lengthening": ["@x-wpah-final-lengthening", "@x-dardic-south-final-lengthening"],
    "@final-shortening": ["@x-cep-final-i-shortening", "@x-wpah-final-shortening"],
    "@length-loss": ["@x-eastern-high-length-loss", "@x-cep-high-vowel-shortening", "@x-bihari-tharu-length-loss",
                     "@x-cep-length-loss", "@x-eastern-length-loss", "@x-dardic-north-length-loss",
                     "@x-dardic-south-length-loss", "@x-marathi-bhil-iu-shortening"],
    "@rt-retroflex": ["@x-rg-rt-retroflex", "@x-hindi-rt-retroflex", "@x-nw-rt-cerebral",
                      "@x-dardic-north-r-retroflex", "@x-mia-insular-r-retroflex"],
    "@kh-x": ["@x-cep-kh-x", "@x-dardic-south-kh-spirant"],
    "@l-r": ["@x-hindi-l-r", "@x-nw-l-r", "@x-bihari-tharu-l-r"],
    "@mr-mb": ["@x-hindi-mr-mb", "@x-rg-mr-mb"],
    "@n-retroflex": ["@x-nw-n-retroflex", "@x-cep-n-retroflex"],
    "@n-dissimilation": ["@x-nw-n-dissimilation", "@x-cep-n-dissimilation", "@x-wpah-retro-dissim"],
    "@nasal-assim": ["@x-eastern-nasal-assim", "@x-cep-nasal-voiced-assim", "@x-nw-mb-mm", "@x-wpah-nd-nn",
                     "@x-bihari-tharu-nd-nn", "@x-mia-insular-nd-nn"],
    "@nc-simplify": ["@x-nw-nc-nasal", "@x-dardic-north-nd-n"],
    "@ts-cluster": ["@x-dardic-north-ts-affricate", "@x-cep-ts-cch"],
    "@tr-affricate": ["@x-dardic-north-tr-affricate", "@x-dardic-south-tr-affricate", "@x-nur-pr-tr-c",
                      "@x-wpah-tr-c"],
    "@tr-lateral": ["@x-dardic-north-tr-lateral", "@x-dardic-south-Cr-lateral", "@x-wpah-tr"],
    "@tr-retroflex": ["@x-nw-tr-cerebral"],
    "@sv-sp": ["@x-mia-insular-sv-sp", "@x-dardic-north-sv-sp"],
    "@ph-f": ["@x-wpah-ph-f", "@x-mia-insular-p-f"],
    "@pretonic-shortening": ["@x-nw-pretonic-shortening", "@x-eastern-pretonic-shortening"],
    "@t-l": ["@x-cep-t-l", "@x-dardic-south-t-l", "@x-dardic-north-t-r"],
    "@s-stop-loss": ["@x-dardic-north-s-stop-loss"],
    "@cluster-simplify": ["@x-dardic-north-cr-loss"],
    "@sibilant-s": ["@x-eastern-s-dental", "@x-dardic-south-s-merger"],
    "@s-x": ["@x-eastern-s-x", "@x-dardic-south-s-x"],
    "@prati-retroflexion": ["@x-mia-insular-prati"],
}
OLD2NEW = {old: new for new, olds in TAG_MAP.items() for old in olds}

NEW_TAG_DOCS = {
    "@anusvara-loss": "anusvāra lost without a nasal vowel (ṁ > ∅)",
    "@aphaeresis": "loss of an initial vowel",
    "@asp-devoicing": "voiced aspirates > voiceless aspirates (Romani, Lomavren, Punjabi-area)",
    "@aspiration-shift": "aspiration displaced onto an initial voiced stop (jambʰa > cʰam, gōdʰūma > gʰōm)",
    "@final-a-short": "shortening of final -ā (Apabhraṁśa-type), before apocope",
    "@deaffrication": "c (cʰ j) > s (z), and j > d in Insular",
    "@deretroflexion": "retroflex stops > dentals",
    "@dv-b": "dv > b(b)",
    "@final-lengthening": "final -i -u > -ī -ū",
    "@final-shortening": "final -ī > -i",
    "@length-loss": "loss of the vowel-length contrast (all vowels or the high vowels)",
    "@rt-retroflex": "r + dental > retroflex (geminate) (rt > ṭṭ)",
    "@kh-x": "kʰ > x",
    "@l-r": "intervocalic l > r",
    "@mr-mb": "m + liquid > mb",
    "@n-dissimilation": "ṇ > n after r ṛ ṇ ḷ",
    "@nasal-assim": "nasal + voiced stop > geminate nasal (mb > mm, nd > nn, ŋg > ŋ)",
    "@nc-simplify": "nasal + stop > nasal (dánta > don)",
    "@ts-cluster": "ts ps (ks) > affricate: cʰ in Central/Romani, ʦ in Dardic (outcome read from the rule)",
    "@tr-affricate": "stop + r > affricate (tr > ʦ̣, c)",
    "@tr-lateral": "stop + r > lateral or lateral affricate (tr > ɬ, ṭ͡ḷ)",
    "@tr-retroflex": "tr > ṭr (Sindhi)",
    "@sv-sp": "śv / sv > śp / sp",
    "@ph-f": "pʰ (p) > f",
    "@pretonic-shortening": "long vowel shortened before a stressed (heavy) syllable",
    "@t-l": "intervocalic dental > l / r",
    "@s-stop-loss": "s + stop > stop (after lenition)",
    "@s-x": "ś / ṣ > x",
    "@prati-retroflexion": "prati- > paṭi-",
}


def languages():
    with open(os.path.join(HERE, "languages.tsv"), encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def rule_files():
    for L in languages():
        p = os.path.join(HERE, "rules", f"OIA-{L['File']}.sc")
        if sca.has_chain(p):
            yield L, p


def retag_line(line: str) -> str:
    if ";" not in line:
        return line
    code, comment = line.split(";", 1)
    def sub(m):
        return OLD2NEW.get(m.group(0), m.group(0))
    comment = re.sub(r"@[\w\-:.̀-ͯʰ-˿]+", sub, comment)
    # collapse duplicates created by the merge (e.g. "@n-retroflex @n-retroflex")
    seen, out = set(), []
    for tok in re.split(r"(\s+)", comment):
        if tok.startswith("@") and tok in seen:
            continue
        if tok.startswith("@"):
            seen.add(tok)
        out.append(tok)
    return code + ";" + "".join(out)


def cmd_apply():
    import evaluate as ev
    changed = 0
    for L, p in rule_files():
        sca.require_one_lect_file(p)
        text = open(p, encoding="utf-8").read()
        new = "\n".join(retag_line(x) for x in text.split("\n"))
        if new == text:
            continue
        rules = sca.load_rules(p)
        _, tr, _ = ev.datasets(p)
        before = ev.metrics(tr, ev.run(rules, tr.etyma))["ned"]
        tmp = p + ".tmp"
        open(tmp, "w", encoding="utf-8").write(new)
        os.replace(tmp, p)
        rules = sca.load_rules(p)
        _, tr, _ = ev.datasets(p)
        after = ev.metrics(tr, ev.run(rules, tr.etyma))["ned"]
        assert abs(before - after) < 1e-12, (p, before, after)
        changed += 1
    print(f"retagged {changed} files; train NED unchanged in every one")
    cat = os.path.join(HERE, "CATALOG.md")
    txt = open(cat, encoding="utf-8").read()
    if "## Shared tags added by reconciliation" not in txt:
        txt += ("\n## Shared tags added by reconciliation\n"
                "These replaced group-specific `@x-…` tags that named the same process in several groups.\n\n"
                "| tag | meaning | replaced |\n|---|---|---|\n")
        for t, doc in NEW_TAG_DOCS.items():
            txt += f"| `{t}` | {doc} | {', '.join('`' + o + '`' for o in TAG_MAP[t])} |\n"
        for t in ("@n-retroflex", "@cluster-simplify", "@sibilant-s"):
            txt += f"| `{t}` | (existing tag) | {', '.join('`' + o + '`' for o in TAG_MAP[t])} |\n"
        open(cat, "w", encoding="utf-8").write(txt)
        print("CATALOG.md updated")


def cmd_audit():
    classes = defaultdict(lambda: defaultdict(list))
    notation_early = []
    xtags = defaultdict(set)
    for L, p in rule_files():
        lines = sca.lect_text(p).split("\n")
        block = False
        seen_phon_after = False
        last_phon = 0
        notation_lines = []
        for n, raw in enumerate(lines, 1):
            s = raw.split(";", 1)[0].strip()
            if raw.lstrip().startswith("#") or not s:
                continue
            if s.startswith("%source"):
                block = True
                continue
            if s.startswith("%end"):
                block = False
                continue
            if s.startswith("class "):
                name, _, members = s[6:].partition("=")
                classes[name.strip()][" ".join(members.split())].append(L["Lang"])
                continue
            if s.startswith("%") or block:
                continue
            tags = re.findall(r"@[\w\-:.̀-ͯʰ-˿]+", raw.split(";", 1)[1]) if ";" in raw else []
            for t in tags:
                if t.startswith("@x-"):
                    xtags[t].add(L["Group"])
            if "@notation" in tags:
                notation_lines.append(n)
            elif tags and not {"@morph", "@acc-loss"} & set(tags):
                last_phon = n
        early = [n for n in notation_lines if n < last_phon]
        if early:
            notation_early.append((L["Lang"], early[:5]))
    print("== class names with different definitions across files")
    for name, defs in sorted(classes.items()):
        if len(defs) > 1:
            print(f"  {name}: {len(defs)} definitions")
            for members, ls in sorted(defs.items(), key=lambda kv: -len(kv[1]))[:4]:
                print(f"      [{len(ls):3d} files] {members[:90]}")
    print(f"\n== @notation rules placed before later phonological rules ({len(notation_early)} files)")
    for lang, ns in notation_early[:40]:
        print(f"  {lang}: lines {ns}")
    print(f"\n== remaining @x- tags ({len(xtags)}); used by >1 group:")
    for t, gs in sorted(xtags.items()):
        if len(gs) > 1:
            print(f"  {t}: {sorted(gs)}")


def cmd_notation():
    """Move mid-chain @notation rules to the chain tail when that leaves train NED unchanged
    (|Δ| ≤ 1e-4); otherwise leave them and report them as position-sensitive (not pure notation)."""
    import evaluate as ev
    moved, kept = 0, []
    for L, p in rule_files():
        sca.require_one_lect_file(p)
        lines = open(p, encoding="utf-8").read().split("\n")
        while True:
            rules = sca.load_rules(p)
            _, tr, _ = ev.datasets(p)
            base = ev.metrics(tr, ev.run(rules, tr.etyma))["ned"]
            own = [r for r in rules if not r.origin]   # included block rules are edited in blocks/
            by_line = {r.line: r for r in own}
            last_phon = max((r.line for r in own if "@notation" not in lines[r.line - 1]), default=0)
            cands = [r.line for r in own if "@notation" in lines[r.line - 1].split(";", 1)[-1]
                     and r.line < last_phon and (L["Lang"], r.line) not in {(k[0], k[1]) for k in kept}]
            if not cands:
                break
            ln = cands[0]
            trial = [r for r in rules if r.line != ln] + [by_line[ln]]
            new = ev.metrics(tr, ev.run(trial, tr.etyma))["ned"]
            if abs(new - base) <= 1e-4:
                # physically move the line to just after the last chain rule (before any %source blocks)
                text = lines[ln - 1]
                rest = lines[:ln - 1] + lines[ln:]
                last = max(i for i, x in enumerate(rest) if x.strip() and not x.lstrip().startswith(("#", ";", "%", "class "))
                           and not _in_block(rest, i))
                rest.insert(last + 1, text)
                tmp = p + ".tmp"
                open(tmp, "w", encoding="utf-8").write("\n".join(rest))
                os.replace(tmp, p)
                lines = rest
                chk = ev.metrics(tr, ev.run(sca.load_rules(p), tr.etyma))["ned"]
                assert abs(chk - new) < 1e-9, (p, chk, new)
                moved += 1
            else:
                kept.append((L["Lang"], ln, lines[ln - 1].split(";")[0].strip(), round(new - base, 4)))
    print(f"moved {moved} notation rules to chain tails (train NED unchanged)")
    print(f"{len(kept)} notation rules are position-sensitive (relabel or keep):")
    for k in kept:
        print("  ", k)


def _in_block(lines, i):
    depth = False
    for x in lines[:i + 1]:
        s = x.split(";", 1)[0].strip()
        if s.startswith("%source"):
            depth = True
        elif s.startswith("%end"):
            depth = False
    return depth


if __name__ == "__main__":
    {"apply": cmd_apply, "audit": cmd_audit, "notation": cmd_notation}[sys.argv[1]]()

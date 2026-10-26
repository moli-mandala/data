"""Lemma-header, attestation and cross-reference parsing for OAD units."""

from __future__ import annotations

import re

from parse import (LOCATOR, SUPERSCRIPT_DIGITS, clean, flatten, grammar_code_tags,
                   label_tags, top_level_groups)

ROOT_GROUP = re.compile(r"^√[¹²³⁴]?[^\s‘’()]+$")


def _first_bold(classes: list[str], text: str, start: int = 0) -> int:
    for i in range(start, len(text)):
        if classes[i] == "B" and text[i].strip() and text[i] not in "[]":
            return i
    return -1


def _headword_region(text: str, classes: list[str], i: int):
    """Read bold headword material from ``i``; return (parts, braces, end)."""
    parts, braces, cur = [], [], ""
    n = len(text)
    while i < n:
        ch, k = text[i], classes[i]
        if k == "B" or ch in SUPERSCRIPT_DIGITS:
            cur += ch
            i += 1
            continue
        if ch in "-°˚" and cur.strip() and not cur.endswith(" "):
            cur += ch
            i += 1
            continue
        m = re.match(r"\s*(/|,|\bor\b)?\s*", text[i:])
        joined = m and (m.group(1) or (m.end() and cur.rstrip().endswith((",", "/"))))
        if joined and i + m.end() < n and classes[i + m.end()] == "B" \
                and text[i + m.end()] not in "(|":
            parts.append(cur)
            cur = ""
            i += m.end()
            continue
        m = re.match(r"\s*\{[^}]*\}", text[i:])
        if m:
            braces.append(m.group().strip())
            i += m.end()
            continue
        break
    parts.append(cur)
    out = []
    for part in parts:
        for piece in re.split(r"\s*/\s*|,\s+", part.replace("[", "").replace("]", "")):
            piece = piece.strip().rstrip(",").strip()
            if piece:
                out.append(piece)
    # an elided root alternant (√š́iiā/š́ā, √mruk/mruc) repeats the root sign
    if out and out[0].startswith("√"):
        out = [p if p.startswith("√") or "√" in p or "-√" in p else "√" + p for p in out]
    return out, braces, i


def header_end(text: str, start: int) -> int:
    """End of the lemma header: a comment (♠), or a depth-0 colon opening passages."""
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "(":
            depth += 1
            if text[i + 1:i + 2] == "♠" and depth == 1:
                return i
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "♠" and depth == 0:
            return i
        elif ch == ":" and depth == 0:
            rest = text[i + 1:].lstrip()
            if rest.startswith("―") or LOCATOR.match(rest):
                return i
    return len(text)


def parse_lemma(unit: dict) -> dict:
    text, classes = flatten(unit)
    marker = text[:1]
    hstart = _first_bold(classes, text, 1)
    bracketed = "[" in text[1:hstart if hstart > 0 else 4]
    rec = {"marker": marker, "text": text, "bracketed": bracketed, "headwords": [],
           "braces": [], "tags": [], "glosses": [], "variants_or": [], "readings": [],
           "root_groups": [], "residue": [], "unparsed_labels": [], "flags": []}
    if hstart < 0:
        rec["flags"].append("no-headword")
        return rec
    parts, braces, i = _headword_region(text, classes, hstart)
    rec["headwords"], rec["braces"] = parts, braces
    end = header_end(text, i)
    rec["header"] = clean(text[:end])
    if text[:end].count("(") != text[:end].count(")"):
        rec["flags"].append("unbalanced parentheses in the printed header")
    rec["after_header"] = clean(text[end:])
    stop_gloss = False
    pending = None  # a second, separately labelled headword: a-hū- (m.), aŋhᵛī- (f.)
    seen_quote = False
    rec["gendered"] = []
    for kind, a, b in top_level_groups(text, i, end):
        chunk = text[a:b]
        if kind == "text" and not seen_quote and rec["tags"] \
                and re.match(r"^\s*(?:,|/|or)\s*\S", chunk):
            bold = re.sub(r"\{[^}]*\}", "", "".join(
                c for c, k in zip(chunk, classes[a:b]) if k == "B")).strip(" ,/")
            if bold and " " not in bold:
                pending = bold
                if re.search(r"\{[^}]*\}", chunk):
                    rec["braces"].extend(re.findall(r"\{[^}]*\}", chunk))
                continue
        if kind == "text" and seen_quote and not stop_gloss:
            words = re.findall(r"[^\W\d_]+", chunk)
            if any(k == "B" for k in classes[a:b]) and words:
                # the prose cites another word; its quotes gloss that word, not the lemma
                stop_gloss = True
        if kind == "text":
            if "→" in chunk:
                stop_gloss = True
                rec["residue"].append(clean(chunk[chunk.index("→"):]))
            elif clean(chunk) not in ("", ",", ";", "[", "]", "]."):
                rec["residue"].append(clean(chunk))
            continue
        if kind == "quote":
            seen_quote = True
            if stop_gloss:
                rec["residue"].append(clean(chunk))
            else:
                rec["glosses"].append(clean(chunk[1:-1]))
            continue
        inner = chunk[1:-1]
        inner_classes = classes[a + 1:b - 1]
        tags, ok = label_tags(inner)
        if not ok:
            tags, ok = grammar_code_tags(inner)
        if ok and pending:
            rec["gendered"].append((pending, tags))
            pending = None
            continue
        if ok and not stop_gloss:
            for tag in tags:
                if tag not in rec["tags"]:
                    rec["tags"].append(tag)
            continue
        stripped = inner.strip()
        if re.match(r"^or\s", stripped) and "B" in inner_classes:
            alts = [p for p in _headword_region(inner, inner_classes,
                                                 _first_bold(inner_classes, inner))[0]]
            rec["variants_or"].extend(alts)
            continue
        if re.match(r"^(v\.l\.|mss\.)", stripped):
            rec["readings"].append(clean(chunk))
            continue
        bold = "".join(c for c, k in zip(inner, inner_classes) if k == "B").strip()
        head = stripped.split("‘", 1)[0].strip()
        if bold and bold == head and ROOT_GROUP.match(bold):
            rec["root_groups"].append(bold)
        rec["residue"].append(clean(chunk))
    if pending:
        rec["gendered"].append((pending, []))
    return rec


def parse_attestation(unit: dict) -> dict:
    text, classes = flatten(unit)
    body = text[1:].strip()
    bi = _first_bold(classes, text, 1)
    form = ""
    if bi >= 0:
        j = bi
        while j < len(text) and (classes[j] == "B" or text[j] in SUPERSCRIPT_DIGITS) \
                and not re.match(r"\s*\(", text[j:j + 2]):
            j += 1
        form = clean(text[bi:j]).strip("[] ")
    code = re.search(r"\(([^()]*)\)", text[bi:] if bi >= 0 else body)
    passages = re.findall(r"(?:YH|Y)?\d+[,.]\d+", body)
    return {"form": form, "code": code.group(1) if code else "", "text": body,
            "passages": passages, "bracketed": body.startswith("[")}


def parse_xref(unit: dict) -> dict:
    text, classes = flatten(unit)
    body = text[1:].strip()
    forms, cur = [], ""
    arrow = text.find("→")
    for ch, cls in zip(text[:arrow if arrow >= 0 else len(text)], classes):
        if cls == "BI" and ch not in ",":
            cur += ch
        elif cur.strip():
            forms.append(cur.strip())
            cur = ""
    if cur.strip():
        forms.append(cur.strip())
    target = body.split("→", 1)[1].strip() if "→" in body else ""
    target_bold = []
    if "→" in text:
        k = text.index("→")
        cur = ""
        for ch, cls in zip(text[k:], classes[k:]):
            if cls == "B":
                cur += ch
            elif cur.strip():
                target_bold.append(cur.strip())
                cur = ""
        if cur.strip():
            target_bold.append(cur.strip())
    return {"text": body, "forms": forms, "target": target, "target_bold": target_bold}

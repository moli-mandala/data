"""Stage 4: Iranian citation items inside a language section.

A section body (up to the first top-level ``◊``) is a list of preverb groups separated by
``||``; a group may open with ``(+ *fra-)``. Inside a group, items are separated by top-level
commas/semicolons. An item is ``[?] [grammar] [Label [(Sub)]] form [/phonemic/] [(notes)]
[‘gloss’]``. Unlabelled items inherit the previous label of the section (or the section's
own language). A gloss scopes back over the preceding gloss-less items of the same group
(Cheung's ``MMP b’m, BMP b’m ‘splendour’``); it never crosses ``||`` or ``;``.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import tokens as T

HERE = Path(__file__).resolve().parent

NON_IRANIAN = {
    "Arab.", "Arm.", "Georg.", "Tk.", "Syr.", "Aram.", "Hebr.", "Akkad.", "Mand.", "Toch.",
    "Udm.", "Hung.", "Russ.", "Ukr.", "Bulg.", "OCS", "Brah.", "BSkt.", "Skt.", "Abkh.", "Rom.",
    "Khow.", "Wai.", "Lah.", "Si.", "Hi.", "Pkt.", "Chuv.", "Gr.", "Lat.", "Goth.", "Lith.",
    "Ved.", "IAr.", "Dard.", "Babyl.", "Elam.", "Chin.", "Mong.", "Tib.", "Uig.", "Pol.", "Cz.",
    "SCr.", "Slov.", "Blruss.", "PSl.", "Engl.", "Germ.", "Lyc.", "Hitt.", "Luw.", "Khetr.",
    "Panj.", "Kafr. LW",
}
NON_IRANIAN_LOAN = "LW"  # "Arm. LW apuṙ": Iranian loanword in a non-Iranian language

GRAMMAR = {
    "m.": ["m"], "f.": ["f"], "n.": ["n"], "pl.": ["pl"], "sg.": ["sg"],
    "caus.": ["caus"], "pass.": ["pass"], "intr.": ["intr"], "tr.": ["tr"],
    "denom.": ["denom"], "denomin.": ["denom"], "adj.": ["adj"], "adv.": ["adv"],
    "impv.": ["impv"], "subj.": ["subjunctive"], "pres.": ["pres"], "inf.": ["inf"],
    "ppp.": ["ppp"], "pret.": ["pret"], "aor.": ["aor"], "impers.": ["impersonal"],
    "redupl.": ["reduplicated"], "intens.": ["intensive"], "neg.": ["neg"], "partic.": ["participle"],
    "1sg.": ["1sg"], "2sg.": ["2sg"], "3sg.": ["3sg"], "1pl.": ["1pl"], "3pl.": ["3pl"],
    "perf.": ["perfect"], "PN": ["proper-noun"], "inch.": ["inchoative"], "inch": ["inchoative"], "ind.": ["ind"], "pass": ["pass"], "intr": ["intr"], "LW": ["loanword"], "(LW)": ["loanword"],
}
# printed grammar labels without a canonical Jambu tag (kept in Notes; see proposals.md)
GRAMMAR_NOTE = {"sec.", "old", "orig.", "supplet.", "iter.", "nomin.", "Gpl.", "Gsg.",
                "Asg.", "Lsg.", "Lpl.", "obl.", "instr.", "loc.", "dial.", "class.", "obs.",
                "stem", "pret. stem", "-iter."}


@dataclass
class Item:
    group: int
    index: int
    preverb: str
    label: str           # resolved lects.csv label ("Kurd. (Kurm.)", "§AVESTAN", …)
    printed_label: str   # label words as printed before the form (may be inherited)
    form: str            # printed form (plain Unicode, NFC)
    phonemic: str = ""
    gloss: str = ""
    gloss_inherited: bool = False
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    doubt: bool = False          # "?" before the item (Cheung: connection uncertain)
    reading: str = ""            # "tentative" (ˣ) / "emended" (⁺) form reading
    status: str = "form"         # form | non_iranian | unmapped_label | not_a_form
    reason: str = ""
    raw: str = ""


def load_lects(path: Path = HERE / "lects.csv") -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["Source_Label"]: row for row in csv.DictReader(handle)}


SECTION_LABEL_OVERRIDES = {("SOGDIAN", "M."): "MSogd."}
RUBRIC = re.compile(r"(?i:widely|well|also|only)\s+(?:attested|borrowed)[A-Za-z ,]*:?")

SECTION_LANG = {"AVESTAN", "OLD PERSIAN", "MIDDLE PERSIAN", "PARTHIAN", "KHOTANESE", "SOGDIAN",
                "CHORESMIAN", "BACTRIAN"}
FORM_SECTIONS = SECTION_LANG | {"NWIR", "NEIR", "MISC", ""}


def clean_form(text: str) -> tuple[str, str, list[str]]:
    """Plain printed form; returns (form, reading flag, notes)."""
    form = T.strip_markers(text).strip()
    form = unicodedata.normalize("NFC", re.sub(r"\s+", " ", form))
    form = re.sub(r"\s*/\s*", "/", form)  # principal parts split by a line break: "bastan/ band-"
    reading = ""
    if form.startswith("ˣ") or "ˣ" in form:
        reading = "tentative"
    if form.startswith("⁺") or "⁺" in form or form.startswith("+"):
        reading = "emended"
    return form.strip(" ,;:/"), reading, []


def _split_items(tokens: list[T.Token]) -> list[list[list[T.Token]]]:
    groups: list[list[list[T.Token]]] = [[[]]]
    for tok in tokens:
        prev = groups[-1][-1][-1] if groups[-1][-1] else None
        if (tok.kind == "sep" and tok.text == "||") or (
                tok.kind == "paren" and tok.plain.lstrip().startswith(("+", "? +"))
                and "*" in tok.plain and prev is not None and prev.kind == "gloss"):
            # "‘to hear’(+ *ham-) LKh. …": a preverb group opened without the printed ||
            groups.append([[]])
            if tok.kind == "paren":
                groups[-1][-1].append(tok)
        elif tok.kind == "sep" and tok.text in ",;":
            groups[-1].append([tok])  # keep the separator to know ';' boundaries
        else:
            groups[-1][-1].append(tok)
    return groups


def _closing_run(after: list[T.Token], k: int) -> int:
    """Index of the word ending in ’ that closes a run of plain words starting at ``k`` (0 if none)."""
    j = k
    while j < len(after) and after[j].kind == "word" and j - k < 12:
        if after[j].plain.strip().endswith("’"):
            return j
        j += 1
    return 0


def _split_on_label(seg: list[T.Token], lects) -> list[list[T.Token]]:
    """Split ``Pash. lwaγza (< …) ? Pash. salwāγa`` (a missing comma) at the second label."""
    out = [[]]
    seen_form = False
    for k, tok in enumerate(seg):
        if seen_form and tok.kind == "word" and (tok.plain.strip() in lects
                                                 or tok.plain.strip() in NON_IRANIAN):
            nxt = [t for t in seg[k + 1:k + 4] if not (t.kind == "word" and t.plain.strip() in ("?", "I.", "D."))
                   and not (t.kind == "paren" and len(t.plain) <= 12)]
            if nxt and nxt[0].kind == "form" and not tok.plain.strip().startswith("§"):
                # a preceding "?" belongs to the new item
                carry = []
                while out[-1] and out[-1][-1].kind == "word" and out[-1][-1].plain.strip() == "?":
                    carry.insert(0, out[-1].pop())
                out.append(carry)
                seen_form = False
        out[-1].append(tok)
        if tok.kind == "form":
            seen_form = True
    return [part for part in out if part]


def _label_from_words(words: list[str], lects, current_main: str) -> tuple[str | None, list[str], list[str], bool]:
    """Return (label key, tags, leftover words, doubt) from the words before a form."""
    tags: list[str] = []
    leftover: list[str] = []
    doubt = False
    main, sub = None, None
    i = 0
    while i < len(words):
        w = words[i]
        two = " ".join(words[i:i + 2])
        if w in ("?", "(?)", "??"):
            doubt = True
        elif two in GRAMMAR and two not in lects:
            tags += GRAMMAR[two]
            i += 1
        elif w in GRAMMAR and w not in lects:
            tags += GRAMMAR[w]
        elif w in ("I.", "D.") and (main or current_main or "").startswith("Oss"):
            sub = f"Oss. {w}"
        elif two in lects and main is None and two.split()[0] not in lects:
            main = two
            i += 1
        elif f"{w} {words[i + 1]}" in lects if i + 1 < len(words) else False:
            main = two
            i += 1
        elif w in lects and main is None and not w.startswith("("):
            main = w
        elif w.startswith("(") and w.endswith(")"):
            inner = w[1:-1].strip()
            inner_words = inner.split()
            if inner_words and inner_words[0] == "also":
                inner_words = inner_words[1:]
                leftover.append("also")
            inner = " ".join(inner_words)
            base = main or current_main
            if base and f"{base.split(' (')[0]} ({inner})" in lects:
                sub = f"{base.split(' (')[0]} ({inner})"
            elif inner in lects:
                sub = inner
            elif inner and all(p.strip() in GRAMMAR for p in re.split(r"[,/]", inner)):
                for p in re.split(r"[,/]", inner):
                    tags += GRAMMAR[p.strip()]
            else:
                leftover.append(w)
        elif w in lects and main is not None and sub is None:
            sub = w
        else:
            leftover.append(w)
        i += 1
    label = sub or main
    return label, tags, leftover, doubt


def parse_section(text: str, section: str, lects) -> tuple[list[Item], str, str]:
    """Return (items, body text, commentary text) for a language section."""
    text = T.normalize_quotes(text)
    toks = T.tokenize(text)
    # italic English rubrics ("Widely attested:", "Well attested but not in NP:") are not forms
    for k, tok in enumerate(toks):
        nxt = toks[k + 1].plain.strip() if k + 1 < len(toks) else ""
        if tok.kind == "form" and RUBRIC.fullmatch(tok.plain.strip()) and (
                nxt.startswith(":") or tok.plain.strip().endswith(":") or len(tok.plain.split()) >= 2):
            tok.kind = "rubric"
    toks = [t for t in toks if t.kind != "rubric" and not (t.kind == "word" and t.plain.strip() == ":")]
    commentary = ""
    for k, tok in enumerate(toks):
        if tok.kind == "diamond":
            commentary = text[tok.start:]
            toks = toks[:k]
            break
    body = text[:len(text) - len(commentary)] if commentary else text
    default = f"§{section}" if section in SECTION_LANG else ""
    current = default
    items: list[Item] = []
    for g, group in enumerate(_split_items(toks)):
        group = [part for seg in group for part in _split_on_label(seg, lects)]
        preverb = ""
        group_items: list[Item] = []
        boundary_before: list[bool] = []
        pending_labels: list[str] = []
        for seg in group:
            sep = ""
            if seg and seg[0].kind == "sep":
                sep = seg[0].text
                seg = seg[1:]
            if not preverb and seg and seg[0].kind == "paren" and seg[0].plain.lstrip().startswith(("+", "? +")):
                preverb = re.sub(r"\s+", " ", seg[0].plain.strip())
                seg = seg[1:]
            forms = [t for t in seg if t.kind == "form"]
            seg_words = [t.plain.strip() for t in seg if t.kind == "word"]
            if not forms and seg and all(t.kind == "word" for t in seg) and seg_words \
                    and all(w in lects or w == "?" for w in seg_words):
                # "Abyan., Abz. yöšta/yǖz-": a label sharing the next item's form
                pending_labels.append(" ".join(w for w in seg_words if w != "?"))
                continue
            if not forms and group_items and seg:
                kinds = {t.kind for t in seg}
                plain_seg = " ".join(T.strip_markers(t.text).strip() for t in seg)
                prev = group_items[-1]
                if kinds <= {"word", "gloss", "paren"} and "gloss" in kinds and all(
                        re.fullmatch(r"\d\.?|\?", t.plain.strip()) for t in seg if t.kind == "word"):
                    # further numbered senses of the same form: "1. ‘…’, 2. ‘…’, 3. ‘…’"
                    extra = "; ".join(clean_gloss(T.strip_markers(t.text)) for t in seg if t.kind == "gloss")
                    prev.gloss = f"{prev.gloss}; {extra}" if prev.gloss else extra
                    continue
                if kinds == {"word"} and plain_seg.endswith("’") and prev.gloss:
                    # a gloss continued after a misplaced closing quote (p. 368 ‘…’, table-cloth’)
                    prev.gloss = f"{prev.gloss}, {clean_gloss(plain_seg)}"
                    continue
            if not forms:
                # a gloss/notes continuation without a form (e.g. "(+ *ā-) ‘to apportion’")
                if group_items and any(t.kind == "gloss" for t in seg) and sep == "":
                    pass
                if seg:
                    glosses = [t for t in seg if t.kind == "gloss"]
                    if glosses and not group_items and section in SECTION_LANG:
                        # Avestan "(+ *ā-) ‘to apportion’": a preverb sense of the section's
                        # headline form; no new printed form.
                        items.append(Item(g, len(items), preverb, current, "", "",
                                          gloss=T.strip_markers(glosses[0].text).strip(),
                                          status="preverb_sense_only", reason="preverb sense of the section headline form; no printed form",
                                          raw=_raw(text, seg, toks)))
                    elif any(t.kind in ("word", "paren", "gloss", "graphemic", "bracket", "xref") for t in seg):
                        items.append(Item(g, len(items), preverb, current, "", "",
                                          notes=[T.strip_markers(t.text).strip() for t in seg],
                                          status="no_form_segment",
                                          reason="segment without a citation form (annotation of the preceding item)",
                                          raw=_raw(text, seg, toks)))
                continue
            first = seg.index(forms[0])
            words = []
            for t in seg[:first]:
                if t.kind == "word":
                    words.append(t.plain.strip())
                elif t.kind == "paren":
                    words.append(f"({t.plain.strip()})")
                elif t.kind == "xref":
                    words.append(t.plain.strip())
            words = [w for w in words if w]
            label, tags, leftover, doubt = _label_from_words(words, lects, current)
            if label is None and words and any(w.rstrip(":") in NON_IRANIAN or w in NON_IRANIAN for w in words):
                label = "NONIR:" + next(w for w in words if w.rstrip(":") in NON_IRANIAN or w in NON_IRANIAN)
            if label is None:
                label = current
            # "M." inside the Sogdian section abbreviates Manichaean Sogdian, not Munji (p. 321)
            label = SECTION_LABEL_OVERRIDES.get((section, label), label)
            # merge italic runs split by a line break or a roman note ("-ihrot/ (impv.) ihroš")
            form_text = forms[0].text
            notes: list[str] = []
            gloss = ""
            phonemic = ""
            after = seg[first + 1:]
            k = 0
            while k < len(after):
                t = after[k]
                if t.kind == "form" and T.strip_markers(form_text).rstrip().endswith("/"):
                    form_text = form_text.rstrip() + T.strip_markers(t.text).strip()
                elif t.kind == "form" and k > 0 and after[k - 1].kind == "paren" \
                        and T.strip_markers(form_text).rstrip().endswith("/"):
                    form_text += T.strip_markers(t.text).strip()
                elif t.kind == "word" and t.plain.strip() == "/" and k + 1 < len(after) \
                        and after[k + 1].kind == "form":
                    form_text = form_text.rstrip() + "/"  # "vēretiš (v[y]ērdiš)/ vēren-"
                elif t.kind == "phon" and not phonemic:
                    phonemic = T.strip_markers(t.text).strip()
                elif t.kind == "gloss" and not gloss:
                    gloss = T.strip_markers(t.text).strip()
                elif t.kind == "paren":
                    inner = T.strip_markers(t.text).strip()
                    if inner in GRAMMAR:
                        tags += GRAMMAR[inner]
                    elif inner and all(p.strip() in GRAMMAR for p in re.split(r"[,/]", inner)):
                        for p in re.split(r"[,/]", inner):
                            tags += GRAMMAR[p.strip()]
                    elif inner in ("?",):
                        doubt = True
                    else:
                        notes.append(f"({inner})")
                elif t.kind == "form":
                    notes.append(T.strip_markers(t.text).strip())
                elif t.kind == "word" and not gloss and not t.plain.strip().startswith("’") \
                        and t.plain.strip()[:1].islower() and _closing_run(after, k):
                    # opening quote missing in the source (p. 39 BSogd. š’m to swallow’)
                    end = _closing_run(after, k)
                    gloss = clean_gloss(" ".join(x.plain.strip() for x in after[k:end + 1]))
                    k = end
                elif t.kind == "word" and not gloss and t.plain.strip().startswith("’") \
                        and len(t.plain.strip()) > 1:
                    # opening quote misprinted as ’ (p. 453 a-xšatā- ’unhurt’; p. 31 ’to realize)
                    words_ = [t.plain.strip()]
                    while not words_[-1].endswith("’") and k + 1 < len(after) and after[k + 1].kind == "word":
                        k += 1
                        words_.append(after[k].plain.strip())
                    gloss = clean_gloss(" ".join(words_).strip("’"))
                elif t.kind == "word":
                    w = t.plain.strip()
                    if w in ("?",):
                        notes.append("?")
                    elif w in GRAMMAR:
                        tags += GRAMMAR[w]
                    elif w:
                        notes.append(w)
                elif t.kind == "bracket":
                    notes.append(f"[{T.strip_markers(t.text).strip()}]")
                elif t.kind == "graphemic":
                    notes.append(f"<{T.strip_markers(t.text).strip()}>")
                elif t.kind == "xref":
                    notes.append(T.strip_markers(t.text).strip())
                k += 1
            gloss = clean_gloss(gloss)
            if "[Arabic script]" in gloss:
                # Arabic-script citations inside a gloss bracket are not lexical gloss text
                notes.append(f"‘{gloss}’")
                gloss = gloss.split("[", 1)[0].strip()
            if re.match(r"\[=?\s*(?:BMP|MMP|Pahl\.|OP|Av\.)", gloss):
                # Avestan citations glossed by their Pahlavi rendering ("‘[BMP] škpt’")
                notes.append(f"‘{gloss}’")
                gloss = ""
            form, reading, _ = clean_form(form_text)
            item = Item(g, len(items), preverb, label, " ".join(words), form, phonemic, gloss,
                        tags=tags, notes=[n for n in [*leftover, *notes] if any(c.isalnum() for c in n)],
                        doubt=doubt, reading=reading,
                        raw=_raw(text, seg, toks))
            if label.startswith("NONIR:"):
                item.status, item.reason = "non_iranian", f"non-Iranian language label {label[6:]}"
                current = label  # "Arab. (LW) baxt ‘luck’, baǧt …, waqt …" share the label
            elif label == "":
                item.status, item.reason = "unmapped_label", "no language label in a modern-language section"
            elif label not in lects:
                item.status, item.reason = "unmapped_label", f"label {label!r} not in lects.csv"
            elif _not_a_form(form):
                item.status, item.reason = "not_a_form", "italic run is prose, not a citation form"
            elif sum(1 for w in leftover if re.fullmatch(r"[a-z]+", w)) >= 4:
                # p. 438: prose after a misprinted || ("Kellens – Pirart … connect the verb to varəzī")
                item.status, item.reason = "not_a_form", "italic word cited inside prose"
            else:
                current = label if not label.startswith("§") else current
            if item.status == "form" and label in lects:
                current = label
            group_items.append(item)
            boundary_before.append(sep == ";")
            items.append(item)
            for shared in pending_labels:
                lect_label, _, _, _ = _label_from_words(shared.split(), lects, current)
                twin = Item(g, len(items), preverb, lect_label or shared, shared, item.form,
                            item.phonemic, item.gloss, tags=list(item.tags), notes=list(item.notes),
                            doubt=item.doubt, reading=item.reading, status=item.status,
                            reason=item.reason, raw=item.raw)
                if lect_label not in lects:
                    twin.status, twin.reason = "unmapped_label", f"label {shared!r} not in lects.csv"
                group_items.append(twin)
                boundary_before.append(False)
                items.append(twin)
            pending_labels = []
        # "id." (idem) repeats the meaning of the preceding glossed item
        last_gloss = ""
        for item in group_items:
            if item.gloss in ("id.", "id"):
                item.gloss = last_gloss
                item.notes.append("gloss printed as ‘id.’")
            elif item.gloss:
                last_gloss = item.gloss
        # gloss scope: a gloss covers the preceding gloss-less items back to the last gloss
        pending: list[Item] = []
        for item, semicolon in zip(group_items, boundary_before):
            if semicolon:
                pending = []
            if item.gloss:
                for prev in pending:
                    prev.gloss, prev.gloss_inherited = item.gloss, True
                pending = []
            else:
                pending.append(item)
    return items, body, commentary


def _raw(text: str, seg: list, toks: list) -> str:
    """The printed source span of an item (from its first token to the next token)."""
    if not seg:
        return ""
    last = seg[-1]
    following = [t.start for t in toks if t.start > last.start]
    end = min(following) if following else len(text)
    if "◊" in text[seg[0].start:end]:
        end = text.index("◊", seg[0].start)
    return text[seg[0].start:end].strip(" ,;")


def clean_gloss(gloss: str) -> str:
    """Drop quotation marks (doubled or nested openers, inner closers, straight quotes);
    keep the apostrophe of *one’s* and an aleph-initial cited form (``[+ ’w ‘to’]``)."""
    gloss = gloss.replace("‘", "").replace("'", "")
    gloss = re.sub(r"’(?![^\W\d_])", "", gloss)
    return re.sub(r"\s+", " ", gloss).strip(" ,;")


def _not_a_form(form: str) -> bool:
    if not form or not any(ch.isalpha() for ch in form):
        return True
    words = form.split()
    return form.endswith(":") or (len(words) >= 3 and all(w.isascii() and w.isalpha() for w in words))

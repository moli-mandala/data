"""Stage 3: entries and sections of the dictionary body (printed pp. 1–475 = PDF 26–500).

An entry starts at a bold ``*root`` line at the left margin that follows a blank separator
line or opens a page. Its sections start with ``•LABEL:``; 7.9 pt lines are paradigm
(inflected-form) blocks belonging to the preceding section, except ``•REFERENCES``.
Text after the ``→`` arrow on a line is the right-aligned reference column.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import decode

FIRST_PDF, LAST_PDF, PAGE_OFFSET = 26, 500, 25
HEADER_BASE = 60.0  # running heads (page number + catch-root) sit above this baseline
LEFT_MARGIN = 60.0
SMALL = 8.5  # 7.9 pt paradigm / reference text

REF_CONTINUATION = re.compile(r"^(?:[A-Z][\w.’ -]{0,25}:\s*)?[\dIVX][\w .,;:|–-]{0,60}$")


@dataclass
class Line:
    pdf: int
    index: int
    base: float
    x0: float
    size: float
    text: str  # decoded, with style markers

    @property
    def page(self) -> int:
        return self.pdf - PAGE_OFFSET

    @property
    def plain(self) -> str:
        return decode.plain(self.text)

    @property
    def blank(self) -> bool:
        return not self.plain.strip()


@dataclass
class Section:
    label: str  # e.g. "AVESTAN", "MIDDLE PERSIAN", "" for unlabelled prose
    page: int
    lines: list[Line] = field(default_factory=list)  # main-text lines
    paradigms: list[list[Line]] = field(default_factory=list)
    refs: list[tuple[int, str]] = field(default_factory=list)  # (page, reference text)

    def text(self) -> str:
        return join_lines(self.lines)


@dataclass
class Entry:
    lines: list[Line]
    head_lines: list[Line] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)

    @property
    def page(self) -> int:
        return self.lines[0].page

    @property
    def last_page(self) -> int:
        return self.lines[-1].page


def body_lines(pages: list[list[dict]]) -> list[Line]:
    out = []
    for pdf in range(FIRST_PDF, LAST_PDF + 1):
        for index, raw in enumerate(pages[pdf - 1]):
            if raw["base"] < HEADER_BASE:
                continue
            text = decode.render(raw)
            if raw["size"] > 13 and len(decode.plain(text).strip()) <= 2:
                continue  # letter divider ("B") of a new alphabet section
            out.append(Line(pdf, index, raw["base"], raw["x0"], raw["size"], text))
    return out


def is_headword(lines: list[Line], i: int) -> bool:
    line = lines[i]
    if not line.text.startswith(decode.B_ON + "*") or line.x0 > LEFT_MARGIN:
        return False
    if i == 0:
        return True
    prev = lines[i - 1]
    return prev.blank or prev.pdf != line.pdf


def split_entries(lines: list[Line]) -> list[Entry]:
    entries: list[Entry] = []
    for i, line in enumerate(lines):
        if is_headword(lines, i):
            entries.append(Entry([line]))
        elif entries:
            if not line.blank:
                entries[-1].lines.append(line)
        elif not line.blank:
            raise ValueError(f"text before first headword: {line.plain!r}")
    return entries


def section_label(text: str) -> tuple[str, str] | None:
    plain = decode.plain(text).strip()
    if not plain.startswith("•"):
        return None
    m = re.match(r"•\s*([A-Z][A-Z .]*[A-Z])\s*(:|\s)\s*", plain)
    if not m:
        raise ValueError(f"unrecognised section label: {plain[:40]!r}")
    label = re.sub(r"\s+", " ", m.group(1))
    # drop the label from the marked text: skip as many visible characters
    visible = len(m.group(0))
    out, seen = [], 0
    for ch in text.lstrip():
        if seen < visible and ch not in decode.MARKERS:
            seen += 1
            continue
        if seen < visible:
            continue
        out.append(ch)
    return label, "".join(out)


def strip_markers_outside(text: str) -> str:
    return text


def split_ref(text: str) -> tuple[str, str]:
    """Split a line at the reference arrow (only when it is the right-hand column)."""
    if "→" not in text:
        return text, ""
    head, _, tail = text.partition("→")
    return head.rstrip(), decode.plain(tail).strip()


def segment(entry: Entry) -> None:
    lines = entry.lines
    i = 0
    head = [lines[0]]
    i = 1
    # the headword gloss can wrap: continue until a section/prose/paradigm line starts
    while i < len(lines) and not lines[i].plain.lstrip().startswith(("•", "◊")) \
            and lines[i].size > SMALL and _open_quote(head):
        head.append(lines[i])
        i += 1
    entry.head_lines = head
    current: Section | None = None
    paradigm: list[Line] | None = None
    ref_open = False
    for line in lines[i:]:
        labelled = section_label(line.text)
        small = line.size < SMALL
        plain = line.plain.strip()
        # wrapped right-hand reference column ("→ SGS: 116, | 74", "→ Ghilain: 96 | | DMMPP: …")
        if not labelled and current is not None and REF_CONTINUATION.match(plain) and (
                ref_open or (current.label == "REFERENCES" and plain[:1].isdigit())):
            if current.label == "REFERENCES" and not ref_open:
                current.lines.append(line)
            else:
                page, ref = current.refs[-1]
                current.refs[-1] = (page, f"{ref} {plain}")
                ref_open = plain.endswith(("|", ",", ":"))
            continue
        ref_open = False
        if labelled:
            label, rest = labelled
            current = Section(label, line.page)
            entry.sections.append(current)
            paradigm = None
            line = Line(line.pdf, line.index, line.base, line.x0, line.size, rest)
            if label == "REFERENCES":
                current.lines.append(line)
                continue
        elif current is None:
            current = Section("", line.page)
            entry.sections.append(current)
        if small and current.label != "REFERENCES":
            if paradigm is None:
                paradigm = []
                current.paradigms.append(paradigm)
            paradigm.append(line)
            continue
        paradigm = None
        if current.label == "REFERENCES" and not small:
            # main-size prose after the reference block (rare): unlabelled prose section
            current = Section("", line.page)
            entry.sections.append(current)
        text, ref = split_ref(line.text)
        if ref:
            current.refs.append((line.page, ref))
            ref_open = ref.endswith(("|", ",", ":"))
        line = Line(line.pdf, line.index, line.base, line.x0, line.size, text)
        if decode.plain(text).strip():
            current.lines.append(line)


def _open_quote(lines: list[Line]) -> bool:
    text = "".join(decode.plain(l.text) for l in lines)
    roman_quotes = text.count("‘") - text.count("’")
    return roman_quotes > 0


HYPHEN_BREAK = re.compile(r"([a-zäöüé])-\s*$")
ITALIC_HYPHEN_END = re.compile(r"(?<=[^\s(/*])-\s*" + re.escape(decode.I_OFF) + r"\s*$")


def join_lines(lines: list[Line]) -> str:
    """Join wrapped lines; drop soft hyphens only in roman prose (``Morgen-|stierne``)."""
    out = ""
    for line in lines:
        text = line.text.strip()
        if not out:
            out = text
            continue
        if ITALIC_HYPHEN_END.search(out) and text.startswith(decode.I_ON):
            # a citation form broken after its hyphen: "haɫ-|āwasīn/āwēs-" (index: haɫ-āwasīn)
            out = ITALIC_HYPHEN_END.sub("-", out) + text[1:]
            continue
        if (HYPHEN_BREAK.search(out) and not out.endswith(decode.I_OFF)
                and decode.plain(text)[:1].islower()
                and not _in_italic(out)):
            out = out[:-1] + text
        else:
            out = out + " " + text
    return re.sub(r"[ \t]+", " ", out)


def _in_italic(text: str) -> bool:
    return text.rfind(decode.I_ON) > text.rfind(decode.I_OFF)


def entries_from_pages(pages) -> list[Entry]:
    entries = split_entries(body_lines(pages))
    for entry in entries:
        segment(entry)
    return entries

"""Tokenise a section's marked text into forms, glosses, parentheticals and punctuation.

Token kinds:
``form``    an italic run (citation form, possibly with ``/`` principal parts)
``gloss``   a roman ‘…’ quotation (a closing ’ must be followed by a delimiter, so the
            apostrophe of *one’s* is not a closer)
``paren``   a balanced roman (…) group (may contain italics; ``(+ *fra-)`` = preverb)
``phon``    a /…/ phonemic transcription around an italic run
``bracket`` a roman […] group
``sep``     one of ``,`` ``;`` ``||`` ``|``
``diamond`` ``◊`` (commentary follows)
``xref``    a bold cross-reference (``*root``)
``word``    any other roman word (language labels, ``?``, grammatical notes, prose)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from decode import B_OFF, B_ON, I_OFF, I_ON, MARKERS, SUB_OFF, SUB_ON, SUP_OFF, SUP_ON

DELIM_AFTER_CLOSE = set(" ,;:.)]|?! ") | set(MARKERS) | {""}


@dataclass
class Token:
    kind: str
    text: str  # marked text (style markers kept)
    start: int

    @property
    def plain(self) -> str:
        return strip_markers(self.text)


def strip_markers(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in (SUP_ON, SUB_ON):
            close = SUP_OFF if ch == SUP_ON else SUB_OFF
            j = text.find(close, i)
            inner = strip_markers(text[i + 1:j if j >= 0 else len(text)])
            out.append(to_script(inner, ch == SUP_ON))
            i = (j + 1) if j >= 0 else len(text)
            continue
        if ch not in MARKERS:
            out.append(ch)
        i += 1
    return "".join(out)


SUPERSCRIPT = str.maketrans("0123456789+-()ixhwyaeouvn", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾ⁱˣʰʷʸᵃᵉᵒᵘᵛⁿ")
SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def to_script(text: str, sup: bool) -> str:
    text = text.strip()
    if not text:
        return ""
    table = SUPERSCRIPT if sup else SUBSCRIPT
    converted = text.translate(table)
    if all(not c.isascii() or not c.isalnum() for c in converted):
        return converted
    return text  # no faithful script code point: keep the base characters


def _style_at(text: str, i: int) -> str:
    """Return 'i' when position ``i`` lies inside an italic run, 'b' bold, else 'r'."""
    io, ic = text.rfind(I_ON, 0, i), text.rfind(I_OFF, 0, i)
    if io > ic:
        return "i"
    bo, bc = text.rfind(B_ON, 0, i), text.rfind(B_OFF, 0, i)
    return "b" if bo > bc else "r"


def normalize_quotes(text: str) -> str:
    """A quotation mark set alone in the italic font is typographic, not an aleph/ayin."""
    text = re.sub(re.escape(I_ON) + r"([‘’])" + re.escape(I_OFF), r"\1", text)
    # an opening ‘ set in the italic font at the end of a form run: "⟨š’m ‘⟩to swallow’"
    text = re.sub(r"(\s*)‘\s*" + re.escape(I_OFF), I_OFF + r"\1‘", text)
    # a closing ’ set in the italic font at the start of the next run: "‘pain⟨’ ||"
    return re.sub(re.escape(I_ON) + r"’(?=[\s,;|=(]|$)", "’" + I_ON, text)


def tokenize(text: str) -> list[Token]:
    text = normalize_quotes(text)
    tokens: list[Token] = []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch.isspace() or ch in (SUP_OFF, SUB_OFF, I_OFF, B_OFF):
            i += 1
            continue
        if ch == I_ON:
            j = text.find(I_OFF, i)
            j = n if j < 0 else j
            body = text[i + 1:j]
            # a superscript opened just before the italic run (^{⟨x}form⟩) belongs to it
            if tokens and tokens[-1].kind == "script":
                body = tokens.pop().text + body
            tokens.append(Token("form", body, i))
            i = j + 1
            continue
        if ch == B_ON:
            j = text.find(B_OFF, i)
            j = n if j < 0 else j
            tokens.append(Token("xref", text[i + 1:j], i))
            i = j + 1
            continue
        if ch in (SUP_ON, SUB_ON):
            close = SUP_OFF if ch == SUP_ON else SUB_OFF
            j = text.find(close, i)
            if j < 0 or I_ON in text[i:j]:
                # ^{⟨x}form⟩: the script opens before the italic run and closes inside it
                k = text.find(I_ON, i)
                if k >= 0 and (j < 0 or k < j):
                    tokens.append(Token("script", text[i:k], i))
                    i = k
                    continue
            j = n if j < 0 else j
            # ^{x}/^{+} (tentative/emended reading) printed right before a citation form
            if strip_markers(text[i:j + 1]).strip() in ("ˣ", "⁺") and text[j + 1:j + 2] == I_ON:
                tokens.append(Token("script", text[i:j + 1], i))
            else:
                tokens.append(Token("word", text[i:j + 1], i))
            i = j + 1
            continue
        if ch == "‘" and _style_at(text, i) == "r":
            j, closer = _gloss_end(text, i + 1)
            tokens.append(Token("gloss", text[i + 1:j], i))
            i = j + closer
            continue
        if ch == "<" and _style_at(text, i) != "i":
            m = GRAPHEMIC.match(strip_markers(text[i:i + 120]))
            close = text.find(">", i)
            if m and close > i:
                # Old Persian graphemic transliteration <b-r-š-n-a> (italic parts allowed)
                tokens.append(Token("graphemic", text[i + 1:close], i))
                i = close + 1
                continue
        if ch == "(" and _style_at(text, i) == "r":
            j = _balanced(text, i, "(", ")")
            tokens.append(Token("paren", text[i + 1:j], i))
            i = j + 1
            continue
        if ch == "[" and _style_at(text, i) == "r":
            j = _balanced(text, i, "[", "]")
            tokens.append(Token("bracket", text[i + 1:j], i))
            i = j + 1
            continue
        if ch == "/" and _style_at(text, i) == "r" and text[i + 1:i + 2] == I_ON:
            j = text.find(I_OFF, i)
            k = text.find("/", j) if j >= 0 else -1
            if j >= 0 and k >= 0 and not strip_markers(text[j + 1:k]).strip():
                tokens.append(Token("phon", text[i + 1:k], i))
                i = k + 1
                continue
        if text.startswith("||", i):
            tokens.append(Token("sep", "||", i))
            i += 2
            continue
        if ch in ",;|":
            tokens.append(Token("sep", ch, i))
            i += 1
            continue
        if ch == "◊":
            tokens.append(Token("diamond", ch, i))
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace() and text[j] not in (
                I_ON, B_ON, SUP_ON, SUB_ON, ",", ";", "|", "‘", "(", "[", "◊"):
            j += 1
        if j == i:
            j = i + 1
        tokens.append(Token("word", text[i:j], i))
        i = j
    return tokens


GRAPHEMIC = re.compile(r"<\s?[^\s<>*‘’]*-[^\s<>*‘’]*>")
# After a gloss: a new item, preverb group, comparison or commentary begins.
STRUCTURAL = re.compile(r"\s*(?:[,;=◊]|\|\||\(\+)")
# An item boundary inside an unclosed gloss: ", " + optional (Sub) + Label + a citation form.
_LABEL = r"\??\s*(?:[A-Z][\w’-]*\.|[A-Z]{2,4}\b)(?:\s*\([^()]{1,12}\))?(?:\s*[A-Z]\.)?"
ITEM_START = re.compile(r",\s+(?:\([^()]{1,12}\)\s*)?(?P<label>(?:" + _LABEL + r",\s*)*" + _LABEL + r")?\s*"
                        + re.escape(I_ON) + r"(?P<form>[^" + re.escape(I_OFF) + r"]*)")


def _gloss_end(text: str, i: int) -> tuple[int, int]:
    """Return (end of gloss content, length of the closing quote: 1, or 0 when unclosed).

    Cheung's closing quote is ’, sometimes a straight ' or missing altogether (e.g. p. 61
    ``drd ‘pain || (+ *u̯i-) …``). A gloss therefore also ends before ``||``, ``(+`` or ``◊``;
    a nested quote closes the whole gloss when an item separator follows it
    (``‘to seek; [+ vā] ‘to find’, Abyan. …``); and an unclosed gloss that runs into
    ``, Label form`` ends at that comma.
    """
    start, depth, n = i, 0, len(text)
    while i < n:
        ch = text[i]
        if _style_at(text, i) != "i" and (text.startswith("||", i) or text.startswith("(+", i)
                                           or ch == "◊"):
            return _unclosed(text, start, i), 0
        if ch == "‘" and _style_at(text, i) == "r":
            if depth == 0:
                cut = _item_boundary(text, start, i)
                if cut is not None:
                    return cut, 0
            depth += 1
        elif ch in "’'" and _style_at(text, i) != "i":
            nxt = text[i + 1:i + 2]
            if nxt in DELIM_AFTER_CLOSE or nxt in ("’", "'", "=", "("):
                if depth and not _closes_whole_gloss(text, i + 1):
                    depth -= 1
                else:
                    return i, 1
        i += 1
    return _unclosed(text, start, n), 0


def _closes_whole_gloss(text: str, i: int) -> bool:
    """After an inner closing quote: does the outer gloss end here too?"""
    m = STRUCTURAL.match(text, i)
    if not m:
        return False
    if m.group(0).strip() in (",", ";"):
        # only when a new item follows (", Abyan., Abz. yöšta"), not more gloss text
        # (‘to bring ‘x’, come into a position of authority’)
        return _item_boundary(text, m.start() + len(m.group(0)) - 1, min(len(text), i + 120)) is not None \
            and _item_boundary(text, m.start() + len(m.group(0)) - 1, min(len(text), i + 120)) <= m.end()
    return True


def _item_boundary(text: str, start: int, end: int) -> int | None:
    for m in ITEM_START.finditer(text, start, end):
        form = strip_markers(m.group("form"))
        # without a language label, only a citation-looking italic (stem hyphen, principal
        # parts, non-ASCII letters) counts; italic French/German gloss words do not
        if m.group("label") or "-" in form or "/" in form or not form.isascii():
            return m.start()
    return None


def _unclosed(text: str, start: int, end: int) -> int:
    cut = _item_boundary(text, start, end)
    return cut if cut is not None else end


def _balanced(text: str, i: int, open_: str, close: str) -> int:
    depth = 0
    for j in range(i, len(text)):
        # brackets are counted in either style: Cheung's notes can close inside a form run
        # ("Zaz. vēretiš (v[y]ērdiš)/ vēren-"), and italic forms carry balanced "(va)" pairs
        if text[j] == open_:
            depth += 1
        elif text[j] == close:
            depth -= 1
            if depth == 0:
                return j
    return len(text)

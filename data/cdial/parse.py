"""Parse the DDSA CDIAL pages (cached in ``cdial.pickle``) into ``cdial.csv``.

    uv run python parse.py                 # full run; outputs are replaced atomically on success
    uv run python parse.py --entry 10132   # parse the named entries only and print their rows
"""

import argparse
import pickle
import os
import sys
import tempfile
import urllib.request
import re
import json
import copy
import csv
import unicodedata
from bs4 import BeautifulSoup
from collections import defaultdict
from enum import Enum
from tqdm import tqdm

from abbrevs import abbrevs
from references import entry_source_field, source_field

TOTAL_PAGES = 836

_cli = argparse.ArgumentParser(description=__doc__.split("\n")[0])
_cli.add_argument("--entry", nargs="+", metavar="NUMBER", help="parse only these CDIAL entry numbers and print their rows as CSV")
ARGS = _cli.parse_args()
ONLY = set(ARGS.entry or [])

# this is such a big brain regex
lang_alternation = "|".join(sorted(list(abbrevs.keys()), key=lambda x: -len(x)))
langs = r'([OM]?(' + lang_alternation + r'))\.'
langs = unicodedata.normalize('NFC', langs)
# A language abbreviation directly followed by another capital initial ("H. W. Bailey") is an
# author citation, not a reflex. Consecutive *known* one-letter language codes ("S. L. P."),
# however, are a normal CDIAL shorthand and must remain available to the language stack.
next_language = r'[OM]?(?:' + lang_alternation + r')\.'
author_guard = r'(?! ?(?!(?:' + next_language + r'))[A-Z]\.)'
regex = re.compile(r'(?<!\w)(?<!← )(?<!→ )' + langs + author_guard + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=([^\(]?(?<!\w)' + langs + r'|</div>|$))')
# Borrower lists begin with an arrow by definition, so their dedicated recursive parse permits a
# language code after the marker while ordinary parsing treats arrow-following codes as donors.
regex_borrowed = re.compile(r'(?<!\w)' + langs + author_guard + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=([^\(]?(?<!\w)' + langs + r'|</div>|$))')
oia = r'((Indo-Aryan))\.'
regex_head = re.compile(r'(?<!\w)' + oia + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=([^\(]?(?<!\w)' + oia + r'|</div>|$))')
# The quoted definition may itself contain parentheses (e.g. 'walnut (or pistacio nut ?)'); match
# any content up to the next definition-closing quote — one NOT followed by "s" (a possessive) —
# rather than forbidding parens, which used to mis-pair quotes onto the inter-gloss source citations.
formatter = re.compile(r'(<i>(.*?)</i>|\'(.*?)\'(?=[^s]|$))(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=$|<i>(.*?)</i>|\'(.*?)\'|\.)')
# In the head, a form is bold (<b>headword / numbered section form) OR italic (<i>alternate spelling
# of the preceding bold form). Match either as a form (char class keeps the capture groups stable);
# the parse loop tags italic head-forms so they become variants and are never promoted to sections.
formatter_head = re.compile(r'(<[bi]>(.*?)</[bi]>|\'(.*?)\'(?=[^s]|$))(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=$|<[bi]>(.*?)</[bi]>|\'(.*?)\'|\.)')
borrowed_terms = re.compile(r'\(→.*?\)')

# CDIAL cites Dravidian forms as comparison/donor evidence.  They belong in the article-level
# cross-family comparison table built by data/cross_family.py, never in the ordinary reflex table.
# ``Go`` (Gondi) lacks the parenthetical family label in abbrevs.py but is Dravidian too.
DRAVIDIAN_COMPARISON_LANGS = {
    "Brah", "Drav", "Ga", "Go", "Kan", "Kol", "Kur", "Mal", "Nk", "Prj", "Tam", "Tel", "Tu",
}

_QUOTE_SENTINEL = "\ue000"
_INNER_DASH_SENTINEL = "\ue001"


_STARRED_FORM = re.compile(r"<i>\*[^<]*</i>-?(?:\s*'[^']*'(?=[^s]|$))?")


def _protect_note_markup(text):
    """Hide note-only italics and quotes from the form/gloss tokenizer."""
    return (text.replace("<i>", "<note-i>")
                .replace("</i>", "</note-i>")
                .replace("'", _QUOTE_SENTINEL))


def _restore_note_markup(text):
    return (text.replace("<note-i>", "<i>")
                .replace("</note-i>", "</i>")
                .replace(_QUOTE_SENTINEL, "'"))


def protect_explanatory_markup(span):
    """Distinguish cited forms/glosses in prose from the reflexes being parsed."""
    # Parenthetical notes use the same italics and quotes as real forms and definitions. Work from
    # innermost parentheses outward; two passes cover the nesting found in the source corpus.
    for _ in range(2):
        span = re.sub(r'\([^()]*\)', lambda match: _protect_note_markup(match.group(0)), span)

    # Explicit prose cues introduce a cited comparison/base form rather than another reflex.
    cue = re.search(
        r"(?:\bdoubtful\b|\bbut\s+<i>|\bposs\.?\s+have\b|\bhave\s+prob\b|\b(?:pret|past tense|pres)\.?\s+(?:tense\s+)?of\b)",
        span,
        re.IGNORECASE,
    )
    if cue:
        span = span[:cue.start()] + _protect_note_markup(span[cue.start():])

    # After a gloss, an etymological relation begins explanatory prose. Demote later markup while
    # retaining the prose in Notes (e.g. "ḍippaï 'rots' < *dīpyatē ... cf. dāpayati 'causes ...'").
    for marker in ("&lt;", "&gt;", "←"):
        start = span.find(marker)
        # Relations inside a parenthetical note are already protected above; do not let one hide
        # the real forms which follow the closing parenthesis.
        inside_parentheses = start >= 0 and span[:start].count("(") > span[:start].count(")")
        if start >= 0 and not inside_parentheses and "'" in span[:start]:
            span = span[:start] + _protect_note_markup(span[start:])
            break
    return span


def _base_character(character):
    return "".join(
        value for value in unicodedata.normalize("NFD", character)
        if not unicodedata.combining(value)
    )


def expand_degree_abbreviation(word, reference):
    """Expand CDIAL's degree-sign ditto notation against the preceding form."""
    if not reference or word == "°":
        return word
    if word.endswith("°") and len(word) > 1:
        target = _base_character(word[-2])
        for index, character in enumerate(reference):
            if _base_character(character) == target:
                return word[:-1] + reference[index + 1:]
    if word.startswith("°") and len(word) > 1:
        target = _base_character(word[1])
        for index in range(len(reference) - 1, -1, -1):
            if _base_character(reference[index]) == target:
                return reference[:index] + word[1:]
    return word


def protect_parenthetical_group_separators(text):
    """Prevent a ``? —`` (etc.) inside a note from splitting the reflex paragraph."""
    depth = 0
    output = []
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")" and depth:
            depth -= 1
        if character == "—" and depth and index >= 2 and text[index - 2] in ";.,:?":
            output.append(_INNER_DASH_SENTINEL)
        else:
            output.append(character)
    return "".join(output)


_MORPHOLOGICAL_BOUNDARY = re.compile(
    r"(?:pass|caus|intr|trans|refl|denom|pres|pret|aor|fut|perf|pp|part|ger|inf|imper|subj|opt)",
    re.IGNORECASE,
)


# A grammatical label printed before a form (``pl. dōnye̯``, ``imper. bēza``, ``obl. hamā``,
# ``3 sg. pres. dātē``) describes that form, but the tokenizer stores it with the preceding
# form's trailing text. It is detached when it closes a comma/semicolon-separated run.
_FORWARD_LABEL = re.compile(
    r"^(?P<own>.*?[,;])\s*(?P<label>(?:(?:[123]\s*)?(?:pl|sg|du|imper|imp|pp|ptc|pres|pret|aor|fut|perf"
    r"|inf|abs|ger|obl|dir|nom|acc|gen|dat|abl|inst|instr|loc|voc|caus|pass|intr|trans|refl|denom"
    r"|subj|opt|adj|adv|sb|vb|f|m|n)\.?\s*)+)$",
    re.DOTALL,
)


def detach_forward_label(trailing):
    """Split ``', pl.'`` into the previous form's own note and a label for the next form."""
    match = _FORWARD_LABEL.match(trailing or "")
    if not match:
        return trailing, ""
    return match.group("own"), match.group("label").strip(" .")


_INDEPENDENT_GROUP = re.compile(
    r"^(?:caus|pass|intr|trans|denom|deriv|cf|x\b|→|.*→)", re.IGNORECASE
)


def section_meaning(info, head_definition, head_definitions, first_group=False, group_gloss=""):
    """The lemma meaning a reflex paragraph inherits, or "" for semantically independent groups."""
    label = re.sub(r"<[^>]+>", "", info or "").strip(" .:")
    if _INDEPENDENT_GROUP.match(label):
        return ""
    numbered = re.match(r"(\d+)\b", label)
    if numbered:
        index = int(numbered.group(1)) - 1
        lemma = head_definitions[index] if index < len(head_definitions) else ""
        return lemma or head_definition
    if first_group or label == "":
        return head_definition
    # Extension, alternation and phonological sub-groups (``ext. -kk-:``, ``With early
    # nasalization:``, ``hypersanskritism with kr-:``) are ordinary reflexes; such a group states
    # its own meaning once if it differs (``Pk. maḍakkiyā- 'earthen jar', M. maḍkī, H. maṭkā``).
    return group_gloss or head_definition


_GENDER_NOTE = re.compile(r"^(?:m|f|n|mn|fn|mf|m\.n|m\.f|f\.n)\.?$")


def is_verbal_lemma(form):
    """OIA verbal lemmas are cited as finite 3 sg. (``mināti``, ``ōvahati``, ``bandháyati``)."""
    plain = unicodedata.normalize("NFD", form or "")
    plain = "".join(ch for ch in plain if not unicodedata.combining(ch)).rstrip("-")
    return bool(re.search(r"(?:a|e|o|ya)te?$|ti$", plain)) and not plain.endswith(("ti-", "ati-"))


def is_verbal_gloss(gloss):
    """Turner glosses verbs as ``'to measure'`` or 3 sg. ``'measures'``, ``'carries down'``."""
    head = re.split(r"[;,(]", gloss, 1)[0].strip()
    if head.startswith("to "):
        return True
    words = head.split()
    if not words or len(words) > 3:
        return False
    verb = words[0]
    return (
        bool(re.fullmatch(r"[a-z]+", verb))
        and verb.endswith("s") and not verb.endswith(("us", "ss"))
        and verb not in {"is", "was", "this", "his", "its", "thus", "yes", "as", "us"}
    )


_VOWEL = re.compile(r"[aeiouəɔεʌɪɛœᵃᵉⁱᵒᵘʸᵊ]")
# marks that carry syllabicity in the source transcriptions (r̥, l̥; Kotgarhi v̄, s̊)
_SYLLABIC_MARKS = {"\u0325", "\u0304", "\u0306", "\u030a", "\u0303", "\u0301", "\u0302", "\u0308"}


def is_sound_fragment(form):
    """A bare consonant run such as ``-kk-``, ``ṇḍ`` or ``ch``: a sound-change label, not a word."""
    core = form.strip("-°*. ")
    if not core or "-" in core or not re.search(r"[a-zA-Z]", core):
        return False
    decomposed = unicodedata.normalize("NFD", core.lower())
    if any(ch in _SYLLABIC_MARKS for ch in decomposed):
        return False
    base = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    if len(base) >= 3 and "y" in base:
        return False  # ``ryč``, ``βyk``: y is vocalic; a bare ``y``/``yy`` is still a fragment
    return len(base) <= 4 and not _VOWEL.search(base)


def is_morphological_boundary_note(note):
    """Whether bare text between two forms changes their grammatical derivation."""
    plain = re.sub(r"<[^>]+>", "", note or "").strip(" ,;:.")
    return bool(_MORPHOLOGICAL_BOUNDARY.fullmatch(plain))


def propagate_single_printed_definition(words):
    """Scope following definitions backward over comma-listed forms in their run.

    CDIAL commonly prints ``<i>x</i>, <i>y</i> 'definition'``. The final quote scopes over both
    forms, not just the immediately preceding italic token. With multiple definitions, each quote
    scopes backward only to the previous quote (``x, y 'A', z 'B'`` gives x/y A and z B).
    """
    start = 0
    segments = []
    for index, word in enumerate(words):
        # ``pass.`` in ``ōvahati, pass. ōvuyhati 'is carried down'`` is detached onto
        # ōvuyhati (word[4]), so it opens a new segment at that word. A run's own trailing
        # label (``helā, hilā intr. 'to lean'``) is not a boundary.
        if index > start and is_morphological_boundary_note(word[4]):
            segments.append(words[start:index])
            start = index
    segments.append(words[start:])

    for segment in segments:
        definitions = list(dict.fromkeys(word[1] for word in segment if word[1]))
        if len(definitions) == 1:
            for word in segment:
                if not word[1]:
                    word[1] = definitions[0]
        elif len(definitions) > 1:
            following_definition = ""
            for word in reversed(segment):
                if word[1]:
                    following_definition = word[1]
                elif following_definition:
                    word[1] = following_definition


def terminal_separator(span):
    """Return the top-level punctuation joining this language span to the next one."""
    plain = re.sub(r"<[^>]+>", "", _restore_note_markup(span)).rstrip()
    return plain[-1] if plain and plain[-1] in ",;." else ""


def blocks_shared_definition(span, forms):
    """Whether leading morphology marks this form as a new semantic/derivational run."""
    prefix = re.sub(r"<[^>]+>", "", span[:forms[0].start()]) if forms else ""
    return bool(re.search(r"\b(?:caus|intr|trans|pass|refl)\.?\b", prefix, re.IGNORECASE))


def blocks_head_definition_propagation(span):
    """Numbered or explicitly derived OIA heads are separate dictionary senses."""
    plain = re.sub(r"<[^>]+>", "", _restore_note_markup(span))
    return bool(
        re.search(r"\b\d+\.\s", plain)
        or re.search(r"\b(?:caus|intr|trans|pass|refl|denom)\.?\b", plain, re.IGNORECASE)
    )

rows = []
params = []
corrupt_forms = []
done = set()

# response caching logic
soups = []
cached = False
if os.path.exists('cdial.pickle'):
    with open('cdial.pickle', 'rb') as fin:
        soups = pickle.load(fin)
    cached = True

def parse(
    subentry,
    subentry_num,
    subnum,
    number,
    info,
    carried="",
    allow_arrow_language=False,
    head_definition="",
    head_definitions=(),
    head_verbal=False,
):
    langs = []
    temp_rows = []
    shared_definition = ""
    head_definition_overridden = False
    valency_blocked_rows = set()  # rows in a ``caus./pass./intr./trans.`` run
    carried_note = ""  # a reconstruction heading (``MIA. *ādariśa-:``) awaiting the form it introduces
    # rows of preceding unglossed spans that ended in a comma: ``S. āṇaṇu, L. āṇaṇ, P. ānnā
    # 'to bring'`` glosses the whole comma-run, across language labels.
    pending_run = []

    # find lemmas in current subgroup
    matches = []
    if subentry_num != 0:
        matcher = regex_borrowed if allow_arrow_language else regex
        matches = list(matcher.finditer(subentry))
    else:
        matches = list(regex_head.finditer(subentry))

    if len(matches) != 0:
        subnum += 1
        info = subentry[:matches[0].span()[0]].strip()
        info = info.strip(':.;')
        # CDIAL addenda split a numbered sub-heading (`4. *kṣāṇayati:`) off with a <br>, leaving the
        # reflex paragraph label-less — fall back to the form number carried from that sub-heading.
        if not info and carried:
            info = carried
        carried = ""
    else:
        # a bare numbered sub-heading with no reflexes → remember its number for the next paragraph
        plain = re.sub(r"<[^>]+>", "", subentry).strip()
        mm = re.match(r"(\d+)\s*[.:]", plain)
        if mm:
            carried = mm.group(1)

    for i in range(len(matches)):

        # grab lang and rest of span
        lang = matches[i].group(1)
        span = matches[i].group(3)

        # In the head paragraph the head-forms (headword, numbered forms, italic alternate spellings)
        # all precede the etymological note ("[…]") and the loan note (" — …"). Italic forms inside
        # those (reconstructed donors, examples, cross-references) are NOT OIA variants, so cut the
        # span there before extracting head-forms.
        if subentry_num == 0 and lang == 'Indo-Aryan':
            span = re.split(r'\[|—', span, 1)[0]

        # formatting
        span = span.replace('ˊ', '́')
        span = span.replace(' -- ', '–')
        span = span.replace('--', '–')
        span = protect_explanatory_markup(span)
        # A starred (reconstructed) form in a reflex paragraph belongs to Turner's etymological
        # prose — ``K. wanun 'to become wet' < MIA. *uvaṇṇa-``, ``MIA. *ukkhuḍa-: Pk. ukkhuḍaï`` —
        # not to the reflex list. Keep it, with its own gloss, as note text so it stays attached
        # to the form it explains; the head paragraph's starred lemmata are the etyma themselves.
        if subentry_num != 0:
            span = _STARRED_FORM.sub(lambda m: _protect_note_markup(m.group(0)), span)
        
        # forms are the actual words (italicised)
        forms = []
        if lang == 'Indo-Aryan':
            forms = list(formatter_head.finditer(span))
        else:
            forms = list(formatter.finditer(span))
        
        if lang == 'mald':
            lang = 'Md'
        # A lowercase code is a dialect qualifier for the immediately preceding parent language,
        # not an additional language sharing the form (Gy. eur., L.awāṇ., Paš.pach., WPah.kṭg.).
        if lang[0].islower():
            if langs:
                langs.pop()

        # langs is a stack of langs, if there are no forms
        # we just add to the stack and continue (means later
        # lang has relevant data)
        langs.append(lang)
        if len(forms) == 0:
            plain = _restore_note_markup(re.sub(r"<[^>]+>", "", span)).strip(" :;,.")
            if "*" in plain:
                # the label belongs to the reconstruction, not to the forms that follow it
                carried_note = f"{lang}. {plain}" if lang != "Indo-Aryan" else plain
                langs.pop()
            continue

        # extract definitions
        # TODO: get morphological labels, notes
        cur = None
        defs = []
        words = []

        forward_label = ""

        def append_to_words(cur, defs):
            if cur:
                for each in cur[0].split(','):
                    definition = '; '.join([d[0] for d in defs]) if defs != [] else ''
                    def_notes = [d[1].strip(' -,;.') for d in defs]
                    notes = '; '.join([n for n in def_notes if n]) if defs != [] else ''
                    own = cur[1].strip(' -,;.')
                    notes = own + ('; ' if (own and notes) else '') + notes
                    words.append([each.strip(), definition, notes, cur[2], cur[3]])

        for form in forms:
            if form.group(0).startswith('<i>') or form.group(0).startswith('<b>'):
                # a label closing the previous trailing text (``, pl.``) belongs to this form
                if defs:
                    defs[-1][1], forward_label = detach_forward_label(defs[-1][1])
                elif cur:
                    cur[1], forward_label = detach_forward_label(cur[1])
                else:
                    forward_label = ""
                append_to_words(cur, defs)
                defs = []
                # an italic form in the head paragraph is an alternate spelling of the preceding bold
                # form → a variant, never a numbered section header (see cognateset marker below)
                is_variant = form.group(0).startswith('<i>') and lang == 'Indo-Aryan'
                trailing = _restore_note_markup(form.group(4)).strip(' -')
                if carried_note:
                    trailing = carried_note + ('; ' if trailing.strip(' -,;.') else '') + trailing
                    carried_note = ""
                cur = [
                    _restore_note_markup(form.group(2)),
                    forward_label + ('; ' if (forward_label and trailing.strip(' -,;.')) else '') + trailing,
                    is_variant,
                    forward_label,
                ]
            else:
                defs.append([
                    _restore_note_markup(form.group(3)).strip(),
                    _restore_note_markup(form.group(4)).strip(' -'),
                ])
        if cur:
            for each in cur[0].split(','):
                append_to_words(cur, defs)

        # A definition printed once scopes over all comma-listed forms in this language span.
        # If the preceding language span ended in a comma, it also scopes forward across language
        # labels until a stronger boundary. Morphological labels start a new run and block carryover.
        printed_definitions = list(dict.fromkeys(word[1] for word in words if word[1]))
        # ``L. āṇaṇ, P. ānnā 'to bring'``: a lone glossed form closing a short comma-run.
        closes_run = len(words) == 1 and bool(words[0][1])
        if printed_definitions:
            head_definition_overridden = True
        if not (lang == "Indo-Aryan" and blocks_head_definition_propagation(span)):
            propagate_single_printed_definition(words)
        shared_definition_blocked = blocks_shared_definition(span, forms)
        if shared_definition and not printed_definitions and not shared_definition_blocked:
            for word in words:
                if not word[1]:
                    word[1] = shared_definition
        # for each language on the stack, add this entry
        span_start = len(temp_rows)
        for l in langs:
            # Preserve the parser's stack and definition state above, but do not emit cited
            # Dravidian comparison forms as ordinary Indo-Aryan reflexes.
            if l in DRAVIDIAN_COMPARISON_LANGS:
                continue
            for word, defn, notes, is_variant, _ in words:
                # drop empty forms (e.g. from a trailing comma inside <b>aṅkōla-,</b>) so they neither
                # emit a blank row nor stand in as the reference for a following "°suffix" expansion
                if not word.strip('.,;-: '):
                    continue

                if '°' in word and word != '°':
                    old = word[:]
                    reference = temp_rows[-1][2] if len(temp_rows) > 0 else rows[-1][2]
                    word = expand_degree_abbreviation(word, reference)
                    if reference == word:
                        word = old[:]

                # normalisation
                word = word.replace('λ', 'ɬ')
                word = word.replace('Λ', 'ʌ')
                word = word.strip('.,;-: ')
                word = word.replace('<? >', '')
                word = word.lower()
                word = word.replace('˜', '̃')
                word = word.replace(f'<smallcaps>i</smallcaps>', 'ɪ')

                # Two DDSA transcriptions contain embedded C1 control characters from a broken
                # HTML-entity decode.  The affected spellings cannot be reconstructed safely from
                # the cached text, while a separate readable form remains in each article.  Keep
                # the source evidence in an audit instead of teaching the sound profile that C1
                # controls and the trailing mojibake are legitimate CDIAL graphemes.
                if any(unicodedata.category(character) == "Cc" for character in word):
                    corrupt_forms.append([
                        number,
                        l,
                        word,
                        defn,
                        notes,
                        source_field(" ".join(filter(None, (defn, notes)))),
                        "excluded",
                        "cached DDSA HTML contains undecodable C1-control mojibake",
                    ])
                    continue
                # ``with MIA. -kk-:`` and ``(S. -ṇḍ- ← Centre)`` italicise sound-change labels
                # exactly like forms. They are audited, not installed as reflexes.
                if l != "Indo-Aryan" and is_sound_fragment(word):
                    corrupt_forms.append([
                        number,
                        l,
                        word,
                        defn,
                        notes,
                        source_field(" ".join(filter(None, (defn, notes)))),
                        "excluded",
                        "sound-change fragment or abbreviation, not a word",
                    ])
                    continue

                # handle macron/breve combo, which we store as two forms (long vowel, short vowel)
                oldest = unicodedata.normalize('NFD', word)
                oldest = oldest.replace('̄˘', '̄̆')
                oldest = oldest.replace('̆̄', '̄̆')
                oldest = oldest.replace('̄̆', '̄̆')
                if '̄̆' in oldest:
                    words.append([oldest.replace('̄̆', '̄'), defn, notes, is_variant, ''])
                    oldest = oldest.replace('̄̆', '')
                    word = oldest
                if '{' in oldest:
                    words.append([re.sub(r'{.*?}', '', oldest), defn, notes, is_variant, ''])
                    oldest = oldest.replace('{', '').replace('}', '')
                    word = oldest
                word = unicodedata.normalize('NFC', word)
                        
                cog = f'{subnum}:@variant' if is_variant else (f'{number}.{subnum}' if info is None else f'{subnum}:{info}')
                citations = " ".join(filter(None, (defn, notes)))
                temp_rows.append([l, number, word, defn, '', '', notes, source_field(citations), cog])

        if shared_definition_blocked:
            valency_blocked_rows.update(range(span_start, len(temp_rows)))

        separator = terminal_separator(span)
        if pending_run and closes_run and len(pending_run) <= 3:
            for index in pending_run:
                if not temp_rows[index][3]:
                    temp_rows[index][3] = words[0][1]
            pending_run = []
        elif pending_run and (printed_definitions or separator != ","):
            pending_run = []
        if separator == "," and not printed_definitions:
            pending_run.extend(
                index for index in range(span_start, len(temp_rows)) if not temp_rows[index][3]
            )
        if separator == ",":
            # A span with several meanings does not establish which one a following unglossed
            # language form shares (e.g. OAw. ``citerā 'painter', citeraï 'paints', lakh. citērā``).
            if len(printed_definitions) == 1:
                shared_definition = printed_definitions[0]
            elif len(printed_definitions) > 1 or shared_definition_blocked:
                shared_definition = ""
            # With no newly printed gloss, keep carrying the established meaning through another
            # comma-linked language span (Mth. ... 'to cut', Aw. ..., H. ..., G. ...).
        else:
            shared_definition = ""

        langs = []

    # Turner prints no meaning for a reflex that keeps its headword's (or its numbered
    # section's lemma's) meaning: ``ṣáṣ 'six': … P. che, chī, … B. chay, Or. cha``. Fill what the
    # in-run rules above left blank from that lemma. Causative/valency runs, derivative,
    # contamination (``X``) and borrowing (``→``) groups are semantically independent.
    group_gloss = next((row[3] for row in temp_rows if row[3]), "")
    section_definition = section_meaning(
        info, head_definition, head_definitions, first_group=subnum == 2, group_gloss=group_gloss
    )
    if section_definition and subentry_num != 0 and not allow_arrow_language:
        for index, row in enumerate(temp_rows):
            if row[3] or index in valency_blocked_rows:
                continue
            first_note = row[6].split(";", 1)[0].strip()
            if is_morphological_boundary_note(first_note) and not re.fullmatch(
                r"(?:pres|pret|aor|fut|perf|pp|part|ger|inf|imper|subj|opt)", first_note.strip(" ."), re.IGNORECASE
            ):
                continue  # caus./pass./intr./trans./denom. forms
            if _GENDER_NOTE.match(first_note) and head_verbal and is_verbal_gloss(section_definition):
                continue  # a noun (``miṇaṇa- n.``) under a verbal lemma has its own meaning
            row[3] = section_definition

    return temp_rows, subnum, info, carried

# go through each entire digitised page
for page in tqdm(range(1, TOTAL_PAGES + 1), disable=bool(ONLY)):

    # a single-entry run only touches the pages that carry that entry
    if ONLY and cached and not any(
        re.search(r"<number>\s*" + re.escape(wanted) + r"\s*</number>", soups[page - 1]) for wanted in ONLY
    ):
        continue

    # get content
    link = "https://dsal.uchicago.edu/cgi-bin/app/soas_query.py?page=" + str(page)
    resp = None
    if not cached: resp = urllib.request.urlopen(link)

    # html parse, split into entries
    soup = None
    if cached: soup = BeautifulSoup(soups[page - 1], 'html.parser')
    else:
        soup = BeautifulSoup(resp, 'html.parser')
        soups.append(str(soup))
    soup = str(soup).split('<number>')

    # for each entry on the page, parse
    for entry in soup:

        # rectify artifacts of the transcription process that hurt parsing
        # e.g. punctuation marks that break italics
        entry = str(entry).replace('\n', ' ')
        # Each chunk ends at </hw>; page-layout tags after it are not dictionary-entry notes.
        if '</hw>' in entry:
            entry = entry.split('</hw>', 1)[0]
        entry = re.sub(r'</i>\(<i>([\w]*?)</i>\)<i>', r'{\1}', entry)
        entry = re.sub(r'</i>\(<i>([\w]*?)</i>\)', r'{\1}</i>', entry)
        entry = re.sub(r'\(<i>([\w]*?)</i>\)<i>', r'<i>{\1}', entry)
        entry = entry.replace('</i><i>', '')
        entry = entry.replace("</i>'<i>", "'")
        # Split italics sometimes surround literal transcription text rather than markup boundaries.
        entry = re.sub(r'</i>([A-Za-z]/[A-Za-z])<i>', r'\1', entry)
        # A few source lines omit the period on a language label immediately before an italic form.
        entry = re.sub(r'(?<!\w)(Si)(?=\s+<i>)', r'\1.', entry)
        entry = entry.replace('WH.bāng.', 'WH.bāṅg.')
        entry = entry.replace('*<b>', '<b>*')
        entry = entry.replace(':</b>', '</b><br>')
        entry = entry.replace('*<i>', '<i>*')
        entry = entry.replace('<i>\'</i>', '\'')

        entry = unicodedata.normalize('NFC', entry)
        # Pin the parser: the default picks lxml when installed, which wraps the fragment in
        # <html><body>…</body></html> and leaks that wrapper into every entry's stored etymology.
        # html.parser keeps it a bare fragment (matching the historical output).
        entry = BeautifulSoup('<number>' + entry, 'html.parser')

        # add entry only if it has a bold member (the headword[s])
        if entry.find('b'):
            lemmas = entry.find_all('b')
            number = entry.find('number').text
            if 'A Comparative Dictionary of Indo-Aryan Languages' in number:
                continue
            if ONLY and number.strip() not in ONLY:
                continue

            # reflexes are grouped into paragraphs or marked by Ext. when they share
            # a common origin that is a derived form from the headword (e.g. -kk- extensions)
            head_split = list(re.split(r'(<br/>)', str(entry)))
            data = head_split
            if len(head_split) > 1:
                tail = protect_parenthetical_group_separators('<br/>'.join(head_split[1:]))
                data = [head_split[0]] + [
                    chunk.replace(_INNER_DASH_SENTINEL, '—')
                    for chunk in re.split(r'(<br/>|Ext.|[;\.,:\?] — )', tail)
                ]

            # store headwords
            # for lemma in lemmas:
            #     rows.append(['Indo-Aryan', number, lemma.text, '', '', '', '', 'CDIAL', ''])
            if number not in done:
                params.append([
                    number,
                    lemmas[0].text,
                    '',
                    data[0],
                    entry_source_field(data[0]),
                ])
            done.add(number)

            # ignore headword from rest of parsing; if no other reflexes ignore this entry
            if (len(data) == 1): continue

            # a subentry is a block of descendants; these are separated by newlines in CDIAL
            subnum = 0
            info = None
            carried = ""  # a numbered sub-heading's form number, held for the next paragraph
            head_definition = ""
            head_definitions = []
            head_verbal = is_verbal_lemma(lemmas[0].text)
            data[0] = 'Indo-Aryan. ' + data[0]
            for subentry_num, subentry in enumerate(data):

                # parse this subentry
                rows_sub, subnum, info, carried = parse(
                    subentry,
                    subentry_num,
                    subnum,
                    number,
                    info,
                    carried,
                    head_definition=head_definition,
                    head_definitions=head_definitions,
                    head_verbal=head_verbal,
                )
                rows.extend(rows_sub)
                if subentry_num == 0:
                    # one slot per numbered head lemma; a gloss printed on the lemma's italic
                    # variant (``*āruhati, ā́ruhat 'ascends'``) belongs to that lemma
                    head_definitions = []
                    for row in rows_sub:
                        if row[0] != "Indo-Aryan":
                            continue
                        if not row[8].endswith("@variant"):
                            head_definitions.append(row[3])
                        elif head_definitions and not head_definitions[-1]:
                            head_definitions[-1] = row[3]
                    head_definition = next((gloss for gloss in head_definitions if gloss), "")

                # find terms borrowed into other langs in the notes of each reflex
                for row in rows_sub:
                    borrowed = list(borrowed_terms.finditer(row[6]))
                    for borrow in borrowed:
                        borrowed_text = borrow.group(0)[1:-1]
                        # A colon followed by contrastive prose ends the borrower list; parsing that
                        # tail creates fake forms from comparison morphemes and language examples.
                        borrowed_text = re.split(r':\s*(?:but|though|while)\b', borrowed_text, 1)[0]
                        rows_borrowed, _, _, _ = parse(
                            row[0] + ' ' + borrowed_text,
                            subentry_num,
                            subnum - 1,
                            number,
                            info,
                            allow_arrow_language=True,
                        )
                        # ``H. akhāṛā 'wrestling ground' (→ K. akahār, N. akhāṛā)``: a borrower
                        # printed without its own gloss keeps the lender's meaning.
                        if "'" not in borrowed_text and row[3]:
                            for borrowed_row in rows_borrowed:
                                if not borrowed_row[3]:
                                    borrowed_row[3] = row[3]
                        rows.extend(rows_borrowed)
    
    if not cached: del resp

# Turner writes ``id.`` for "same meaning as the preceding form"; expand it within each entry.
import itertools
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dedr'))
from parser_utils import resolve_idem  # noqa: E402
RELATIVE_GLOSS = re.compile(
    r"^(?:the|its|a) (?:tree|plant|shrub|grass|herb|creeper|fruit|berry|nut|seed|wood|flower|bird|fish"
    r"|animal|root|leaf|leaves|oil|bark)(?: (?:and|or) (?:its|the) \w+)?$"
)


def anchor_relative_glosses(entry_rows):
    """``'the tree'`` / ``'its fruit'`` refer to the headword's referent; name it in parentheses.

    Under akṣōṭa- 'walnut' Turner writes ``Phal. ac̣hū́ṛī 'the tree'`` and ``Sh. ac̣hói f.
    'the tree'``. Out of the entry that is unreadable, so the head's first sense is appended:
    ``the tree (walnut)``. Heads whose own gloss is relative (``'made of its wood'``) are skipped.
    """
    def first_sense(gloss):
        sense = re.split(r"[;,(]", re.sub(r"<[^>]+>", "", gloss), 1)[0].strip()
        if not sense or len(sense.split()) > 5 or RELATIVE_GLOSS.match(sense) or sense.startswith(("made of", "its ")):
            return ""
        return sense

    head = first_sense(next((row[3] for row in entry_rows if row[0] == "Indo-Aryan" and row[3]), ""))
    previous = head  # ``its X`` refers to the form just before; ``the X`` to the headword's referent
    for row in entry_rows:
        gloss = row[3].strip()
        if row[0] != "Indo-Aryan" and RELATIVE_GLOSS.match(gloss):
            referent = previous if gloss.startswith("its ") else head
            if referent and referent not in gloss:
                row[3] = f"{gloss} ({referent})"
        elif first_sense(gloss):
            previous = first_sense(gloss)


for _, entry_rows in itertools.groupby(rows, key=lambda row: row[1]):
    entry_rows = list(entry_rows)
    resolve_idem(entry_rows)
    anchor_relative_glosses(entry_rows)

if ONLY:
    csv.writer(sys.stdout, lineterminator='\n').writerows(rows)
    sys.exit(0)


def write_atomic(path, header, body):
    """Write to a sibling temp file and rename, so a crash never leaves a stale output in place."""
    fd, temporary = tempfile.mkstemp(prefix=path + '.', dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, 'w') as fout:
            writer = csv.writer(fout, lineterminator='\n')
            if header:
                writer.writerow(header)
            writer.writerows(body)
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


write_atomic('cdial.csv', None, rows)
write_atomic('params.csv', None, params)
write_atomic('corrupt_forms.csv', [
    'Entry_ID', 'Language_ID', 'Raw_Form', 'Gloss', 'Notes', 'Source', 'Status', 'Reason'
], corrupt_forms)

if not cached:
    with open('cdial.pickle', 'wb') as fout:
        pickle.dump(soups, fout)

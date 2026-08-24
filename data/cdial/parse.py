import pickle
import os
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
    for index, word in enumerate(words[:-1]):
        # Inter-form text is stored with the preceding word. ``pass.`` in
        # ``ōvahati, pass. ōvuyhati 'is carried down'`` therefore ends its segment here.
        if is_morphological_boundary_note(word[2]):
            segments.append(words[start:index + 1])
            start = index + 1
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
):
    langs = []
    temp_rows = []
    shared_definition = ""
    head_definition_overridden = False

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
            continue

        # extract definitions
        # TODO: get morphological labels, notes
        cur = None
        defs = []
        words = []

        def append_to_words(cur, defs):
            if cur:
                for each in cur[0].split(','):
                    definition = '; '.join([d[0] for d in defs]) if defs != [] else ''
                    notes = '; '.join([d[1] for d in defs if d[1] != '']) if defs != [] else ''
                    notes = cur[1] + ('; ' if (cur[1] and notes) else '') + notes
                    words.append([each.strip(), definition, notes, cur[2]])

        for form in forms:
            if form.group(0).startswith('<i>') or form.group(0).startswith('<b>'):
                append_to_words(cur, defs)
                defs = []
                # an italic form in the head paragraph is an alternate spelling of the preceding bold
                # form → a variant, never a numbered section header (see cognateset marker below)
                is_variant = form.group(0).startswith('<i>') and lang == 'Indo-Aryan'
                cur = [
                    _restore_note_markup(form.group(2)),
                    _restore_note_markup(form.group(4)).strip(' -,;.'),
                    is_variant,
                ]
            else:
                defs.append([
                    _restore_note_markup(form.group(3)).strip(),
                    _restore_note_markup(form.group(4)).strip(' -,;.'),
                ])
        if cur:
            for each in cur[0].split(','):
                append_to_words(cur, defs)

        # A definition printed once scopes over all comma-listed forms in this language span.
        # If the preceding language span ended in a comma, it also scopes forward across language
        # labels until a stronger boundary. Morphological labels start a new run and block carryover.
        printed_definitions = list(dict.fromkeys(word[1] for word in words if word[1]))
        if printed_definitions:
            head_definition_overridden = True
        if not (lang == "Indo-Aryan" and blocks_head_definition_propagation(span)):
            propagate_single_printed_definition(words)
        shared_definition_blocked = blocks_shared_definition(span, forms)
        if shared_definition and not printed_definitions and not shared_definition_blocked:
            for word in words:
                if not word[1]:
                    word[1] = shared_definition
        # The first descendant group often omits quotes because its forms retain the headword's
        # meaning (``jananī 'mother': Pa. jananī-, Pk. jaṇaṇī-, P. jaṇṇī``). Apply that
        # fallback only to wholly unglossed spans in the direct-reflex group. Later numbered,
        # extended, causative, and comparison groups are semantically independent.
        if (
            head_definition
            and subnum == 2
            and not printed_definitions
            and not head_definition_overridden
            and not shared_definition_blocked
        ):
            for word in words:
                if not word[1]:
                    word[1] = head_definition

        # for each language on the stack, add this entry
        for l in langs:
            # Preserve the parser's stack and definition state above, but do not emit cited
            # Dravidian comparison forms as ordinary Indo-Aryan reflexes.
            if l in DRAVIDIAN_COMPARISON_LANGS:
                continue
            for word, defn, notes, is_variant in words:
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

                # handle macron/breve combo, which we store as two forms (long vowel, short vowel)
                oldest = unicodedata.normalize('NFD', word)
                oldest = oldest.replace('̄˘', '̄̆')
                oldest = oldest.replace('̆̄', '̄̆')
                oldest = oldest.replace('̄̆', '̄̆')
                if '̄̆' in oldest:
                    words.append([oldest.replace('̄̆', '̄'), defn, notes, is_variant])
                    oldest = oldest.replace('̄̆', '')
                    word = oldest
                if '{' in oldest:
                    words.append([re.sub(r'{.*?}', '', oldest), defn, notes, is_variant])
                    oldest = oldest.replace('{', '').replace('}', '')
                    word = oldest
                word = unicodedata.normalize('NFC', word)
                        
                cog = f'{subnum}:@variant' if is_variant else (f'{number}.{subnum}' if info is None else f'{subnum}:{info}')
                citations = " ".join(filter(None, (defn, notes)))
                temp_rows.append([l, number, word, defn, '', '', notes, source_field(citations), cog])

        separator = terminal_separator(span)
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

    return temp_rows, subnum, info, carried

# go through each entire digitised page
for page in tqdm(range(1, TOTAL_PAGES + 1)):
    
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
                )
                rows.extend(rows_sub)
                if subentry_num == 0:
                    head_definition = next(
                        (row[3] for row in rows_sub if row[0] == "Indo-Aryan" and row[3]),
                        "",
                    )

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
                        rows.extend(rows_borrowed)
    
    if not cached: del resp

with open(f'cdial.csv', 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerows(rows)

with open(f'params.csv', 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerows(params)

with open('corrupt_forms.csv', 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    writer.writerow([
        'Entry_ID', 'Language_ID', 'Raw_Form', 'Gloss', 'Notes', 'Source', 'Status', 'Reason'
    ])
    writer.writerows(corrupt_forms)

if not cached:
    with open('cdial.pickle', 'wb') as fout:
        pickle.dump(soups, fout)

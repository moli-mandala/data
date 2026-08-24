import pickle
import os
import urllib.request
import re
import json
import copy
import csv
from bs4 import BeautifulSoup
from collections import defaultdict
from enum import Enum
from tqdm import tqdm

from abbrevs import (
    abbrevs, dialects, replacements, fixes, shared_gloss_boundaries,
    shared_section_glosses, source_markup_repairs,
)
from cleanup import footer_note, is_footer_misparse
from parser_utils import (
    clean_form_html,
    detach_leading_grammatical_note,
    detach_trailing_forward_grammatical_note,
    grammatical_tag,
    grammatical_tags,
    is_prose_misparse,
    marker_only_tags,
    mark_unbolded_compact_forms,
    mark_unbolded_gender_forms,
    propagate_shared_glosses,
    split_alternatives,
    split_forms,
    split_language_spans,
    split_tagged_forms,
    strip_grammatical_markers,
)

TOTAL_PAGES = 514
APPENDIX = 509
ERR = False

# useful regexes
l = '(' + "|".join(sorted([re.escape(x) for x in abbrevs], key=lambda x: -len(x))) + r')'
langs_regex = re.compile(r'(' + l + r')(\.|$)')
l += r'\.?'
regex = re.compile(r'(<i>|<b>|^)*' + l + r'(([^\(\)\[\]]*?(\[.*?\]|\(.*?\)))*?[^\(\)\[\]]*?)(?=((<i>|<b>)*' + l + r'|DED|DEN|</div>|$))')
lemmata = re.compile(r'(<b>|^)(.*?)(</b>|$)(.*?)((?=<b>)|$)')
formatter = re.compile(r'<.*?>')
comma_split = re.compile(r',(?![^\(]*?\))')
_LEXICAL_SLASH_SENTINEL = '\ue010'


def split_entry_sections(value):
    """Split source-level slash notes while preserving slashes inside bold lexical forms."""
    pieces = []
    cursor = 0
    for match in re.finditer(r'(?:\s+|(?<=[.;:]))/\s*', value):
        pieces.append(value[cursor:match.start()])
        inside_bold = (
            value.rfind('<b>', 0, match.start())
            > value.rfind('</b>', 0, match.start())
        )
        following = BeautifulSoup(value[match.end():], 'html.parser').get_text(
            ' ', strip=True
        )
        comparison_cue = bool(re.match(
            r'(?:cf|perhaps|prob|poss|borrow|loan|influenc|relationship|areal)\b',
            following,
            re.IGNORECASE,
        ))
        source_separator = (
            not inside_bold
            and bool(following)
            and (not following[0].islower() or comparison_cue)
        )
        pieces.append(
            ' / ' if source_separator
            else match.group().replace('/', _LEXICAL_SLASH_SENTINEL)
        )
        cursor = match.end()
    pieces.append(value[cursor:])
    protected = ''.join(pieces)
    return [
        part.replace(_LEXICAL_SLASH_SENTINEL, '/')
        for part in re.split(r'( / |(?=Cf\.))', protected)
    ]


def strip_embedded_cross_family_notes(value):
    """Remove balanced parenthetical/bracketed CDIAL notes but retain their HTML boundaries.

    Several DEDR comparisons occur inside the gloss of the final Dravidian form instead of in a
    slash section.  Leaving them there both truncates that gloss and can emit the cited IA lemma
    as another reflex.  The source wording is retained by cross_family.py.
    """
    opening = {'(': ')', '[': ']'}
    closing = {')': '(', ']': '['}
    stack = []
    ranges = []
    for index, character in enumerate(value):
        if character in opening:
            stack.append((character, index))
        elif character in closing:
            for stack_index in range(len(stack) - 1, -1, -1):
                if stack[stack_index][0] != closing[character]:
                    continue
                _, start = stack[stack_index]
                del stack[stack_index:]
                fragment = value[start:index + 1]
                if re.search(r'\bCDIAL\b', BeautifulSoup(fragment, 'html.parser').get_text(' '), re.I):
                    ranges.append((start, index + 1))
                break

    # Prefer the outer comparison group when nested parentheses occur inside the note.
    selected = []
    for start, end in sorted(ranges, key=lambda pair: (pair[0], -pair[1])):
        if selected and selected[-1][0] <= start and end <= selected[-1][1]:
            continue
        selected.append((start, end))
    for start, end in reversed(selected):
        # Preserve tags which opened or closed inside the note so the surrounding lexical HTML
        # remains balanced (DEDR 195 closes its headword's <b> inside the comparison bracket).
        tag_tokens = []
        open_tags = []
        for match in re.finditer(r'<(/?)\s*([A-Za-z0-9]+)[^>]*>', value[start:end]):
            token = [match.group(2).lower(), bool(match.group(1)), match.group(), True]
            tag_tokens.append(token)
            if not token[1]:
                open_tags.append(len(tag_tokens) - 1)
                continue
            paired = next(
                (position for position in reversed(open_tags)
                 if tag_tokens[position][0] == token[0]),
                None,
            )
            if paired is not None:
                tag_tokens[paired][3] = False
                token[3] = False
                open_tags.remove(paired)
        tags = ''.join(token[2] for token in tag_tokens if token[3])
        value = value[:start] + tags + value[end:]
    return value

def is_bold_or_italic(tag):
    return tag.name in ('b', 'i') and not (any([x.name in ('b', 'i') for x in tag.children]))

# response caching logic
soups = []
cached = False
if os.path.exists('dedr.pickle'):
    with open('dedr.pickle', 'rb') as fin:
        soups = pickle.load(fin)
    cached = True
print('Caching?', cached)

# file
fout = open('dedr_new.csv', 'w')
writer = csv.writer(fout)
footer_notes = defaultdict(list)
if os.path.exists('footer_notes.csv'):
    with open('footer_notes.csv') as fin:
        for param, note_html in csv.reader(fin):
            note_soup = BeautifulSoup(note_html, 'html.parser')
            footer_notes[param].extend(str(note_soup.body or note_soup).removeprefix('<body>').removesuffix('</body>').split('<br>'))

count = 1

ref_ct = defaultdict(int)

# go through each entire digitised page
for page in tqdm(range(1, TOTAL_PAGES + 1)):
    if ERR: print(page)
    
    # get content
    link = "https://dsal.uchicago.edu/cgi-bin/app/burrow_query.py?page=" + str(page)
    resp = None
    if not cached: resp = urllib.request.urlopen(link)
    if ERR: print('fetched page')

    # html parse, split into entries
    soup = None
    if cached: soup = BeautifulSoup(soups[page - 1], 'html5lib')
    else:
        soup = BeautifulSoup(resp, 'html5lib').find(class_='hw_result')
        soups.append(str(soup))
    soup = str(soup).split('<number>')
    if ERR: print('made soup')

    # for each entry on the page, parse
    for entry_index, entry in enumerate(soup):
        # prettify
        entry = entry.replace('\n', '')
        entry = BeautifulSoup('<number>' + entry, 'html.parser')

        # only if this is an actual entry
        if entry.find('number'):

            # store and get rid of number
            number: str = entry.find('number').text
            entry.find('number').decompose()
            if page >= APPENDIX:
                number = 'a' + number

            # Some page snapshots contain a short entry followed by a longer
            # continuation carrying the same number.  Parse only the final,
            # complete occurrence to avoid duplicate rows.
            later_numbers = [
                chunk.split('</number>', 1)[0].strip()
                for chunk in soup[entry_index + 1:]
                if '</number>' in chunk
            ]
            raw_number = number[1:] if page >= APPENDIX else number
            if raw_number in later_numbers:
                continue

            if ERR: print(entry)
            # Source-level slashes introduce comparison commentary; a slash inside a bold form is
            # a lexical alternative and must remain in the reflex parser.
            entry_str = split_entry_sections(str(entry))
            for i in range(1, len(entry_str)):
                for f in sorted(fixes, key=lambda x: -len(x)):
                    entry_str[i] = entry_str[i].replace(f, f'<b><i>{f}</i></b>')
            
            for section_num, section in enumerate(entry_str):
                # Numeric ``Cf.`` tails point to another DEDR entry; their
                # bold headword is not another reflex in the current entry.
                preceded_by_slash = (
                    section_num > 0 and entry_str[section_num - 1].strip() == '/'
                )
                # cross_family.py owns the comparison tail (including its wording and cited
                # forms). A later explicitly labelled DEDR subsection can follow that note in
                # the same HTML run, however; resume only at that subsection (e.g. d1110 (b)).
                if preceded_by_slash:
                    next_subsection = re.search(
                        r'(?=(?:<b>)?<i>\([a-z]\)(?:\s|</i>))', section
                    )
                    if not next_subsection:
                        continue
                    section = section[next_subsection.start():]
                if re.match(r'\s*Cf\.', section) and not preceded_by_slash:
                    next_subsection = re.search(r'(?=<i>\([a-z]\))', section)
                    if not next_subsection:
                        continue
                    section = section[next_subsection.start():]
                section = strip_embedded_cross_family_notes(section)
                entry = BeautifulSoup(section, 'html.parser')

                section_label, spans = split_language_spans(section, abbrevs)
                section_rows = []

                for span in spans:
                    lang = abbrevs[span[0].strip('.')]
                    # Indo-Aryan/Sanskrit forms in a DEDR comparison tail are evidence for an
                    # article-level cross-family claim, not Dravidian reflexes of this etymon.
                    # data/cross_family.py resolves their printed CDIAL IDs into the dedicated
                    # comparisons table and audits unresolved prose.
                    if lang == 'OIA':
                        continue
                    for old, new in source_markup_repairs.get(str(number), []):
                        span[1] = span[1].replace(old, new)
                    span[1] = mark_unbolded_gender_forms(span[1].strip())
                    boundary = shared_gloss_boundaries.get(str(number))
                    if boundary and f', {boundary}' in span[1]:
                        span[1] = span[1].replace(f', {boundary}', f'</b> {boundary}', 1)
                    span[1] = mark_unbolded_compact_forms(span[1])

                    # get every forms + gloss pairing (delineated by bold tags)
                    rows = []
                    last_paren = False
                    pending_tags = []
                    pending_form_tags = []
                    for y in lemmata.finditer(span[1]):
                        if ERR: print('    lemma', y)
                        gloss = y.group(4).strip(' ').split('\t')[0]

                        if last_paren:
                            lemma_tags = marker_only_tags(y.group(2))
                            extra_tags = list(dict.fromkeys(
                                lemma_tags + grammatical_tags(gloss)
                            ))
                            rows[-1][14] = " ".join(
                                dict.fromkeys(rows[-1][14].split() + extra_tags)
                            )
                            marker = '' if lemma_tags else y.group(2)
                            rows[-1][3] += marker + strip_grammatical_markers(gloss)
                            last_paren = rows[-1][3].count('(') > rows[-1][3].count(')')
                            continue

                        marker_tags = marker_only_tags(y.group(2))
                        if y.group(2) == 'Voc.' or marker_tags:
                            if rows:
                                rows[-1][14] = " ".join(
                                    dict.fromkeys(rows[-1][14].split() + marker_tags)
                                )
                            else:
                                pending_tags = list(dict.fromkeys(pending_tags + marker_tags))
                            marker = y.group(2) if not marker_tags else ''
                            if rows:
                                rows[-1][3] += marker + strip_grammatical_markers(gloss)
                                last_paren = rows[-1][3].count('(') > rows[-1][3].count(')')
                            continue
                        
                        form_html = y.group(2).strip()
                        previous_tags, form_html = detach_leading_grammatical_note(form_html)
                        if previous_tags and rows:
                            rows[-1][14] = " ".join(
                                dict.fromkeys(rows[-1][14].split() + previous_tags)
                            )
                        initial_form_tags = pending_form_tags
                        pending_form_tags, gloss = detach_trailing_forward_grammatical_note(gloss)
                        if initial_form_tags and re.fullmatch(
                            r"[^\s<>]+\.", formatter.sub('', form_html).strip()
                        ):
                            form_html = form_html.rstrip().removesuffix('.')
                        tags = list(dict.fromkeys(pending_tags + grammatical_tags(gloss)))
                        pending_tags = []
                        gloss = strip_grammatical_markers(gloss)
                        cleaned_form, gloss = clean_form_html(form_html, gloss)
                        is_gender_form = re.match(
                            r'(?:fem|masc)\.', formatter.sub('', form_html).strip()
                        )
                        if is_gender_form and rows and not gloss.strip(';,./ '):
                            gloss = rows[-1][3]
                        if is_prose_misparse(form_html):
                            if rows:
                                rows[-1][3] += (' ' if rows[-1][3] else '') + cleaned_form
                            continue
                        row = [
                            lang, 'd' + str(number), cleaned_form, gloss, '', '', '', 'dedr',
                            f'{section_num}:{section_label}', '', '', '', '', '', ' '.join(tags),
                            split_tagged_forms(form_html, initial_form_tags),
                        ]

                        # extract parentheticals from previous row--they are sources or notes about this one
                        if rows:
                            if rows[-1][3].endswith(')'):
                                paren = rows[-1][3].rfind('(')
                                trailing = rows[-1][3][paren:]
                                trailing_tags = grammatical_tags(trailing)
                                if trailing_tags:
                                    rows[-1][14] = " ".join(
                                        dict.fromkeys(rows[-1][14].split() + trailing_tags)
                                    )
                                    rows[-1][3] = rows[-1][3][:paren].rstrip()
                                else:
                                    row[6] = trailing[1:-1]
                                    rows[-1][3] = rows[-1][3][:paren]
                        
                        # extract parentheticals from this row
                        if row[2].startswith('('):
                            paren = row[2].find(')')
                            row[6] += (' ' if row[6] else '') + row[2][:paren].strip(' ()')
                            row[2] = row[2][paren + 1:].strip()

                        note_tag = grammatical_tag(row[6])
                        if note_tag:
                            row[14] = " ".join(dict.fromkeys(filter(None, row[14].split() + [note_tag])))
                            row[6] = ""

                        rows.append(row)

                        if gloss.count('(') > gloss.count(')'):
                            last_paren = True

                        if ERR: print('        done with forms')
                    
                    # Source/dialect markers sometimes sit between a group of
                    # forms and their shared gloss.  Carry that gloss backwards
                    # over otherwise empty rows in the same language span.
                    for row in rows:
                        row[3] = re.sub(r'<([ib])>\s*</\1>', '', row[3])
                        row[3] = re.sub(r'\(\s*\)', '', row[3])
                        row[3] = re.sub(r' {2,}', ' ', row[3])
                        row[3] = row[3].strip(';,./ ').lstrip(') ')
                    for pos, row in enumerate(rows):
                        if not row[3]:
                            for later in rows[pos + 1:]:
                                if later[3]:
                                    row[3] = later[3]
                                    break

                    for pos, row in enumerate(rows):
                        # fix Tamil (-pp-, -tt-)
                        if row[0] == 'Tam' and row[2] == '' and row[6] == '-pp-, -tt-':
                            row[2] = rows[pos - 1][2].split(' (')[0] + ' (-pp-, -tt-)'
                            row[6] = ""

                        forms = row[15] or [(form, []) for form in split_forms(row[2])]
                        row[3] = re.sub(r'\s*\?\s*$', '', row[3]).strip(';,./ ')

                        for replacement in replacements:
                            row[6] = row[6].replace(replacement, replacements[replacement])

                        # refs and dialects
                        dial_forms = []
                        for ref in row[6].split():
                            ref = ref.strip(' ,;')
                            if (ref, row[0]) in dialects:
                                ref, dial = dialects[(ref, row[0])]
                                if dial:
                                    dial_forms.append(dial)
                                if ref:
                                    row[7] += ';' + ref
                            else:
                                ref_ct[(ref, row[0])] += 1

                        dial_forms = list(dict.fromkeys(dial_forms))
                        if not dial_forms:
                            dial_forms.append(row[0])

                        # add forms for each dialect
                        for dial in dial_forms:
                            for form, form_tags in forms:
                                new_row = row[::]
                                new_row[0] = dial
                                new_row[14] = " ".join(
                                    dict.fromkeys(new_row[14].split() + form_tags)
                                )

                                if ERR: print('        form', form)
                                raw_form = form.strip()
                                if is_prose_misparse(raw_form):
                                    continue
                                form = formatter.sub('', form).strip()

                                # extract parentheticals from this row
                                if form.startswith('('):
                                    paren = form.find(')')
                                    new_row[6] += (' ' if new_row[6] else '') + form[:paren].strip(' ()')
                                    form = form[paren + 1:].strip()

                                # handle parse fails for Turner cognates
                                if lang == 'OIA' and (form == '' or 'no.' in form):
                                    continue

                                # Repair a common source typo where the closing
                                # parenthesis falls just outside the bold lemma.
                                if form.count('(') == form.count(')') + 1 and new_row[3].startswith(')'):
                                    form += ')'
                                    new_row[3] = new_row[3][1:].lstrip()

                                for altform in split_alternatives(form):
                                    new_row[2] = altform.strip(" ;.,/")
                                    if new_row[2] and not is_footer_misparse(new_row[2]):
                                        section_rows.append(new_row[:15])
                                    elif new_row[2]:
                                        note = footer_note(raw_form, new_row[3])
                                        param = 'd' + str(number)
                                        if note and note not in footer_notes[param]:
                                            footer_notes[param].append(note)
                                    count += 1

                shared_fallback = shared_section_glosses.get((str(number), section_num), '')
                for output_row in propagate_shared_glosses(section_rows, shared_fallback):
                    writer.writerow(output_row)

                if ERR: print('    done with spans')
    
    if ERR: print('deleting')
    if not cached: del resp
    if ERR: print('deleted')

# print top values in ref_ct
for key in sorted(ref_ct, key=lambda x: ref_ct[x], reverse=True)[:100]:
    print(key, ref_ct[key])

# close file
fout.close()

with open('footer_notes.csv', 'w') as fout:
    writer = csv.writer(fout, lineterminator='\n')
    for param, notes in footer_notes.items():
        writer.writerow([param, f"<html><body>{'<br>'.join(notes)}</body></html>"])

if not cached:
    with open('dedr.pickle', 'wb') as fout:
        pickle.dump(soups, fout)

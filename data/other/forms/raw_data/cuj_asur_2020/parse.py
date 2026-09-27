"""Candidate semantic extraction, deliberately not an installer.

Every physical article receives an audit record; missing transcription, missing
English gloss and unresolved native glyphs stay visible for review. Full senses,
variants and relationship resolution must be reviewed before source installation.
"""
import argparse
import collections
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NATIVE = 'KohinoorDevanagariBold'
ROMAN = 'CharisSIL-Bold'
PLAIN = 'CharisSIL'
ITALIC = 'CharisSIL-Italic'
HINDI = 'AnnapurnaSIL-Regular'
POS = {'n', 'v', 'adj', 'adv', 'interj', 'quant', 'pro-form', 'pro', 'sfx',
       'part', 'num', 'conj', 'post', 'prep', 'pron', 'aux', 'ideo', 'clf',
       'nomprt', 'conn', 'Vce', 'prt', 'dem', 'pfx', 'det'}


def native_join(tokens):
    result = ''
    previous = None
    for token in tokens:
        same_line = (previous and previous['pdf_page'] == token['pdf_page']
                     and previous['column'] == token['column']
                     and abs(previous['top'] - token['top']) < 2)
        gap = token['x0'] - previous['x1'] if same_line else 100
        result += (' ' if result and gap > 1.5 else '') + token['text']
        previous = token
    # A PDF word break can fall between the pre-base i glyph and its base.
    # Repeat logical ordering after rejoining adjacent native pieces.
    result = re.sub(r'\ue001([क-हक़-य़](?:़)?(?:्[क-हक़-य़](?:़)?)*)', r'\1ि', result)
    return re.sub(r'^([-=]) +(?=[\u0900-\u097f])', r'\1', result)


def native_target(tokens, start):
    native, homonym = [], []
    while start < len(tokens) and tokens[start]['font'] == NATIVE:
        token = tokens[start]
        (homonym if token['size'] < 7 else native).append(token)
        start += 1
    sense = ''
    if (start < len(tokens) and tokens[start]['font'] == 'AnnapurnaSIL-Bold'
            and tokens[start]['text'].isdigit()):
        sense = tokens[start]['text']
        start += 1
    return {'native': native_join(native), 'printed_sense': sense,
            'homonym': ''.join(t['text'] for t in homonym)}, start


def source_references(tokens):
    """Keep lexical relations separate from etymological ancestry claims."""
    refs = []
    depth = 0
    for index, token in enumerate(tokens):
        label, start, kind = '', index + 1, ''
        text = token['text'].lstrip('(')
        # A nested variant can describe a referenced word, not this headword.
        # Preserve its scope so an importer cannot silently attach it here.
        depth += token['text'].count('(')
        reference_depth = depth
        depth -= token['text'].count(')')
        if text in {'unspec.', 'fr.', 'sp.', 'dial.'} and index + 1 < len(tokens):
            if tokens[index + 1]['text'] == 'var.':
                label = text + ' var.'
                start = index + 2
                kind = 'variant-list'
                if start < len(tokens) and tokens[start]['text'] == 'of':
                    start += 1
                    kind = 'variant-of'
            elif [t['text'] for t in tokens[index + 1:index + 4]] == ['comp.', 'form', 'of']:
                label, kind, start = text + ' comp. form of', 'complex-form-of', index + 4
        elif token['font'] == ITALIC and text in {'wh', 'pt', 'gen', 'spec', 'syn', 'ant'}:
            label, kind = text, 'lexical-' + text
            if start < len(tokens) and tokens[start]['text'] == ':':
                start += 1
        elif text == 'comp.' and index + 1 < len(tokens) and tokens[index + 1]['text'] == 'of':
            label, kind, start = 'comp. of', 'compound-of', index + 2
        if not kind:
            continue
        targets = []
        while start < len(tokens):
            target, end = native_target(tokens, start)
            if not target['native']:
                break
            targets.append(target)
            start = end
            if start < len(tokens) and tokens[start]['text'] in {',', ';'}:
                start += 1
            else:
                break
        follows_target = (token['text'].startswith('(') and index > 0
                          and tokens[index - 1]['font'] in {NATIVE, 'AnnapurnaSIL-Bold'})
        refs.append({'kind': kind, 'label': label, 'targets': targets,
                     'token_index': index, 'parenthesis_depth': reference_depth,
                     'scope': 'nested-reference' if reference_depth > 1 or follows_target else 'entry'})
    return refs


def leading_annotations(tokens):
    annotations, start = [], 0
    while start < len(tokens):
        if tokens[start]['font'] == 'SegoeUIEmoji':
            start += 1
            continue
        if not tokens[start]['text'].startswith('('):
            break
        depth, words = 0, []
        while start < len(tokens):
            text = tokens[start]['text']
            depth += text.count('(') - text.count(')')
            words.append(text)
            start += 1
            if depth <= 0:
                break
        annotations.append(' '.join(words))
    return annotations, start


def inline_compounds(tokens):
    starts = []
    for i, token in enumerate(tokens):
        if token['text'] != 'comp.':
            continue
        label_start = i - 1 if i and tokens[i - 1]['text'] == 'unspec.' else i
        start = label_start
        while start and tokens[start - 1]['font'] == NATIVE:
            start -= 1
        if start < label_start:
            end = i + 1
            if end < len(tokens) and tokens[end]['text'] == 'form':
                end += 1
            starts.append((start, label_start, end))
    result = []
    for n, (start, label, end) in enumerate(starts):
        limit = starts[n + 1][0] if n + 1 < len(starts) else len(tokens)
        for i in range(end, limit):
            if tokens[i]['font'] == ROMAN and re.fullmatch(r'\d+\)', tokens[i]['text']):
                limit = i
                break
        result.append({'native': native_join(tokens[start:label]),
                       'label': ' '.join(t['text'] for t in tokens[label:end]),
                       'hindi_raw': ' '.join(t['text'] for t in tokens[end:limit] if t['font'] == HINDI),
                       'pos': [t['text'] for t in tokens[end:limit] if t['font'] == ITALIC and t['text'] in POS],
                       'token_start': start, 'token_end': limit})
    return result


def parse(record):
    tokens = record['tokens']
    index = 0
    native, ipa, homonyms = [], [], []
    while index < len(tokens) and tokens[index]['font'] == NATIVE:
        token = tokens[index]
        if token['size'] < 7:
            homonyms.append(token['text'])
        else:
            native.append(token)
        index += 1
    if index < len(tokens) and tokens[index]['font'] == 'SegoeUI-Bold' and tokens[index]['text'] == 'ʔ':
        native.append(tokens[index])
        index += 1
    while index < len(tokens):
        token = tokens[index]
        if token['size'] < 7 and token['font'] == NATIVE:
            index += 1
            continue
        if token['font'] != ROMAN or re.fullmatch(r'\d+\)', token['text']):
            break
        ipa.append(token['text'])
        index += 1
    body = tokens[index:]
    body_text = ' '.join(t['text'] for t in body)
    annotations, definition_start = leading_annotations(body)
    references = source_references(body)
    # Botanical/zoological identifications are italic like examples, but
    # binomials remain useful definition evidence even without English prose.
    scientific_names = []
    for first, second in zip(body, body[1:]):
        if (first['font'] == second['font'] == ITALIC
                and re.fullmatch(r'[A-Z][a-z]+', first['text'])
                and re.fullmatch(r'[a-z]+', second['text'])):
            scientific_names.append(first['text'] + ' ' + second['text'])
    markers = [i for i, t in enumerate(body)
               if t['font'] == ROMAN and re.fullmatch(r'\d+\)', t['text'])]
    boundaries = markers or [0]
    if markers and markers[0] > 0:
        preamble = body[:markers[0]]
    else:
        preamble = []
    default_pos = [t['text'] for t in preamble if t['font'] == ITALIC and t['text'] in POS]
    senses = []
    for n, start in enumerate(boundaries):
        end = boundaries[n + 1] if n + 1 < len(boundaries) else len(body)
        part = body[start:end]
        printed_sense = part[0]['text'][:-1] if markers else ''
        eligible = bool(default_pos or ipa)
        pos, hindi, gloss = [], [], []
        collecting = False
        annotation_depth = 0
        for token in part:
            text, font = token['text'], token['font']
            if not collecting and (annotation_depth or (font == PLAIN and text.startswith('('))):
                annotation_depth += text.count('(') - text.count(')')
                continue
            if font == 'SegoeUIEmoji':
                continue
            if font == ROMAN and re.fullmatch(r'\d+\)', text):
                eligible = True
                continue
            if font == ITALIC and text in POS and not collecting:
                eligible = True
                pos.append(text)
                continue
            if font == HINDI and not collecting:
                eligible = True
                hindi.append(text)
                continue
            if font == PLAIN and eligible:
                if text.startswith('[') or text == 'comp.':
                    break
                collecting = True
                gloss.append(text)
            elif collecting:
                # Source examples use native regular + roman italic; labels
                # introducing cross-references use italic. Neither is gloss.
                break
        senses.append({'printed_sense': printed_sense,
                       'pos': pos or default_pos,
                       'english_gloss': ' '.join(gloss),
                       'hindi_raw': ' '.join(hindi),
                       'scientific_names': [a['text'] + ' ' + b['text']
                          for a, b in zip(part, part[1:])
                          if a['font'] == b['font'] == ITALIC
                          and re.fullmatch(r'[A-Z][a-z]+', a['text'])
                          and re.fullmatch(r'[a-z]+', b['text'])]})
    headword = native_join(native)
    flags = []
    if '(cid:' in headword or '\ue000' in headword or '\ue001' in headword or '◌' in headword:
        flags.append('unresolved-native-glyph')
    if not ipa:
        flags.append('no-printed-IPA-head')
    if any(not s['english_gloss'] for s in senses):
        flags.append('missing-English-gloss')
    relation_text = ' '.join(t['text'] for t in body[definition_start:])
    relation = re.match(r'(unspec\. var\.|fr\. var\.|sp\. var\.|dial\. var\.|comp\.|der\.) of\b', relation_text)
    return {'entry_key': record['entry_key'], 'printed_page': record['printed_page'],
            'pdf_page': record['pdf_page'], 'column': record['column'],
            'native': headword, 'ipa': ' '.join(ipa), 'homonym': ''.join(homonyms),
            'kind': 'cross-reference' if relation else 'article',
            'relation_label': relation.group(1) if relation else '',
            'senses': senses, 'review_flags': flags,
            'annotations': annotations, 'references': references,
            'scientific_name_candidates': scientific_names,
            'inline_compounds': inline_compounds(body),
            'relationship_body': body_text if relation else '',
            'status': 'candidate-unreviewed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    counts = collections.Counter()
    with (HERE / 'entries.jsonl').open() as stream, args.output.open('w') as out:
        for line in stream:
            parsed = parse(json.loads(line))
            out.write(json.dumps(parsed, ensure_ascii=False) + '\n')
            counts['entries'] += 1
            counts[parsed['kind']] += 1
            counts['senses'] += len(parsed['senses'])
            counts['with_IPA'] += bool(parsed['ipa'])
            counts.update(parsed['review_flags'])
    print(json.dumps(counts, indent=2))

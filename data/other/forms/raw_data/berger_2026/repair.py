"""Conservative, source-positioned repairs to the pinned Berger extraction.

Native PDF typography supplies scope; OCR supplies spelling. Uncertain lexical
readings and references remain explicit in the audit, never fuzzy graph links.
"""
from __future__ import annotations

import csv
import difflib
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLASSES = {'h': ['H'], 'hm': ['HM'], 'hf': ['HF'], 'hmf': ['HM', 'HF'],
           'x': ['X'], 'y': ['Y'], 'hx': ['H', 'X'], 'hy': ['H', 'Y'], 'xy': ['X', 'Y']}
LABELS = {'h', 'hm', 'hf', 'hmf', 'x', 'y', 'hz', 'ng', 'hz.ng', 'NH', 'sg', 'pl',
          'D.pl', 'fem', 'mask', 'adj', 'Adj', 'adv', 'Adv', 'Pron', 'Postp', 'Konj',
          'Interj', 'trs', 'intr', 'trans', 'intrans', 'K', 'L', 'auch', 'und', 'oder', 'usw'}
LABELS.update(CLASSES)
POS = {'adj': 'adj', 'adv': 'adv', 'pron': 'pron', 'postp': 'postp', 'konj': 'conj',
       'interj': 'interj', 'trs': 'tr', 'trans': 'tr', 'intr': 'intr', 'intrans': 'intr'}
GRAMMAR = {'noun', 'verb', 'adj', 'adv', 'pron', 'postp', 'conj', 'interj', 'sg', 'pl', 'f', 'm', 'tr', 'intr'}


def column_starts(page):
    """Recover four entry margins, including shifted scans and indented text."""
    xs = [line['left'] for line in page['lines']]
    centers = [page['width'] * x for x in (.09, .29, .53, .73)]
    for _ in range(20):
        groups = [[] for _ in centers]
        for x in xs:
            groups[min(range(4), key=lambda i: abs(x - centers[i]))].append(x)
        next_centers = [sum(g) / len(g) if g else centers[i] for i, g in enumerate(groups)]
        if centers == next_centers:
            break
        centers = next_centers
    if min(b - a for a, b in zip(centers, centers[1:])) < page['width'] * .1:
        raise ValueError(f"Cannot identify four separated columns on PDF page {page['pdf_page']}")
    # A fragment containing only the definition may begin far to the right of
    # its headword. Boundaries belong just before the next column's margin,
    # not halfway between the column centers.
    return [sorted(g)[max(0, round((len(g) - 1) * .08))] if g else centers[i]
            for i, g in enumerate(groups)]


def column(left, starts):
    # "davon/dazu" heads can project about 50 px left of the normal entry
    # margin at the pinned 300 dpi. Keep that hanging indent in its column.
    return max([0] + [i for i in range(1, 4) if left >= starts[i] - 60])


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.casefold()) if c.isalnum())


def aligned_tokens(ocr, native):
    """Transfer font evidence through character alignment, with confidence.

    Never transfer the hidden text's degraded spelling to the lexical form.
    """
    src, fonts = [], []
    for word, font in native:
        for c in norm(word):
            src.append(c)
            fonts.append(font)
    tokens = re.findall(r'\S+', ocr)
    dst, owners = [], []
    for i, token in enumerate(tokens):
        for c in norm(token):
            dst.append(c)
            owners.append(i)
    votes = defaultdict(Counter)
    for block in difflib.SequenceMatcher(None, ''.join(src), ''.join(dst), autojunk=False).get_matching_blocks():
        for k in range(block.size):
            votes[owners[block.b + k]][fonts[block.a + k]] += 1
    output = []
    for i, token in enumerate(tokens):
        counts = votes[i]
        font, count = counts.most_common(1)[0] if counts else ('', 0)
        output.append((token, font, sum(counts.values()) / max(1, len(norm(token)))))
    return output


def load_layout():
    with gzip.open(HERE / 'layout.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return {(r['pdf'], r['left'], r['top']): r for r in map(json.loads, stream)}


def verify_inputs():
    """Fail closed when a published snapshot's editorial inputs drift."""
    manifest = HERE / 'manifest.json'
    if not manifest.exists():
        return  # Initial editorial preparation precedes the first manifest.
    for relative, expected in json.loads(manifest.read_text())['inputs'].items():
        path = HERE / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Berger pinned input changed: {relative}; review and update manifest explicitly')


def header_tokens(unit, layout):
    output = []
    for line in unit.lines:
        r = layout.get((line.get('pdf_page', unit.pdf_page), line['left'], line['top']))
        if not r:
            break
        tokens = aligned_tokens(line['text'], r['words'])
        for token, font, confidence in tokens:
            label = token.strip(' ,;:()').rstrip('.')
            lexical = 'Italic' in font or 'Oblique' in font
            allowed = label in LABELS or not re.search(r'[A-Za-zÀ-ž]', token)
            if not output and label.lower() in {'davon', 'dazu'}:
                continue
            if font in {'Times-Roman', 'Helvetica'} and confidence >= .65 and not allowed:
                return output
            if not lexical and confidence < .5 and re.search(r'[A-Za-z]', token):
                return output
            output.append((token, lexical, confidence))
    return output


def scoped_tags(header, form, old_tags, *, variant=False):
    """Read only the current form's labels, stopping at a subsequent full form."""
    tags = [t for t in old_tags if t not in GRAMMAR and not t.startswith(('Burushaski-class-', 'dialect:'))]
    target = norm(form)
    start = 0
    for i, (token, lexical, _) in enumerate(header):
        if lexical and norm(token) == target:
            start = i
            break
    # Dialect labels at the end of a paradigm scope the entry; a label preceding
    # a different full form scopes that alternative only.
    dialects = []
    for i in range(start + 1, len(header)):
        token, lexical, confidence = header[i]
        label = token.strip(' ,;:()').rstrip('.')
        if lexical and not token.startswith('-') and norm(token) and label not in LABELS:
            break
        if label in CLASSES and not lexical:
            tags.extend('Burushaski-class-' + c for c in CLASSES[label])
        if not lexical and label.lower() in POS:
            tags.append(POS[label.lower()])
        if label in {'hz', 'ng', 'hz.ng', 'NH'}:
            dialects.extend({'hz': ['Hunza'], 'ng': ['Nager'], 'hz.ng': ['Hunza', 'Nager'], 'NH': ['NH']}[label])
        if label in {'sg', 'pl', 'D.pl'}:
            following = []
            for item in header[i + 1:]:
                if not item[1] and item[0].strip('.,;') in {'sg', 'pl', 'D.pl'}:
                    break
                following.append(item)
            # "pl. agóyo" labels the following full form; "pl." alone labels this one.
            has_following_form = any(lex and norm(tok) and tok.strip('.,;') not in LABELS
                                     for tok, lex, _ in following)
            if not has_following_form:
                tags.append('pl' if label != 'sg' else 'sg')
                if label == 'D.pl':
                    tags.append('double-plural')
    if any(t.startswith('Burushaski-class-') for t in tags) and not set(tags) & {'pron', 'adj', 'adv', 'verb'}:
        tags.append('noun')
    if form.endswith('-'):
        tags.append('verb')
    if not dialects:
        if variant:
            dialects = [t.split(':')[-1] for t in old_tags if t.startswith('dialect:')]
        else:
            # Keep a documented default, never harvest a dialect from examples.
            dialects = ['Hunza']
    tags.extend('dialect:' + d for d in dialects)
    return list(dict.fromkeys(tags))


def apply(entries):
    layout = load_layout()
    overrides = json.loads((HERE / 'reviewed.json').read_text(encoding='utf-8'))
    for entry in entries:
        unit = entry.unit
        header = header_tokens(unit, layout)
        entry.header = header
        grammar = ' '.join(t for t, _, _ in header)
        # The older gloss splitter removed only some labels. Remove any
        # surviving *complete suffix* of the independently font-scoped header.
        for offset in range(1, len(header)):
            tail = ' '.join(t for t, _, _ in header[offset:]).strip(' ,;:')
            if tail and entry.gloss_de.casefold().startswith(tail.casefold() + ' '):
                entry.gloss_de = entry.gloss_de[len(tail):].lstrip(' ,;:')
                entry.english_gloss = ''
                entry.review.append('grammar-separated-from-definition')
                break
        if not entry.variant_of_stable and header:
            lexical_head = []
            for token, lexical, confidence in header:
                if not lexical or confidence < .7:
                    break
                if lexical_head and (token.strip(' ,;:.') in LABELS | {'s', 'ds'}
                                     or re.match(r'^(?:s|ds)[.]', token)):
                    break
                if token.startswith('-') and lexical_head and not re.fullmatch(r'[-:·+]*[tcsśṣ]-', token):
                    break
                if token.endswith(',') or re.search(r'[!?{}<>]', token):
                    break
                lexical_head.append(token)
            head = ' '.join(lexical_head)
            if (len(lexical_head) > 1 and norm(lexical_head[0]) == norm(entry.form)
                    and len(head) < 75):
                entry.form = head
                entry.review.append('font-supported-complete-headword')
                tail = ' '.join(lexical_head[1:])
                if entry.gloss_de.startswith(tail):
                    entry.gloss_de = entry.gloss_de[len(tail):].lstrip(' ,;:')
                    entry.english_gloss = ''
                    entry.review.append('headword-separated-from-definition')
            elif (lexical_head and lexical_head[0].startswith('-')
                  and norm(lexical_head[0]) == norm(entry.form)):
                entry.form = lexical_head[0]
        # Keep the entire printed first part too: imperfect font alignment must
        # never silently discard a class label or suffix-only paradigm.
        entry.notes = 'Source grammar/context (OCR): ' + '\n'.join(l['text'] for l in unit.lines)
        entry.tags = scoped_tags(header, entry.form, entry.tags, variant=bool(entry.variant_of_stable))
        entry.review.append('grammar-scoped-to-header' if header else 'grammar-scope-unresolved')
        if grammar:
            entry.notes += '\nParsed header (OCR): ' + grammar
        fix = overrides.get(unit.stable_key)
        if not fix:
            if 'g' in entry.form:
                entry.review.append('legacy-ocr-g-versus-g-dot-unreviewed')
            continue
        if entry.variant_of_stable and fix.get('replace_variants'):
            entry.review.extend(['suspicious-form', 'superseded-by-image-reviewed-form'])
            continue
        if fix.get('exclude'):
            entry.review.extend(['suspicious-form', fix['reason']])
            continue
        if not entry.variant_of_stable:
            entry.form = fix.get('form', entry.form)
            if fix.get('accept'):
                entry.review = [r for r in entry.review if r != 'suspicious-form']
            if 'tags' in fix:
                entry.tags = list(dict.fromkeys(fix['tags'].split() + ['uncertain']))
        if 'german' in fix:
            entry.gloss_de = fix['german']
            entry.english_gloss = fix['english']
            entry.review = [r for r in entry.review if r not in {'missing-gloss', 'missing-translation', 'stale-translation'}]
        if 'grammar' in fix:
            entry.notes = 'Source grammar (image reviewed): ' + fix['grammar'] + '\n' + entry.notes
        entry.review.append('source-image-reviewed-20260914')
    return overrides


def apply_translations(entries):
    path = HERE / 'translations.json'
    if not path.exists():
        return
    translated = json.loads(path.read_text(encoding='utf-8'))
    for entry in entries:
        if entry.english_gloss:
            continue
        digest = hashlib.sha256(entry.gloss_de.encode()).hexdigest()
        record = translated.get(digest)
        if record:
            if record['german'] != entry.gloss_de:
                raise ValueError('Translation source mismatch')
            entry.english_gloss = record['english']
            entry.review = [r for r in entry.review if r not in {'missing-translation', 'stale-translation'}]
            entry.review.append('machine-translated-unreviewed-20260914')


def preserve_variant_identities(entries):
    with gzip.open(HERE / 'installed-before.csv.gz', 'rt', encoding='utf-8') as stream:
        before = {r[10]: r for r in csv.reader(stream)}
    parents = {e.unit.stable_key: e.installed_key for e in entries if not e.variant_of_stable}
    used = {e.installed_key for e in entries if not e.variant_of_stable}
    for entry in entries:
        if not entry.variant_of_stable:
            continue
        old = before.get(entry.installed_key)
        if old and (old[0], old[2]) == (entry.language, entry.form) and entry.installed_key not in used:
            used.add(entry.installed_key)
            continue
        parent = parents[entry.variant_of_stable]
        candidates = [r for r in before.values() if r[11] == parent
                      and (r[0], r[2]) == (entry.language, entry.form) and r[10] not in used]
        if len(candidates) == 1:
            entry.installed_key = candidates[0][10]
        else:
            suffix = hashlib.sha256((entry.language + '|' + entry.form).encode()).hexdigest()[:12]
            entry.installed_key = parent + ':variant:' + suffix
        used.add(entry.installed_key)


def separate_misaligned_gold(entries):
    fixes = json.loads((HERE / 'gold-alignments.json').read_text())
    for entry in entries:
        fix = fixes.get(entry.installed_key)
        if fix and entry.unit.stable_key != fix['stable']:
            entry.installed_key = entry.unit.stable_key + ':source-article'
            entry.review.append('separated-from-misaligned-historical-gold')


def source_aliases(entries, rows, gold, old_units):
    """Redirect retired parse fragments to the article owning their source line.

    Coordinate containment, never nearest word or phonetic similarity, decides
    the target. Unavailable targets remain explicit pending identity review.
    """
    with gzip.open(HERE / 'installed-before.csv.gz', 'rt', encoding='utf-8') as stream:
        before = {r[10]: r for r in csv.reader(stream)}
    installed = {r[10] for r in rows + gold}
    old_positions = {u.stable_key: (u.pdf_page, u.left, u.top) for u in old_units}
    with gzip.open(HERE.parent / '20260828-berger-audit.csv.gz', 'rt', encoding='utf-8') as stream:
        old_audit = {a['Installed_Key']: a for a in csv.DictReader(stream)}
    # Old audits have no Left column; recover it through the frozen layout's OCR
    # first line, which distinguishes equal baselines in different columns.
    native = load_layout()
    owners = {}
    for e in entries:
        if e.variant_of_stable or e.installed_key not in installed:
            continue
        for line in e.unit.lines:
            owners[(line['pdf_page'], line['left'], line['top'])] = e.installed_key
    position_index = defaultdict(list)
    for position, line in native.items():
        position_index[(position[0], position[2])].append((position, line))
    aliases, unresolved = [], []
    for key in sorted(before.keys() - installed):
        a = old_audit.get(key) or old_audit.get(key.split(':cdial:', 1)[0])
        candidates = set()
        if key + ':legacy-graph' in installed:
            candidates.add(key + ':legacy-graph')
        elif a and a['Stable_Key'] in old_positions:
            position = old_positions[a['Stable_Key']]
            if position in owners:
                candidates.add(owners[position])
        elif a and a['PDF_Page'] and a['Top']:
            for position, line in position_index[(int(a['PDF_Page']), int(a['Top']))]:
                if norm(line['ocr']) == norm(a['Raw_OCR'].split('\n')[0]) and position in owners:
                    candidates.add(owners[position])
        if len(candidates) == 1:
            target = candidates.pop()
            reason = ('2026-09-14: preserved the original catalog evidence in a separate compatibility record.'
                      if target == key + ':legacy-graph' else
                      '2026-09-14 image/font/column repair: surviving article contains the original source line; retired parsing fragment or variant.')
            aliases.append({'Retired_Source_Key': key, 'Target_Source_Key': target, 'Reason': reason})
        else:
            unresolved.append(key)
    return aliases, unresolved


def add_reviewed_forms(rows, gold, overrides):
    by_key = {r[10]: r for r in rows + gold}
    for fix in overrides.values():
        parent = by_key.get(fix.get('key'))
        if not parent:
            continue
        for f in fix.get('forms', []):
            if any(r[2] == f['form'] and r[10] != parent[10] and r[3] == f.get('english', parent[3])
                   and (r[11] == parent[10] or 'english' in f) for r in rows + gold):
                continue
            row = list(parent)
            row[1], row[9], row[10], row[11], row[13] = '', '', f.get('key', parent[10] + ':' + f['suffix']), parent[10], ''
            row[2] = f['form']
            row[14] = f['tags'] + (' uncertain' if f.get('separate_entry') or f.get('derived') else ' alternate uncertain')
            if f.get('derived'):
                row[13], row[11] = parent[10], ''
            if 'english' in f:
                row[3] = f['english']
                # A separately printed index entry is not a variant of its neighbor.
                if f.get('separate_entry'):
                    row[11] = ''
            rows.append(row)


def add_paradigms(entries, rows, gold, overrides):
    by_key = {r[10]: r for r in rows + gold}
    seen = {(r[11], r[2]) for r in rows + gold}
    for entry in entries:
        if entry.variant_of_stable or entry.unit.stable_key in overrides:
            continue
        parent = by_key.get(entry.installed_key)
        if not parent:
            continue
        header = entry.header
        active_number = ''
        for i, (token, lexical, confidence) in enumerate(header):
            label = token.strip(' ,;:()').rstrip('.')
            if not lexical and label in {'pl', 'sg', 'D.pl'}:
                active_number = label
                continue
            if not lexical or confidence < .7 or not active_number:
                continue
            form = token.strip(' ,;:')
            if (not form or form.startswith('-') or re.search(r'[\d?!{}<>]', form)
                    or norm(form) == norm(entry.form) or label in LABELS
                    or (entry.installed_key, form) in seen):
                continue
            row = list(parent)
            row[1], row[9], row[13] = '', '', ''
            row[2] = form
            suffix = hashlib.sha256((form + '|' + active_number).encode()).hexdigest()[:10]
            row[10] = parent[10] + ':paradigm:' + suffix
            row[11] = parent[10]
            tags = scoped_tags(header[i:], form, entry.tags)
            tags = [t for t in tags if t not in {'sg', 'pl', 'double-plural'}]
            tags += ['pl' if active_number != 'sg' else 'sg', 'alternate', 'uncertain']
            if active_number == 'D.pl':
                tags.append('double-plural')
            row[14] = ' '.join(dict.fromkeys(tags))
            row[6] += '\nFull printed paradigm form (OCR, unreviewed): ' + active_number + '. ' + form
            rows.append(row)
            seen.add((parent[10], form))


def resolve_relations(rows, gold):
    """Require emitted endpoints; keep excluded-target evidence in the notes."""
    available = {r[10] for r in rows + gold}
    audit = []
    for row in rows + gold:
        for column, kind in [(11, 'variant'), (12, 'borrowing'), (13, 'derivation')]:
            good = []
            for target in filter(None, row[column].split('|')):
                if target in available:
                    good.append(target)
                else:
                    audit.append({'child': row[10], 'relation': kind, 'target': target,
                                  'status': 'unresolved-excluded-target'})
                    row[6] += f'\nUnresolved source {kind} relation: {target} (target excluded; source reading requires review).'
            row[column] = '|'.join(good)
    return audit


def crossreferences(rows, gold, entries):
    """Retain literal targets; match exact complete forms only, never stems.

    Homonyms, relative references and damaged targets remain unlinked. Glosses
    propagate only along a unique acyclic chain to an independently defined row.
    """
    all_rows = rows + gold
    by_form = defaultdict(list)
    for r in all_rows:
        if 'graph-evidence' not in r[14].split():
            by_form[unicodedata.normalize('NFC', r[2])].append(r)
    candidates = {}
    audit = []
    entry_by_key = {e.installed_key: e for e in entries}
    installed_keys = {r[10] for r in all_rows}
    excluded_forms = {e.form for e in entries if e.installed_key not in installed_keys}
    for row in all_rows:
        if 'graph-evidence' in row[14].split():
            continue
        relative = re.match(r'^(?:same as preceding entry|see (?:the )?following entry)\b', row[3], re.I)
        if relative:
            row[6] += '\nSource relative cross-reference (unresolved): ' + row[3] + '.'
            row[3] = ''
            audit.append({'key': row[10], 'target_text': relative[0], 'candidate_keys': [],
                          'resolved_key': '', 'status': 'unresolved-relative'})
            continue
        match = re.fullmatch(r'(?:see\s+|s\.?\s*|ds\.?\s*|sS\.?\s*|cf\.?\s+|=\s*)(.+?)\s*[.;]?', row[3], re.I)
        # A literal "s" needs punctuation/space to distinguish English words.
        if match and not re.match(r'^(?:see\s|s[.\s]|ds[.\s]|sS[.\s]|cf[.\s]|=)', row[3], re.I):
            match = None
        if not match:
            continue
        entry = entry_by_key.get(row[10])
        source = re.fullmatch(r'(?:s\.|ds\.|=)\s*(.+?)\s*[.;]?', entry.gloss_de, re.I) if entry else None
        target = (source[1] if source else match[1]).strip().rstrip('.')
        found = [r for r in by_form.get(target, []) if r[10] != row[10] and r[0] == row[0]]
        if target in excluded_forms:
            found = []
        candidates[row[10]] = (row, target, found)
    def definition(key, seen):
        if key in seen:
            return None
        row, target, found = candidates[key]
        if len(found) != 1:
            return None
        other = found[0]
        if other[10] in candidates:
            return definition(other[10], seen | {key})
        return other if other[3] else None
    for key, (row, target, found) in candidates.items():
        resolved = definition(key, set())
        row[6] += '\nSource cross-reference: ' + target + '.'
        if resolved:
            row[3] = resolved[3]
            row[6] += '\nDefinition supplied by source entry: ' + resolved[10] + '.'
            if not row[11]:
                row[11] = resolved[10]
        else:
            row[3] = ''
            row[6] += '\nCross-reference unresolved: ' + ('ambiguous target' if len(found) > 1 else 'no unique defined target') + '.'
        audit.append({'key': key, 'target_text': target, 'candidate_keys': [r[10] for r in found],
                      'resolved_key': resolved[10] if resolved else '',
                      'status': 'resolved-exact' if resolved else 'unresolved'})
    return audit


def repair_gold(gold):
    for row in gold:
        if row[10] in {'berger:gold:cdial11406:balan-man', 'berger:gold:cdial11406:balanees-man'}:
            row[3] = 'writhe, roll about (especially horses and cows); dangle; (breasts) hang loosely'
            row[6] = 'Source grammar (image reviewed): balán man-́ und balaneéś man-́; printed p. 33, PDF p. 19. The two forms share the printed definition.'
            row[9] = 'ys. phalán-, sh. balán, phalán; zu T 11406? (comparison, not an asserted etymology)'
            row[1] = ''
            row[14] = 'verb dialect:Hunza uncertain'


def finish_rows(rows, gold):
    for row in rows + gold:
        row[:] = [unicodedata.normalize('NFC', value) for value in row]
        if row[0] == 'Werch':
            row[0] = 'Bur'
        # These canonical source-qualified tags register through dialects.py.
        row[14] = ' '.join(t.replace('dialect:Hunza', 'dialect:Bur:Berger-HZ:Hunza')
                          .replace('dialect:Nager', 'dialect:Bur:Berger-NG:Nager')
                          .replace('dialect:Yasin', 'dialect:Bur:Berger-YS:Yasin')
                          .replace('dialect:NH', 'dialect:Bur:Berger-NH:NH') for t in row[14].split())


def complete_audit(audit, rows, gold, preserved):
    by_key = {r[10]: r for r in rows + gold}
    emitted = defaultdict(list)
    for r in rows + gold:
        emitted[r[10].split(':cdial:', 1)[0]].append(r[10])
    covered = set()
    for a in audit:
        key = a['Installed_Key']
        row = by_key.get(key)
        if row:
            a.update(Final_Form=row[2], English_Gloss=row[3], Tags=row[14], Notes=row[6])
            a['Status'] = 'gold' if row in gold else 'installed'
            a['Emitted_Keys'] = '|'.join(emitted[key])
            covered.update(a['Emitted_Keys'].split('|'))
        else:
            a['Status'] = 'excluded'
            a['Emitted_Keys'] = ''
        a['Snapshot_Date'] = '2026-09-14'
    preservation = {r[10] for r in preserved}
    for key in sorted(by_key.keys() - covered):
        row = by_key[key]
        a = {k: '' for k in audit[0]}
        a.update(Snapshot_Date='2026-09-14', Stable_Key=key, Installed_Key=key,
                 Final_Form=row[2], English_Gloss=row[3], Tags=row[14], Notes=row[6],
                 Status='installed-preserved' if key in preservation else 'gold' if row in gold else 'installed-paradigm' if ':paradigm:' in key else 'installed-reviewed-form',
                 Review='legacy-ocr-unreviewed' if key in preservation else 'legacy-gold-source-alignment-review' if row in gold else 'paradigm-ocr-unreviewed' if ':paradigm:' in key else 'source-image-reviewed-20260914',
                 Raw_OCR=row[6], Emitted_Keys=key, Source=row[7])
        audit.append(a)
    for a in audit:
        emitted_rows = [by_key[k] for k in a['Emitted_Keys'].split('|') if k in by_key]
        for field, column in [('Final_Turner_IDs', 1), ('Final_Source', 7),
                              ('Final_Variant_Of', 11), ('Final_Derivation_Parents', 13)]:
            a[field] = '|'.join(dict.fromkeys(r[column] for r in emitted_rows if r[column]))
        a['Record_SHA256'] = hashlib.sha256(json.dumps({k: v for k, v in a.items() if k != 'Record_SHA256'},
                                                     sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return audit


def summary(units, entries, rows, gold, audit, relations, references):
    return {'snapshot_date': '2026-09-14', 'raw_units': len(units), 'parsed_entries': len(entries),
            'auto_rows': len(rows), 'gold_rows': len(gold), 'audit_rows': len(audit),
            'status': dict(Counter(a['Status'] for a in audit)),
            'class_tagged_rows': sum('Burushaski-class-' in r[14] for r in rows + gold),
            'grammar_notes': sum('Source grammar' in r[6] for r in rows + gold),
            'unresolved_source_relations': len(relations),
            'crossreferences': dict(Counter(r['status'] for r in references)),
            'review': dict(Counter(reason for a in audit for reason in a['Review'].split(';') if reason)),
            'full_build': 'deferred: no authorized suitable remote full-build runner',
            'browser_refresh': 'not requested'}


def write_artifacts(destination, audit, relations, references, summary):
    with gzip.open(destination / 'audit.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for row in audit:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    for name, value in [('relations.json', relations), ('crossreferences.json', references), ('summary.json', summary)]:
        (destination / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

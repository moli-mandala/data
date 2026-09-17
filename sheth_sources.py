"""Verified Sheth source abbreviations, with reference-only matching and scope."""
from __future__ import annotations
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATALOG = ROOT / 'sheth_sources.tsv'
with CATALOG.open(encoding='utf-8', newline='') as stream:
    WORKS = list(csv.DictReader(stream, delimiter='\t'))


def work_tag(work):
    prefix = 'Sheth:' if work['Scope'] == 'main' else 'Sheth:नाट:'
    return prefix + work['Abbreviation'].replace(' ', '-')


LABELS = {}
for work in WORKS:
    tag = work_tag(work)
    label = work['Title'] + (' (नाट)' if work['Scope'] == 'nat' else '')
    LABELS[tag] = label
    LABELS[tag + ':commentary'] = label + ' — टीका/भाष्य'
SOURCE_TAGS = set(LABELS)
DIGITS = str.maketrans('0123456789', '०१२३४५६७८९')
PATTERNS = [(work, re.compile('^' + re.escape(work['Abbreviation']).replace(r'\ ', r'\s*') + r'(?=$|[\s०-९0-9.,;:()—–-])'))
            for work in sorted(WORKS, key=lambda w: len(w['Abbreviation']), reverse=True)]


def _match(text, scope):
    for work, pattern in PATTERNS:
        if work['Scope'] != scope:
            continue
        match = pattern.match(text.translate(DIGITS))
        if match:
            return work, text[match.end():].strip(' .,:—–-')
    return None, text


def _modifier(locator):
    if re.search(r'(?:^|[\s(,])(?:टी|भा)(?=$|[\s.,)०-९0-9])', locator):
        return 'commentary'
    if re.search(r'(?:^|[\s(,])टि(?=$|[\s.,)०-९0-9])', locator):
        return 'variant-reading'
    return ''


def parse_reference(raw):
    """Never call on glosses, etymology, grammar or language-label fields.

    Unknown codes remain unresolved; no fuzzy matching or era inference. Numeric
    continuations may reuse the immediately preceding work within this reference.
    नाट establishes its own nested abbreviation namespace for the remaining group.
    """
    text = raw.strip()
    if text.startswith('(') and text.endswith(')'):
        text = text[1:-1]
    claims, previous = [], None
    scope = 'main'
    for part in re.split(r'[;；]', text):
        part = part.strip(' ,')
        if not part:
            continue
        if re.fullmatch(r'\(?[०-९0-9][०-९0-9\s,().—–-]*(?:(?:टी|टि|भा)[०-९0-9\s.,()]*)?', part) and previous:
            modifier = _modifier(part)
            base = previous['tag'].removesuffix(':commentary')
            claim = dict(previous, raw=part, locator=part, continuation=True,
                         modifier=modifier, tag=base + (':commentary' if modifier == 'commentary' else ''))
            claims.append(claim)
            previous = claim
            continue
        if part.startswith('नाट') and re.match(r'^नाट(?=$|[\s.—–-])', part):
            scope = 'main'
            work, rest = _match(part, scope)
            tag = work_tag(work)
            claims.append({'raw': part, 'status': 'resolved', 'tag': tag, 'work': work['Title'],
                           'locator': rest, 'modifier': '', 'scope': 'main', 'catalog_pdf_page': work['PDF_Page']})
            scope = 'nat'
            if not rest or re.match(r'^[०-९0-9]', rest):
                previous = claims[-1]
                continue
            part = rest
        work, locator = _match(part, scope)
        if work is None:
            claims.append({'raw': part, 'status': 'unresolved', 'scope': scope,
                           'reason': 'abbreviation absent from verified catalogue'})
            previous = None
            continue
        modifier = _modifier(locator)
        tag = work_tag(work) + (':commentary' if modifier == 'commentary' else '')
        claim = {'raw': part, 'status': 'resolved', 'tag': tag, 'work': work['Title'],
                 'locator': locator, 'modifier': modifier, 'scope': scope,
                 'catalog_pdf_page': work['PDF_Page']}
        claims.append(claim)
        previous = claim
    return claims


def reference_tags(references):
    claims = [dict(claim, reference=ref) for ref in references for claim in parse_reference(ref)]
    return list(dict.fromkeys(c['tag'] for c in claims if c['status'] == 'resolved')), claims


def write_frontend(path):
    path.write_text(json.dumps(LABELS, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    write_frontend(ROOT.parent / 'jambu-static/src/lib/shethSourceLabels.json')

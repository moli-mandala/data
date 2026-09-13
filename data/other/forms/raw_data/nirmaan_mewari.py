"""Extract Nirmaan's January 2018 Mewari dictionary; no OCR or data build.

Usage: python3 nirmaan_mewari.py --pdf /tmp/mewari-dictionary.pdf --output /tmp/review
       python3 nirmaan_mewari.py --pdf /tmp/mewari-dictionary.pdf --install
Requires PyMuPDF for extraction only. The source PDF is intentionally not committed.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata as ud

SOURCE = 'nirmaan2018mewari'
PDF_SHA256 = '9adf6f4aefdd0705d29c255b664ac1bf193df2f7ab85cf0bfa33af30532344bd'
STEM = '20260911-nirmaan-mewari'
ROOT = Path(__file__).resolve().parents[4]
DIALECT = 'dialect:mewari_dholpura:mewari_kapasan:Kapasan%20area'
URL = 'https://eternalmewarblog.com/documents/mewari_dictionary.pdf'
# Krishna is a legacy Devanagari font: replacements apply to this font ONLY.
KRISHNA = {'Dd': 'क्क', 'Uu': 'न्न', 'Yy': 'ल्ल', 'Pp': 'च्च', 'V~Vh': 'ट्टी',
           'fDr': 'क्ति', 'Uu¨': 'न्नो', 'p': 'च', 'Dr': 'क्त', 'Luk': 'स्ना',
           'Uuh': 'न्नी', 'Pps': 'च्चे', 'fYy': 'ल्लि', 'Yyh': 'ल्ली', 'Rr¨': 'त्तो'}
POS = {'क्रि.वि.': 'adv', 'क्रि.': 'verb', 'सं.': 'noun', 'विशे.': 'adj',
       'सर्व.': 'pron', 'पर.': 'postp', 'संयो.': 'conj', 'विसम.': 'interj',
       'पू.संखया': 'num', 'क्र.संखया': 'num ord', 'बहु.': 'pl', 'सम्मा.': 'honorific'}
VARIANT_RE = re.compile(r'(मु\.\s*रू\.|बो\.\s*रू\.|वर्त\.\s*रू\.)\s*')


def clean_native(text):
    # Verified text-layer duplications, not phonological emendations.
    text = text.replace('यय', 'य')
    text = re.sub(r'न्(?=[ािीुूेैोौ])', 'न्द', text)
    text = text.replace('छ्ो', 'छ्यो').replace('ड्ो', 'ड्डो')
    text = re.sub(r'([ािीुूेैोौंँ])\1+', r'\1', text)
    text = re.sub(r'\s+([ािीुूेैोौंँ])', r'\1', text)
    return ud.normalize('NFC', text.strip())


def span_text(span):
    if span['font'] == 'Krishna':
        token = span['text'].strip()
        if token not in KRISHNA:
            raise ValueError(f'Unmapped Krishna token: {token!r}')
        return KRISHNA[token] + (' ' if span['text'].endswith(' ') else '')
    return span['text']


def ordered_spans(spans):
    """Cluster physical baselines before sorting x: IPA is printed 2pt higher.

    Decimal rounding bins split virtually identical baselines at bin boundaries.
    Superscript homograph numbers are within 1pt of the corrected baseline.
    """
    groups = []
    for s in sorted(spans, key=lambda s: s['y']):
        if not groups or s['y'] - groups[-1][0]['y'] > 3:
            groups.append([])
        groups[-1].append(s)
    return [s for group in groups for s in sorted(group, key=lambda s: s['x'])]


def extract(pdf):
    import pymupdf
    pdf = Path(pdf)
    if not pdf.is_file():
        raise FileNotFoundError(f'Required source PDF is absent: {pdf}; acquire {URL}')
    if hashlib.sha256(pdf.read_bytes()).hexdigest() != PDF_SHA256:
        raise ValueError('Source PDF hash differs from the pinned January 2018 edition')
    doc = pymupdf.open(pdf)
    assert len(doc) == 389, 'Wrong edition/page count'
    records = []
    counts = collections.Counter()
    for pi in range(18, len(doc)):
        spans = []
        for block in doc[pi].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for s in line['spans']:
                    x, y = s['origin']
                    if s['font'] == 'DoulosSIL':
                        y += 2
                    if 52 < y < 550 and s['size'] < 20:
                        spans.append(dict(s, x=x, y=y, col=1 if x < 198 else 2, page=pi+1))
        for col in (1, 2):
            ss = ordered_spans([s for s in spans if s['col'] == col])
            anchors = [s for s in ss if s['font'] == 'AnnapurnaSIL-Bold'
                       and 11 < s['size'] < 13
                       and abs(s['x'] - (42.52 if col == 1 else 205.51)) < 2]
            for s in ss:
                eligible = [a for a in anchors if a['y'] <= s['y'] + 3]
                a = eligible[-1] if eligible else None
                if a:
                    coord = (pi+1, col, round(a['y'], 1))
                    if not records or records[-1]['coord'] != coord:
                        counts[(pi+1, col)] += 1
                        ordinal = counts[(pi+1, col)]
                        key = f'{SOURCE}:p{pi-15:03}:c{col}:e{ordinal:02}'
                        records.append(dict(key=key, coord=coord, ordinal=ordinal, spans=[]))
                if records:
                    records[-1]['spans'].append(s)
    assert len(records) == 6406, 'Headword anchors do not match the source count'
    return records


def join_spans(spans):
    chunks = []
    previous = None
    for s in spans:
        if previous and (s['page'], s['col'], round(s['y'], 0)) != (previous['page'], previous['col'], round(previous['y'], 0)):
            if abs(s['y']-previous['y']) > 3 or s['page'] != previous['page'] or s['col'] != previous['col']:
                chunks.append(' ')
        chunks.append(span_text(s))
        previous = s
    return re.sub(r'\s+', ' ', ''.join(chunks)).strip()


def parse_article(record):
    ss = record['spans']
    first_ipa = next(i for i, s in enumerate(ss) if s['font'] == 'DoulosSIL')
    ipa = ''.join(s['text'] for s in ss if s['font'] == 'DoulosSIL')
    assert re.fullmatch(r'\[[^\[\]]+\]', ipa), (record['key'], ipa)
    raw_head = join_spans(ss[:first_ipa])
    head = clean_native(raw_head)
    homograph = ''.join(s['text'] for s in ss[:first_ipa] if s['size'] < 8).strip()
    native = re.sub(r'\d+$', '', head).strip()
    body = ss[first_ipa+1:]
    # Only source typography explicitly marking numbered senses is a boundary.
    parts = [[]]
    numbers = ['']
    for s in body:
        marker = re.fullmatch(r'([1-9])\)\s*', s['text']) if s['font'] == 'AnnapurnaSIL-Bold' and s['size'] > 11 else None
        if marker:
            parts.append([])
            numbers.append(marker[1])
        else:
            parts[-1].append(s)
    prefix = parts[0] if len(parts) > 1 else []
    sense_parts = list(zip(numbers[1:], parts[1:])) if len(parts) > 1 else [('', parts[0])]
    # POS labels use regular 10pt type; examples use the same font, so stop at
    # the first 12pt definition. Variant labels are bold and cannot become POS.
    def grammar(spans):
        pieces = []
        for span in spans:
            text = span['text'].strip()
            if '‘' in text:
                break
            if span['font'] == 'TimesNewRomanPSMT' and re.search('[A-Za-z]', text):
                if text not in {'unspec.', 'var.', 'of', 'unspec. var. of'}:
                    break
            if span['font'] == 'AnnapurnaSIL' and 9 < span['size'] < 11:
                pieces.append(text)
        label = ''.join(pieces).replace(' ', '')
        hits = [(label.index(k), k) for k in POS if k in label]
        if not hits:
            return [], '', ''
        start = min(hits)[0]
        label = label[start:]
        tags, labels = [], []
        for k in sorted(POS, key=len, reverse=True):
            if label.startswith(k):
                tags.extend(POS[k].split())
                labels.append(k)
                label = label[len(k):]
        return tags, ''.join(labels), ''
    prefix_tags, prefix_pos, prefix_unknown = grammar(prefix)
    senses = []
    for number, part in sense_parts:
        tags, raw_pos, unknown = grammar(part)
        if not tags:
            tags, raw_pos, unknown = prefix_tags, prefix_pos, prefix_unknown
        # Latin 12pt spans are the English definition. Scientific names are
        # separated in the source, rather than appended to the lexical gloss.
        english = []
        scientific = []
        sci = False
        for s in part:
            if s['text'].strip() in {'unspec.', 'var.', 'of'} or 'unspec. var. of' in s['text']:
                continue
            if 'वैज्ा.' in s['text'] or 'वैज्ञा.' in s['text']:
                sci = True
            if s['font'] == 'TimesNewRomanPSMT':
                (scientific if sci else english).append(s)
        gloss = join_spans(english).strip(' ,;')
        gloss = re.sub(r'(?<=\w)-\s+(?=\w)', '-', gloss)
        source_gloss = gloss
        for label, tag in [('feminine', 'f'), ('masculine', 'm'), ('slang', 'colloquial')]:
            if f'({label})' in gloss:
                gloss = gloss.replace(f'({label})', '').strip()
                tags = tags + [tag]
        senses.append(dict(number=number, gloss=gloss, source_gloss=source_gloss, tags=tags, raw_pos=raw_pos,
                           unparsed_pos=unknown, scientific=join_spans(scientific)))
    text = clean_native(join_spans(body))
    variants = []
    # Source variant labels precede POS/definitions or form a whole alias entry.
    # Explicitly retain all targets, including ambiguous and reciprocal ones.
    for m in VARIANT_RE.finditer(text):
        tail = text[m.end():]
        tail = re.split(r'[()]|(?:मु\.|बो\.|वर्त\.)\s*रू\.|(?:सं\.|क्रि\.|विशे\.|सर्व\.|विलो\.|समा\.)|[‘\[]', tail, maxsplit=1)[0]
        tail = tail.split('unspec. var. of')[0]
        for target in re.split(r'[,;]', tail):
            target = target.strip(' .')
            if target:
                variants.append(dict(kind=m[1].replace(' ', ''), target=target))
    for m in re.finditer(r'unspec\.\s*var\.(?:\s+of)?\s+([^‘)]+)', text):
        variants.append(dict(kind='unspecified-variant', target=m[1].strip()))
    compounds = re.findall(r'\(समा\.\s*(.*?)\)', text)
    if '(मुहा.' in text:
        for sense in senses:
            sense['tags'].append('multiword-expression')
    problems = []
    if re.search(r'[A-Za-z�]', native) or re.search(r'्[ेैोौंँािीुू]', native):
        problems.append('transcription:native-text-layer')
    if any(s['unparsed_pos'] for s in senses):
        problems.append('grammar:unparsed-label')
    return dict(key=record['key'], status='installed-raw', raw_extracted_text=''.join(s['text'] for s in ss), pdf_page=record['coord'][0], printed_page=record['coord'][0]-16,
                column=record['coord'][1], baseline=record['coord'][2], ordinal=record['ordinal'],
                raw_head=raw_head, head=head, native=native, homograph=homograph,
                ipa=ud.normalize('NFC', ipa[1:-1]), raw_text=join_spans(ss), senses=senses,
                variants=variants, compounds=compounds, problems=problems)


def match_key(text):
    return re.sub(r'\s*-\s*', '-', clean_native(text)).replace('\u200d', '').replace('\u200c', '')


def emit(articles):
    index = collections.defaultdict(list)
    by_key = {a['key']: a for a in articles}
    for a in articles:
        a['problems'] = [p for p in a['problems'] if not p.startswith(('variant:', 'compound:'))]
        a['additional_variant_keys'] = []
        index[match_key(a['head'])].append(a)
        if a['homograph']:
            index[match_key(a['native'])].append(a)
        a['row_keys'] = [a['key'] + (f':s{s["number"]}' if s['number'] else '') for s in a['senses']]
        a['variant_parent'] = ''
        a['edge_evidence'] = ''
    def candidates(target):
        target = match_key(target)
        exact = index.get(target, [])
        if exact:
            return exact
        # Some references append a Hindi gloss after a head/homograph number.
        prefixes = [k for k in index if target.startswith(k+' ')]
        if prefixes:
            return index[max(prefixes, key=len)]
        return []
    adjacency = collections.defaultdict(dict)
    for a in articles:
        for v in a['variants']:
            found = [b for b in candidates(v['target']) if b['key'] != a['key']]
            v['candidates'] = [b['key'] for b in found]
            v['resolved'] = found[0]['key'] if len(found) == 1 else ''
            v['status'] = 'ambiguous-or-unmatched'
            if len(found) == 1:
                target = found[0]
                if len(a['senses']) == len(target['senses']) == 1:
                    adjacency[a['key']][target['key']] = a['key']
                    adjacency[target['key']][a['key']] = a['key']
                    v['status'] = 'represented-in-variant-family'
                else:
                    v['status'] = 'sense-scope-unresolved'
        a['component_keys'] = []
        a['compound_candidates'] = []
        for text in a['compounds']:
            parts = [candidates(t.strip()) for t in text.split(',')]
            a['compound_candidates'].append([[b['key'] for b in part] for part in parts])
            if len(parts) >= 2 and all(len(part)==1 and len(part[0]['senses'])==1 for part in parts):
                a['component_keys'] = [part[0]['row_keys'][0] for part in parts]
            else:
                a['problems'].append('compound:parent-or-sense-scope')
    # Free/spelling/dialect variants are equivalence claims, not directional
    # ancestry. Use a deterministic spanning tree of explicit claims, preferring
    # a defined article as root; every edge keeps its source assertion locator.
    visited = set()
    priority = lambda key: (not bool(by_key[key]['senses'][0]['gloss']), key)
    for root in sorted(adjacency, key=priority):
        if root in visited:
            continue
        queue = collections.deque([root])
        visited.add(root)
        while queue:
            parent = queue.popleft()
            for child in sorted(adjacency[parent], key=priority):
                if child in visited:
                    continue
                visited.add(child)
                queue.append(child)
                by_key[child]['variant_parent'] = by_key[parent]['row_keys'][0]
                by_key[child]['edge_evidence'] = adjacency[parent][child]
    def citation(a, sense=''):
        return (f'{SOURCE}[p. {a["printed_page"]}, col. {a["column"]}, entry {a["ordinal"]}'
                + (f', sense {sense}' if sense else '') + ']')
    rows = []
    for a in articles:
        if any(v['status'] != 'represented-in-variant-family' for v in a['variants']):
            a['problems'].append('variant:target-or-sense-scope')
        for key, sense in zip(a['row_keys'], a['senses']):
            sources = [citation(a, sense['number'])]
            if a['edge_evidence'] and a['edge_evidence'] != a['key']:
                sources.append(citation(by_key[a['edge_evidence']]))
            tags = list(dict.fromkeys(sense['tags'] + [DIALECT] + (['uncertain'] if a['problems'] else [])))
            for vi, pronunciation in enumerate(a['ipa'].split('/')):
                child_key = key if vi == 0 else key + f':v{vi+1}'
                rows.append(['mewari_dholpura', '', pronunciation, sense['gloss'], a['native'], pronunciation, '', ';'.join(sources),
                             '', '', child_key, a['variant_parent'] if vi == 0 else key, '', '|'.join(a['component_keys']), ' '.join(tags)])
                if vi:
                    a.setdefault('additional_variant_keys', []).append(child_key)
    return rows


def write_outputs(pdf, output, install=False):
    articles = [parse_article(r) for r in extract(pdf)]
    rows = emit(articles)
    forms_dir = ROOT / 'data/other/forms' if install else output
    audit_dir = forms_dir / 'raw_data' if install else output
    forms_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (forms_dir / f'{STEM}.csv').open('w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(rows)
    with (audit_dir / f'{STEM}-audit.jsonl').open('w', encoding='utf-8') as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + '\n')
    manifest = dict(source=SOURCE, url=URL, edition='Second publication, January 2018',
                    pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(), pdf_pages=389,
                    article_count=len(articles), form_count=len(rows), variant_edges=sum(bool(r[11]) for r in rows),
                    problems=dict(collections.Counter(p for a in articles for p in a['problems'])),
                    ipa_symbols=sorted(set(''.join(a['ipa'] for a in articles))),
                    license='All rights reserved, Nirmaan Society 2017; local data preparation; no redistribution licence established',
                    excluded='PDF pages 1–18 (front matter/title/blank); headers, footers, examples and scientific names excluded from Gloss; Hindi definitions retained in audit',
                    extraction='PyMuPDF positioned/font spans; no OCR; 6406 bold left-margin headword anchors',
                    deferred=['full CLDF build', 'full test suite', 'persistent ID assignment', 'generated CLDF and references', 'browser database and browser QA'])
    (audit_dir / f'{STEM}-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('article_count','form_count','variant_edges','problems')}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pdf', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('/tmp/nirmaan-mewari-review'))
    ap.add_argument('--install', action='store_true')
    ap.add_argument('--sample-seed', type=int, help='Render 20 seeded source entries instead of installing')
    args = ap.parse_args()
    if args.sample_seed is not None:
        import random
        import pymupdf
        doc = pymupdf.open(args.pdf)
        args.output.mkdir(parents=True, exist_ok=True)
        selected = random.Random(args.sample_seed).sample(extract(args.pdf), 20)
        for record in selected:
            groups = collections.defaultdict(list)
            for span in record['spans']:
                groups[(span['page'], span['col'])].append(span)
            for (page, col), spans in groups.items():
                x = 40 if col == 1 else 203
                rect = pymupdf.Rect(x, min(s['y'] for s in spans)-15, x+156, max(s['y'] for s in spans)+5)
                doc[page-1].get_pixmap(matrix=pymupdf.Matrix(2.5,2.5), clip=rect).save(str(args.output / f'{record["key"]}-pdf{page}-c{col}.png'))
        (args.output / 'selection.json').write_text(json.dumps([parse_article(r) for r in selected], ensure_ascii=False, indent=2))
    else:
        write_outputs(args.pdf, args.output, args.install)

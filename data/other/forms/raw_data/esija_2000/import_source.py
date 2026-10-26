"""ESIJa lemma layer: Rastorgueva & Edel'man, Etimologičeskij slovar' iranskix jazykov, vols. 1-3.

Usage (from the data git root):
    .venv/bin/python -I data/other/forms/raw_data/esija_2000/import_source.py --extract   # PDF -> OCR cache
    .venv/bin/python -I data/other/forms/raw_data/esija_2000/import_source.py             # dry run
    .venv/bin/python -I data/other/forms/raw_data/esija_2000/import_source.py --install   # write outputs

The scan (copyrighted, not redistributed) is read from PDF_PATH; --extract fails clearly if it is
absent. The extraction cache (text-layer lines, headword crops, Tesseract hOCR) lives outside the
repository in ESIJA_CACHE (default ../tmp/iranic/esija); the checked-in audit keeps every raw
reading, so the installed rows can be reviewed without the scan.

Only the lemma layer is installed (PIr etymon rows in data/other/params/20261025-esija.csv).
Reflex citations stay a deferred stage: the ABBYY Russian-mode layer garbles Latin-script forms
(see proposals.md), so every article span is recorded in the audit with its page range only.
"""
import argparse, csv, hashlib, json, os, random, re, subprocess, sys, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import esija_ocr as E  # noqa: E402

DATA_ROOT = HERE.parents[4]  # .../jambu-all/data (git root)
PDF_PATH = Path(os.environ.get('ESIJA_PDF', '/Users/aryamanarora/Documents/Linguistics/Indo-European/'
                'Iranic/Etimologicheskiy slovar irananskih yazikov. Tom 1-3 (А-H).pdf'))
PDF_SHA256 = 'c5b37d2017df63795bc404699426d8096bd4ef306452642ae43bb603a948dc10'
CACHE = Path(os.environ.get('ESIJA_CACHE', str(HERE.parents[5] / 'tmp' / 'iranic' / 'esija')))
SOURCE = 'rastorgueva2000esija'
STEM = '20261025-esija'
PARAMS = DATA_ROOT / 'data/other/params' / f'{STEM}.csv'
AUDIT = HERE / f'{STEM}-audit.csv'
SAMPLE = HERE / f'{STEM}-sample.csv'
MANIFEST = HERE / f'{STEM}-manifest.json'
IDS = HERE / 'ids.csv'
GLOSSES = HERE / 'glosses_en.csv'
MANUAL = HERE / 'manual-lemmas.csv'  # manual readings of headword crops for OCR-disputed/failed lemmas
SEED = 20261029
SEED_MANUAL = 20261030

# PDF page -> (volume, printed page); verified against OCR of the printed folios (manifest).
VOLUMES = {1: (1, 327, +1), 2: (328, 830, -326), 3: (831, 1324, -829)}
DICT_PDF = {1: (62, 326), 2: (366, 828), 3: (873, 1322)}  # dictionary body (letters a-ā, b-d, f-h)
STAR = re.compile(r"^[*•#^\"'«»+ф†t%]")
SIG = re.compile(r"^[\dЮЗSз8'\s\-–—«»*.,и]+$")
STOP = re.compile(r"^(['‘’\"„“(—–\-]|см\.|[А-Яа-яЁё])")


def printed(pdf_page):
    for vol, (a, b, off) in VOLUMES.items():
        if a <= pdf_page <= b:
            return vol, pdf_page + off
    raise ValueError(pdf_page)


# ---------------------------------------------------------------- extraction (needs the scan)
def extract_lines(pdf):
    import pdfplumber
    res = {}
    with pdfplumber.open(pdf) as doc:
        for i, pg in enumerate(doc.pages, 1):
            s = 1 if pg.width < 300 else 2
            lines = []
            for w in pg.extract_words(extra_attrs=['fontname', 'size'], use_text_flow=True):
                rec = [w['text'], w['fontname'].split('-')[-1][:6], round(w['size'], 1), round(w['x0'], 1)]
                if lines and abs(w['top'] - lines[-1]['top']) < 1.5 * s and w['x0'] >= lines[-1]['x1'] - 1:
                    lines[-1]['words'].append(rec); lines[-1]['x1'] = w['x1']
                else:
                    lines.append({'top': round(w['top'], 1), 'x0': round(w['x0'], 1), 'x1': w['x1'], 'words': [rec]})
            for ln in lines:
                ln['x1'] = round(ln['x1'], 1)
            res[str(i)] = {'w': float(pg.width), 'h': float(pg.height), 'lines': lines}
            pg.close()
    return res


def _fit(pts):
    n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - mx) ** 2 for p in pts) or 1e-9
    b = sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx
    return my - b * mx, b


def _margin(lines, m0, s):
    a0, b0, win = m0, 0.0, 3 * s
    for _ in range(4):
        pts = [(ln['top'], ln['x0']) for ln in lines if abs(ln['x0'] - (a0 + b0 * ln['top'])) < win]
        if len(pts) < 3:
            break
        a0, b0 = _fit(pts); win = 1.6 * s
    return a0, b0


def detect_headwords(L):
    """Articles open with a hanging (outdented) bold lemma line; derivatives are indented."""
    from collections import Counter
    out = []
    for vol, (a, b) in DICT_PDF.items():
        for i in range(a, b + 1):
            pg = L[str(i)]; s = 1 if pg['w'] < 300 else 2
            lines = [ln for ln in pg['lines'] if 20 * s < ln['top'] < pg['h'] - 12 * s]
            if len(lines) < 5:
                continue
            m0 = Counter(round(ln['x0'] / s) for ln in lines).most_common(1)[0][0] * s
            a0, b0 = _margin(lines, m0, s)
            low = [ln for ln in lines if ln['x0'] - (a0 + b0 * ln['top']) < -2.8 * s]
            if len(low) >= 4 and sum(bool(STAR.match(ln['words'][0][0])) for ln in low) < 0.5 * len(low):
                a0, b0 = _margin(lines, a0 - 5.5 * s, s)  # the mode was the paragraph indent
            for ln in lines:
                if ln['x0'] - (a0 + b0 * ln['top']) < -2.8 * s:
                    txt = ' '.join(w[0] for w in ln['words'])
                    if re.fullmatch(r"[\d\s\-–«»*'.,з]+\d", txt):
                        continue
                    out.append((i, ln['top']))
    return out


def head_words(words):
    out = []
    for k, w in enumerate(words):
        if k > 0 and STOP.match(w[0]) and not w[0].startswith('*'):
            break
        out.append(w)
    return out or words[:1]


def segment(L, starts):
    starts = set(starts); arts = []; seen = set()
    for vol, (a, b) in DICT_PDF.items():
        cur = None
        for i in range(a, b + 1):
            pg = L[str(i)]; s = 1 if pg['w'] < 300 else 2
            for ln in pg['lines']:
                if ln['top'] <= 20 * s:
                    continue  # running head
                txt = ' '.join(w[0] for w in ln['words'])
                if ln['top'] >= pg['h'] - 30 * s and SIG.match(txt) and re.search(r'\d', txt):
                    continue  # folio / printer's signature
                if (i, ln['top']) in starts and (i, ln['top']) not in seen:
                    seen.add((i, ln['top']))  # a headword line split into two layout lines opens one article
                    cur = {'vol': vol, 'pdf': i, 'top': ln['top'], 'lines': []}; arts.append(cur)
                if cur is not None:
                    cur['lines'].append([i, txt])
    for art in arts:
        t = ''
        for _, x in art['lines']:
            if re.search(r'[а-яё]-$', t) and re.match(r'^[а-яё]', x):
                t = t[:-1] + x
            else:
                t = (t + ' ' + x) if t else x
        art['text'] = t
        art['pages'] = sorted({p for p, _ in art['lines']})
        del art['lines']
    return arts


def extract():
    if not PDF_PATH.exists():
        sys.exit(f'ESIJa scan not found at {PDF_PATH}; set ESIJA_PDF to the PDF path.')
    sha = hashlib.sha256(PDF_PATH.read_bytes()).hexdigest()
    if sha != PDF_SHA256:
        sys.exit(f'unexpected PDF sha256 {sha}')
    import pypdfium2 as pdfium
    from PIL import Image
    CACHE.mkdir(parents=True, exist_ok=True); (CACHE / 'hwt').mkdir(exist_ok=True)
    lp = CACHE / 'lines.json'
    if not lp.exists():
        json.dump(extract_lines(str(PDF_PATH)), open(lp, 'w'), ensure_ascii=False)
    L = json.load(open(lp))
    hp = CACHE / 'hocr.json'
    hocr = json.load(open(hp)) if hp.exists() else {}
    env = dict(os.environ, OMP_THREAD_LIMIT='1')
    doc = pdfium.PdfDocument(str(PDF_PATH))
    bypage = {}
    for p, top in detect_headwords(L):
        bypage.setdefault(p, []).append(top)
    for p in sorted(bypage):
        todo = [t for t in bypage[p] if f'{p}_{t}' not in hocr]
        if not todo:
            continue
        page = doc[p - 1]; w, _ = page.get_size(); sc = 7.4 if w < 300 else 3.7  # ~native scan dpi
        img = page.render(scale=sc, grayscale=True).to_pil(); page.close()
        for top in todo:
            ln = next(x for x in L[str(p)]['lines'] if x['top'] == top)
            hw = head_words(ln['words'])
            nxt = ln['words'][len(hw)][3] if len(hw) < len(ln['words']) else ln['x1'] + 1
            size = max(x[2] for x in hw)
            box = (int((ln['x0'] - 1.5 * size / 5) * sc), int((top - 0.6 * size) * sc),
                   int((nxt - 0.3 * size / 5) * sc), int((top + 1.45 * size) * sc))
            im = img.crop(box); im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
            im = E.band(im)
            f = CACHE / 'hwt' / f'{p}_{top}.png'; im.save(f)
            o = subprocess.run(['tesseract', str(f), '-', '-l', 'script/Latin', '--psm', '7', '-c',
                                'hocr_char_boxes=1', 'hocr'], capture_output=True, env=env).stdout.decode('utf-8', 'replace')
            import html
            chars = [(html.unescape(m.group(5)), [int(m.group(i)) for i in (1, 2, 3, 4)]) for m in
                     re.finditer(r"x_bboxes (\d+) (\d+) (\d+) (\d+); x_conf [\d.]+'>([^<]*)</span>", o)]
            hocr[f'{p}_{top}'] = {'chars': chars, 'text': ''.join(c for c, _ in chars), 'size': im.size}
        json.dump(hocr, open(hp, 'w'), ensure_ascii=False)
    print('extracted', len(hocr), 'headword crops into', CACHE)


# ---------------------------------------------------------------- build
def slug(lemma):
    s = lemma.replace('ϑ', 'th').replace('ᵛ', 'v')
    s = ''.join(ch for ch in unicodedata.normalize('NFD', s) if not unicodedata.combining(ch))
    return re.sub(r'[^a-z0-9]', '', s.lower()) or 'x'


ALPHABET = set('abčdfghiīǰklmnprsśštϑuūxzźžāᵛáíú') | {'̥', '̯', '(', ')', '-'}
VOWELS = set('aāiīuū̥')
# manual readings may also carry printed breves (ă ĭ ŭ, ā̆) and word-internal '/' alternation notation
MANUAL_ALPHABET = ALPHABET | set('ăĭŭ/') | {'\u0306'}


def first_lemma(lem):
    f = re.split(r'\s*[:,<>~]\s*|\s*/\s*(?=\*)|(?<=-)\s*/\s*|\s+/\s+|\s+или\s+|\s*\d\)', lem.strip(), 1)[0]
    return f.strip(' ,;')


def clean_form(lem, abbyy):
    """Mechanical repairs of the first lemma; returns (lemma, flags, ok)."""
    flags = []
    if '(?)' in lem:
        lem = lem.replace('(?)', ''); flags.append('source-query')
    if re.search(r'[\d.}{|]', lem):
        lem = re.sub(r'[\d.}{|]', '', lem); flags.append('stray-mark-removed')
    if re.match(r"^\W*-", abbyy) and not lem.startswith('-'):
        lem = '-' + lem
    lem = re.sub('-{2,}', '-', lem)
    # ABBYY sets hyphens reliably; Tesseract drops some: take final-hyphen presence from ABBYY
    abt = [t for t in abbyy.split() if re.search(r'[A-Za-zА-Яа-я$§£{]', t) and not re.fullmatch(r'[lIn]', t)]
    if abt:
        t0 = abt[0].rstrip(',;')
        if t0.endswith('-') and not lem.endswith('-'):
            lem += '-'; flags.append('hyphen-from-abbyy')
        elif not t0.endswith('-') and lem.endswith('-') and not t0.endswith(')'):
            lem = lem.rstrip('-'); flags.append('hyphen-from-abbyy')
    nfc = unicodedata.normalize('NFC', lem)
    nfd = unicodedata.normalize('NFD', lem)
    bad = [ch for ch in nfc if ch not in ALPHABET
           and ch not in '̄̌́']
    ok = not bad and any(ch in VOWELS or ch == 'r' for ch in nfd) and len(re.sub(r'[^a-zāīūčǰšśźžϑ]', '', lem)) >= 1
    if bad:
        flags.append('non-alphabet:' + ''.join(bad))
    return unicodedata.normalize('NFC', lem.strip(' -') and lem), flags, ok


def load_csv(path, key):
    if not path.exists():
        return {}
    with open(path, encoding='utf-8', newline='') as f:
        return {r[key]: r for r in csv.DictReader(f)}


def build():
    L = json.load(open(CACHE / 'lines.json'))
    hocr = json.load(open(CACHE / 'hocr.json'))
    starts = detect_headwords(L)
    arts = segment(L, starts)
    gl = {r['Russian']: r['English'] for r in load_csv(GLOSSES, 'Russian').values()}
    ids = {k: r['ID'] for k, r in load_csv(IDS, 'Entry_Key').items()}
    used = set(ids.values())
    manual = load_csv(MANUAL, 'Entry_Key')
    ordinal = {}
    audit, params = [], []
    for art in arts:
        p, top = art['pdf'], art['top']
        vol, page = printed(p)
        ordinal[(vol, page)] = ordinal.get((vol, page), 0) + 1
        key = f'{SOURCE}:v{vol}:p{page}:h{ordinal[(vol, page)]}'
        ln = next(x for x in L[str(p)]['lines'] if x['top'] == top)
        words = head_words(ln['words'])
        ab = ' '.join(w[0] for w in words)
        h = hocr.get(f'{p}_{top}')
        g = E.analyse(str(CACHE / 'hwt' / f'{p}_{top}.png'), h['chars']) if h and h['chars'] else None
        if g:
            rc = E.reconcile(g, words)
        else:
            rc = dict(lemma='', first='', homonym=E.abbyy_homonym(words), flags=['no-geometry'], abbyy=ab, status='ocr_failed')
        gp = E.parse(art['text'], ab)
        lem = first_lemma(rc['lemma']) if rc['lemma'] else ''
        lem, cflags, form_ok = clean_form(lem, rc.get('abbyy_clean', ab)) if lem else ('', [], False)
        rc['flags'] = rc['flags'] + cflags
        if not form_ok and rc['status'].startswith('ocr_consensus'):
            rc['status'] = 'ocr_disputed'
        form = '*' + lem if lem else ''
        ru = gp.get('ru', '')
        en = gl.get(ru, '') if ru else ''
        review = ['unreviewed-ocr']
        if gp['kind'] == 'cross_reference':
            status, reason = 'cross_reference', 'printed see-reference lemma; target article carries the record'
        elif rc['status'] == 'ocr_failed' or not lem:
            status, reason = 'deferred_ocr_failed', 'no usable Tesseract character geometry for the headword line'
        elif not rc['status'].startswith('ocr_consensus'):
            status, reason = 'deferred_ocr_disputed', 'Tesseract/geometry reading disagrees with the ABBYY letter skeleton'
        else:
            status, reason = 'installed', 'first printed lemma agrees across Tesseract+geometry and ABBYY skeleton'
            if rc['status'] == 'ocr_consensus_first':
                review.append('variants-unverified')
            if not en:
                review.append('gloss-blank')
            if 'source-query' in cflags:
                review.append('source-query')
        hom, lemma_source = rc['homonym'], ('ocr' if status == 'installed' else '')
        m = manual.get(key)
        if m and status.startswith('deferred'):
            lemma_source = 'manual'
            if m['Decision'] == 'lemma':
                form, lem, hom = m['Form'], m['Form'].lstrip('*'), m['Homonym']
                bad = [ch for ch in unicodedata.normalize('NFC', lem) if ch not in MANUAL_ALPHABET]
                if bad or not m['Form'].startswith('*'):
                    raise ValueError(f'manual lemma {key}: unexpected characters {bad}')
                if m.get('Gloss_RU'):
                    ru = m['Gloss_RU']; en = gl.get(ru, '')
                status, reason = 'installed', 'manual reading of the headword crop (manual-lemmas.csv)'
                review = ['manual-collation']
                if not en:
                    review.append('gloss-blank')
                if '(?)' in m['Note']:
                    review.append('source-query')
            elif m['Decision'] == 'not_headword':
                status, reason = 'layout_false_positive', m['Note']
            elif m['Decision'] == 'cross_reference':
                status, reason = 'cross_reference', m['Note']
            else:
                status, reason = 'excluded_ambiguous', m['Note']
        if status == 'installed':
            pid = ids.get(key)
            if not pid:
                base = f"esija-{vol}-{page}-{slug(lem)}" + (f"-{hom}" if hom not in ('', '?') else '')
                pid, n = base, 1
                while pid in used:
                    n += 1; pid = f'{base}-{chr(96 + n)}'
                ids[key] = pid; used.add(pid)
            src = f'{SOURCE}[vol. {vol}, p. {page}]'
            params.append([pid, 'PIr', form, en, src])
        else:
            pid = ids.get(key, '')
        audit.append({
            'Entry_Key': key, 'ID': pid, 'Status': status, 'Reason': reason, 'Volume': vol, 'Printed_Page': page,
            'PDF_Page': p, 'Line_Top': top, 'Raw_Headword_ABBYY': ab, 'Raw_Headword_Tesseract': h['text'] if h else '',
            'Fused_Headword': rc['lemma'], 'Homonym': hom, 'Form': form, 'Lemma_Source': lemma_source, 'OCR_Status': rc['status'],
            'OCR_Flags': ' '.join(rc['flags']), 'Gloss_Kind': gp['kind'], 'Gloss_RU': ru, 'Gloss_EN': en,
            'Cross_Reference_Raw': gp.get('target', ''), 'Article_Pages': ' '.join(str(printed(x)[1]) for x in art['pages']),
            'Reflex_Stage': 'deferred', 'Review': ' '.join(review) if status == 'installed' else '',
            'Article_Head_Raw': art['text'][:400], 'Source': f'{SOURCE}[vol. {vol}, p. {page}]'})
    return audit, params, ids


AUDIT_FIELDS = ['Entry_Key', 'ID', 'Status', 'Reason', 'Volume', 'Printed_Page', 'PDF_Page', 'Line_Top',
                'Raw_Headword_ABBYY', 'Raw_Headword_Tesseract', 'Fused_Headword', 'Homonym', 'Form', 'Lemma_Source', 'OCR_Status',
                'OCR_Flags', 'Gloss_Kind', 'Gloss_RU', 'Gloss_EN', 'Cross_Reference_Raw', 'Article_Pages',
                'Reflex_Stage', 'Review', 'Article_Head_Raw', 'Source']


def write(audit, params, ids, root):
    root = Path(root)
    with open(root / AUDIT.name, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, AUDIT_FIELDS); w.writeheader(); w.writerows(audit)
    with open(root / IDS.name, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['Entry_Key', 'ID'])
        for k in sorted(ids, key=lambda k: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', k)]):
            w.writerow([k, ids[k]])
    target = PARAMS if Path(root) == HERE else root / PARAMS.name
    with open(target, 'w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerows(params)
    write_sample(audit, root)
    write_manifest(audit, params, root)


SAMPLE_FIELDS = ['Round', 'Seed', 'Entry_Key', 'ID', 'Printed_Page', 'Raw_Headword_ABBYY', 'Raw_Headword_Tesseract',
                 'Homonym', 'Form', 'Gloss_RU', 'Gloss_EN', 'Verdict', 'Material_Error', 'Note']


def write_sample(audit, root):
    """Seeded raw-vs-output sample of installed rows; verdicts come from REVIEW (manual image check)."""
    review = load_csv(HERE / 'sample-review.csv', 'Entry_Key')
    rows = []
    for rnd, seed, src in (('final', SEED, 'ocr'), ('manual', SEED_MANUAL, 'manual')):
      inst = [r for r in audit if r['Status'] == 'installed' and r['Lemma_Source'] == src]
      for r in random.Random(seed).sample(inst, 20):
        v = review.get(r['Entry_Key'], {})
        rows.append({'Round': rnd, 'Seed': seed, 'Entry_Key': r['Entry_Key'], 'ID': r['ID'],
                     'Printed_Page': r['Printed_Page'], 'Raw_Headword_ABBYY': r['Raw_Headword_ABBYY'],
                     'Raw_Headword_Tesseract': r['Raw_Headword_Tesseract'], 'Homonym': r['Homonym'], 'Form': r['Form'],
                     'Gloss_RU': r['Gloss_RU'], 'Gloss_EN': r['Gloss_EN'], 'Verdict': v.get('Verdict', 'unreviewed'),
                     'Material_Error': v.get('Material_Error', ''), 'Note': v.get('Note', '')})
    with open(Path(root) / SAMPLE.name, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, SAMPLE_FIELDS); w.writeheader(); w.writerows(rows)


def write_manifest(audit, params, root):
    man = {
        'bibliography': "Rastorgueva, V. S. & Edel'man, D. I. Ètimologičeskij slovar' iranskix jazykov. "
                        "Vol. 1 (a–ā), 2000; vol. 2 (b–d), 2003; vol. 3 (f–h), 2007. Moscow: Vostočnaja literatura.",
        'source_key': SOURCE, 'pdf_name': PDF_PATH.name, 'pdf_sha256': PDF_SHA256, 'pdf_pages': 1324,
        'pdf_redistributed': False,
        'rights': 'Copyrighted scan used for research extraction; neither the scan nor page images are checked in.',
        'volumes': {
            '1': {'pdf_pages': [1, 327], 'printed_page': 'pdf + 1', 'dictionary_pdf': [62, 326], 'dictionary_printed': [63, 327], 'letters': 'a, ā'},
            '2': {'pdf_pages': [328, 830], 'printed_page': 'pdf - 326', 'dictionary_pdf': [366, 828], 'dictionary_printed': [40, 502], 'letters': 'b, č, d'},
            '3': {'pdf_pages': [831, 1324], 'printed_page': 'pdf - 829', 'dictionary_pdf': [873, 1322], 'dictionary_printed': [44, 493], 'letters': 'f, g, h'}},
        'page_mapping_verification': 'Tesseract digit OCR of the printed folio strip of every PDF page: 1236 folios read '
                                     'agree with the mapping, 28 partial reads (e.g. "4" for 44), 0 contradictions; no '
                                     'missing or duplicated leaves. Vol. 3 pages are scanned at double size (389x599 pt).',
        'extraction': {
            'text_layer': 'ABBYY FineReader Russian layer via pdfplumber words (font size used for superscripts)',
            'headword_detection': 'hanging-indent (outdented) lines vs a skew-fitted left margin; derivatives are indented',
            'headword_ocr': 'pypdfium2 render at native resolution (7.4 px/pt, vol. 3 3.7 px/pt), 2x Lanczos, '
                            'band-trimmed; tesseract 5 -l script/Latin --psm 7 hocr_char_boxes=1',
            'diacritics': 'connected-component geometry: detached flat marks over a/i/u = macron, háček vs acute by '
                          'top-left ink share (0.16), detached/under-baseline marks under i/u/r = i̯ u̯ r̥, raised '
                          'glyphs = homonym numerals / ᵛ',
            'acceptance': 'OCR lemmas installed only when the fused first lemma agrees with the ABBYY letter skeleton; '
                          'all other headwords read manually from the headword crops (manual-lemmas.csv)',
            'gloss': 'first quoted Russian gloss(es) of the headword paragraph; English by editorial translation table glosses_en.csv'},
        'outputs': {'params': str(PARAMS.relative_to(DATA_ROOT)), 'audit': str(AUDIT.relative_to(DATA_ROOT)),
                    'sample': str(SAMPLE.relative_to(DATA_ROOT)), 'ids': str(IDS.relative_to(DATA_ROOT))},
        'counts': counts(audit, params),
        'reflex_stage': 'deferred: no reflex rows installed (see proposals.md)',
    }
    with open(Path(root) / MANIFEST.name, 'w', encoding='utf-8') as f:
        json.dump(man, f, ensure_ascii=False, indent=1, sort_keys=True); f.write('\n')


def counts(audit, params):
    from collections import Counter
    c = Counter((r['Volume'], r['Status']) for r in audit)
    return {'articles': len(audit), 'params': len(params),
            'by_volume_status': {f'{v}:{s}': n for (v, s), n in sorted(c.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--install', action='store_true')
    a = ap.parse_args()
    if a.extract:
        extract(); return
    audit, params, ids = build()
    if a.install:
        write(audit, params, ids, HERE)
    else:
        out = CACHE / 'dry-run'; out.mkdir(parents=True, exist_ok=True)
        write(audit, params, ids, out)
    print(json.dumps(counts(audit, params), indent=1))


if __name__ == '__main__':
    main()

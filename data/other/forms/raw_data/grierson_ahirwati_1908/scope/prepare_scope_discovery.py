"""Scope discovery from retained OCR/XML and originals; emits no lexical rows."""
from pathlib import Path
import bz2, hashlib, json, re, subprocess, unicodedata
import xml.etree.ElementTree as ET
from PIL import Image

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[6]
DJVU = ROOT / 'tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu'
XML = ROOT / 'tmp/pdfs/lsi-sprakbanken.xml.bz2'
SCRATCH = ROOT / 'tmp/jambu-next-source/ahirwati-scope-20260927'
SCRATCH.mkdir(exist_ok=True)

def sha(p):
    with p.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def folded(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s).lower() if not unicodedata.combining(c))

def immutable(name, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    p = BASE / name
    if p.exists():
        assert p.read_bytes() == data, f'Historical scope artifact differs: {p}'
    else:
        p.write_bytes(data)

raw = subprocess.check_output(['/opt/homebrew/bin/djvutxt', str(DJVU)])
text_path = SCRATCH / 'LSI-V9-2-retained-djvu-text.txt'
if not text_path.exists(): text_path.write_bytes(raw)
else: assert text_path.read_bytes() == raw
pages = raw.decode('utf-8').split('\f')
if not pages[-1].strip(): pages.pop()
pattern = re.compile(r'ahir|hirwat|hirwal|irwati|irwali')
leads = []
for physical, text in enumerate(pages, 1):
    lines = text.splitlines()
    matches = [i for i, line in enumerate(lines) if pattern.search(folded(line))]
    if matches:
        leads.append({'key': f'djvu:physical:{physical}', 'physical_one_based': physical,
            'match_line_numbers': [i+1 for i in matches],
            'raw_contexts': [{'line': i+1, 'raw': '\n'.join(lines[max(0,i-2):i+4])} for i in matches]})
immutable('djvu-volume-wide-leads.json', leads)
errata = [{'physical_one_based': i, 'raw': t} for i,t in enumerate(pages,1)
          if re.search(r'addend|corrigend|errat', folded(t))]
immutable('djvu-errata-leads.json', errata)
stack = []
xml_leads = []
volume_pages = []
with bz2.open(XML, 'rb') as h:
    for event, e in ET.iterparse(h, events=('start','end')):
        if event == 'start' and e.tag == 'corpus': stack.append(e.attrib.copy())
        elif event == 'end' and e.tag == 'page':
            if stack[-1].get('vol') == '9' and stack[-1].get('part') == '2':
                sentences = [{'id': s.get('id'), 'raw': ' '.join(w.text or '' for w in s.findall('w'))}
                             for s in e.findall('sentence')]
                volume_pages.append(e.attrib.copy())
                hits = [s for s in sentences if pattern.search(folded(s['raw']))]
                if hits:
                    xml_leads.append({'key': 'nordic:p'+e.get('pno','unknown'),
                        'page_attributes':e.attrib.copy(), 'matching_sentences':hits})
            e.clear()
        elif event == 'end' and e.tag == 'corpus': stack.pop()
immutable('nordic-volume-wide-leads.json', xml_leads)
immutable('discovery-inputs.json', {'stage':'scope discovery only',
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [DJVU,XML]},
    'djvu_ocr_pages':len(pages), 'nordic_volume_pages':len(volume_pages),
    'djvu_lead_pages':len(leads), 'nordic_lead_pages':len(xml_leads),
    'errata_lead_pages':len(errata), 'normalization':'NFKD accent-folded lowercase search only; raw evidence unchanged',
    'search_pattern':pattern.pattern})
for printed in range(304,322):
    physical = printed+15
    target = SCRATCH / f'table-header-footer-printed-{printed}-physical-{physical}.png'
    if target.exists(): continue
    pnm = subprocess.check_output(['/opt/homebrew/bin/ddjvu','-format=pnm',
        f'-page={physical}','-size=1600x2200',str(DJVU)])
    import io
    im = Image.open(io.BytesIO(pnm))
    top=im.crop((0,0,im.width,int(im.height*.24)))
    bottom=im.crop((0,int(im.height*.9),im.width,im.height))
    joined=Image.new('RGB',(im.width,top.height+bottom.height),'white')
    joined.paste(top,(0,0));joined.paste(bottom,(0,top.height));joined.save(target)
print(json.dumps({'djvu_lead_pages':len(leads),'nordic_lead_pages':len(xml_leads),
    'errata_lead_pages':len(errata), 'djvu_ocr_pages':len(pages)},indent=2))

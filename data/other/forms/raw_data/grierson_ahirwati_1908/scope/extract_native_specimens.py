"""Extract seven selected specimen witnesses from an already cached PDF.

The selected +17 mapping requires visual folio verification. It is not a
global pagination rule and this command does not transcribe or install forms.
"""
import hashlib
import io
import json
from pathlib import Path
from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[7]
BASE = Path(__file__).resolve().parent
SOURCE = ROOT / 'tmp/pdfs/lsi-v9-2/LSI-V9-2-alternate.pdf'
PIN = 'a9fbdec72119b8a6aab236541782342c62318cb1eb7cfddf2d1f5cf5891cbcb1'
DIRECTORY = ROOT / 'tmp/jambu-next-source/ahirwati-native-specimens-20260927'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    assert SOURCE.exists(), 'Required cached original PDF absent; do not silently download.'
    assert sha(SOURCE) == PIN, 'Cached witness changed'
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(SOURCE)
    pages = []
    for printed in range(233, 240):
        physical = printed + 17
        page = reader.pages[physical - 1]
        images = list(page.images)
        assert len(images) == 1, (physical, 'Unexpected image topology')
        original = images[0]
        path = DIRECTORY / f'printed-{printed}-pdf-{physical}-{original.name}'
        if path.exists():
            assert path.read_bytes() == original.data
        else:
            path.write_bytes(original.data)
        native = Image.open(io.BytesIO(original.data))
        preview = path.with_suffix('.png')
        if not preview.exists():
            native.save(preview)
        else:
            with Image.open(preview) as prior:
                assert prior.size == native.size and prior.mode == native.mode
                assert prior.tobytes() == native.tobytes(), 'Prior preview pixel drift'
        pages.append({'printed_page_expected':printed, 'physical_pdf_onebased':physical,
                      'path':str(path.relative_to(ROOT)), 'sha256':sha(path),
                      'bytes':path.stat().st_size, 'native_dimensions':list(native.size),
                      'lossless_png_path':str(preview.relative_to(ROOT)),
                      'lossless_png_sha256':sha(preview),
                      'verification':'Requires direct folio and layout review'})
        native.close()
    report = {'status':'seven_selected_embedded_witnesses_extracted_not_transcribed',
              'pdf_path':str(SOURCE.relative_to(ROOT)), 'pdf_sha256':PIN,
              'physical_pdf_pages':len(reader.pages), 'pages':pages,
              'method':'Sequential pypdf native image extraction, no OCR or resampling',
              'limits':['No global printed/physical offset claim.', 'No source acceptance or canonical mutations.']}
    target = BASE / 'native-specimen-image-manifest.json'
    serialized = json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    if target.exists():
        assert target.read_text() == serialized, 'Preserve prior manifest'
    else:
        target.write_text(serialized)
    print(json.dumps({'manifest':str(target),'sha256':sha(target),'pages':len(pages),
                      'embedded_bytes':sum(p['bytes'] for p in pages)}))

if __name__ == '__main__':
    main()

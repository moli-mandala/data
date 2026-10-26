"""Pin the complete, unaccepted specimen37 owner reading and crop continuity."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
PACKAGE=HERE.parents[1]
CACHE=ROOT/'tmp/jambu-next-source/ahirwati-native-specimens-20260927'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,value):
    p=HERE/name
    text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if p.exists():
        assert p.read_text()==text, f'Historical evidence differs: {p}'
    else: p.write_text(text)

manifest=json.loads((PACKAGE/'scope/native-specimen-image-manifest.json').read_text())
originals={r['printed_page_expected']:r for r in manifest['pages']}
crops=[]
spec=[(233,'top',(245,1090,2015,1680)),(233,'middle',(245,1670,2015,2200)),
      (233,'bottom',(245,2190,2015,2860)),(234,'top',(240,905,2020,1550)),
      (234,'middle',(240,1540,2020,2180)),(234,'bottom',(240,2170,2020,2810)),
      (235,'top',(240,365,2020,1110)),(235,'middle',(240,1100,2020,1900)),
      (235,'bottom',(240,1890,2020,2800)),(236,'whole',(240,400,2020,1020))]
for pg,label,box in spec:
    image=CACHE/f'specimen37-p{pg}-owner-{label}.png'
    original=ROOT/originals[pg]['lossless_png_path']
    assert sha(original)==originals[pg]['lossless_png_sha256']
    crops.append({'path':str(image.relative_to(ROOT)),'sha256':sha(image),
                  'source':str(original.relative_to(ROOT)),'source_sha256':sha(original),
                  'rectangle_in_original_pixels':box,'method':'native-resolution crop; no enhancement'})
for name,pg,box,scope in [
    ('near-form-detail',235,(615,500,985,685),'Disputed first vowel: owner ā/root provisional ī; current pīlᵃnē retained with both readings.'),
    ('khōṭ-gloss-detail',234,(1600,2520,1930,2640),'Historical clipped gloss crop; not sufficient to certify middle English letters.'),
    ('all-native-first-detail',233,(1070,1305,1350,1420),'Historical mislocated crop; not evidence for all-form.'),
    ('all-native-last-detail',233,(950,2560,1990,2690),'Native all-form and preceding/following script context; क/ळ and तकी remain font holds.')]:
    image=CACHE/f'specimen37-{name}.png'
    crops.append({'path':str(image.relative_to(ROOT)),'sha256':sha(image),
                  'source':originals[pg]['lossless_png_path'],
                  'source_sha256':originals[pg]['lossless_png_sha256'],
                  'rectangle_in_original_pixels':box,'method':'native-resolution crop','evidence_scope':scope})
root_controls=PACKAGE/'specimen38/root-specimen37-two-control-evidence.json'
controls=json.loads(root_controls.read_text())
for row in controls:
    assert sha(ROOT/row['path'])==row['sha256']
    assert sha(ROOT/row['source'])==row['source_sha256']
put('first-reading-crop-provenance.json',{'owner_crops':crops,
     'root_control_manifest':str(root_controls.relative_to(ROOT)),
     'root_control_manifest_sha256':sha(root_controls),'root_controls':controls,
     'scope':'Crop continuity only; narrow ambiguous glyphs remain typed holds.'})

names=['roman-first-reading.tsv','native-first-reading.tsv','prepare_first_reading.py',
       'freeze_first_reading.py','roman-physical-lines-first-reading.jsonl','roman-atoms-first-reading.jsonl',
       'native-physical-lines-first-reading.jsonl','native-atoms-first-reading.jsonl',
       'physical-break-joins-proposal.jsonl','script-alignment-proposal.jsonl',
       'script-alignment-review.tsv','whole-utterances-first-reading.jsonl',
       'first-reading-draft-status.json','first-reading-crop-provenance.json']
status=json.loads((HERE/'first-reading-draft-status.json').read_text())
for name,digest in status['output_hashes'].items(): assert sha(HERE/name)==digest
put('first-reading-freeze.json',{'status':'complete owner first reading frozen for independent original review; unaccepted',
    'counts':status['counts'],'files':{name:sha(HERE/name) for name in names},
    'source_manifest_sha256':sha(PACKAGE/'scope/native-specimen-image-manifest.json'),
    'originals':{str(pg):{'path':originals[pg]['lossless_png_path'],'sha256':originals[pg]['lossless_png_sha256']} for pg in (233,234,235,236)},
    'coverage_checks':['Every Roman/native physical key retained exactly once in script-span accounting.',
                       'Every Roman physical atom occurs in at least one manual whole/speech quotation span.',
                       '42 Roman lines and27 native lines are complete; script differences not normalized.',
                       'Three physical line-break joins retain both raw parts; three local spacing/comparative differences retain joint spans.'],
    'limits':['No independent lexical or boundary acceptance yet; no canonical emission or installation.',
              'Two narrow Roman/English glyph disputes and all keyed native/font holds remain explicit.',
              'No author-supplied free translation on these pages; concatenated interlinear analysis remains a distinct field.',
              'Full source additionally requires specimen38, grammar/outside reconciliation, metadata/profile/graph/testing gates.']})
print(json.dumps({'freeze_sha256':sha(HERE/'first-reading-freeze.json'),**status['counts']}))

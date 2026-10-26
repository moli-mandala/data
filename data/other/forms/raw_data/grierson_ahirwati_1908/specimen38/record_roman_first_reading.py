"""Record whole No38 Roman interlinear first reading; not accepted installation."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=Path(__file__).resolve().parents[7]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((BASE/'scope/native-specimen-image-manifest.json').read_text())
native=next(r for r in manifest['pages'] if r['printed_page_expected']==238)
assert sha(ROOT/native['lossless_png_path'])==native['lossless_png_sha256']
census=[json.loads(l) for l in (BASE/'scope/physical-source-unit-census.jsonl').read_text().splitlines()]
parents=[r for r in census if r['printed_page']==238 and r['kind']=='physical_interlinear_line']
assert len(parents)==15
manual=HERE/'roman-first-reading.tsv'
rows=list(csv.DictReader(manual.open(),delimiter='\t'))
assert len(rows)==15
holds={
    (3,1):'Raised-a and following n underdot require narrow occurrence control; no phonemic substitution.',
    (3,5):'Source nasal mark attachment and quantity on this pronoun require narrow occurrence control.',
    (3,6):'Raised medial a and nasal/lateral glyph details require narrow occurrence control.',
    (3,7):'Interrogative final n underdot requires native occurrence check; do not propagate grammar kauṇ automatically.',
    (6,2):'Quantity/nasal encoding on yes needs narrow occurrence control.',
    (8,2):'Raised medial a and following retroflex glyph require narrow occurrence control.',
    (12,2):'Single/doubled n and possible underdot in beautiful require narrow occurrence control.',
    (13,8):'Personal name medial raised a versus other small source glyph must be verified.',
    (14,2):'Lateral underdot and medial raised a need narrow occurrence control.',
    (14,8):'Raised medial a and final nasal quantity need narrow occurrence control.',
}
output=[];counts=[]
for index,(record,parent) in enumerate(zip(rows,parents),1):
    assert int(record['page'])==238 and int(record['line'])==index
    assert parent['key']==f'lsi9.2:p238:aligned-line:{index}'
    forms=record['source_forms_pipe_separated'].split('|')
    glosses=record['printed_glosses_pipe_separated'].split('|')
    assert len(forms)==len(glosses)==parent['source_atom_count_initial'],(index,len(forms),len(glosses))
    atoms=[]
    for word,(form,gloss) in enumerate(zip(forms,glosses),1):
        form=unicodedata.normalize('NFC',form);gloss=unicodedata.normalize('NFC',gloss)
        assert form and gloss and '\ufffd' not in form+gloss
        key=f"{parent['key']}:word:{word}"
        atoms.append({'source_unit_key':key,'physical_parent_key':parent['key'],'printed_page':238,'physical_line':index,'physical_word':word,
                      'raw_form':form,'raw_gloss':gloss,'typed_uncertainties':[{'type':'source-glyph','reason':holds[(index,word)]}] if (index,word) in holds else [],
                      'review_status':'root complete first reading; independent exact-literal review pending'})
    output.append({'source_unit_key':parent['key'],'printed_page':238,'specimen_number':38,'source_lect':'Ahirwati of Jhajjar/Rohtak, explicitly mixed with Bangaru',
                   'raw_target_line':' '.join(forms),'raw_aligned_english_line':' '.join(glosses),'target_atoms':atoms,'original_witness':native,
                   'literal_boundary_policy':'Printed spaces retain physical atoms; Jāi-kaḷi-/kai continuation remains two physical records until separately reviewed joined emission.'})
    counts.append(len(atoms))
target=HERE/'p238-roman-first-reading.jsonl'
serialized=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output)
assert not target.exists(),'Preserve immutable first reading'
target.write_text(serialized)
status={'status':'complete root first reading of all No38 Roman aligned lines; independent review pending','physical_lines':15,'physical_atoms':sum(counts),'line_counts':counts,
        'input_sha256':{'roman-first-reading.tsv':sha(manual),'scope/physical-source-unit-census.jsonl':sha(BASE/'scope/physical-source-unit-census.jsonl'),
                       'scope/native-specimen-image-manifest.json':sha(BASE/'scope/native-specimen-image-manifest.json'),'scope/root-specimen38-native-crops.json':sha(BASE/'scope/root-specimen38-native-crops.json')},
        'output_sha256':{'p238-roman-first-reading.jsonl':sha(target)},'glyph_holds':len(holds),
        'pending':['Native Persian transcription/alignment on237','Documentary quotations/comparison routing on237','Complete free translation238–239','Independent raw-literal and metadata review','Separate joined-word/expression conservation','Whole-source integration and all applicable focused gates'],
        'limits':['Initial space-group census matches fine reading, not an independent accepted-error audit.','No source-wide completion or canonical installation.','No arbitrary reassignment of mixed-lect narrative tokens.','English spelling/punctuation/source analyses retained literally; no inferred tags or phonemic values.']}
(HERE/'first-reading-status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'lines':15,'atoms':sum(counts),'holds':len(holds),'sha256':sha(target)}))

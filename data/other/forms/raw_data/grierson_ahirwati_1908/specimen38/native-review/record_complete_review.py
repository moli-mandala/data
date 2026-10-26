"""Complete source-only No38 native reading and independent Latin review."""
import csv,hashlib,json,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[7]
OWNER=HERE.parent
PACKAGE=HERE.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(name,rows):
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    p=HERE/name
    if p.exists():assert p.read_text()==text, f'Preserve historical review: {name}'
    else:p.write_text(text)
def report(name,value):
    p=HERE/name;text=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    if p.exists():assert p.read_text()==text
    else:p.write_text(text)

owner_lines=[json.loads(l) for l in (OWNER/'p238-roman-first-reading.jsonl').read_text().splitlines()]
latin=[r for line in owner_lines for r in line['target_atoms']]
assert len(latin)==133
original_manifest=json.loads((PACKAGE/'scope/native-specimen-image-manifest.json').read_text())
originals={r['printed_page_expected']:r for r in original_manifest['pages']}
for page in (237,238):assert sha(ROOT/originals[page]['lossless_png_path'])==originals[page]['lossless_png_sha256']
crop_manifest=json.loads((OWNER/'native-persian-crop.json').read_text())
assert sha(ROOT/crop_manifest['path'])==crop_manifest['sha256']
for filename in ['original-crop-provenance.json','narrow-control-provenance.json','followup-control-provenance.json']:
    for r in json.loads((HERE/filename).read_text()):
        assert sha(ROOT/r['path'])==r['sha256'] and sha(ROOT/r['source'])==r['source_sha256']

corrections={
 (1,8):('bērē-mai','bērē-nai','n after hyphen has one hump; matches other printed nai occurrences. Independent Persian بیرے نے is corroborating, not the replacement authority.','narrow-bere-nai.png'),
 (4,10):('Jāi-kaḷi-','Jai-kaḷi-','No horizontal quantity mark over first a in the printed name; source i tittle distinct.','original-roman-top.png'),
 (5,8):('Jāi-kaḷi-kai','Jai-kaḷi-kai','Native-pixel complete name shows plain first a, not ā.','followup-jai-full.png'),
}
holds={
 (9,3):('kai-ṭarah','kai-ṭaraḥ','Visible lower mark at final h may be an explicit underdot; exact encoding requires root font-control adjudication.','narrow-kai-tarah-first.png'),
 (15,1):('kai-ṭarah','kai-ṭaraḥ','Same lower final-h mark recurs; retain candidate and frozen reading separately.','followup-kai-tarah-last-full.png'),
}
decisions=[]
for r in latin:
    location=(r['physical_line'],r['physical_word'])
    decision={'key':r['source_unit_key'],'old_form':r['raw_form'],'old_gloss':r['raw_gloss'],
       'printed_page':238,'physical_line':r['physical_line'],'physical_word':r['physical_word'],
       'review':'independent complete Roman+aligned-English original reading; source script differences not harmonized',
       'status':'agrees','proposed_form':r['raw_form'],'proposed_gloss':r['raw_gloss'],
       'original_sha256':originals[238]['lossless_png_sha256'],
       'original_hold_review':'Native-pixel control supports current spelling/mark placement; no phonemic interpretation.' if r['typed_uncertainties'] else None}
    if location in corrections:
        old,new,why,crop=corrections[location];assert r['raw_form']==old
        decision.update(status='proposed-literal-correction',proposed_form=new,reason=why,
                        evidence=str((HERE/crop).relative_to(ROOT)),evidence_sha256=sha(HERE/crop))
    if location in holds:
        old,candidate,why,crop=holds[location];assert r['raw_form']==old
        decision.update(status='typed-source-glyph-hold',alternative_form=candidate,reason=why,
                        evidence=str((HERE/crop).relative_to(ROOT)),evidence_sha256=sha(HERE/crop))
    decisions.append(decision)
put('independent-roman-all-133-review.jsonl',decisions)

native_rows=list(csv.DictReader((HERE/'native-first-reading.tsv').open(),delimiter='\t'))
assert [int(r['line']) for r in native_rows]==list(range(1,12))
native=[];native_lines=[]
specific_holds={
 'اینہ':['Source initial pronoun glyph sequence needs independent native reading; paired Latin aĩh is separate authority.'],
 'جیکلی':['Printed personal-name spelling differs from Latin Jai-kaḷi; do not invent omitted native letters.'],
 'گھروالے':['Printed final native spelling may differ from earlier گھر والو; preserve source occurrence.'],
 'سونا':['Native continuous group corresponds to separate Latin sō/nā; word division not silently normalized.'],
 'جینہم':['Final native joined letters are unresolved; candidates جینھم/جینھہ remain script-glyph alternatives, not Roman-driven emendations.'],
 'تین':['Final nasal/nūn shape needs occurrence reading where Latin Taĩ-nai differs; ordinary numeral تین elsewhere remains distinct context.'],
 'دیدون':['Final native nasal/vowel letter shape remains uncertain; Latin dē-dū̃ remains a separate witness.'],
 'حکملے':['Native name vowel/letter sequence differs from Latin Hukamᵃlā; preserve script-specific spelling.'],
}
for r in native_rows:
    line=int(r['line']);key=f'lsi9.2:p237:native-line:{line}';keys=[]
    for rank,word in enumerate(r['source_text'].split(),1):
        k=f'{key}:word:{rank}';keys.append(k)
        role='separator-glyph' if word=='—' else 'redundant-printed-numeral' if word=='۲۱' else 'native-script-physical-space-group'
        uncertainties=list(specific_holds.get(word,[]))
        if word=='تین' and line!=11: uncertainties=[]
        if role=='separator-glyph':uncertainties=['Horizontal printed separator is encoded provisionally as em dash; no inferred sentence/free-translation boundary.']
        native.append({'key':k,'parent_key':key,'printed_page':237,'physical_line':line,'word_rank':rank,
          'raw_native':word,'source_script':'Persian','unit_role':role,'typed_uncertainties':uncertainties,
          'source_image':originals[237]['lossless_png_path'],'source_image_sha256':originals[237]['lossless_png_sha256'],
          'source_lect':'Ahirwati of Jhajjar/Rohtak; source explicitly qualifies mixed Bangaru usage',
          'stage':'complete owner native first reading; independent native-glyph review pending'})
    native_lines.append({'key':key,'raw_native_line':r['source_text'],'native_word_keys':keys,
      'source_image':originals[237]['lossless_png_path'],'source_image_sha256':originals[237]['lossless_png_sha256']})
put('native-lines-first-reading.jsonl',native_lines)
put('native-atoms-first-reading.jsonl',native)

# Joint spans avoid splitting Latin compounds at native physical line endings.
spans=[([1],(1,1),(2,3)),([2],(2,4),(3,4)),([3],(3,5),(4,7)),
       ([4],(4,8),(6,1)),([5],(6,2),(7,5)),([6,7],(7,6),(10,3)),
       ([8],(10,4),(11,8)),([9,10,11],(11,9),(15,2))]
where={(r['physical_line'],r['physical_word']):i for i,r in enumerate(latin)}
align=[]
for rank,(lines,first,last) in enumerate(spans,1):
    romans=latin[where[first]:where[last]+1]
    natives=[r for r in native if r['physical_line'] in lines]
    align.append({'key':f'lsi9.2:specimen38:script-span:{rank}','native_physical_lines':lines,
      'native_keys':[r['key'] for r in natives],'roman_keys':[r['source_unit_key'] for r in romans],
      'raw_native_span':' '.join(r['raw_native'] for r in natives),
      'raw_roman_span':' '.join(r['raw_form'] for r in romans),
      'printed_interlinear_analysis':' '.join(r['raw_gloss'] for r in romans),
      'mapping_type':'conservative contiguous source-span alignment; not per-word equivalence',
      'source_differences':['Native unvocalized spellings/spacing and additional numeral retained; no inferred transliteration.'],
      'typed_uncertainties':[u for r in natives for u in r['typed_uncertainties']],
      'status':'owner span proposal from both originals; independent native and boundary review pending'})
assert [k for r in align for k in r['roman_keys']]==[r['source_unit_key'] for r in latin]
assert [k for r in align for k in r['native_keys']]==[r['key'] for r in native]
assert sum(r['raw_native']=='۲۱' for r in native)==1
assert not any(unicodedata.category(c)=='Co' or c=='\ufffd' for r in native for c in r['raw_native'])
put('native-roman-span-alignment-proposal.jsonl',align)
files=['native-first-reading.tsv','record_complete_review.py','native-lines-first-reading.jsonl',
       'native-atoms-first-reading.jsonl','native-roman-span-alignment-proposal.jsonl',
       'independent-roman-all-133-review.jsonl','original-crop-provenance.json',
       'narrow-control-provenance.json','followup-control-provenance.json']
report('complete-review-status.json',{'status':'complete independent Latin/gloss review plus owner native first reading; proposed repairs and individual native glyph holds unadjudicated',
 'counts':{'roman_atoms_reviewed':133,'agrees':128,'proposed_latin_corrections':3,'latin_glyph_holds':2,
           'native_lines':11,'native_physical_space_groups':len(native),'redundant_printed_numerals':1,
           'conservative_script_span_groups':8},
 'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [OWNER/'p238-roman-first-reading.jsonl',OWNER/'roman-first-reading.tsv',OWNER/'native-persian-crop.json',PACKAGE/'scope/native-specimen-image-manifest.json']},
 'output_hashes':{name:sha(HERE/name) for name in files},
 'limits':['No owner input or canonical rows changed.',
           'Arabic/Persian letter encoding and typography holds require a separate independent native reading.',
           'Joint spans preserve all physical keys; no forced word-to-word pairing, donor inference, or free English translation.',
           'Root owns documentary/comparator accounting and source-supplied free translation; this packet does not replace those records.']})
print(json.dumps({'latin':133,'native_lines':11,'native_groups':len(native),'spans':8,'status_sha256':sha(HERE/'complete-review-status.json')}))

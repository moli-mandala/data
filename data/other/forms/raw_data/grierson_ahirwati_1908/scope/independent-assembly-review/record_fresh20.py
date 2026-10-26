"""Record the independent visual observations after frozen seed selection."""
import csv, hashlib, json
from pathlib import Path
here=Path(__file__).resolve().parent
package=here.parents[1]
assembly=package/'integration/review-assembly-r1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
selection_path=here/'fresh20-selection-20260927510.json'
selection=json.loads(selection_path.read_text())
assert sha(selection_path)=='09f715bed24dda189f15ade6ebfabf7e67c6f57a9f1694056e0568eecf31a03d'
assert sha(assembly/'combined-review-candidate.csv')==selection['csv_sha256']
assert sha(assembly/'combined-review-audit.jsonl')==selection['audit_sha256']
actual={x[10]:x for x in csv.reader((assembly/'combined-review-candidate.csv').open())}
grammar='tmp/jambu-next-source/ahirwati-native-grammar-20260927/'
specimens='tmp/jambu-next-source/ahirwati-native-specimens-20260927/'
outside=str(package/'outside-review')+'/'
commentary=str(package/'specimen38/root-documentary-review/p237-linguistic-commentary-native.png')
observations=[
 ('p51-first-reading-pronouns.png','Printed demonstrative yō or yoh, “this”; sampled yoh preserves the distinct listed alternate without inventing length on its plain o.'),
 ('p51-first-reading-copulas.png','Second-singular present column prints sā beside sai. The cell supplies no separately printed lexical English; blank Gloss is appropriate.'),
 ('p51-first-reading-pronouns.png','Parenthesis explicitly prints feminine yā, “this”; final ā and y are clear.'),
 ('p51-first-reading-copulas.png','Running prose prints second-plural saĩ and attributes it to Bagri. The nasal sign is retained; blank lexical Gloss does not substitute the grammatical label.'),
 ('specimen37-p235-owner-top.png','Printed dēkhᵃtā̃- / hī continues over two physical rows; English on-seeing- / even. Joined literal dēkhᵃtā̃-hī and on-seeing-even conserve both keys and the raised a/nasal/quantity marks.'),
 ('specimen37-p234-owner-top.png','par-dēsā̃-nē̃ is printed above foreign-countries-to; internal hyphens, long ē, nasal ā and final nē̃ are retained. No plural-tag acceptance is inferred from this literal match.'),
 ('specimen37-p235-owner-bottom.png','Capital Ib is printed directly above Now. Following jab-taĩ belongs to the next word and is not merged.'),
 ('specimen37-p234-owner-middle.png','dhan is the separately printed word above fortune after apᵃṇū. No inherent sound normalization is substituted.'),
 ('specimen37-p236-owner-whole.png','thō, has a macron o and printed comma; interlinear English is was, with the same punctuation.'),
 ('specimen37-p236-owner-whole.png','The speech begins at printed ‘bēṭā, and extends through final milyō-sai-gō. All contributing words and English pairs agree in order, including kimaĩ, sā, Tū-nē̃, karᵃṇō, khūyō and (he) found-is. Missing terminal close quote is not invented; this is printed interlinear English, not smooth translation.'),
 ('root-p238-roman-bottom.png','kāḷᵃjē-kī occurs directly above liver-of. Long ā/ē/ī, retroflex underdot and raised a are retained.'),
 ('root-p238-roman-middle.png','Hīr-kē is capitalized and printed directly above The-Ahīr-of; quantity and internal hyphen retained.'),
 ('root-p238-roman-middle.png','tō is the separate token above then in ji tam dē, tō maĩ-nai…; not the earlier Tō beginning another quoted span.'),
 ('root-p238-roman-bottom.png','jaĩh-nai is printed above which, before ikīs. Preserve this occurrence and its nasal sign; no inferred case or repaired smooth English.'),
 ('root-p238-roman-top.png','The bounded speech ‘tērō mehmān sai.’ is printed with opening and closing quote and full stop; exact aligned English ‘thy guest is.’ is preserved.'),
 (commentary,'The explanatory prose prints Ahirwati bōlyō in contrast to bōlō. This is its separate explanatory occurrence, with no independently supplied lexical Gloss; blank Gloss retained.'),
 (commentary,'The first quoted sentence explicitly contains Ahīr with capital A and long ī. It is separately printed in the commentary; not a forced reuse of narrative ahīr.'),
 ('p151-auxiliary-native.png','Under the Sirsa and other affected-parts heading, first-person plural cell prints sā̃ with long nasal ā. No per-cell lexical English printed, so blank Gloss retained.'),
 ('p151-finite-context-native.png','The prose explicitly contrasts Hissar third-plural mārē̃ with the general mārē table form. Long ā/ē and final nasal preserved; strike is the scoped table-heading lexical meaning, not an invented full translation.'),
 ('p149-phonology-native.png','Bikaner prose gives bō, not wō, he. Sampled bō/he follows that actual literal claim; wider Wilson/Sirsa comparison does not create extra copies of this occurrence.'),
]
results=[]
for sample,(image,observation) in zip(selection['samples'],observations):
    assert actual[sample['key']]==sample['row']
    if image.startswith('p51-'): image=grammar+image
    elif image.startswith('specimen37-') or image.startswith('root-p238-'): image=specimens+image
    elif image.startswith('p151-') or image.startswith('p149-'): image=outside+image
    p=Path(image)
    page=int(sample['key'].split(':')[1][1:])
    if page==51: primary=Path(grammar+'printed-51-pdf-68-Im68.png')
    elif page in [234,235,236,237,238]: primary=Path(specimens+f'printed-{page}-pdf-{page+17}-Im{page+17}.png')
    else: primary=Path(outside+f'candidate-printed-{page}-pdf-{page+17}-Im{page+17}.png')
    results.append({'sample':sample['sample'],'entry_key':sample['key'],'kind':sample['kind'],
                    'actual_form':sample['row'][2],'actual_gloss':sample['row'][3],
                    'row_sha256':hashlib.sha256(json.dumps(sample['row'],ensure_ascii=False).encode()).hexdigest(),
                    'status':'literal/provenance pass','material_errors':[], 'observation':observation,
                    'primary_image':str(primary),'primary_image_sha256':sha(primary),
                    'viewed_crop':str(p),'viewed_crop_sha256':sha(p)})
assert len(results)==20
report={'status':'passed scoped Form/Gloss original-image audit','reviewer':'/root/asur_finish',
        'seed':selection['seed'],'sampled_count':20,'material_errors':[],
        'csv_sha256':selection['csv_sha256'],'audit_sha256':selection['audit_sha256'],
        'selection_sha256':sha(selection_path),'samples':results,
        'source_glyph_policy':'Preserve literal raised letters, vowel quantities, nasalization, hyphens and printed punctuation. Existing scoped glyph/English holds remain in the full audit and are not closed by unsampled matches.',
        'limits':['Metadata r4/residual decisions remain separately pending; this report is not metadata acceptance.',
                  'Native literal fields were not recertified by this Form/Gloss sample; Persian194 remain broad documentary provenance.',
                  'No claim of whole-source random certification, final profile/installation/build/browser pass.',
                  'Candidate input column2 is the literal source Form field (parsed Original); generic display normalization is a separate parser gate.'],
        'canonical_changes':False}
(here/'fresh20-original-review-20260927510.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'sampled':20,'material_errors':0,'report_sha256':sha(here/'fresh20-original-review-20260927510.json')},indent=2))

from pathlib import Path
import json,hashlib,csv
H=Path(__file__).resolve().parent;P=H.parent
selection=json.loads((H/'fresh-original-selection.json').read_text())
observations=[
'p9 final-vowel table prints Malayalam aama opposite aame, with shared tortoise gloss. Malayalam routing and attributed Kannada-influence note are properly scoped.',
'p9 cart pair prints baṇṭi with two lower dots, opposite Malayalam vaṇṭi. Target cart gloss and source contact claim agree.',
'p9 stream line prints puje/puye. Selected second alternative puye and stream agree; variant parent is the explicitly co-listed puje, without donor inference.',
'p9 measure line prints naaji/naayi with aa, first j and shared a measure. Selected first form and retained source prose discrepancy agree.',
'p9 medial-length table prints ureelu opposite ural and mortar. Double ee is retained literally.',
'p10 gemination table prints necci opposite neṛṛi and forehead. cc, initial ne and source dotted-r comparison agree.',
'p10 second sentence prints n̲iin naaḷe baa, with underline on initial n and dot beneath l. Whole translation you come tomorrow and clause metadata agree.',
'p10 first sentence prints n̲iin eppoo pooggaa with doubled oo/gg/aa and the translated question when do you go?; interrogative scope is the whole clause.',
'p10 inanimate example prints kuṛe kallu and stones. Quantitative plural phrase, not a reconstructed inflected single word.',
'p10 prose explicitly calls naam first-person plural and associates it with a large group while denying inclusive/exclusive difference. Blank lexical gloss and scoped note are appropriate.',
'p10 present-suffix list prints atu without a leading dash. Output preserves that omission while structuring suffix and present function.',
'p10 prose explicitly calls kuṛe a quantitative adjectival form. Dotted r and adj/quantifier metadata agree; no separate lexical gloss is invented.',
'p10 dative prose wraps the printed dash and kku across lines. Rejoined –kku and dat/suffix agree, and Notes retain its absence after pronouns.',
'p10 past suffix series includes –ppe after -ippe. Printed dash, pp and e agree; past function is mapped to pret without lexical verb POS.',
'p10 row6 future column prints tallum, aligned with tall to beat. Form/gloss and verb/future metadata agree.',
'p10 row1 past column prints tinuppe, aligned with tinj to eat. No n/j harmonization or person/number inference appears.',
'p10 row5 present column prints kondatu, aligned with koll to kill. The actual allomorph is preserved and present heading correctly scoped.',
'p10 row8 present column prints kalaŋgtu, aligned with kalaŋg to talk. Eng and following gt cluster are preserved literally.',
'p11 final item prints kummen hunch back. Double m and final en agree, with no extra invented grammatical label.',
'p11 preceding item prints arccu fin of fishes. Double c and plural fishes in the definition agree.'
]
current=list(csv.reader((P/'integration/review-candidate.csv').open()));bykey={r[10]:r for r in current}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'integration/review-candidate.csv')==selection['csv_sha256']
assert sha(P/'integration/review-audit.jsonl')==selection['audit_sha256']
for i,(s,obs) in enumerate(zip(selection['samples'],observations),1):
 assert bykey[s['key']]==s['row'];s.update(sample_number=i,observation=obs,verdict='pass',original_image=f"../scope/original-page-{int(s['key'].split(':p')[1].split(':')[0]):02d}-150dpi.png")
report=dict(status='passed',reviewer='asur_finish',seed=selection['seed'],method=selection['method'],sampled_count=20,material_errors=[],material_error_count=0,csv_sha256=selection['csv_sha256'],audit_sha256=selection['audit_sha256'],source_sha256='4d2588987dab5d15c62ecc95330ccf7e925a9bce80d08068b91ca3f48c8cc5af',selection_sha256=sha(H/'fresh-original-selection.json'),samples=selection['samples'],original_evidence=[dict(path=f'../scope/original-page-{p:02d}-150dpi.png',sha256=sha(P/f'scope/original-page-{p:02d}-150dpi.png')) for p in [9,10,11]],limits='Fresh stratified original-vs-output sample, complemented by exhaustive frozen-reading/metadata validator. Does not claim full database or browser validation.')
(H/'fresh-original-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS 20; CSV',report['csv_sha256'])

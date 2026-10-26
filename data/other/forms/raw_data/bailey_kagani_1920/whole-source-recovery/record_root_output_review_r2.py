"""Record root's original-page comparison of fresh r2 sample and regressions."""
import csv,hashlib,json
from pathlib import Path
R=Path(__file__).resolve().parent/'whole-proposal-r2'
sample=json.loads((R/'root-output-sample-20260927.json').read_text())
manifest=json.loads((R/'review-manifest.json').read_text())
assert hashlib.sha256((R/'review-manifest.json').read_bytes()).hexdigest()==sample['manifest_sha256']
for name,digest in manifest['output_sha256'].items():
    assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest
rows={r[10]:r for r in csv.reader((R/'proposal.csv').open())}
checks={
 'bailey1920kagani:whole:p103:parable:line12:word1':({'verb','inf'},{'pres','participle'}),
 'bailey1920kagani:whole:p102:parable:line1:word4':({'pron','third-person','sg'},set()),
 'bailey1920kagani:whole:p104:parable:line12:word2':({'pron','second-person','sg','gen'},set()),
 'bailey1920kagani:whole:p103:parable:line15:word5':({'verb','caus'},{'pass','participle'})
}
for key,(required,forbidden) in checks.items():
    tags=set(rows[key][14].split())
    assert required<=tags and not forbidden&tags,key
report={'status':'passed','sample_size':20,'sample_rows':sample['rows'],'material_errors':0,
 'seed':sample['seed'],'frozen_manifest_sha256':sample['manifest_sha256'],
 'frozen_output_sha256':manifest['output_sha256'],
 'reviews':[{'printed_page':s['printed_page'],'entry_keys':[r[10] for r in s['actual_rows']],
             'status':'passed','evidence':'Compared frozen actual rows with original page image: literal forms, shared gloss scope, grammatical headings, exact locators and residual notes.'} for s in sample['sample']],
 'targeted_prior_failure_checks':[{'entry_key':k,'status':'passed'} for k in checks],
 'notable_checks':['Source88 literal tone alternatives; source89 bound ending with blank gloss.',
   'Source92–93 printed abbreviation expansion and paradigm heading scope.',
   'Source99 plain n and literal tone stroke retained in pronunciation witness.',
   'Source103 full expression spans lines3–9; complete printed interlinear translation retained without invented free translation.',
   'Source109 adjective label applies to second answer only.'],
 'installation_approval':'Approved for guarded source-stage installation of exact r2 rows/profile with planned metadata and focused canonical validation.',
 'limits':'Full CLDF pipeline, global compiled graph/identity/reference checks, full suite and browser/database gates remain deferred. This is not complete compiled ingestion.'}
assert len(report['reviews'])==20
(R/'root-output-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(f"Fresh r2 audit passed:20 units/{sample['rows']} rows; four targeted checks passed. Source-stage installation approved.")

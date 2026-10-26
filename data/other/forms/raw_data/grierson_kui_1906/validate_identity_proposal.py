"""Bounded source-specific identity proposal checks; no canonical changes."""
import csv
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'source_meta.py').exists())
WORKSPACE=ROOT.parent
proposal=json.loads((P/'source-lect-mapping-proposal-20260927.json').read_text())
languages={r['ID']:r for r in csv.DictReader((ROOT/'cldf/languages.csv').open())}
assert languages['Kui']['Glottocode']=='kuii1252'
assert languages['Kuwi']['Glottocode']=='kuvi1243'
rows=proposal['mappings']
assert len(rows)==len({r['key'] for r in rows})==11
assert all(r['accepted'] is False for r in rows)
assert all(r['proposed_base'] is None or r['proposed_base'] in languages for r in rows)
assert set(map(int,proposal['page_accounting']))==set(range(457,472))
kalahandi=next(r for r in rows if r['key']=='kalahandi')
assert kalahandi['proposed_base']=='Kui' and kalahandi['alternative_modern_base']=='Kuwi'
assert kalahandi['uncertainty_type']=='conflicting-modern-language-identification'
assert not any(r['proposed_base']=='Kuwi' for r in rows)
images=['tmp/kui-457.png','tmp/kui-459.png','tmp/kui-printed460-review.png',
        'tmp/kui-printed462-review.png','tmp/kui-printed464-review.png','tmp/kui-465.png',
        'tmp/kui-467.png','tmp/kui-468.png','tmp/kui-469.png','tmp/kui-470.png','tmp/kui-471.png','tmp/kui-643.png']
sourcefiles=['SCOPE.md','nordic-scope-snapshot.json','scope-snapshot-status.json',
             'original-specimen-scope-review-20260927.json','source-lect-mapping-proposal-20260927.json',
             'identity-evidence-review-20260927.md']
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'status':'proposal validated; not installed or accepted',
        'mapping_rules':len(rows),'proposed_named_dialects':sorted({r['proposed_dialect_id'] for r in rows if r['proposed_dialect_id']}),
        'modern_identification_conflicts':['historical Kalahandi'],
        'existing_base_rows':{k:languages[k] for k in ['Kui','Kuwi']},
        'source_sha256':{n:digest(P/n) for n in sourcefiles},
        'original_images_independently_viewed_sha256':{n:digest(WORKSPACE/n) for n in images},
        'external_evidence_access_date':'2026-09-27',
        'primary_paper_existing_bibliography_key':'K-Kui',
        'canonical_writes':False,'lexical_transcription_performed':False,
        'limitation':'Identity/attribution review only. Complete cell reading and table-header mapping remain separate tasks.'}
(P/'identity-proposal-validation-20260927.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))

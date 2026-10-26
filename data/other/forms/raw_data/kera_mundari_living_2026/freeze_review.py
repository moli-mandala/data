"""Pin the complete source-stage review inputs after focused validation."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
files = ['proposal.csv','proposal-audit.jsonl','proposal-profile.txt','proposal.yaml',
         'about-attestations.json','lexical-snapshot.jsonl','lexical-context.json',
         'prepare_source.py','proposed-reference.bib','proposed-dialect.csv',
         'media-url-provenance.json','snapshot-completeness.json','snapshot-acquisition.json',
         'independent-about-whole-scope-review-20260927.json',
         'snapshot/kera-mundari.db','snapshot/about.html','snapshot/dictionary.html','snapshot/terms.html']
manifest = {'status':'frozen for independent output review; no installation',
            'counts':{'entries':446,'senses':446,'dictionary_rows':462,'about_rows':17,'rows':479,'audit_units':466},
            'hashes':{f:hashlib.sha256((P/f).read_bytes()).hexdigest() for f in files},
            'previous_selection':'independent-selection-20260927281.json was an owner-generated provisional selection before complete about-page corrections; not an independent passed audit.',
            'focused_validation':(P/'focused-tests.log').read_text().strip(),
            'deferred':'Canonical installation follows passing review. Full database, compiled graph and browser checks deferred by current instruction.'}
assert '8 passed' in manifest['focused_validation']
target = P/'review-manifest-r1.json'
assert not target.exists(), 'Do not overwrite an existing freeze.'
target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))

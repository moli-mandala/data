"""Record already verified source-stage results; no lexical or database writes."""
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent
ROOT=next(p for p in P.parents if (p/'source_meta.py').exists())
tests=(P/'installed-focused-tests.log').read_text()
assert '10 passed' in tests and 'failed' not in tests
assert (P/'regeneration-parity.log').read_text().strip()=='479 approved rows verified; no writes'
receipt=json.loads((P/'installation-report-20260927.json').read_text())
ledger=ROOT/'source_checklists/20260925-source-completeness-ledger.tsv'
key='20260927-kera-mundari-living'
if not any(line.startswith(key+'\t') for line in ledger.read_text().splitlines()):
    row=[key,'data/other/forms/raw_data/kera_mundari_living_2026','whole_published_snapshot_and_about',
         'complete_source_stage','asur_finish/root',
         '446 entries/senses -> 462 dictionary readings; complete about-page target-family evidence adds 17 readings, total 479 rows / 466 audit units. All 440 notes, 443 POS, 28 borrowing claims and 445 audio references retained. Four nonlexical about context units; no omitted target-family forms. Kera and Hasada separately qualified, unnamed other-Mundari -keda remains generic mu. Independent fresh20 and whole-about/alternate controls passed; 10 focused tests and actual 479-row parser pass, byte-identical regeneration. Durable identities unchanged. Full database/CLDF, compiled graph and browser gates deferred; new source outside original cohort.']
    with ledger.open('ab') as f: f.write(('\t'.join(row)+'\r\n').encode())
evidence={'status':'complete_source_stage_only','rows':479,'audit_units':466,
          'focused_tests':tests.strip(),'regeneration':'byte-identical to approved frozen CSV/audit',
          'installation_receipt_sha256':hashlib.sha256((P/'installation-report-20260927.json').read_bytes()).hexdigest(),
          'canonical_sha256':receipt['canonical_sha256'],'canonical_audit_sha256':receipt['audit_sha256'],
          'durable_identity_sha256_unchanged':receipt['durable_identity_sha256_unchanged'],
          'ledger_key':key,'deferred':['full database/CLDF build','compiled graph and identity integration','browser QA'],
          'no_commits_remotes_or_publication':True}
(P/'source-stage-validation-20260927.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(evidence,ensure_ascii=False,indent=2))

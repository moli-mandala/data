"""Record independent original-page decisions; never modify owner inputs."""
from pathlib import Path
import collections
import hashlib
import json

ROOT = Path(__file__).resolve().parents[7]
PACKAGE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
NATIVE = ROOT / 'tmp/jambu-next-source/ahirwati-native-grammar-20260927'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

child_path = PACKAGE / 'grammar/child-witnesses-first-reading.jsonl'
parent_path = PACKAGE / 'grammar/parent-coverage-first-reading.jsonl'
assert sha(child_path) == '96ccc5261f3ddca49a5fe96b08675f800d64173c7d590a7eda4048882cb6f6d7'
assert sha(parent_path) == '53ab13bfe22a064db76cb551ceae2e61e850abc6163ebf9c346773098f0e7f22'
children, parents = rows(child_path), rows(parent_path)
assert len(children) == 95 and len(parents) == 51
by_key = {r['witness_key']: r for r in children}
assert len(by_key) == 95
assert collections.Counter(k for p in parents for k in p['child_keys']) == collections.Counter(by_key.keys())
for parent in parents:
    assert all(by_key[k]['parent_key'] == parent['key'] for k in parent['child_keys'])

corrections = [
    {'key': 'lsi9.2:p50:nominal-ending-comparison-continuation:w3',
     'field': 'source_form', 'old': 'ē', 'new': 'ā',
     'evidence': 'peer-p50-nominal-ending-native.png',
     'reason': 'Native p50 explicitly prints an oblique singular in ā, contrasted with Bangaru ē. This is not the locative ē on p51.',
     'context_repair': 'Correct this parent source_context and its child source_quote summaries from Ahirwati oblique ē to ā; preserve first-reading history.'},
    {'key': 'lsi9.2:p51:qualified-oblique-relative-interrogative:w2',
     'field': 'source_form', 'old': 'kaun', 'new': 'kauṇ',
     'evidence': 'p51-first-reading-hypothesis-detail.png',
     'reason': 'Separate underdot is visible below the final n. The base word is printed; the following four obliques are explicitly unattested hypotheses.',
     'context_repair': 'Correct kaun to kauṇ in the parent and child context summaries without changing hypothetical dispositions.'},
]
for number in (4, 5):
    corrections.append({'key':f'lsi9.2:p50:nominal-ending-comparison-continuation:w{number}',
        'field':'language_label','old':'Bagri','new':'Bangaru',
        'evidence':'peer-p50-nominal-ending-native.png',
        'reason':'Original label is Bāṅgarū, with ṅ and medial ar sequence. This is distinct from the explicitly Bāgrī borrowing claim on p51.',
        'context_repair':'Replace Bagri with Bangaru only in the p50 nominal-ending parent and its child summaries.'})
for c in corrections:
    assert by_key[c['key']][c['field']] == c['old']
    c['evidence_sha256'] = sha(NATIVE / c['evidence'])

coverage_additions = [
    {'parent_key': 'lsi9.2:p50:nominal-ending-comparison-continuation', 'source_form': 'ō',
     'language_label': 'Mewati', 'source_analysis': 'strong masculine noun nominative singular ending',
     'physical_token': 'shared with existing w2', 'source_evidence': 'thus agreeing with Mēwātī',
     'decision': 'Add explicitly shared comparator interpretation, with blank lexical gloss.'},
    {'parent_key': 'lsi9.2:p50:nominal-ending-comparison-continuation', 'source_form': 'ā',
     'language_label': 'Mewati', 'source_analysis': 'strong masculine noun oblique singular ending',
     'physical_token': 'shared with corrected existing w3', 'source_evidence': 'thus agreeing with Mēwātī',
     'decision': 'Add explicitly shared comparator interpretation, with blank lexical gloss.'},
    {'parent_key': 'lsi9.2:p51:genitive-suffix', 'source_form': 'kō',
     'language_label': 'Mewati', 'source_analysis': 'genitive suffix',
     'physical_token': 'shared with existing w1', 'source_evidence': 'The genitive suffix is kō, as in Mēwātī.',
     'decision': 'Add explicitly shared comparator interpretation, with blank lexical gloss.'},
]

resolved = {
    'lsi9.2:p51:occasional-neuter-example:w1': 'Separate round dot over initial i supports diṇū̃; retain occasional-neuter source qualification, archive the former quantity-review hold.',
    'lsi9.2:p51:demonstrative-this-oblique:w1': 'Printed aĩh has the nasal mark over i and no separate length bar over first a.',
    'lsi9.2:p51:demonstrative-that-oblique:w1': 'Printed waĩh has the nasal mark over i and no separate length bar over a.',
    'lsi9.2:p51:qualified-oblique-relative-interrogative:w3': 'Printed jaĩh supported as a source hypothesis only; retain unattested disposition.',
    'lsi9.2:p51:qualified-oblique-relative-interrogative:w5': 'Printed kaĩh supported as a source hypothesis only; retain unattested disposition.',
}
greek = 'lsi9.2:p49:Ahir-history-6:w1'
correction_keys = {c['key'] for c in corrections}
reviews = []
for r in children:
    evidence = f"printed-{r['printed_page']}-pdf-{r['printed_page']+17}-Im{r['printed_page']+17}.png"
    status = 'literal_correction_required' if r['witness_key'] in correction_keys else ('documentary_greek_typography_hold' if r['witness_key'] == greek else 'original_agrees_with_first_reading')
    reviews.append({'key': r['witness_key'], 'parent_key': r['parent_key'], 'printed_page': r['printed_page'],
        'input_record_sha256': hashlib.sha256(json.dumps(r, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
        'reviewed_fields': ['source_form', 'source_gloss', 'source_analysis', 'language_label', 'disposition', 'qualification'],
        'status': status, 'evidence': evidence, 'evidence_sha256': sha(NATIVE/evidence),
        'typed_hold_decision': resolved.get(r['witness_key'], 'Preserve existing source qualification and historical reading.')})
review_path = OUT/'all-95-keyed-review.jsonl'
review_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in reviews))
images = [f'printed-{p}-pdf-{p+17}-Im{p+17}.png' for p in (49,50,51)]
images += ['p51-first-reading-nominals.png','p51-first-reading-pronouns.png','p51-first-reading-copulas.png','p51-first-reading-passive-detail.png','p51-first-reading-hypothesis-detail.png','p51-first-reading-own-correct-span.png','peer-p50-nominal-ending-native.png','peer-p49-greek-name-native.png','peer-p51-neuter-native.png']
report = {
    'reviewer': '/root/asur_finish', 'status': 'complete_original_review_corrections_required',
    'review_date': '2026-09-27', 'scope': 'Every native original page 49–51, all 51 physical census parents and all 95 owner child interpretations; not a sample audit.',
    'inputs': {str(p.relative_to(ROOT)): sha(p) for p in [child_path,parent_path,PACKAGE/'scope/native-grammar-image-manifest.json',PACKAGE/'grammar/first-reading-crop-provenance.json']},
    'native_evidence': [{'path': str((NATIVE/p).relative_to(ROOT)), 'sha256': sha(NATIVE/p)} for p in images],
    'counts': {'parents':51,'child_interpretations':95,'literal_corrections':2,'language_label_corrections':2,'unchanged_literal_and_attribution_agreements':90,'documentary_greek_typography_holds':1,'explicit_unattested_obliques':4,'recommended_shared_comparator_interpretations':3},
    'dispositions_as_received': dict(collections.Counter(r['disposition'] for r in children)),
    'corrections': corrections, 'completeness_recommendations': coverage_additions,
    'resolved_transcription_holds': resolved,
    'retained_hold': {'key':greek,'reason':'Ptolemy Greek ethnonym stays documentary with typography uncertainty; no target lexical routing or variant edge.'},
    'source_claims': [
      'Historical Abhira/Bhil etymology is explicitly tentative; retain documentary attribution, no modern genealogy or borrowed edge inferred.',
      'Gurgaon is the grammatical standard; Bawal/Kot Kasam regional sō/sī/sā must preserve locality qualification without guessed gender associations.',
      'Personal pronouns and conjugation generally as in Mewati do not license copying an unprinted Mewati paradigm.',
      'Four proposed jaĩh/jā̃h/kaĩh/kā̃h are explicitly unencountered by the author and must remain audit-only.',
      'Past thō and the future-passive construction share a form but have distinct source occurrence/function scope.',
      'Second-plural saĩ is explicitly said to be borrowed from Bagri. Preserve source-attributed borrowing; any donor occurrence or edge needs exact identified endpoint. The source claim is not a new separately printed donor token.',
      'gō/gā and sai-gō/hai-gā are explicit Ahirwati/Panjabi comparisons, not automatic borrowing edges.',
      'Bound endings ē/ō/ai/i/nai/nē̃/kō are source-attested morphology, not abstract a-base notation; keep appropriate structured functions.',
      'Whole mhārē ghari and tū-nē̃ karᵃṇō thō and their atom interpretations share printed spans; do not count them as separately printed occurrences.',
      'No invented English person translations for the present paradigm; printed person/number labels are analysis, not lexical gloss.',
      'Population, ethnographic names, bibliography absence and unprinted Gurmukhi specimens remain documentary; parent closure covers all source blocks.'
    ],
    'review_limits': 'This first-reading review does not certify final CSV tags, profiles, identity routing, graph implementation, parser, full build or browser gates. No owner inputs or canonical files changed.',
    'keyed_review_path': str(review_path.relative_to(ROOT)), 'keyed_review_sha256':sha(review_path),
}
(OUT/'independent-full-grammar-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'counts':report['counts'],'report_sha256':sha(OUT/'independent-full-grammar-review.json')},ensure_ascii=False))

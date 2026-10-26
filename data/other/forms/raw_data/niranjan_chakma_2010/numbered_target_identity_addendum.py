"""Keyed decisions supporting the research addendum; no input mutations."""
import hashlib
import json
from collections import Counter
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    baseline = P / 'numbered-language-mapping-proposal.json'
    original = json.loads(baseline.read_text())
    decisions = []
    for r in original['comparators']:
        label = r['lookup_label']
        if label not in {'প্রা', 'প্রাক', 'চে', 'মু', 'সান', 'কাই', 'বৈ'}:
            continue
        base, status, caveat = None, 'unresolved lect retained', r['decision_basis']
        proposed_record = None
        if label in {'প্রা', 'প্রাক'}:
            base, status = 'Pk', 'existing broad base supported by actual registry usage'
            caveat = 'The source labels this comparison Prakrit without specifying a variety. No subtype, place or Maharashtri attribution inferred.'
        elif label == 'চে':
            base, status = 'Chenchu', 'named language resolved; new registry approval required'
            caveat = 'Source-declared Chenchu; not collapsed into Telugu. Original compact-glyph review remains separate.'
            proposed_record = {'ID': 'Chenchu', 'Name': 'Chenchu', 'Glottocode': 'chen1255',
                               'Clade': 'S. Dravidian II', 'Latitude': '', 'Longitude': '',
                               'ISO_639_3_evidence': 'cde'}
        elif label == 'বৈ':
            base, status = 'Sk', 'base supported subject to preserving Vedic source stage'
            caveat = 'The source attributes this comparison to Vedic. No early/late stage, text or date is specified.'
        elif label == 'মু':
            status = 'group reference only; parent decision needed for Mu emission'
            caveat = 'Printed Munda does not establish Mundari or a reconstructed proto-form. Mu is an existing group reference, not a resolved spoken lect.'
        elif label == 'সান':
            caveat = 'Printed Sawar has Sora as a possible exonym match, but source provenance does not exclude other Sabar speech communities.'
        elif label == 'কাই':
            caveat = 'Kaithi/Kayethi is not a uniquely resolved language; script-name and Bihar evidence do not select a lect.'
        decisions.append({'comparison_key': r['comparison_key'], 'raw_label': r['raw_label'],
                          'literal_form': r['literal_form'], 'prior_base_id': r['proposed_base_id'],
                          'proposed_base_id': base, 'decision_status': status,
                          'source_scope_note': caveat, 'proposed_registry_record': proposed_record,
                          'candidate_only_ids': ['Mu'] if label == 'মু' else (['so'] if label == 'সান' else []),
                          'accepted_for_installation': False})
    assert len(decisions) == 40
    assert sum(x['proposed_base_id'] == 'Pk' for x in decisions) == 21
    assert sum(x['proposed_base_id'] == 'Chenchu' for x in decisions) == 4
    assert sum(x['proposed_base_id'] == 'Sk' for x in decisions) == 5
    report = {'date': '2026-09-27', 'reviewer': 'asur_finish',
              'status': 'research proposals; parent adjudication and any registration remain pending',
              'inputs': {baseline.name: sha(baseline),
                         'numbered-structured-review.jsonl': sha(P / 'numbered-structured-review.jsonl'),
                         'numbered-target-identity-research-addendum.md': sha(P / 'numbered-target-identity-research-addendum.md')},
              'local_primary_evidence': ['data/cdial/abbrevs.py: Pk=Prakrit; Mu=Munda',
                                         'cldf/dialects.csv: multiple Prakrit varieties under Pk',
                                         'data/other/forms/20260914-sheth.csv: general Prakrit usage',
                                         'cldf/languages.csv: separate Mu and mu; Chenchu absent'],
              'counts': dict(Counter(x['decision_status'] for x in decisions)),
              'decisions': decisions,
              'unchanged': 'All source literals, declarations, first mapping proposal and canonical data remain unchanged.'}
    (P / 'numbered-target-identity-research-addendum.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['counts']))


if __name__ == '__main__':
    main()

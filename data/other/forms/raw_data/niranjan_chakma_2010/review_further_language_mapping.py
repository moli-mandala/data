"""Build keyed identity proposals; never mutate first readings or the registry."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REGISTRY = ROOT.parents[4] / 'cldf/languages.csv'
INPUT = ROOT / 'further-reviewed-reading-proposal.jsonl'
DIRECT = {
    'অসমীয়া': 'A', 'রাজবংশী': 'MH', 'সংস্কৃত': 'Sk', 'সং.': 'Sk',
    'তামিল': 'Tamil', 'সিংহলী': 'Si', 'হিন্দি': 'H', 'উর্দু': 'H',
    'মুন্ডারি': 'mu', 'প্রাচীন বাংলা': 'OB', 'আঞ্চলিক বাংলা': 'B',
    'পালি': 'Pa', 'কুরুখ': 'Kurux', 'সাঁওতালি': 'sa', 'হাজং': 'Hajong',
    'হো': 'ho', 'বাংলা': 'B', 'নেপালি': 'N', 'মৈথিলী': 'Mth', 'সাদরি': 'Sadri',
}
NOTES = {
    'হিন্দি': 'Existing combined Hindi-Urdu record; preserve explicit Hindi attribution.',
    'উর্দু': 'Existing combined Hindi-Urdu record; preserve explicit Urdu attribution.',
    'আঞ্চলিক বাংলা': 'Bengali base only; unspecified regional variety retained, no invented locality.',
    'প্রাচীন বাংলা': 'Retain source historical-stage attribution; do not add dating.',
    'মুন্ডা': 'Group label does not prove Mundari; neither mu nor group node Mu assigned.',
    'প্রাকৃত': 'Unspecified Prakrit; registry Pk has maha1305, which would add unsupported specificity.',
    'বৈদিক': 'Vedic attribution retained; precise Sanskrit-stage metadata needs review before mapping.',
    'কায়েথী': 'Unresolved lect. Kaithi is also a script used for multiple languages; no unique identity follows.',
    'কায়েথী (বিহার)': 'Bihar narrows provenance but not the language; do not map to unrelated kaithal.',
    'কাকমারা (দ্রাবিড়)': 'Bhowmik describes Midnapore Kakmara as closely related to Telugu. This supports a research lead, but does not yet establish that the Chakma book refers to exactly that variety; retain named variety pending cross-source comparison.',
    'শবর': 'Historical/community label is insufficient to assign Sora automatically.',
    'বাংজবংশী': 'Possible source-label error; do not silently normalize to Rajbanshi.',
    'গুজরাট': 'Geographic label, likely Gujarati; explicit adjudication required.',
    'সেরবাদিয়া': 'No verified registry identity; preserve original attribution.',
    'চাঁই (পঃবঙ্গ)': 'Named West Bengal variety needs identity evidence; no automatic regional Bengali assignment.',
}
BROK = {'ব্রোক্সা', 'ব্রোক্সা (লাদাখ)'}
OUTSIDE = {'অহোম', 'তুর্কি', 'আরাকানি', 'ককবরক', 'চাক (মিয়ানমার)', 'তিব্বতি', 'মিজো', 'চাংনাগা'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with REGISTRY.open() as stream:
        registry = {r['ID']: r for r in csv.DictReader(stream)}
    rows = [json.loads(line) for line in INPUT.read_text().splitlines()]
    assert len(rows) == 84
    witnesses = [w for r in rows for w in r['witnesses'][1:] + r['embedded_comparisons']]
    assert len(witnesses) == 89
    records = []
    for w in witnesses:
        label = w['source_language_label']
        language = DIRECT.get(label)
        reason = NOTES.get(label, 'Explicit language name corresponds to existing registry record.')
        evidence = ['source printed language attribution', 'cldf/languages.csv']
        status = 'supported base-language proposal' if language else 'identity unresolved'
        if label in BROK:
            language = 'bro'
            status = 'context-supported base-language proposal'
            reason = 'Dedicated Brok-Skad chapter and Ladakh attribution support Brokskat; no new site or dialect inferred.'
            evidence.append('brok-skad-identity-review.json')
        elif label in OUTSIDE:
            status = 'comparison retained; outside requested language families'
            reason = 'Preserve comparison evidence without creating an IA, Dravidian or Munda lexical record.'
        elif not language and label not in NOTES:
            reason = 'No identity assignment yet; explicit external identity evidence required.'
        if label.startswith('কায়েথী'):
            evidence.append('https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-15/ (15.2)')
        if label == 'কাকমারা (দ্রাবিড়)':
            evidence.append('kakmara-identity-research-note.md')
        reg = registry[language] if language else None
        records.append({
            'witness_key': w['witness_key'], 'source_language_label': label,
            'reviewed_label_proposal': w.get('reviewed_label_proposal', label),
            'proposed_language_id': language, 'identity_status': status,
            'registry_evidence': {k: reg[k] for k in ('ID', 'Name', 'Glottocode', 'Clade')} if reg else None,
            'decision_basis': reason, 'evidence': evidence,
            'accepted_for_installation': False,
        })
    assert len({r['witness_key'] for r in records}) == 89
    labels = Counter(w['source_language_label'] for w in witnesses)
    output = ROOT / 'further-language-mapping-review.json'
    output.write_text(json.dumps({
        'date': '2026-09-27', 'reviewer': 'root',
        'status': 'complete witness-level identity inventory; unresolved identities and glyph adjudication remain',
        'inputs': {INPUT.name: sha(INPUT), 'cldf/languages.csv': sha(REGISTRY)},
        'coverage': {'physical_rows': 84, 'main_comparators': 84, 'embedded_comparators': 5,
                     'source_labels': dict(labels), 'identity_status_counts': dict(Counter(r['identity_status'] for r in records))},
        'policy': ['No ancestry or borrowing inferred from comparison.',
                   'No coordinates or precise dialects inferred from broad source labels.',
                   'Unresolved comparisons retained for completion; never silently excluded.',
                   'Identity proposals do not approve literal readings or installation.'],
        'witnesses': records, 'canonical_changes': False,
    }, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'output': output.name, 'sha256': sha(output), 'witnesses': len(records),
                      'status_counts': dict(Counter(r['identity_status'] for r in records))}))


if __name__ == '__main__':
    main()

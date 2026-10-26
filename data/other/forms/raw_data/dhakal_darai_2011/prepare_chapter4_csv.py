"""Review-only Chapter 4 export; intentionally has no installation command."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LECTS = {
    'Chitwan': 'dialect:Darai:darai_chitwan:Chitwan',
    'Tanahun': 'dialect:Darai:darai_tanahun:Tanahun',
    'Nawalparasi': 'dialect:Darai:darai_nawalparasi:Nawalparasi',
}


def build():
    records = [json.loads(line) for line in (HERE/'chapter4-metadata-proposal.jsonl').read_text().splitlines()]
    output = []
    for record in records:
        if record['disposition'] != 'proposed-attestation':
            continue
        grammar = record['proposed_grammar']
        notes = [record.get('proposed_notes', '')]
        if record.get('text_ids'):
            notes.append('Source corpus reference: '+', '.join(record['text_ids'])+'.')
        tags = [LECTS[record['lect']], *grammar['tags']]
        phonemic = record.get('proposed_phonemic', record['form'])
        # These literal accent shapes are attested but their phonetic meaning is
        # unresolved. Do not invent a replacement or claim a phonemic reading.
        if '\u0301' in record['form'] or 'í' in record['form'] or '\u033d' in record['form']:
            phonemic = ''
            tags.append('uncertain')
            notes.append('The original has an acute and an x-shaped mark above i; their phonological interpretation is unresolved.')
        if record['form'] == '-otạ':
            phonemic = ''
        notes = list(dict.fromkeys(note for note in notes if note))
        output.append([
            'Darai', '', record['form'], grammar['gloss'], '', phonemic,
            ' '.join(notes), f"dhakal2011darai[p. {record['printed_page']}]",
            '', '', record['source_unit_key'], '', '', '',
            ';'.join(dict.fromkeys(tags)),
        ])
    assert len(output) == 1282
    assert len({row[10] for row in output}) == len(output)
    return output


if __name__ == '__main__':
    rows = build()
    path = HERE/'chapter4-proposed.csv'
    with path.open('w', newline='') as handle:
        csv.writer(handle).writerows(rows)
    report = {
        'status': 'Review-only export; not installed or independently accepted',
        'rows': len(rows),
        'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'metadata_sha256': hashlib.sha256((HERE/'chapter4-metadata-proposal.jsonl').read_bytes()).hexdigest(),
        'pending': ['Source-profile validation, including literal unresolved marks',
                    'Register proposed Tanahun and Nawalparasi dialect tags after parent review',
                    'Independent original-versus-output review',
                    'Whole-thesis closure and all installation/downstream gates'],
        'canonical_changed': False,
    }
    (HERE/'chapter4-export-status-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report, ensure_ascii=False))

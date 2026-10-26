"""Account for all101 original numbered groups using reviewed physical boundaries."""
import json
from collections import Counter
from pathlib import Path
from record_brokskat_first_reading import sha

ROOT = Path(__file__).resolve().parent
# Pairs are (printed group number, column-separated OCR line number).
# Read against all11 original pages, including malformed OCR numerals and headless48.
BOUNDARIES = {
    (79, 'left'): [(1,1),(2,6),(3,17),(4,22)],
    (79, 'right'): [(5,4),(6,8),(7,13),(8,19)],
    (80, 'left'): [(9,4),(10,12),(11,25)],
    (80, 'right'): [(12,4),(13,8),(14,13),(15,19),(16,23)],
    (81, 'left'): [(17,1),(18,11),(19,18),(20,25)],
    (81, 'right'): [(21,3),(22,7),(23,11),(24,17),(25,22)],
    (82, 'left'): [(26,5),(27,10),(28,20),(29,24)],
    (82, 'right'): [(30,1),(31,6),(32,11),(33,16),(34,21),(35,25)],
    (83, 'left'): [(36,1),(37,7),(38,13),(39,20),(40,25),(41,28)],
    (83, 'right'): [(42,10),(43,16),(44,20),(45,27)],
    (84, 'left'): [(46,1),(47,6),(48,10),(49,17),(50,22),(51,26)],
    (84, 'right'): [(52,8),(53,13),(54,24)],
    (85, 'left'): [(55,1),(56,6),(57,14),(58,18),(59,22),(60,26)],
    (85, 'right'): [(61,1),(62,6),(63,13),(64,18),(65,22),(66,28)],
    (86, 'left'): [(67,3),(68,7),(69,12),(70,23),(71,28)],
    (86, 'right'): [(72,4),(73,9),(74,14),(75,26)],
    (87, 'left'): [(76,8),(77,15),(78,19),(79,24),(80,28)],
    (87, 'right'): [(81,5),(82,13),(83,17),(84,20),(85,24)],
    (88, 'left'): [(86,2),(87,25)],
    (88, 'right'): [(88,4),(89,10),(90,17),(91,21),(92,27)],
    (89, 'left'): [(93,4),(94,10),(95,20),(96,24)],
    (89, 'right'): [(97,1),(98,6),(99,10),(100,16),(101,21)],
}


def main():
    columns_path = ROOT / 'numbered-column-raw-evidence.jsonl'
    columns = list(map(json.loads, columns_path.read_text().splitlines()))
    assert [(c['pdf_page'], c['column']) for c in columns] == list(BOUNDARIES)
    assert [g for pairs in BOUNDARIES.values() for g, _ in pairs] == list(range(1,102))
    groups, current = {}, None
    identities = []
    for column in columns:
        page, side = column['pdf_page'], column['column']
        starts = {line: group for group, line in BOUNDARIES[page, side]}
        assert max(starts) <= len(column['lines'])
        for line in column['lines']:
            index = line['column_line_number']
            if index in starts:
                current = starts[index]
                groups[current] = {
                    'source_unit_key': f'niranjan2010chakma:numbered-comparison:group{current}',
                    'printed_group': current, 'start_pdf_page': page,
                    'start_printed_page': page-2, 'start_column': side,
                    'printed_head_present': current != 48,
                    'raw_fragments': [], 'accepted_for_installation': False,
                    'status': 'complete raw group allocation; original literal reading and witness parsing pending',
                    'source_claim_type': 'explicit origin analysis requiring literal review' if current in (65,87) else 'comparison, not an inferred ancestry edge',
                }
            assert current is not None
            groups[current]['raw_fragments'].append(dict(line, pdf_page=page, printed_page=page-2, column=side))
            identities.append((page, side, index))
    assert len(identities) == len(set(identities)) == sum(len(c['lines']) for c in columns)
    cross = [g for g,r in groups.items() if len({(f['pdf_page'],f['column']) for f in r['raw_fragments']}) > 1]
    assert cross == [4,8,11,20,25,41,51,66,71,75,80,85,87,92], cross
    target = ROOT / 'numbered-group-raw-inventory.jsonl'
    target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in groups.values()))
    status = {
        'status': 'all101 numbered groups have complete coordinate-linked raw evidence; not accepted lexical transcription',
        'groups': len(groups), 'column_fragments': len(identities),
        'raw_word_occurrences': sum(len(f['words']) for r in groups.values() for f in r['raw_fragments']),
        'cross_column_or_page_groups': cross, 'headless_groups': [48],
        'explicit_origin_groups': [65,87], 'source_blank_comparison_groups': [7,19],
        'fragment_counts': {g:len(r['raw_fragments']) for g,r in groups.items()},
        'inputs': {columns_path.name:sha(columns_path), Path(__file__).name:sha(Path(__file__))},
        'output': {target.name:sha(target)},
        'next': 'Full literal reading of every header, definition, witness, parenthesis and reference; contextual label mapping and hasanta review.',
        'installed_rows_changed': 0,
    }
    (ROOT/'numbered-group-inventory-status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(status, ensure_ascii=False))


if __name__ == '__main__':
    main()

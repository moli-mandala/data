"""Compiled graph assertions for sources subsequently annotated by reviewed sidecars."""
import csv
from pathlib import Path
from etymology_assignments import read_assignments

ROOT = Path(__file__).resolve().parents[1]


def assert_reviewed_source_graph(forms, source_edges=()):
    """No unreviewed outgoing edge; exact approved edges and linked status survive."""
    by_id = {r['ID']: r for r in forms}
    expected = set(source_edges)
    for row in read_assignments():
        if (row['Form_ID'] in by_id or row['Etymon_ID'] in by_id) and row.get('Status', 'accepted').lower() in {'accepted','yes','active'} and row['Etymon_ID']:
            expected.add((row['Form_ID'],row['Etymon_ID'],row['Kind'],row['Rank'] or '1',row.get('Pos','')))
    with (ROOT/'cldf/edges.csv').open() as stream:
        actual_rows = [(r['Child_ID'],r['Parent_ID'],r['Kind'],r['Rank'],r['Pos']) for r in csv.DictReader(stream) if r['Child_ID'] in by_id or r['Parent_ID'] in by_id]
    actual = set(actual_rows)
    assert len(actual_rows) == len(actual), 'Duplicate compiled edge tuples'
    assert actual == expected
    linked = {r[0] for r in expected}
    for fid, row in by_id.items():
        assert row['Status'] == ('' if fid in linked else 'unlinked')
        assert not row['Redirect']

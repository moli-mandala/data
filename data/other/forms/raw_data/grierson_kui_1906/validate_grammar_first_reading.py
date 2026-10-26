"""Small structural check, explicitly not a glyph or source-completeness audit."""
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

P = Path(__file__).parent
path = P / 'grammar-first-reading.jsonl'
rows = [json.loads(line) for line in path.read_text().splitlines()]
assert len({r['source_unit_key'] for r in rows}) == len(rows)
assert all(r['form'] == unicodedata.normalize('NFC', r['form']) for r in rows)
assert all(r['physical_page'] == r['printed_page']+20 for r in rows)
expected = {('noun-table-',462):18, ('pronoun-table1-',462):25,
            ('pronoun-table2-',462):31, ('principal-',462):50,
            ('principal-',463):25, ('personal-',463):57,
            ('copula-',463):24}
for (prefix,page),n in expected.items():
    found = sum(r['printed_page']==page and r['block'].startswith(prefix) for r in rows)
    assert found == n, (prefix,page,found,n)
empty = [r for r in rows if r.get('disposition')=='source-empty-cell']
assert len(empty)==16 and all(not r['form'] and not r['tags'] for r in empty)
assert all(r['form'] or r in empty for r in rows)
report = {'status':'structural checks passed; source review pending',
          'rows':len(rows), 'empty_source_cells':len(empty),
          'by_page':dict(Counter(r['printed_page'] for r in rows)),
          'typed_uncertainties':sum(bool(r['uncertainty']) for r in rows),
          'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
(P/'grammar-first-reading-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

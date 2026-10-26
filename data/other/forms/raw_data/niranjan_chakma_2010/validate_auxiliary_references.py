"""Validate work-only identities without asserting primary lexical consultation."""
import hashlib
import json
from pathlib import Path
from pybtex.database import parse_file

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[4]
mapping = json.loads((ROOT / 'numbered-reference-mapping-20260927.json').read_text())
bib_path = DATA / 'cldf/sources.bib'
bib = parse_file(str(bib_path))
checked = []
for work in mapping['works']:
    entry = bib.entries[work['key']]
    assert entry.type == 'misc'
    assert entry.fields['title'] == work['title']
    assert entry.fields['url'] == work['evidence_url']
    assert not ({'year', 'edition', 'publisher', 'pages'} & set(entry.fields))
    assert 'not independently collated' in entry.fields['note']
    assert 'no independent lexical ingestion' in entry.fields['included']
    assert entry.fields['etymology_provenance'] == 'none'
    assert work['independently_consulted_text'] is False
    checked.append({'key': work['key'], 'groups': work['groups'],
                    'status': 'registered work identity; secondary citation only'})
assert len(checked) == 4
result = {'checked': checked, 'sources_bib_sha256': hashlib.sha256(bib_path.read_bytes()).hexdigest(),
          'compiled_reference_generation': 'deferred with full pipeline',
          'lexical_witness': 'niranjan2010chakma with original page/group locator',
          'primary_passage_consultation': False}
(ROOT / 'auxiliary-reference-registration-20260927.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'registered_auxiliary_works': len(checked), 'validation': 'passed'}))

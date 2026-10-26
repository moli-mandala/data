"""Bounded source-local reference rendering; no shared bibliography writes."""
import hashlib
import json
from pathlib import Path
from pybtex.database import parse_file
from pybtex.style.formatting.plain import Style

HERE = Path(__file__).resolve().parent
bib = parse_file(str(HERE/'proposed-references.bib'))
rendered = {e.key:e.text.render_as('plaintext') for e in Style().format_entries(bib.entries.values())}
assert len(rendered)==2
assert 'Tribal Languages of Kerala' in rendered['nair2013mullukurumba']
assert 'A Descriptive Study of Mullu Kurumba' in rendered['robertsathyajoseph1982mullukurumba']
assert '1982' in rendered['robertsathyajoseph1982mullukurumba']
(HERE/'reference-format-review.json').write_text(json.dumps(dict(status='Both source-local Bib records format successfully',
    bibliography_sha256=hashlib.sha256((HERE/'proposed-references.bib').read_bytes()).hexdigest(),
    rendered=rendered, auxiliary_identity='Exact reported author/title/year/institution; no invented thesis type or thesis page attribution.',
    canonical_changes=False),ensure_ascii=False,indent=2)+'\n')
print(json.dumps(rendered,ensure_ascii=False))

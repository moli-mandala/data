"""Validate proposed reported authorities without changing the bibliography."""
import hashlib
import json
from pathlib import Path

from pybtex.database import parse_file
from pybtex.plugin import find_plugin

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
proposal = HERE / 'auxiliary-source-proposal.bib'
canonical = DATA / 'cldf/sources.bib'
new = parse_file(str(proposal))
old = parse_file(str(canonical))
assert set(new.entries) == {'abraham1995sadri', 'ctdp2002hillkorwa'}
additional = HERE / 'outside-review/additional-authorities-proposal.bib'
extra = parse_file(str(additional))
assert set(extra.entries) == {'powercity_korba_ndc', 'singh1994scheduledtribes'}
for key, entry in extra.entries.items():
    assert key not in new.entries
    new.add_entry(key, entry)
assert not set(new.entries).intersection(old.entries)
assert 'rensch-decker-hallberg1992' in old.entries
style = find_plugin('pybtex.style.formatting', 'plain')()
formatted = {entry.key: entry.text.render_as('text')
             for entry in style.format_bibliography(new)}
assert len(formatted) == 4
report = {
    'status': 'Draft references parse and format; no canonical changes',
    'proposal_sha256': hashlib.sha256(proposal.read_bytes()).hexdigest(),
    'additional_proposal_sha256': hashlib.sha256(additional.read_bytes()).hexdigest(),
    'canonical_sha256_at_check': hashlib.sha256(canonical.read_bytes()).hexdigest(),
    'formatted': formatted,
    'reused_methodology_reference': 'rensch-decker-hallberg1992',
    'primary_evidence': 'Behera printed104 bibliography; printed18 Abraham attribution',
    'limitation': 'Earlier originals not collated; use only as reported authorities',
}
(HERE / 'auxiliary-reference-check.json').write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report, ensure_ascii=False, indent=2))

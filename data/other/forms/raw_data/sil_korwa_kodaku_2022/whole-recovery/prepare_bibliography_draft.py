"""Create a full bibliography draft with a single replaced source and four additions."""
import hashlib
import json
from pathlib import Path

from pybtex.database import BibliographyData, parse_file, parse_string
from pybtex.plugin import find_plugin

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[5]
OUT = HERE / 'integration-proposal'
canonical = DATA / 'cldf/sources.bib'
raw = canonical.read_text()
before = parse_string(raw, 'bibtex')
key = 'behera2022korwakodaku'
entry = before.entries[key]
entry.fields['title'] = (
    'A Sociolinguistic Profile of {Korwa} and {Kodaku} Tribes in '
    '{Chhattisgarh} and {Jharkhand}, {India}')
entry.fields['included'] = (
    'Whole-report coverage: 6,864 rows comprising 4,459 repaired legacy Korwa/Kodaku rows '
    'and 2,405 additions from six language bases. All 25 Appendix B.5 lists, four Appendix C texts '
    'and outside lexical evidence are accounted for. Tanmai identity conflicts, 66 internal text-alignment '
    'spans, 20 outside language-attribution holds, documentary material, blank responses and two undefined '
    'site codes remain explicitly audited; their evidence is not silently discarded')
entry.fields['provenance'] = (
    'Pinned SIL PDF SHA-256 a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c; '
    '115 physical pages, recovered from the archived publisher PDF. The original PDF is not redistributed. '
    'Source evidence, reviewed transcriptions, per-record decisions and reproducible preparation scripts '
    'are under data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-recovery. '
    'Source CSVs: data/other/forms/20260828-sil-korwa-kodaku.csv and '
    'data/other/forms/20260927-behera-whole-additions.csv. Whole-source audit: '
    'data/other/forms/raw_data/sil_korwa_kodaku_2022/whole-source-audit.jsonl')
entry.fields['ocr'] = 'No'
entry.fields['etymology_provenance'] = 'source'
entry.fields['note'] = (
    'Fieldwork 2004--2005. Source-specific language and locality identities remain separate from speakers '
    'and test audiences. The Standard Sadri Brave Hunter text is reported from Abraham 1995 Appendix C-11, '
    'not independently collated. Birjia remains distinct from Birhor/Asuri/Korwa. Literal additions preserve '
    'field-transcription distinctions without inferred phonemic analysis; the legacy profile remains '
    'separate. New literal evidence uses the embedded text layer and original-page review; older OCR '
    'was a comparison aid, not installed transcription. Qualified introductory word-origin claims '
    'remain attributed commentary, not historical edges. Source conflicts and uncertain alignments '
    'are explicitly recorded')
replacement = BibliographyData(entries={key: entry}).to_string('bibtex')
start = raw.index('@techreport{' + key + ',')
end = raw.index('\n}', start) + 2
draft = raw[:start] + replacement.rstrip() + raw[end:]
proposals = [HERE / 'auxiliary-source-proposal.bib',
             HERE / 'outside-review/additional-authorities-proposal.bib']
added = []
for path in proposals:
    extra = parse_file(path)
    assert not set(extra.entries) & set(before.entries)
    added.extend(extra.entries)
    draft += '\n\n' + path.read_text().rstrip() + '\n'
after = parse_string(draft, 'bibtex')
original = parse_string(raw, 'bibtex')
assert set(after.entries) == set(original.entries) | set(added)
for old_key, old_entry in original.entries.items():
    if old_key == key:
        continue
    new = after.entries[old_key]
    assert old_entry.type == new.type and dict(old_entry.fields) == dict(new.fields)
    assert {role: [str(p) for p in people] for role, people in old_entry.persons.items()} == {
        role: [str(p) for p in people] for role, people in new.persons.items()}
style = find_plugin('pybtex.style.formatting', 'plain')()
formatted = {item.key: item.text.render_as('text') for item in style.format_bibliography(
    BibliographyData(entries={k: after.entries[k] for k in [key, *added]}))}
target = OUT / 'sources.bib'
target.write_text(draft)
report = {'status': 'Draft only; source scope awaits final acceptance',
          'canonical_sha256': hashlib.sha256(raw.encode()).hexdigest(),
          'draft_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'replaced_keys': [key], 'added_keys': added,
          'unrelated_entries_preserved': len(original.entries) - 1,
          'formatted': formatted, 'canonical_changes': False}
(OUT / 'bibliography-draft-check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'formatted'}, indent=2))

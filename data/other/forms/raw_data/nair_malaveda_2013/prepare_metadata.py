"""Source-local metadata proposals only; no registry or bibliography writes."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
META = HERE / 'metadata'
META.mkdir(exist_ok=True)
status = json.loads((HERE / 'proposal-status.json').read_text())
lect = next(csv.reader((HERE / 'proposed.csv').open()))[14].split()[-1]
with (META / 'proposed-language.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['ID', 'Name', 'Glottocode', 'Latitude', 'Longitude', 'Clade', 'Location', 'Quality'])
    writer.writerow(['Malaveda', 'Malaveda', '', '', '', 'S. Dravidian I', 'Kerala; source Veda/Malaveda examples have unresolved item-level locality', ''])
with (META / 'proposed-dialect.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['ID', 'Tag', 'Language_ID', 'Source_Language_ID', 'Name', 'Glottocode', 'Latitude', 'Longitude', 'Clade', 'Location', 'Quality'])
    writer.writerow(['nair2013_malaveda_unlocated', lect, 'Malaveda', 'nair2013-malaveda',
                     'Veda (Nair 2013; locality unresolved)', '', '', '', 'S. Dravidian I',
                     'Kerala; chapter contrasts Pathanamthitta 1977 and Kollam 1991 but does not locate individual examples', ''])
(META / 'proposed-settings.yaml').write_text('''file: 20260927-nair-malaveda.csv
defaults:
  transcription:
    convert: false
    preserve_hyphens: true
  identity:
    dedupe_by_entry_key: true
sources:
  nair2013malaveda:
    forms:
      split_alternates: false
    reference:
      editor: OpenAI Codex
      ocr: "No"
      etymology_provenance: source
''')
bib = '''@article{nair2013malaveda,
  author = {Nair, Ravi Sankar S.},
  title = {{Tribal Languages of Kerala}},
  journal = {Language in India},
  year = {2013},
  volume = {13},
  number = {7},
  month = {July},
  pages = {40--44},
  issn = {1930-2940},
  url = {https://languageinindia.com/july2013/ravisankarkeralatriballanguages.pdf},
  included = {Complete Malaveda/Veda chapter target forms, including morphology, paradigms, clauses, vocabulary and community terms. Ten comparison forms and five documentary labels are accounted for in the audit and excluded from canonical rows},
  note = {Original chapter heading prints MALADEVA/VEDA; contents prints Malaveda/Veda. Source warns that Veda/Malaveda/Vettuvar communities may differ and contrasts Pathanamthitta 1977 speech with Kollam 1991 speech. Individual examples are not explicitly assigned to a locality. No explicit reuse licence stated. Literal Roman transcription retained without inferred IPA},
  provenance = {data/other/forms/20260927-nair-malaveda.csv; data/other/forms/raw_data/nair_malaveda_2013/proposal-audit.jsonl},
  jambu_editor = {OpenAI Codex},
  ocr = {No},
  etymology_provenance = {source}
}

'''
reported = [
    ('vimalakumari1977veda', 'Vimalakumari, V. S.', 'Syntactic Analysis of Veda', '1977', 'M.A.'),
    ('vimala1984malaveda', 'Vimala, M.', 'Noun Morphology of Malaveda', '1984', 'M.A.'),
    ('kumarinirmala1977veda', 'Kumari Nirmala, B.', 'Internal Reconstruction of Veda Language', '1977', 'M.A.'),
    ('thankachi1977veda', 'Thankachi, R.', 'Noun Morphology of Veda Language', '1977', 'M.A.'),
    ('radhakrishnan1977veda', 'Radhakrishnan, S.', 'Verb Morphology of Veda Language', '1977', 'M.A.'),
    ('hyrunnisabeegam1991malaveda', 'Hyrunnisa Beegam, A.', 'Language of Malavedas', '1991', 'Ph.D.'),
]
for key, author, title, year, degree in reported:
    corrections = {
        'kumarinirmala1977veda': 'Internal Recounstruction of Veda Language',
        'thankachi1977veda': 'Noun Morophology of Veda Language',
    }
    correction_note = ''
    if key in corrections:
        correction_note = f". Nair's printed title is {corrections[key]}; the display title records an editorial spelling correction, not independent verification of the original thesis title"
    bib += f'''@misc{{{key},
  author = {{{author}}},
  title = {{{{{title}}}}},
  year = {{{year}}},
  note = {{Unpublished {degree} thesis, Department of Linguistics, University of Kerala. Reported in Nair 2013, p. 44; not independently consulted or used as a lexical attestation{correction_note}}},
  included = {{Reported auxiliary authority only}},
  jambu_editor = {{OpenAI Codex}},
  ocr = {{No}},
  etymology_provenance = {{none}}
}}

'''
(META / 'proposed-references.bib').write_text(bib)
(META / 'raw-title-evidence.json').write_text(json.dumps({
    'printed_and_pdf_page': 44,
    'reported_works_not_independently_consulted': True,
    'decisions': [
        {'key': 'kumarinirmala1977veda', 'printed_title': 'Internal Recounstruction of Veda Language',
         'display_title': 'Internal Reconstruction of Veda Language', 'decision': 'explicit editorial typo correction; raw title retained in bibliography note'},
        {'key': 'thankachi1977veda', 'printed_title': 'Noun Morophology of Veda Language',
         'display_title': 'Noun Morphology of Veda Language', 'decision': 'explicit editorial typo correction; raw title retained in bibliography note'},
    ],
}, ensure_ascii=False, indent=2) + '\n')
(META / 'identity-decision.json').write_text(json.dumps({
    'decision': 'Source-named Malaveda base with source-qualified Veda lect; blank Glottocode and coordinates',
    'source_claim': 'Nair p40 considers Pathanamthitta 1977 speech an independent Southern Dravidian language and Kollam 1991 speech close to Malayalam; examples are generically Veda.',
    'not_asserted': ['All named Veda/Malaveda/Vettuvar communities speak one language', 'Item-level Pathanamthitta or Kollam locality', 'Independent equivalence to Glottolog mala1463'],
    'independent_review': '../independent-identity-review-20260927.md',
    'source_mapping_uncertainty': 'Unresolved source-to-modern-language equivalence remains explicit in metadata and per-record audit.',
    'profile': 'convert false; preserve hyphens, spacing, lower dots, drawn lowlines, breve and doubled vowels; no IPA claim',
    'licence': 'No explicit redistribution licence in source; included material is extracted linguistic facts, not republished pages.',
    'scope': status,
}, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'proposed_metadata_files': 6, 'references': 7, 'canonical_changes': False}))

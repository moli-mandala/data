"""Prepare review-only Naiki metadata and a conservative Roman display profile."""
import csv
import hashlib
import json
import unicodedata as ud
from pathlib import Path

HERE = Path(__file__).resolve().parent


def clusters(text):
    current = ''
    for char in ud.normalize('NFC', text):
        if ud.combining(char) and current:
            current += char
        else:
            if current:
                yield current
            current = char
    if current:
        yield current


def main():
    rows = list(csv.reader((HERE / 'whole-review-proposal.csv').open()))
    graphemes = sorted({g for row in rows for g in clusters(row[2])})
    mapping = {}
    for grapheme in graphemes:
        display = '#' if grapheme == ' ' else grapheme.lower().replace('ṅ', 'ŋ')
        for normalization in ['NFC', 'NFD']:
            alias = ud.normalize(normalization, grapheme)
            assert alias not in mapping or mapping[alias] == display
            mapping[alias] = display
    (HERE / 'proposal-profile.txt').write_text('Grapheme\tIPA\n' + ''.join(
        f'{g}\t{mapping[g]}\n' for g in sorted(mapping)))
    decisions = dict(stage='review-only; no registry or canonical edits',
        source_rows=len(rows), input_graphemes=graphemes,
        mapping=mapping,
        basis='Original grammar pp570–571 and specimen spelling; conservative display conventions parallel the existing reviewed LSI Holiya/Kaikadi profiles.',
        policy=['Lowercase display initials while retaining exact source spelling as Original.',
                'Render printed velar nasal ṅ as house ŋ; preserve other consonant spellings and digraphs literally.',
                'Retain printed e/o quantities and the author’s conjectured quantities as separate qualified evidence; no harmonization.',
                'Preserve meaningful hyphens, spaces, aspiration spelling, doubled consonants and nasal diacritics.',
                'No separate IPA analysis supplied: Phonemic and Native remain blank.',
                'Whole-expression Original retains printed punctuation; the shared display converter trims terminal commas, semicolons and full stops only at expression boundaries.',
                'Canonical NFC/NFD equivalents must yield the same NFC display.'],
        proposal_sha256=hashlib.sha256((HERE / 'whole-review-proposal.csv').read_bytes()).hexdigest())
    (HERE / 'profile-decisions.json').write_text(json.dumps(decisions, ensure_ascii=False, indent=2)+'\n')
    (HERE / 'whole-review-proposal.yaml').write_text('''file: whole-review-proposal.csv
defaults:
  transcription:
    profile: grierson-naiki-1906
    input: form
    preserve_hyphens: true
  identity:
    dedupe_by_entry_key: true
  forms:
    split_alternates: false
  gloss:
    grammar_tags: false
sources:
  grierson1906naiki:
    identity:
      dedupe_by_entry_key: true
    forms:
      split_alternates: false
    gloss:
      grammar_tags: false
    reference:
      editor: OpenAI Codex
      ocr: true
      etymology_provenance: none
''')
    with (HERE / 'proposed-dialect.csv').open('w', newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['ID','Tag','Language_ID','Source_Language_ID','Name','Glottocode','Latitude','Longitude','Clade','Location','Quality'])
        writer.writerow(['naiki_chanda_lsi1906','dialect:Naiki:naiki_chanda_lsi1906:Chanda','Naiki','Naiki-Chanda-LSI1906','Chanda (LSI 1906)','','','','C. Dravidian','Historical Chanda district; exact collection locality unspecified',''])
    print(f'Prepared review profile for {len(rows)} rows, {len(graphemes)} graphemes; no canonical installation')


if __name__ == '__main__':
    main()

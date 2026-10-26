"""Owner original-image first reading of every lexical/morphological unit on173."""
import json
import unicodedata
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    records = []
    def add(key, form, gloss, tags, note=''):
        records.append({'source_unit_key': f'bailey1920northjubbal:p173:{key}',
                        'printed_page': 173, 'pdf_page': 199, 'lect': 'Barari',
                        'form': unicodedata.normalize('NFC', form), 'gloss': gloss,
                        'tags': tags.split(), 'source_analysis': note,
                        'status': 'owner-original-first-reading; independent review pending'})
    add('aspiration:to', 'khi', 'to', 'prep', 'Explicitly labelled Barari in shared phonology paragraph.')
    add('aspiration:from', 'khu', 'from', 'prep', 'Explicitly labelled Barari in shared phonology paragraph.')
    for index, (form, gloss, case) in enumerate([('rū','of','gen'), ('rā','of','gen'),
                                               ('khi','to','dat'), ('khu','from','abl'), ('du','from','abl')], 1):
        add(f'declension:{index}', form, gloss, f'postp {case}', 'Separate printed occurrence; grammatical relation explicitly named.')
    add('pronoun:remote', 'tīã', 'that', 'pron 3sg f obl demonstrative', 'Remote feminine singular oblique; first i macron and final a tilde.')
    add('pronoun:near', 'ī̃au', 'this', 'pron 3sg f obl demonstrative', 'Near feminine singular oblique; i carries macron/tilde stack. No extra mark supplied on final u.')
    for group, forms, gloss, extra in [
        ('here', ['īdā','īā','īchā'], 'here', ''),
        ('there', ['tēidā','tēā','tēchā'], 'there', ''),
        ('where-interrogative', ['kēidā','kēā','kēchā'], 'where', 'interr'),
        ('where-relative', ['jēidā','jēā','jēchā'], 'where', 'relative'),
    ]:
        for index, form in enumerate(forms, 1):
            add(f'place:{group}:{index}', form, gloss, f'adj {extra}',
                'Source calls these place adverbs but explicitly analyzes them as adjectives agreeing with the sentence nominative; no gender ending inferred.')
    for key, form, gloss in [('tomorrow','kāl','tomorrow'), ('day-after','pōrs̲h̲ī','the day after tomorrow'),
                             ('yesterday','hīz','yesterday'), ('day-before','phrēz','the day before yesterday')]:
        add(f'time:{key}', form, gloss, 'adv')
    add('copula:negative-present', 'anthi', 'not be', 'verb copula pres neg indecl', 'Source expressly labels the negative present indeclinable.')
    for index, form in enumerate(['lā','li','le','li'], 1):
        add(f'future-marker:{index}', form, 'future marker', 'fut',
            'Printed added element, with no hyphen inserted. Parenthesis repeats li; preserve both source positions without inventing gender/number assignments.')
    add('copula:past', 'thīā', 'be', 'verb copula pret', 'Past copula added to present form to make imperfect indicative.')
    assert len(records) == len({r['source_unit_key'] for r in records}) == 31
    report = {'status': 'page173 first original reading; not installed', 'reviewer': 'root',
              'evidence': 'Original page at3x, pronoun/adverb and verbal paragraphs additionally at8x.',
              'target_units': records,
              'controls': [{'source_unit_key': 'bailey1920northjubbal:p173:control:khe', 'form':'khe', 'gloss':'to', 'lect':'Bishshau', 'decision':'Other-lect control, not Barari'},
                           {'type':'sound-symbol discussion', 'forms':['h','‘'], 'decision':'Notation explanation, not free lexical items; Bishshau h is described as tone.'}],
              'pending': 'Independent literal/tag review and whole-source reconciliation. Repeated attestations retain separate source keys; this page is not the completed source.'}
    (OUT / 'intro-p173-first-reading-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(records)} source units recorded from printed173; no canonical writes')


if __name__ == '__main__':
    main()

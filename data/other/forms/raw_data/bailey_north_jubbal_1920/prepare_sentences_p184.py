"""Original-image first reading of final sentences and full commentary."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    rows = []
    def add(key, forms, gloss, tags, note=''):
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p184:{key}',
                         printed_page=184, pdf_page=210, forms=forms, gloss=gloss,
                         tags=tags, source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    for number, forms, gloss in [
        (19, ['Mu khu āgo călo'], 'Me from before walk.'),
        (20, ['Kauh rā tshōṭā tā’re pătshu āo'], 'Whom of boy your behind comes?'),
        (21, ['Tūē kau khe lauā mūlle'], 'By-you whom from was-taken in price.'),
        (22, ['Gaũŏ re dŭkāndāro du lauū', 'Gaũŏ re dŭkāndāro du gīnū'], 'Village of shopkeeper from was taken.'),
    ]:
        add(f'sentence:{number}', forms, gloss,
            ['sentential'] + (['interr'] if number in (20, 21) else []),
            'Complete expression and literal source translation. Sentence 21 is interrogative in content but source prints a period.' if number == 21 else
            'Printed or replaces final predicate; both complete expressions retained without a lexical variant edge.' if number == 22 else
            'Complete printed expression and literal translation retained.')
    add('note:2:printed', ['ō’le'], '', ['verb'],
        'Source cites sentence 2 and says final e and i are often interchanged.')
    add('note:2:possible', ['ō’li'], '', ['verb', 'uncertain'],
        'Source explicitly says this might be ō’li. Typed source-analysis uncertainty; no asserted variant edge.')
    add('note:7:actual', ['gauhro'], 'house', ['noun'],
        'Source attributes h to influence of neighbouring dialects, without identifying a specific donor.')
    add('note:7:comparison', ['gau’ro'], 'house', ['noun'],
        'Source says gauhro for gau’ro; no specific donor or ancestry edge claimed.')
    add('note:9', ['lāī'], '', ['verb'],
        'Agrees with an understood word for blow; comparative Hindi example retained only as control. No gender inferred here.')
    add('note:10', ['gōrū'], 'cattle', ['noun', 'collective'],
        'Explicit source characterization: collective word, cattle.')
    add('note:11', ['bōṭhondā'], 'seated', ['verb', 'participle', 'stative'],
        'Explicit source label stat. part.; tag registry reconciliation pending.')
    add('note:13', ['cījje ṭhănni'], 'two-and-a-half rupees', ['multiword'],
        'Gloss supported by sentence 13. Source cross-refers to the note on this sentence in the Kŭār dialect; resolve exact locator before export.')
    controls = [dict(forms=['lăgāī'], source_lect='Hindi', printed_page=184,
                     reason='Explicit Hindi comparison for sentence 9, excluded from North Jubbal target.')]
    report = dict(status='complete page owner first reading; independent audit pending',
                  reviewer='root', evidence='Original full page at 3x and complete target/commentary crop at 8x.',
                  source_units=rows, controls=controls, remaining_on_page=[],
                  unresolved=['Resolve Kŭār sentence 13 cross-reference.',
                              'Reconcile canonical tags and source-wide breath notation before export.'])
    assert len(rows) == 12 and sum(len(r['forms']) for r in rows) == 13
    (OUT / 'sentences-p184-first-reading-20260927.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('4 sentences and 8 commentary units / 13 forms; 1 Hindi control; no installation')


if __name__ == '__main__':
    main()

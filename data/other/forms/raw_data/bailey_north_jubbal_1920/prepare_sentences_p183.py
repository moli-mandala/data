"""Original-image first reading of all numbered sentences on printed page 183."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    data = [
        (['Tēro nāũ kā sŏ', 'Tēro nāũ kā ŏsso'], 'Thy name what is?'),
        (['Ē gō’ṛe ri kētti ŏmbăr ō’le'], 'This horse of how-much age will be?'),
        (['Ītthau Kăs̲h̲mīr kēti dūr ŏ'], 'From-here Kashmir how-much far is?'),
        (['Tā’re bappŏ re kētti lŏṛke'], 'Your father of how-many boys?'),
        (['Āũ ādz dūrŏ du hoṇḍĕo'], 'I to-day far from walked.'),
        (['Mēre kāke rā bēṭā tēhri bauhṇī sāthi dzādzṇā ō’ā'], 'My uncle of son his sister with married became.'),
        (['Gauhro dē tsīṭe dzīn gō’ṛe rī'], 'House in white saddle horse of.'),
        (['Ēhri pīṭṭhe gāi dzīn kŏs̲h̲au'], 'His back upon saddle tighten.'),
        (['Māẽ tēh re tshōṭe dī khūb lāī'], 'By-me him of boy on well attached-was (i.e. beat).'),
        (['Pai’ṛo rī tīro dī gōrū bē’ṛi tsāro'], 'Hill of top on cows sheep he-is-grazing.'),
        (['Ē bīkhlo nīṭhā gō’ṛe gāī bōṭhondā thīa'], 'He tree under horse upon seated was.'),
        (['Ēh rā bā’i ăpṇī bauhṇī du jēṭhā'], 'This of brother own sister than elder.'),
        (['Ēh rā mūl cījje ṭhănni'], 'This-of price two-and-a-half rupees'),
        (['Mērā bāp tshōṭe gau’ro dī rauo'], 'My father small house in remains (lives).'),
        (['Ē rŭpŏye dēo'], 'Him-to rupees give.'),
        (['Ĕs du ōrū māṅgu rupŏye'], 'Him from hither ask rupees.'),
        (['Ĕs khūb pīṭĕau rŏs̲h̲io dā baunho'], 'Him well having-beaten ropes with tie.'),
        (['Kūe dū pāṇī ōrū gāṛo'], 'Well from water hither draw.'),
    ]
    rows = []
    for number, (forms, gloss) in enumerate(data, 1):
        note = 'Complete printed expression; source literal translation retained.'
        if number == 1:
            note += ' Printed or substitutes final copula; two full expressions expanded without claiming a lexical variant edge.'
        if number == 13:
            note += ' Printed translation ends with (see note); corresponding commentary still to be reconciled.'
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p183:sentence:{number}',
                         printed_page=183, pdf_page=209, item=number, forms=forms,
                         gloss=gloss, tags=['sentential'] + (['interr'] if number <= 4 else []),
                         source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    assert len(rows) == 18 and sum(len(row['forms']) for row in rows) == 19
    report = dict(status='complete page owner first reading; independent audit pending',
                  reviewer='root', evidence='Original full page at 3x and three 8x crops covering all 18 sentences.',
                  source_units=rows, controls=[], remaining_on_page=[],
                  unresolved=['Source-wide breath typography reconciliation remains pending.',
                              'Sentence 13 commentary must be joined from following page.'])
    (OUT / 'sentences-p183-first-reading-20260927.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('18 sentence units / 19 full expressions; independent audit pending; no installation')


if __name__ == '__main__':
    main()

"""Preserve40 printed paradigm cells and auditable shorthand expansions."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'
CASES = [('nom-acc','Nom., Acc.',['nom','acc']), ('gen','Gen.',['gen']),
         ('dat','Dat.',['dat']), ('abl','Abl.',['abl']), ('agent','Agent',['erg'])]


def main():
    paradigms = [
        ('horse','m','gō’ṛ',
         ['gō’ṛ-o','-e ru','-e khi','-e khu or du','-e'],
         ['-e','-e ru','-e khi','-e khu or du','-e']),
        ('bear','m','rīkh',
         ['rīkh-','-ŏ ru','-ŏ khi','-ŏ khu or du','-ē'],
         ['rīkh-','-ŏ ru','-ŏ khi','-ŏ khu or du','-ē']),
        ('daughter','f','dī’',
         ['dī’-ī','-iŏ rā','-iŏ khi','-iŏ khu or du','-iŏ'],
         ['-ī','-ī ru','-ī khi','-ī khu or du','-iē']),
        ('sister','f','bauih',
         ['bauih-ṇ','-ṇī ru','-ṇī khi','-ṇī khu or du','-ṇī'],
         ['-ṇī','-ṇī ru','-ṇī khi','-ṇī khu or du','-ṇīē']),
    ]
    cells = []
    for gloss, gender, stem, singular, plural in paradigms:
        for number, patterns in [('sg',singular),('pl',plural)]:
            for (case, label, tags), pattern in zip(CASES, patterns):
                if pattern.startswith('-'):
                    expansion = stem + pattern[1:]
                else:
                    expansion = pattern.replace('-', '')
                # The second ablative postposition replaces khu, sharing the
                # explicitly supplied oblique noun form rather than bare du.
                if ' or du' in expansion:
                    first = expansion.removesuffix(' or du')
                    assert first.endswith(' khu')
                    forms = [first, first[:-3] + 'du']
                else:
                    forms = [expansion]
                cells.append({'source_unit_key':f'bailey1920northjubbal:p175:noun:{gloss}:{number}:{case}',
                              'printed_page':175, 'pdf_page':201, 'source_stem':stem,
                              'source_pattern':pattern, 'source_case':label,
                              'source_number':number, 'gloss':gloss,
                              'tags':['noun',gender,number] + tags,
                              'expanded_form_proposals':forms,
                              'review':'owner-original-first-reading; independent expansion and glyph review pending',
                              'note':'Horse meaning is explicitly supplied by the same source on172.' if gloss == 'horse' else ''})
    assert len(cells) == len({c['source_unit_key'] for c in cells}) == 40
    assert sum(len(c['expanded_form_proposals']) for c in cells) == 48
    report = {'status':'review-only40cell inventory; no canonical changes', 'reviewer':'root',
              'evidence':'Original175 full page at3x; separate masculine/feminine crops at8x.',
              'cells':cells,
              'expansion_policy':'Preserve exact table fragments in audit. Expansion replaces only explicit stem hyphen; identical forms under different case/number prompts retain separate keys. Combined Nom./Acc. label remains one source cell. Both ablative postpositions retain full oblique noun.',
              'case_policy':'Source Agent label retained verbatim; proposed erg tag requires independent review before installation.',
              'notation_policy':'Right-curving raised breath sign recorded as U+2019 pending explicit reconciliation with old glossary convention. Breves, macrons and underdots preserved.',
              'controls':[{'scope':'Above NORTH JUBBAL OR BARARI heading',
                           'items':['continued Bishshau ability sentence','Bishshau pronominal suffix s','Bishshau temporal expressions'],
                           'decision':'Continues preceding Bishshau section despite page running head; excluded from Barari.'}]}
    (OUT / 'nouns-p175-first-reading-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('40 source cells /48expanded proposals; other-lect continuation excluded; no installation')


if __name__ == '__main__':
    main()

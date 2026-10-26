"""Literal owner reading of original page 177; source-local review only."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    rows = []
    def add(key, forms, gloss, tags, note=''):
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p177:{key}',
                         printed_page=177, pdf_page=203, forms=forms, gloss=gloss,
                         tags=tags, source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    add('relative-feminine-gen', ['jīõ ru'], 'who (relative)', ['pron','rel','f','sg','gen'],
        'Continuation of preceding page. Printed etc. does not license invented case forms. Final o mark read as tilde; independent review pending.')
    add('what', ['kā'], 'what', ['pron','interr'])
    add('something', ['kīcch'], 'something', ['pron'])
    for key, form, gloss, tags in [
        ('this-kind','īṇo','of this kind',[]), ('that-kind','tīṇo','of that kind',[]),
        ('what-kind','kīṇo','of what kind?',['interr']), ('which-kind','jīṇo','of which kind (rel.)',['rel']),
        ('this-quantity','ētī','so much or many',[]), ('that-quantity','tētī','so much or many (correl.)',[]),
        ('what-quantity','kētī','how much or many?',['interr']), ('relative-quantity','jētī','as much or many (rel.)',['rel']),
    ]:
        add(key,[form],gloss,['adj']+tags,'Source heading: Pronominal Adjectives.')
    add('adjective-ending-a',['-ā'],'',['suffix','adj'], 'Agreement class ending; no separate lexical translation supplied.')
    add('adjective-ending-e',['-e'],'',['suffix','adj','m','sg','pl'],
        'Source says masc. sing. and all masc. plur.; preserve wording without silently repairing it.')
    add('adjective-ending-i',['-i'],'',['suffix','adj','f','sg','pl'],'Source feminine ending in both numbers.')
    add('adjective-ending-e-alternative',['e'],'',['suffix','adj','f'],
        'Source says this i sometimes changes to e; literal e has no printed hyphen. No extra paradigm generated.')
    add('comparison-du',['du'],'',['postp'], 'Source explicitly identifies du as comparison marker used with positive; no distinct comparative/superlative form.')
    add('comparison-positive',['ē tāto ŏ'],'this is hot',['sentential'])
    add('comparison-comparative',['ē ĕs du tāto ŏ'],'this is hot from this, hotter than this',['sentential'])
    add('comparison-superlative',['ē bādlhe du tāto ŏ'],'this is hot from all, hotter than all, hottest',['sentential'])
    for key, forms, gloss, tags, note in [
        ('now',['ēbṛe','ēbbī'],'now',[],''),
        ('then',['taubṛe','tŏbbe'],'',[],'Printed without English gloss. Position suggests then, but no translation supplied as fact.'),
        ('when',['kaubṛe','kŏbbe'],'when?',['interr'],''),
        ('when-relative',['jaubṛe','jŏbbe'],'when (rel.)',['rel'],''),
        ('today',['ādz'],'to-day',[],''), ('tomorrow',['kāl'],'to-morrow',[],''),
        ('day-after',['pōrs̲h̲ī'],'day after to-morrow',[],'Printed underlined sh retained literally.'),
        ('fourth-day',['tsōuthe'],'on fourth day',[],''),
        ('yesterday',['hīz'],'yesterday',[],''),
        ('day-before',['phrēz'],'yesterday, day before',[],'Printed English word order retained.'),
        ('fourth-day-back',['tsōuthe'],'on fourth day back',[],'Distinct printed occurrence and meaning retained despite identical form above.'),
        ('sometimes',['kŏbbe'],'sometimes',[],''),
        ('sometime',['kŏbe kŏbe'],'some time or other; sometimes',['multiword-expression'], 'Source prints one b per repeated component, unlike adjacent kŏbbe.') ,
        ('never',['kŏbbe na'],'never',['multiword-expression','neg'],''),
    ]:
        add('time-'+key, forms, gloss, ['adv']+tags,note)
    assert len(rows) == 33
    assert len({r['source_unit_key'] for r in rows}) == len(rows)
    report = dict(status='complete page owner first reading; independent audit pending',
                  reviewer='root', evidence='Original full page at 3x and three overlapping 8x crops.',
                  source_units=rows, remaining_on_page=[],
                  policy='Literal text and explicit alternatives retained; no paradigms generated from etc. No phonemic interpretation supplied.')
    (OUT/'misc-p177-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(rows)} units / {sum(len(r['forms']) for r in rows)} forms; no installation")


if __name__ == '__main__':
    main()

"""Owner first reading of complete original page 178; not canonical data."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    rows = []
    def add(key, forms, gloss, tags, note=''):
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p178:{key}',
                         printed_page=178,pdf_page=204,forms=forms,gloss=gloss,
                         tags=tags,source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    for key, forms, gloss, tags in [
        ('here',['īdā','īā','īchā'],'here',[]),
        ('there',['tēidā','tēā','tēchā'],'there',[]),
        ('where',['kēidā','kēū','kēā','kēchā'],'where?',['interr']),
        ('where-relative',['jēiḍā','jēā','jēchā'],'where (rel.)',['rel']),
    ]:
        add('place-'+key, forms,gloss,['adv','adj']+tags,
            'Each printed form is asterisked: source explicitly says these adverbs agree with subject. Asterisk is annotation, not form character. Relative first form has apparent underdot on d; preserve pending independent audit.')
    for key, forms, gloss in [
        ('up-to-here',['īchā tāĩ'],'up to here'), ('from-here',['ītthau'],'from here'),
        ('inside',['bītre'],'inside'), ('outside',['bāhre'],'outside'),
        ('up',['ūbhā'],'upwards'), ('down',['ūṭā'],'downwards'),
        ('near',['nēṛo'],'near'), ('far',['dūr'],'far'),
        ('forward',['gŏū̃','āgo'],'forward, in front'), ('backward',['tshŏū̃'],'backwards'),
        ('beyond',['pare','pōrū'],'beyond'), ('this-side',['wār','ōrū'],'on this side'),
    ]:
        tags = ['adv'] + (['multiword-expression'] if any(' ' in f for f in forms) else [])
        add('place-'+key,forms,gloss,tags,
            'Stacked macron and tilde on final u require independent glyph audit.' if key in ('forward','backward') else '')
    add('other-why',['kĕōi'],'why?',['adv','interr'])
    add('other-quickly',['phēṭī'],'quickly',['adv'])
    add('other-yes',['ō'],'yes',['interj'])
    add('other-well',['khūb','accho'],'well',['adv'])
    add('discourse-thither',['pōrū'],'thither',['adv'],'Explicit gloss in prose; source says often used with little meaning.')
    add('discourse-hither',['ōrū'],'hither',['adv'],'Explicit gloss in prose; source says often used with little meaning.')
    add('example-ask',['māṅg ōrū'],'ask hither, ask for it',['sentential','imp'])
    add('example-give-him',['dē pōrū'],'give thither, give it to him',['sentential','imp'])
    add('example-give-me',['dē ōrū'],'give hither, give it to me',['sentential','imp'])
    for key, forms, gloss in [
        ('to',['khi'],'to'), ('from-khu',['khū'],'from'), ('from-du',['dū'],'from'),
        ('other-side',['pāre'],'on the other side of'), ('this-side',['wār'],'on this side of'),
        ('behind',['pătshu'],'behind'), ('front',['āgo'],'in front of'), ('with',['dā'],'with (instru.)'),
        ('beside',['khe'],'beside'), ('along',['sāthi','sātthe'],'along with'),
        ('sake',['khi','khe'],'for sake of'), ('in',['dā','dē'],'in'),
        ('upon',['gāī'],'upon'), ('below',['nīṭhā'],'below'), ('in-mandz',['māndz'],'in'),
    ]:
        add('postposition-'+key,forms,gloss,['postp'],
            'Printed heading Prepositions; source explicitly says normally after nouns and pronouns.')
    for key, forms in [
        ('left-1',['ŏsso','ŏ']), ('left-2',['ŏssĕ','ŏ']), ('left-3',['ŏsso','ŏ']),
        ('right-1',['ŏsso','ŏ']), ('right-2',['ŏsso','ŏ']), ('right-3',['ŏssōĕ','ōĕ']),
    ]:
        add('copula-present-'+key,forms,'be',['verb','prs'],
            'Source two-column three-row paradigm has no person or number labels on this page. Cell position retained; person/number assignment deferred pending adjacent paradigm evidence.')
    assert len({r['source_unit_key'] for r in rows}) == len(rows)
    report=dict(status='complete page owner first reading; independent audit pending',reviewer='root',
                evidence='Original full page at 3x and overlapping 8x crops.',source_units=rows,
                remaining_on_page=[],controls=[
                    'Most adjectives can be used as adverbs: descriptive rule, no generated lexical forms.',
                    'Asterisk agreement rule retained as metadata on each affected unit.',
                ])
    (OUT/'misc-p178-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(rows)} units / {sum(len(r['forms']) for r in rows)} forms; no installation")


if __name__ == '__main__':
    main()

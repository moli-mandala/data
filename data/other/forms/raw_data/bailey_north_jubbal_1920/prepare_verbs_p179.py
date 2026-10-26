"""Owner transcription of complete original page 179, with explicit expansions."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    rows=[]
    def add(key, forms, gloss, tags, note='', raw=None):
        row=dict(source_unit_key=f'bailey1920northjubbal:p179:{key}',printed_page=179,
                 pdf_page=205,forms=forms,gloss=gloss,tags=tags,source_analysis=note,
                 review='owner-original-first-reading; not installed')
        if raw is not None:
            row['raw_printed_fragments']=raw
        rows.append(row)
    add('copula-negative',['naĩ anthi','nī anthi'],'be',['verb','neg','multiword-expression'],
        'Source says indeclinable. First i upper mark requires independent comparison with dot/tilde; literal preferred reading retained.')
    for column, masculine in [('left','thīā'),('right','thīe')]:
        for pos in range(1,4):
            for gender,form in [('m',masculine),('f','thī')]:
                add(f'copula-past-{column}-{pos}-{gender}',[form],'be',['verb','pst',gender],
                    'Two columns and three rows lack printed person/number labels; position retained pending paradigm-wide review.')
    add('fall-heading',['pŏṛno'],'fall',['verb'])
    add('fall-imperative-left',['pŏṛ'],'fall',['verb','imp'])
    add('fall-imperative-right',['pŏṛo','pŏṛuu'],'fall',['verb','imp'],
        'Second alternative literally uu, without length marks in original crop.')
    for column,endings in [('left',['ū','e','au']),('right',['ū','au','au'])]:
        for pos,ending in enumerate(endings,1):
            raw='pŏṛ-ū' if column=='left' and pos==1 else '-'+ending
            add(f'fall-present-{column}-{pos}',['pŏṛ'+ending],'fall',['verb','prs'],
                'Source heading jointly present indicative and present conditional. Stem expanded from explicit pŏṛ-; no extra paradigm forms generated.',[raw])
    for pos,ending in enumerate(['-lā','-li','-le','-li'],1):
        add(f'future-marker-{pos}',[ending],'',['suffix','fut'],
            'Four positions explicitly printed; duplicate li occurrence retained. No inferred gender label.')
    for column,pairs in [('left',[('ūlā','ūli'),('ēlā','ēli'),('ōlā','ōli')]),
                         ('right',[('ūle','ūli'),('ōle','ōli'),('ōle','ōli')])]:
        for pos,pair in enumerate(pairs,1):
            for gender,ending in zip(['m','f'],pair):
                raw='pŏṛ-ūlā' if (column,pos,gender)==('left',1,'m') else '-'+ending
                add(f'fall-future-{column}-{pos}-{gender}',['pŏṛ'+ending],'fall',['verb','fut',gender],
                    'Expanded only the printed stem abbreviation; raw fragment retained.',[raw])
    for key,form,tags in [
        ('1sg-m','pŏṛū thā',['1sg','m']),('1sg-f','pŏṛū thi',['1sg','f']),
        ('2sg-m','pŏṛe thā',['2sg','m']),('2sg-f','pŏṛe thī',['2sg','f']),
    ]:
        add('fall-imperfect-'+key,[form],'fall',['verb','ipfv','pst','multiword-expression']+tags,
            'Only explicitly printed examples captured, not the forms implied by etc. First feminine thi has plain i in original; second has macron.')
    for tense,forms in [('past',['pŏṛā','pŏṛe','pŏṛe','pŏṛī']),
                        ('past-conditional',['pŏṛdā','pŏṛde','pŏṛde','pŏṛdī'])]:
        for label,form in zip(['sg-m','sg-f','pl-m','pl-f'],forms):
            number,gender=label.split('-')
            add('fall-'+tense+'-'+label,[form],'fall',['verb','pst',number,gender]+(['cond'] if tense.endswith('conditional') else []))
    add('fall-pluperfect-m',['pŏṛā thā'],'fall',['verb','plup','m','multiword-expression'],'Explicit example only; etc. not expanded.')
    add('fall-pluperfect-f',['pŏṛe thī'],'fall',['verb','plup','f','multiword-expression'],'Explicit example only; etc. not expanded.')
    add('fall-conjunctive',['pŏṛĕau'],'having fallen',['verb','converb'],
        'Breve on e preferred from original crop; independent glyph review pending.')
    for key,forms,tags in [
        ('heading',['ō’ṇo'],[]),('imperative-left',['ō’'],['imp']),('imperative-right',['ō’au'],['imp']),
        ('present',['ō’ū'],['prs','ind']),('future',['ō’ūlu'],['fut']),
        ('past-conditional',['ū’ndā'],['pst','cond']),('past',['ū’ō'],['pst']),
    ]:
        add('become-'+key,forms,'be; become',['verb']+tags,
            'Raised breath mark transcribed U+2019 pending source-wide reconciliation; no phonemic claim.')
    for key,forms,tags in [
        ('heading',['āṇo'],[]),('imperative-left',['ā'],['imp']),('imperative-right',['āo'],['imp']),
        ('negative-imperative',['nai ō','nĭ aiō'],['neg','imp','multiword-expression']),
        ('past-conditional',['āndā'],['pst','cond']),('past',['āā','ās̲h̲ā'],['pst']),
    ]:
        add('come-'+key,forms,'come',['verb']+tags,
            'Negative imperative checked in narrow original crop: first nai has dotted i, second nĭ breve; do not silently harmonize with copula negative.' if key=='negative-imperative' else '')
    assert len({r['source_unit_key'] for r in rows})==len(rows)
    report=dict(status='complete page owner first reading; independent audit pending',reviewer='root',
                evidence='Original full page at 3x, upper/lower at 8x, negative imperative narrow crop at 8x.',
                source_units=rows,remaining_on_page=[],
                policy='Printed abbreviation expansions recorded with raw fragments; no generation from etc.; unlabeled table positions retained without invented person labels.')
    (OUT/'verbs-p179-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(rows)} units / {sum(len(r['forms']) for r in rows)} forms; no installation")


if __name__=='__main__':
    main()

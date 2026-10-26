"""Original page 180 owner reading, prior to independent audit."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'whole-source-recovery'


def main():
    rows=[]
    def add(key,forms,gloss,tags,note=''):
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p180:{key}',
                         printed_page=180,pdf_page=206,forms=forms,gloss=gloss,
                         tags=tags,source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    groups=[
        ('go-action','go',[
            ('heading',['ḍēūṇo'],[]),('imperative-left',['ḍēo'],['imp']),('imperative-right',['ḍēo'],['imp']),
            ('future-left-1',['ḍēūlā','ḍēūlau'],['fut']),('future-left-2',['ḍēuela'],['fut']),
            ('future-left-3',['ḍēōla'],['fut']),('future-right-1',['ḍēūle'],['fut']),
            ('future-right-2',['ḍēōle'],['fut']),('future-right-3',['ḍēōlē'],['fut']),
            ('past-conditional',['ḍēūnda'],['pst','cond']),('past',['ḍēūa'],['pst']),
        ]),
        ('go-composition','go',[
            ('heading',['jāṇo'],[]),('imperative-left',['jā'],['imp']),('imperative-right',['jāo'],['imp']),
            ('future',['jāūlā'],['fut']),('past-conditional',['jāndā'],['pst','cond']),('past',['gōā'],['pst']),
        ]),
        ('remain','remain',[
            ('heading',['rauhṇo'],[]),('future',['rauhūla'],['fut']),
        ]),
        ('sit','sit',[
            ('heading',['būṭhṇo'],[]),('imperative-left',['būṭh'],['imp']),('imperative-right',['buṭṭho'],['imp']),
            ('past-conditional',['buṭhdā'],['pst','cond']),('future',['buṭṭhūlā'],['fut']),
            ('past',['bōṭṭhā'],['pst']),
        ]),
        ('beat','beat',[
            ('heading',['pīṭṇo'],[]),('imperative-left',['pīṭ'],['imp']),('imperative-right',['pīṭā'],['imp']),
            ('present',['pīṭū'],['prs']),('future',['pīṭūla'],['fut']),('imperfect',['pīṭū thā'],['pst','ipfv']),
            ('past-conditional',['pīṭdā'],['pst','cond']),('past',['pīṭā'],['pst']),
            ('present-perfect',['pīṭā ai'],['prs','prf']),('pluperfect',['pīṭā thā'],['plup']),
        ]),
        ('eat','eat',[('heading',['khāṇo'],[])]),
    ]
    for group,gloss,entries in groups:
        for key,forms,tags in entries:
            note=''
            if group=='go-composition':
                note='Source explicitly says used in composition with other verbs.'
            if group=='go-action' and key.startswith('future'):
                note='Unlabeled table position retained; no invented person assignment. Printed vowel lengths preserved, including distinct final e/ē.'
            if group=='beat' and key=='present':
                note='Source explicitly says present indicative or conditional; both analyses retained in this note.'
            add(group+'-'+key,forms,gloss,['verb']+tags+(['multiword-expression'] if any(' ' in f for f in forms) else []),note)
    add('sit-stative',['bōṭhŏndā'],'in the state of having sat, seated',['verb','ptcp'],
        'Source label Stat. part.; single ṭ in this form contrasts printed double ṭ in preceding past.')
    add('eat-stative',['khāṛū'],'in the state of having been eaten',['verb','ptcp','pass'],
        'Source label Stat. part. and explicit passive English translation; not an inferred active meaning.')
    add('beat-conjugation-reference',['pŏṛno'],'fall',['verb'],
        'Explicit cross-reference to fall conjugation. Do not generate unprinted beat forms; source separately states past transitive verbs agree with subject.')
    assert len({r['source_unit_key'] for r in rows})==len(rows)
    report=dict(status='complete page owner first reading; independent audit pending',reviewer='root',
                evidence='Original full page at 3x; upper and lower paradigms at 8x.',
                source_units=rows,remaining_on_page=[],
                controls=['Conjugation-same-as rule does not license unattested paradigm generation.',
                          'Source statement about past transitive subject agreement preserved without editorial correction.'])
    (OUT/'verbs-p180-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(rows)} units / {sum(len(r['forms']) for r in rows)} forms; no installation")


if __name__=='__main__':
    main()

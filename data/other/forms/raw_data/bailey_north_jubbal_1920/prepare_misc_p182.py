"""Owner original reading of complete ability and numeral page 182."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'whole-source-recovery'


def main():
    rows=[]
    def add(key,forms,gloss,tags,note=''):
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p182:{key}',printed_page=182,
                         pdf_page=208,forms=forms,gloss=gloss,tags=tags,source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    for key,forms,gloss,neg in [
        ('give',['mēre na dēūo'],'I cannot give',True),
        ('read',['ĕsre na pŏṛūo'],'he cannot read',True),
        ('read-book',['ēdza kĭtāb mēre na pŏṛīda','ēdza kĭtāb mēre na pŏṛūa'],'I cannot read this book',True),
        ('eat-bread',['mēre na rōṭī khāīndi'],'I cannot eat bread',True),
        ('eat-rice',['bāt mēre khāīo'],'I can eat rice',False),
    ]:
        add('ability-'+key,forms,gloss,['sentential']+(['neg'] if neg else []),
            'Whole printed expression retained. For read-book, source or replaces final predicate only; both full expressions recorded.' if key=='read-book' else '')
    for key,form,note in [
        ('ending-ido','-īdo','One of two stated passive-participle endings.'),
        ('ending-uo','-ūo','One of two stated passive-participle endings.'),
        ('ending-io','-īo','Explicit exceptional ending of khāīo.'),
        ('ending-ida','īda','Printed without hyphen in prose; vowel-final roots insert n.'),
    ]:
        add(key,[form],'',['suffix','participle','pass'],note)
    add('inserted-n',['n'],'',['morphological-marker'], 'Explicit inserted segment in source analysis, not a lexical word.')
    add('passive-khaio',['khāīo'],'',['verb','participle','pass'],'Explicit citation of exceptional passive-participle form; gloss inherited only from full sentence, left blank here.')
    add('passive-khainda',['khāīndā'],'',['verb','participle','pass'],'Source example illustrating inserted n; no generated paradigm.')
    for key,form,gloss,gender in [('bread','rōṭī','bread','f'),('book','kĭtāb','book','m'),('rice','bāt','rice','m')]:
        add('gender-'+key,[form],gloss,['noun',gender],'Source explicitly identifies gender in prose; gloss supported by preceding translated sentence.')
    cardinal=['ēk','dūī','cīn','tsār','pānc','tshau','sāt','āṭh','nau','daus̲h̲','gĕro',
              'bāro','ṭēro','tsŏūdau','pŏndrau','sōḷau','sŏttrau','ṭhārau','ŭṇīs̲h̲','bīs̲h̲']
    for n,form in enumerate(cardinal,1):
        add('cardinal-'+str(n),[form],str(n),['num'])
    add('cardinal-100',['s̲h̲au'],'100',['num'])
    ordinal=[['paihlau'],['dŭjjau','dŭjjā'],['cījjā'],['tsōuthā'],['panjūau'],
             ['tshŏūau'],['sātūau'],['āṭhūau'],['nŏūau'],['dŏs̲h̲ūau']]
    for n,forms in enumerate(ordinal,1):
        add('ordinal-'+str(n),forms,str(n),['num','ordinal'])
    add('fraction-1.5',['dēōṛh'],'1½',['num'])
    add('fraction-2.5',['ḍā’e'],'2½',['num'],'Raised mark retained as U+2019 pending source-wide breath typography reconciliation.')
    controls=[
        dict(forms=['khāīdā'],source_lect='Panjabi',reason='Explicit comparative control, not North Jubbal target.'),
        dict(forms=['khāīṇā'],gloss='to be eaten',source_lect='Panjabi',reason='Explicit comparative passive verb, not target lexical assertion.'),
    ]
    assert len({r['source_unit_key'] for r in rows})==len(rows)
    report=dict(status='complete page owner first reading; independent audit pending',reviewer='root',
                evidence='Original page3x; ability and full numeral tables in8x crops.',source_units=rows,
                controls=controls,remaining_on_page=[])
    (OUT/'misc-p182-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(rows)} target units / {sum(len(r['forms']) for r in rows)} forms,2 comparative controls; no installation")


if __name__=='__main__':
    main()

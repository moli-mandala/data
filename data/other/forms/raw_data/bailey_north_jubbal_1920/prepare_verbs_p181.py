"""Complete page 181 owner original reading; installation remains gated."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'whole-source-recovery'


def main():
    rows=[]
    def add(key,forms,gloss,tags,note=''):
        rows.append(dict(source_unit_key=f'bailey1920northjubbal:p181:{key}',printed_page=181,
                         pdf_page=207,forms=forms,gloss=gloss,tags=tags,source_analysis=note,
                         review='owner-original-first-reading; not installed'))
    for group,gloss,entries in [
        ('drink','drink',[('heading','pīṇo',[]),('stative','pīēṛu',['participle'])]),
        ('give','give',[('heading','dēṇo',[]),('future','dēūla',['fut']),('past-conditional','dīnda',['pst','cond']),
                        ('past','dīṇā',['pst']),('stative','dēēṛu',['participle'])]),
        ('take','take',[('heading','lauṇo',[]),('future','lauūla',['fut']),('past-conditional','laundā',['pst','cond']),
                        ('stative','lauēṛu',['participle'])]),
        ('do-breath','do',[('heading','ē’rno',[]),('past-conditional','ē’ddā',['pst','cond']),('past','ē’ro',['pst'])]),
        ('do-k','do',[('heading','kŏrno',[]),('past-conditional','kŏrdā',['pst','cond']),('past','kīō',['pst'])]),
        ('know','knew',[('heading','jāṇṇo',[]),('past-conditional','jāṇdā',['pst','cond'])]),
        ('bring','bring',[('heading','āṇṇo',[]),('past-conditional','aṇdā',['pst','cond']),('past','āṇō',['pst'])]),
        ('take-away','take away',[('heading','nīṇo',[]),('past-conditional','nīndā',['pst','cond']),
                                 ('past','nīō',['pst']),('stative','nīēṛu',['participle'])]),
    ]:
        for key,form,tags in entries:
            note=''
            if group=='know':
                note='Printed English heading is knew; do not silently modernize to know.'
            if group=='do-breath':
                note='Raised breath mark represented as U+2019 pending source-wide typographic reconciliation.'
            add(group+'-'+key,[form],gloss,['verb']+tags,note)
    add('stative-ending',['ēṛu'],'',['suffix','participle'],
        'Source says ending can be separated and placed before root; no printed leading hyphen here.')
    for key,left,right,gloss in [('drink','ēṛu pīe','pīēṛu','drunk'),
                               ('give','ēṛu dē','dēēṛu','given'),
                               ('take','ēṛū laue','lauēṛu','taken')]:
        add('stative-fronted-'+key,[left],gloss,['verb','participle','multiword-expression'],
            'Explicitly paired by source with '+right+'. Take example has printed final ū in fronted ending; preserve occurrence-specific quantity.')
        add('stative-joined-'+key,[right],gloss,['verb','participle'],
            'Explicitly paired by source with '+left+'. Separate attestation and explicit gloss retained.')
    add('purpose-khi',['tsārne khi nēo'],'take to graze',['multiword-expression'],
        'Source literal gloss: grazing for take-away. No unprinted words supplied.')
    add('purpose-go',['tsārde ḍēo'],'take to graze',['multiword-expression'],
        'Source literal gloss: grazing go. Preserved as printed, not harmonized with preceding phrase.')
    add('uncontracted-question',['kauh rā ō’lā ēdzā tshōṭā'],'whose is this boy?',['sentential','interr'],
        'Source explicit uncontracted counterpart; breath-mark orientation pending source-wide reconciliation.')
    add('contracted-question',['kauhra lādza tshōṭā'],'whose is this boy?',['sentential','interr'],
        'Source explicitly describes this as rapid-speech pronunciation of preceding question; preserve both complete expressions.')
    assert len({r['source_unit_key'] for r in rows})==len(rows)
    report=dict(status='complete page owner first reading; independent audit pending',reviewer='root',
                evidence='Original full page at3x and overlapping top/bottom 8x crops.',source_units=rows,
                remaining_on_page=[],controls=['Negative-sentence past-conditional rule is descriptive; no unattested negative forms generated.'])
    (OUT/'verbs-p181-first-reading-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{len(rows)} units / {sum(len(r['forms']) for r in rows)} forms; no installation")


if __name__=='__main__':
    main()

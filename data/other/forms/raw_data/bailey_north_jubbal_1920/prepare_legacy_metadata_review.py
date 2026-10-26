"""Keyed metadata reconciliation after original pp185–188 review; no installation."""
import hashlib
import json
from pathlib import Path
import import_source

HERE = Path(__file__).resolve().parent


def main():
    rows, audit = import_source.generate()
    changes = {}
    by_key = {r[10]: r for r in rows}
    def amend(page, item, answer=1, *, tags=(), gloss=None, note=None, reason):
        key=f'bailey1920northjubbal:p{page}:item:{item}' + (f':answer{answer}' if answer > 1 else '')
        row=by_key[key]
        record=changes.setdefault(key, dict(entry_key=key, before=dict(gloss=row[3],notes=row[6],tags=row[14]),
                                            proposed=dict(gloss=row[3],notes=row[6],tags=row[14]), reasons=[]))
        target=record['proposed']
        target['tags']=' '.join(dict.fromkeys(target['tags'].split()+list(tags)))
        if gloss is not None: target['gloss']=gloss
        if note: target['notes']=' '.join(filter(None,[target['notes'],note]))
        record['reasons'].append(reason)
    amend(185,35,2,tags=['collective'],reason='Printed col., cattle explicitly qualifies second answer.')
    amend(186,51,3,tags=['correlative'],reason='Printed correl. explicitly qualifies tētī.')
    amend(186,1,2,gloss='go',note='Source restricts this form to use in composition.',
          reason='Usage restriction belongs in source commentary rather than lexical gloss.')
    amend(185,28,note='Source explicitly contrasts its printed ending with -no.',reason='Retain source notation warning beside the accepted form.')
    amend(186,7,note='Source explicitly contrasts the final dental n with retroflex ṇ.',reason='Preserve source parenthetical spelling distinction.')
    for page,item in [(186,51),(188,9)]:
        amend(page,item,note='Source explicitly warns that the printed plain r is not retroflex ṛ.',reason='Preserve source parenthetical rhotic distinction.')
    amend(187,2,note='Source explicitly contrasts plain r with retroflex ṛ.',reason='Preserve source parenthetical rhotic distinction.')
    amend(185,62,note='Source cross-reference: see leg (p. 186, vocabulary item 34).',reason='Recover explicit foot-to-leg reference; not a synonym or variant edge.')
    for answer in [1,2]:
        amend(187,33,answer,note='Age is relative to the person referred to.',reason='Retain explicit scope of older/younger sister distinction.')
    # Explicit numeral glosses; no inferred lexical senses or extra forms.
    numerals={185:[48,49,51,58,61,63,64],186:[19,56,57,67],187:[24,25,35,36,52,59,62],188:[2,3,4]}
    for row in rows:
        key=row[10]
        parts=key.split(':')
        page=int(parts[1][1:]); item=int(parts[3]); answer=int(parts[4][6:]) if len(parts)>4 else 1
        if item in numerals.get(page,[]) and 'num' not in row[14].split():
            amend(page,item,answer,tags=['num'],reason='Printed cardinal/fraction/ordinal sense explicitly identifies numeral category.')
        if ' ' in row[2] and 'multiword-expression' not in row[14].split():
            amend(page,item,answer,tags=['multiword-expression'],reason='Retain scope of the complete printed multiword expression.')
    report=dict(status='owner original-page metadata review; independent review pending; no literal or canonical changes',
                reviewer='root', evidence='All four original glossary pages185–188 inspected at3x; compared complete243-cell ledger.',
                canonical_sha256=hashlib.sha256(import_source.OUTPUT.read_bytes()).hexdigest(),
                changes=list(changes.values()),
                audit_only=[dict(source_cell_key='bailey1920northjubbal:p185:item:1',
                                 reason='Printed above has two cross-references and no new form.',
                                 targets=['bailey1920northjubbal:p188:item:8','bailey1920northjubbal:p188:item:7'])],
                breath_policy='Earlier glossary uses U+2018 for source raised breath typography; newer drafts use U+2019. Preserve audit history and both literal conventions in profile pending explicit final reconciliation.',
                pending='Independent review of this keyed metadata delta; whole-source output audit.')
    (HERE/'whole-source-recovery/legacy-metadata-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(changes)} keyed metadata corrections, one recovered cross-reference record; no canonical edits')


if __name__ == '__main__':
    main()

"""Apply keyed independent readings; retain the immutable first-reading evidence."""
import csv
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent / 'whole-source-recovery'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (P/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    review_path = P/'independent-parable-p101-p104-review-20260927.json'
    review = json.loads(review_path.read_text())
    draft = P/'parable-aligned-first-reading.tsv'
    assert sha(draft) == review['input_sha256']
    rows = list(csv.DictReader(draft.open(), delimiter='\t'))
    indexed = {(int(row['page']), int(row['line'])): row for row in rows}
    for correction in review['corrections']:
        row = indexed[correction['page'], correction['line']]
        forms = row['forms'].split('|')
        index = correction['word']-1
        assert forms[index] == correction['old_form'], correction
        forms[index] = correction['form']
        row['forms'] = '|'.join(forms)
    assert len(review['corrections']) == 28
    # Historical comments remain first-reading evidence, not unresolved flags.
    units, lookup = [], {}
    for row in rows:
        page, line = int(row['page']), int(row['line'])
        forms, glosses = row['forms'].split('|'), row['glosses'].split('|')
        assert len(forms) == len(glosses)
        for word, (form, gloss) in enumerate(zip(forms, glosses), 1):
            key = f'bailey1920kagani:whole:p{page}:parable:line{line}:word{word}'
            unit = dict(source_unit_key=key, printed_page=page, pdf_page=page+26,
                line=line, token=word, raw_form=form, raw_alignment=gloss,
                first_reading_comment=row['comment'],
                review='Independent original glyph/alignment review applied; grammatical integration pending',
                review_report=review_path.name)
            lookup[page,line,word] = len(units)
            units.append(unit)
    assert len(units) == 362
    (P/'parable-aligned-reviewed.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
    # All printed parenthetical witnesses are independently aligned records.
    anchors = [(102,1,1), (102,3,1), (102,4,5), (102,9,4), (103,1,1), (104,11,4)]
    parentheticals = []
    for number, (evidence, anchor) in enumerate(zip(review['parenthetical_evidence'], anchors), 1):
        page,line,word = anchor
        assert (evidence['page'], evidence['line']) == (page,line)
        target = units[lookup[anchor]]
        parentheticals.append(dict(source_unit_key=target['source_unit_key']+':parenthetical:1',
            printed_page=page,pdf_page=page+26,line=line,after_token=word,
            raw_form=evidence['form'],raw_alignment=evidence['gloss'],
            source_analysis=evidence['scope'],anchor_key=target['source_unit_key'],
            relation='printed lexical alternative' if number in (1,2) else 'distinct aligned parenthetical witness',
            review_report=review_path.name))
    assert len(parentheticals) == 6
    write('parable-parentheticals-reviewed.json', parentheticals)
    # Source punctuation, quotations, and finite clause boundaries; never split at a page break.
    ends = [(101,1,6),(101,4,3),(101,5,2),(101,7,7),(101,9,2),
            (102,1,3),(102,2,1),(102,5,3),(102,12,5),(102,13,5),
            (103,2,6),(103,9,4),(104,2,3),(104,4,1),(104,9,7),(104,15,2)]
    spans, start = [], 0
    for number, end in enumerate(ends, 1):
        stop = lookup[end]+1
        selected = units[start:stop]
        assert selected
        spans.append(dict(source_unit_key=f'bailey1920kagani:whole:parable:expression{number}',
            first_key=selected[0]['source_unit_key'],last_key=selected[-1]['source_unit_key'],
            primary_token_keys=[u['source_unit_key'] for u in selected],
            parenthetical_keys=[p['source_unit_key'] for p in parentheticals if p['anchor_key'] in {u['source_unit_key'] for u in selected}],
            printed_pages=sorted({u['printed_page'] for u in selected}),
            tags=['multiword-expression','sentential'],
            translation_policy='No free translation invented: preserve concatenated printed interlinear gloss and label it as such. Keep parenthetical forms with their own distinct glosses.',
            review='Root original-page punctuation and clause-scope reading; final output audit pending'))
        start = stop
    assert start == 362
    write('parable-expression-spans-reviewed.json', spans)
    write('parable-review-application-20260927.json', dict(
        status='independent literal review applied; whole-source integration pending',
        original_draft_sha256=sha(draft), independent_review_sha256=sha(review_path),
        literal_corrections=28, primary_tokens=362, parenthetical_witnesses=6,
        contiguous_expression_spans=len(spans),
        historical_comments='Draft comments retained as evidence only; resolution is in the independent review report.',
        outputs={name:sha(P/name) for name in ['parable-aligned-reviewed.jsonl','parable-parentheticals-reviewed.json','parable-expression-spans-reviewed.json']}))
    print('Applied28 independent corrections;362 primary tokens,6 parentheticals,16 contiguous expressions; no installation.')


if __name__ == '__main__':
    main()

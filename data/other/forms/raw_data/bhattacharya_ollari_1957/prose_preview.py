"""Export reviewed source prose for keyed sidecar integration; does not install."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIELDS = ['Form_ID', 'Entry_Key', 'Position', 'Kind', 'Format', 'Content', 'Source']


def build(output):
    rows, audit = [], []
    reference_ledger = json.loads((HERE / 'explicit-reference-resolution.json').read_text())
    references = {(r['entry_key'], r['position']): r for r in reference_ledger['records']}
    if len(references) != len(reference_ledger['records']):
        raise ValueError('Duplicate passage-level reference assignment')
    used_references = set()
    for page in range(48, 78):
        lexical_path = HERE / f'reviewed-p{page}.json'
        lexical = json.loads(lexical_path.read_text())
        review_path = HERE / f'comparison-reviewed-p{page}.json'
        review = json.loads(review_path.read_text())
        if review['lexical_record_sha256'] != hashlib.sha256(lexical_path.read_bytes()).hexdigest():
            raise ValueError(f'Stale prose evidence on page {page}')
        if [r['entry_key'] for r in lexical] != [r['entry_key'] for r in review['records']]:
            raise ValueError(f'Prose coverage differs on page {page}')
        for source, record in zip(lexical, review['records']):
            start = len(rows)
            for passage in record['passages']:
                target = (record['entry_key'], passage['position'])
                citation = f"bhattacharya1957ollari[p. {page}, col. {source['column']}]"
                if target in references:
                    citation += ';' + references[target]['citation']
                    used_references.add(target)
                rows.append(dict(Form_ID='', Entry_Key=record['entry_key'],
                                 Position=passage['position'], Kind=passage['kind'],
                                 Format='text', Content=passage['text'],
                                 Source=citation))
            audit.append(dict(entry_key=record['entry_key'], printed_page=page,
                              row_start=start, row_count=len(rows)-start,
                              review_file=review_path.name,
                              review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
                              review_record=record,
                              auxiliary_reference_review=[references[(record['entry_key'], p['position'])]
                                  for p in record['passages'] if (record['entry_key'], p['position']) in references],
                              status='preview-only; auxiliary references and final audit pending'))
    if used_references != set(references):
        raise ValueError('Auxiliary reference targets an absent prose passage')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '20260921-bhattacharya-ollari-entry-texts.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output / 'prose-row-audit.jsonl').write_text(''.join(
        json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in audit))
    summary = dict(status='preview-not-installed', physical_records=len(audit),
                   passages=len(rows), entries_with_prose=sum(bool(r['row_count']) for r in audit),
                   uncertain_entries=sum(bool(r['review_record'].get('reading_uncertainty')) for r in audit),
                   explicitly_cited_passages=len(used_references),
                   attachment_policy='Physical headword only; no copying comparisons to inflections or alternates',
                   comparison_edges_emitted=0)
    (output / 'prose-preview-summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False, indent=2))

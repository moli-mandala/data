"""Prepare a review-only 15-column source CSV and accounting; never install/build."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def prepare():
    metadata = json.loads((ROOT/'metadata-review.json').read_text())
    source = metadata['source_key_proposal']
    raw = {r['entry_key']:r for r in map(json.loads, (ROOT/'positioned-candidates.jsonl').read_text().splitlines())}
    analyses = [json.loads(s) for s in (ROOT/'analysis-proposals.jsonl').read_text().splitlines()]
    rows, audit = [], []
    for analysis in analyses:
        key = analysis['entry_key']; entry = raw[key]
        citation = f"{source}[p. {entry['printed_page']}, entry {entry['page_item']}]"
        # Raw IPA drives the future source-local profile; do not preconvert Form
        # or duplicate it in Phonemic. No historical links are asserted.
        row = [analysis['language'], '', analysis['form'], analysis['gloss'],
               analysis['native'], '', ' '.join(analysis['notes']), citation,
               '', '', key, '', '', '', ' '.join(analysis['tags'])]
        rows.append(row)
        audit.append({'entry_key':key, 'citation':citation,
                      'pdf_page':entry['pdf_page'], 'printed_page':entry['printed_page'],
                      'page_item':entry['page_item'], 'language':'Mahali',
                      'raw_ipa':entry['raw_ipa'], 'raw_english':entry['candidate_english'],
                      'form':analysis['form'], 'gloss':analysis['gloss'],
                      'native':analysis['native'], 'tags':analysis['tags'],
                      'notes':analysis['notes'],
                      'description_review':'description-review-20260926.json',
                      'native_status':analysis['native_status'],
                      'issues':analysis['issues'], 'domain':analysis['domain'],
                      'status':'included in 2451-row source CSV; compiled build deferred',
                      'raw_evidence':'positioned-candidates.jsonl',
                      'native_evidence':'native-recovery-review.jsonl',
                      'graph_decision':'unlinked; no source-supported historical or donor relation assigned'})
    assert len(rows)==len(raw)==len({r[10] for r in rows})==2451
    return rows, audit

if __name__ == '__main__':
    rows, audit = prepare()
    with (ROOT/'review-draft.csv').open('w', newline='') as stream:
        csv.writer(stream).writerows(rows)
    (ROOT/'draft-audit.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in audit))
    print(f'{len(rows)} review-only rows; {sum(bool(r[4]) for r in rows)} reviewed Native values; not installed')

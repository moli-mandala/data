import csv,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
INPUT=HERE.parent/'grammar-metadata-proposal-r2'
rows=[json.loads(s) for s in (INPUT/'all98-keyed-grammar-metadata-proposal.jsonl').read_text().splitlines()]
assert hashlib.sha256((INPUT/'all98-keyed-grammar-metadata-proposal.jsonl').read_bytes()).hexdigest()=='89025ac3ebf122a8db117de76dd408116eaf7f88c499f46c98cf95fc007ff9a3'
with (HERE/'all98-review-view.tsv').open('w',newline='') as stream:
    out=csv.writer(stream,delimiter='\t')
    out.writerow(['Key','Form','Gloss','Label','Disposition','Route','Tags','Notes','Etymology','Source analysis','Source qualification'])
    for r in rows:
        e=r['source_grammatical_evidence']
        out.writerow([r['source_unit_key'].replace('lsi9.2:',''),r['exact_form_guard'],r['exact_gloss_guard'],r['exact_language_label_guard'],r['disposition'],r['proposed_language_id'],' '.join(r['proposed_tags']),' | '.join(r['proposed_notes']),r['proposed_etymology'],e['source_analysis'],e['raw_qualification']])

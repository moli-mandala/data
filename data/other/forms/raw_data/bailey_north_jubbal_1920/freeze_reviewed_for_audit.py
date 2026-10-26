"""Pin the reviewed North Jubbal output and sample actual emitted source units."""
import csv
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    stem = 'whole-reviewed-preview-20260927'
    with (P/f'{stem}.csv').open(newline='') as stream:
        rows = list(csv.reader(stream))
    by_key = {row[10]:row for row in rows}
    units = [json.loads(line) for line in (P/f'{stem}-audit.jsonl').read_text().splitlines()]
    assert len(rows) == len(by_key) == 837 and len(units) == 688
    files = [f'{stem}{suffix}' for suffix in ['.csv','-audit.jsonl','-profile.txt','-report.json','-parser-validation.json']]
    files += ['prepare_reviewed_preview.py','prepare_whole_preview.py','whole-metadata-plan.json']
    frozen = dict(status='frozen for independent original-page audit; not installed',
                  source_pdf='tmp/pdfs/bailey1920/bailey1920.pdf',
                  source_pdf_sha256='7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39',
                  files={name:sha(P/name) for name in files})
    assert sha(P.parents[5]/frozen['source_pdf']) == frozen['source_pdf_sha256']
    freeze = P/'whole-reviewed-output-freeze-20260927.json'
    freeze.write_text(json.dumps(frozen,ensure_ascii=False,indent=2)+'\n')
    pools = {kind:[] for kind in ['glossary','grammar','sentence','commentary']}
    for index,unit in enumerate(units):
        keys = unit.get('entry_keys',[])
        if not keys:
            continue
        key = unit['source_cell_key']
        kind = ('glossary' if index < 243 else 'sentence' if ':sentence:' in key
                else 'commentary' if ':note:' in key or ':comparison:' in key else 'grammar')
        pools[kind].append(index)
    rng = random.Random(20260927271)
    selected = []
    for kind,count in [('glossary',4),('grammar',10),('sentence',4),('commentary',2)]:
        for index in rng.sample(pools[kind],count):
            unit = units[index]
            selected.append(dict(audit_index=index,kind=kind,audit_unit=unit,
                                 actual_csv_rows=[by_key[key] for key in unit['entry_keys']],
                                 verdict='pending original-image review'))
    sample = dict(status='no verdict yet',freeze_sha256=sha(freeze),seed=20260927271,selected_units=selected)
    (P/'root-whole-output-sample-20260927-pass1.json').write_text(json.dumps(sample,ensure_ascii=False,indent=2)+'\n')
    for unit in selected:
        print(unit['audit_index'],unit['kind'],[(r[2],r[3],r[14],r[10]) for r in unit['actual_csv_rows']])


if __name__ == '__main__':
    main()

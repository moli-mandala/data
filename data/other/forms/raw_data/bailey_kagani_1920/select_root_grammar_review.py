"""Pin a page-stratified sample of grammar readings for independent review."""
import hashlib
import json
import random
from pathlib import Path

P = Path(__file__).resolve().parent/'whole-source-recovery'


def main():
    rng = random.Random(20260927141)
    files, selected = {}, []
    for page, count in [(92,3),(93,3),(94,3),(97,3),(98,2),(99,2),(100,2),(101,2)]:
        path = P/f'grammar-p{page}-first-reading-20260927.jsonl'
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        units = [json.loads(line) for line in path.read_text().splitlines()]
        selected.extend(rng.sample(units,count))
    report = dict(status='selected_not_reviewed',seed=20260927141,
        scope='Independent original-vs-reading audit, not final installed-output audit',
        input_sha256=files,source_units=len(selected),selected=selected)
    path = P/'root-grammar-sample-20260927.json'
    if path.exists():
        assert json.loads(path.read_text()) == report
    else:
        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    for unit in selected:
        print(json.dumps(unit,ensure_ascii=False))


if __name__ == '__main__':
    main()

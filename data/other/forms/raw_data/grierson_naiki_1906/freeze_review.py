"""Pin source-stage evidence and select reproducible actual-output audit units."""
import csv
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20260927232


def main():
    manifest_path = HERE / 'review-manifest-r2.json'
    sample_path = HERE / f'independent-selection-r2-{SEED}.json'
    if manifest_path.exists() or sample_path.exists():
        raise SystemExit('Frozen review exists; use a new revision rather than overwrite it.')
    rows = list(csv.reader((HERE / 'whole-review-proposal.csv').open()))
    audit = [json.loads(line) for line in
             (HERE / 'whole-review-proposal-audit.jsonl').read_text().splitlines()]
    assert len(rows) == 694 and len(audit) == 699
    by_key = {row[10]: row for row in rows}
    assert len(by_key) == len(rows)
    assert set(by_key) == {key for unit in audit for key in unit['emitted_keys']}
    quotas = {'prose-group': 4, 'pronoun-table-cell': 3, 'aligned-token': 8,
              'whole-utterance': 3, 'whole-translated-block': 2}
    rng = random.Random(SEED)
    previous=json.loads((HERE/'revision1/independent-selection-r1-20260927231.json').read_text())
    previous_keys={u['source_unit_key'] for u in previous['samples']}
    selected = []
    for kind, count in quotas.items():
        candidates = sorted((unit for unit in audit if unit['kind'] == kind
                             and unit['emitted_keys'] and unit['source_unit_key'] not in previous_keys), key=lambda u: u['source_unit_key'])
        selected.extend(rng.sample(candidates, count))
    assert len(selected) == 20
    samples = [dict(source_unit_key=u['source_unit_key'], kind=u['kind'],
                    printed_page=u.get('printed_page'), audit=u,
                    actual_rows=[by_key[key] for key in u['emitted_keys']])
               for u in selected]
    excluded = {manifest_path.name, sample_path.name}
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(HERE.iterdir()) if path.is_file()
             and path.suffix in {'.py', '.json', '.jsonl', '.csv', '.yaml', '.bib', '.txt', '.md'}
             and path.name not in excluded}
    manifest = dict(stage='Frozen source-stage proposal, not installed or approved',
                    revision=2, rows=694, audit_units=699, sha256=files,
                    original_images_manifest='dsal-originals-manifest.json',
                    deferred=['Full data pipeline and compiled graph checks',
                              'Full test suite', 'User-triggered database and browser QA'])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    selection = dict(seed=SEED, manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                     quotas=quotas, samples=samples,
                     targeted_controls=['All exclusions and p623 extra-chapter genitive witness',
                       'annē my/me conflict; bare English object glosses must not imply case',
                       'Qualified iūn, r-final copulas and kaktēn analyses',
                       'All seven free-translation paragraph boundaries and six speaker labels',
                       'Whole utterances preserve interlinear English only in audit',
                       'Bound morphology, source nasal uncertainty, complete profile coverage'])
    sample_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(manifest=manifest_path.name, selection=sample_path.name,
                          pinned_files=len(files), selected_units=len(samples))))


if __name__ == '__main__':
    main()

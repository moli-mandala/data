"""Resolve source-keyed prose against the forms emitted by make_cldf."""
import csv

FIELDS = ['Form_ID', 'Position', 'Kind', 'Format', 'Content', 'Source']


def read_entry_text_sources(paths, forms_path):
    """Keep legacy Form_ID sidecars; resolve optional Entry_Key without guessed IDs.

    Only requested keys are indexed, so the large forms table is streamed once.
    Missing, ambiguous, or contradictory targets fail before output is written.
    """
    paths = list(paths)
    wanted = set()
    for path in paths:
        with open(path, encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream)
            if 'Entry_Key' in (reader.fieldnames or []):
                wanted.update(row['Entry_Key'] for row in reader if row['Entry_Key'])
    targets = {}
    if wanted:
        with open(forms_path, encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                key = row.get('Entry_Key', '')
                if key in wanted:
                    targets.setdefault(key, set()).add(row['ID'])
        for key in sorted(wanted):
            if len(targets.get(key, set())) != 1:
                raise ValueError(f'Entry-text key {key!r} has missing or ambiguous form target')
    for path in paths:
        with open(path, encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                key = row.get('Entry_Key', '')
                target = row.get('Form_ID', '')
                if key:
                    resolved = next(iter(targets[key]))
                    if target and target != resolved:
                        raise ValueError(f'Entry-text Form_ID conflicts with Entry_Key {key!r}')
                    target = resolved
                if not target:
                    raise ValueError(f'Entry text in {path} lacks Form_ID or Entry_Key')
                yield [target] + [row[field] for field in FIELDS[1:]]

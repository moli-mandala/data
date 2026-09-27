import csv

import pytest

from entry_text_sources import FIELDS, read_entry_text_sources


def write(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_keyed_prose_follows_identity_not_row_order_and_keeps_legacy(tmp_path):
    fields = FIELDS + ['Entry_Key']
    base = dict(Position='1', Kind='comparison', Format='text', Content='cf. x', Source='source[p. 1]')
    sidecar = write(tmp_path/'texts.csv', fields, [dict(base, Entry_Key='head-a'), dict(base, Form_ID='legacy')])
    forms = tmp_path/'forms.csv'
    for rows in [[{'ID':'second', 'Entry_Key':'head-b'}, {'ID':'first', 'Entry_Key':'head-a'}],
                 [{'ID':'first', 'Entry_Key':'head-a'}, {'ID':'second', 'Entry_Key':'head-b'}]]:
        write(forms, ['ID','Entry_Key'], rows)
        resolved = list(read_entry_text_sources([sidecar], forms))
        assert [r[0] for r in resolved] == ['first','legacy']
        assert all(r[1:] == [base[f] for f in FIELDS[1:]] for r in resolved)


@pytest.mark.parametrize('forms,target', [([], ''),
    ([{'ID':'one','Entry_Key':'key'}, {'ID':'two','Entry_Key':'key'}], ''),
    ([{'ID':'one','Entry_Key':'key'}], 'conflicting')])
def test_bad_prose_targets_fail(tmp_path, forms, target):
    sidecar = write(tmp_path/'texts.csv', FIELDS+['Entry_Key'], [dict(Form_ID=target, Entry_Key='key')])
    form_path = write(tmp_path/'forms.csv', ['ID','Entry_Key'], forms)
    with pytest.raises(ValueError):
        list(read_entry_text_sources([sidecar], form_path))

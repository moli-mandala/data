"""Reuse the established Mannan interruption-recovery fixtures for Irula."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
DATA=next(p for p in HERE.parents if (p/'source_meta.py').exists())
template=DATA/'data/other/forms/raw_data/nair_mannan_2013/installation/test_installer.py'
text=template.read_text().replace('Mannan','Irula').replace('mannan','irula').replace('== 154','== 256').replace('== 140','== 261')
needle="        original = workspace / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'"
assert text.count(needle)==1
text=text.replace(needle,"""        import csv
        language = data / 'cldf/languages.csv'
        language.parent.mkdir(parents=True, exist_ok=True)
        with language.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(approval['existing_irula_identity']))
            writer.writeheader(); writer.writerow(approval['existing_irula_identity'])
"""+needle)
text+='''\n\ndef test_disabled_approval_cannot_mutate(installed_fixture):
    module, approval = installed_fixture
    approval['installation_approved'] = False
    module.APPROVAL.write_text(json.dumps(approval))
    before = {name: sha(module.DATA / name) for name in module.TARGETS.values()}
    with pytest.raises(AssertionError): module.main(install=True)
    assert before == {name: sha(module.DATA / name) for name in module.TARGETS.values()}
    assert not (module.HERE / 'installation-transaction.json').exists()
'''
(HERE/'test_installer.py').write_text(text)
print('prepared eight isolated guarded transaction fixtures')

"""Copy bounded, standalone parser/transaction checks for the Mannan route."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
SOURCE = DATA / 'data/other/forms/raw_data/nair_malaveda_2013/installation'
assert not (HERE / 'root-acceptance.json').exists()
text = (SOURCE / 'check_fixture.py').read_text().replace('Malaveda', 'Mannan').replace('malaveda', 'mannan').replace('223', '154')
start, end = text.index('    proposed_language ='), text.index('    old_discovery =')
text = text[:start] + '''    registry = {r['ID'] for r in csv.DictReader((DATA / 'cldf/languages.csv').open(newline=''))}
    assert {'Mannan', 'Malayalam'} <= registry
    assert all(r[0] in {'Mannan', 'Malayalam'} and set(r[14].split()) <= GRAMMATICAL_TAGS | GENDER_TAGS for r in rows)
''' + text[end:]
text = text.replace('len(meta.sources) == 7', 'len(meta.sources) == 2').replace('source_reference_flags=7', 'source_reference_flags=2')
old_draft = "    draft = json.loads((HERE / 'root-acceptance-draft.json').read_text())\n    assert draft['installation_approved'] is False"
assert old_draft in text
text = text.replace(old_draft, '''    if (HERE / 'root-acceptance-draft.json').exists():
        draft = json.loads((HERE / 'root-acceptance-draft.json').read_text())
        assert draft['installation_approved'] is False
    else:
        freeze = json.loads((PACKAGE / 'review-freeze.json').read_text())
        assert sha(PACKAGE / 'review-freeze.json') == '8fe6620df601435704e65f2f66b1382f34dbb507392123b3ee48ad30cea4829a'
        draft = {'files': dict(freeze['files_sha256'])}
        for name in ['installation/20260927-nair-mannan.yaml', 'installation/accepted-audit-proposal.jsonl']:
            draft['files'][name] = sha(PACKAGE / name)''')
text = text.replace("proposed_language_and_lect='explicit new source-named base/lect, blank catalogue and coordinates; not canonically registered yet'",
                    "registered_languages='existing Mannan/mann1246 and Malayalam; no new base, lect or coordinates'")
(HERE / 'check_fixture.py').write_text(text)
text = (SOURCE / 'test_installer.py').read_text().replace('Malaveda', 'Mannan').replace('malaveda', 'mannan').replace('223', '154').replace('240', '140')
text = text.replace("len(result['installed_sha256']) == 6", "len(result['installed_sha256']) == 4")
old = "    languages = module.DATA / module.TARGETS['languages']\n    with languages.open('a') as stream:\n        stream.write('Fixture,Fixture,,,,Other,,\\n')\n"
assert old in text
text = text.replace(old, '')
(HERE / 'test_installer.py').write_text(text)
print('Standalone staged parser and seven transaction fixtures prepared; no canonical writes')

"""Temporary Jambu-only transaction fixtures; never activate the real draft."""
import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


@pytest.fixture
def installed_fixture(monkeypatch):
    real_draft_sha = sha(HERE / 'root-acceptance-draft.json')
    assert not (HERE / 'root-acceptance.json').exists()
    with tempfile.TemporaryDirectory(dir=DATA.parent / 'tmp/jambu-next-source') as temporary:
        workspace = Path(temporary)
        data = workspace / 'data'
        package = data / 'data/other/forms/raw_data/nair_mannan_2013'
        installation = package / 'installation'
        installation.mkdir(parents=True)
        approval = json.loads((HERE / 'root-acceptance-draft.json').read_text())
        for name in approval['files']:
            destination = package / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PACKAGE / name, destination)
        # Simulated fixture authorization only; the real disabled draft is unchanged.
        approval['installation_approved'] = True
        approval['result'] = 'pass'
        for name in approval['canonical_baselines']:
            current = DATA / name
            fixture = data / name
            fixture.parent.mkdir(parents=True, exist_ok=True)
            if current.exists():
                shutil.copyfile(current, fixture)
            approval['canonical_baselines'][name] = sha(fixture)
        identity = data / 'data/form-identities.csv'
        identity.parent.mkdir(parents=True, exist_ok=True)
        identity.write_text('Source_Key,Form_ID\nfixture,1\n')
        approval['durable_identity_sha256'] = sha(identity)
        original = workspace / 'tmp/pdfs/tribal-languages-kerala-nair2013.pdf'
        original.parent.mkdir(parents=True)
        original.write_bytes(b'fixture-original-pin')
        approval['source_pdf_sha256'] = sha(original)
        auth = installation / 'root-acceptance.json'
        auth.write_text(json.dumps(approval))
        spec = importlib.util.spec_from_file_location('mannan_transaction_fixture', HERE / 'install_source.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(module, 'HERE', installation)
        monkeypatch.setattr(module, 'PACKAGE', package)
        monkeypatch.setattr(module, 'DATA', data)
        monkeypatch.setattr(module, 'APPROVAL', auth)
        yield module, approval
    assert sha(HERE / 'root-acceptance-draft.json') == real_draft_sha
    assert not (HERE / 'root-acceptance.json').exists()


def test_clean_install_and_unrelated_shared_addition_noop(installed_fixture):
    module, approval = installed_fixture
    assert module.main()['canonical_mutations'] is False
    result = module.main(install=True)
    assert len(result['installed_sha256']) == 4
    assert result['rows'] == 154 and result['audit_units'] == 140
    bib = module.DATA / module.TARGETS['bib']
    with bib.open('a') as stream:
        stream.write('\n@misc{unrelated_fixture, title={Other}, year={2026}}\n')
    identity = module.DATA / 'data/form-identities.csv'
    with identity.open('a') as stream:
        stream.write('later-build,2\n')
    before = {name: sha(module.DATA / name) for name in module.TARGETS.values()}
    assert module.main(install=True)['canonical_mutations'] is False
    assert before == {name: sha(module.DATA / name) for name in module.TARGETS.values()}


def test_input_and_shared_baseline_drift_rejected(installed_fixture):
    module, approval = installed_fixture
    bib = module.DATA / module.TARGETS['bib']
    bib.write_bytes(bib.read_bytes() + b'\n% unexpected shared change\n')
    with pytest.raises(AssertionError, match='canonical baseline drift'):
        module.main(install=True)
    assert not (module.HERE / 'installation-transaction.json').exists()


def test_receipt_interruption_recovers_without_recomputing_shared_assets(installed_fixture, monkeypatch):
    module, approval = installed_fixture
    original_replace = Path.replace
    state = {'interrupted': False}
    def replace(path, target):
        if path == module.HERE / 'installation-receipt.tmp' and not state['interrupted']:
            state['interrupted'] = True
            raise OSError('simulated receipt interruption')
        return original_replace(path, target)
    monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises(OSError, match='receipt interruption'):
        module.main(install=True)
    assert not (module.HERE / 'installation-receipt.json').exists()
    assert (module.HERE / 'installation-transaction.json').exists()
    assert module.main(install=True)['rows'] == 154
    assert module.main(install=True)['canonical_mutations'] is False


def test_rollback_restores_exact_baselines(installed_fixture):
    module, approval = installed_fixture
    module.main(install=True)
    assert module.main(rollback=True)['status'] == 'rollback dry run validated'
    module.main(install=True, rollback=True)
    assert all(sha(module.DATA / name) == expected for name, expected in approval['canonical_baselines'].items())
    with pytest.raises(AssertionError, match='new root authorization'):
        module.main(install=True)


def test_unexpected_postwrite_change_not_certified_or_rolled_back(installed_fixture, monkeypatch):
    module, approval = installed_fixture
    original_replace = Path.replace
    csv_target = module.DATA / module.TARGETS['csv']
    yaml_target = module.DATA / module.TARGETS['yaml']
    def replace(path, target):
        result = original_replace(path, target)
        if Path(target) == yaml_target:
            csv_target.write_bytes(b'unexpected concurrent edit')
        return result
    monkeypatch.setattr(Path, 'replace', replace)
    with pytest.raises(AssertionError, match='partial transaction drift'):
        module.main(install=True)
    assert not (module.HERE / 'installation-receipt.json').exists()
    with pytest.raises(AssertionError, match='partial transaction drift'):
        module.main(install=True, rollback=True)
    assert csv_target.read_bytes() == b'unexpected concurrent edit'


def test_approved_input_tamper_rejected(installed_fixture):
    module, approval = installed_fixture
    asset = module.PACKAGE / approval['assets']['audit']['input']
    asset.write_bytes(asset.read_bytes() + b'\n')
    with pytest.raises(AssertionError, match='approved input drift'):
        module.main(install=True)
    assert not (module.HERE / 'installation-transaction.json').exists()


def test_initial_identity_drift_rejected(installed_fixture):
    module, approval = installed_fixture
    identity = module.DATA / 'data/form-identities.csv'
    identity.write_text('Source_Key,Form_ID\nunexpected,9\n')
    with pytest.raises(AssertionError):
        module.main(install=True)
    assert not (module.HERE / 'installation-transaction.json').exists()

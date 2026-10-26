"""Canonical whole-source contract, independent of compiled database refresh."""
import csv,hashlib,io,json
from pathlib import Path
import yaml
D=Path(__file__).resolve().parents[1];P=D/'data/other/forms/raw_data/bailey_kagani_1920';F=P/'whole-source-recovery/whole-proposal-r2';FINAL=P/'whole-source-recovery/uncertainty-addendum';CSV=D/'data/other/forms/20260925-bailey-kagani.csv'
def test_exact_approved_installation_and_legacy_key_delta():
    assert CSV.read_bytes()==(FINAL/'proposal.csv').read_bytes()
    assert (P/'audit.jsonl').read_bytes()==(FINAL/'proposal-audit.jsonl').read_bytes()
    assert (D/'conversion/bailey-kagani-1920.txt').read_bytes()==(FINAL/'profile-proposal.txt').read_bytes()
    data=list(csv.reader(CSV.open()));assert len(data)==1306
    baseline=json.loads((P/'whole-source-recovery/legacy-baseline-20260927.json').read_text())
    assert [r[10] for r in data[:329]]==baseline['entry_keys']
    keyed={r[10]:r for r in data}
    assert keyed['bailey1920kagani:p106:right:line:33'][2]=='pălĕādā'
    uncertain=keyed['bailey1920kagani:p97:substantive-past:3-pl-f']
    assert uncertain[2]=='äsīā̃' and 'uncertain' in uncertain[14].split()
    assert 'damaged ā remains possible' in uncertain[6]
    assert keyed['bailey1920kagani:p108:right:line:34:answer2'][6]
    assert 'cross-reference p. 108' in keyed['bailey1920kagani:p109:right:line:4:answer3'][7]
    report=json.loads((F/'root-output-review-20260927.json').read_text())
    assert report['status']=='passed' and report['material_errors']==0 and report['sample_size']>=20
    assert report['frozen_manifest_sha256']==hashlib.sha256((F/'review-manifest.json').read_bytes()).hexdigest()

def test_canonical_parser_and_coordinated_import_route():
    import make_cldf
    errors=io.StringIO();data=list(csv.reader(CSV.open()))
    parsed,stats=make_cldf.parse_file(str(CSV),errors,name=CSV.stem)
    assert len(parsed)==stats['converted']==1306 and not errors.getvalue()
    assert [r.entry_key for r in parsed]==[r[10] for r in data]
    assert [(r.old_form,r.notes,r.source,r.tags) for r in parsed]==[(r[2],r[6],r[7],r[14]) for r in data]
    meta=yaml.safe_load(CSV.with_suffix('.yaml').read_text())
    assert meta['defaults']['transcription']['input']=='form'
    assert meta['defaults']['transcription']['preserve_hyphens'] is True
    assert meta['defaults']['importer']['commands']==[['data/other/forms/raw_data/bailey_kagani_1920/install_whole_source.py','--install']]
    # Old CLI remains callable, but its --install branch delegates before the
    # historical 329-row generator could write into the canonical installation.
    script=(P/'import_source.py').read_text()
    assert script.index('from install_whole_source import install')<script.index('rows, audit = generate()')

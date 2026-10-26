"""Actual installed Naiki routing and source identity; no compiled build."""
import csv,hashlib,io,json
from pathlib import Path
import make_cldf,make_refs,pybtex
from pybtex.database import parse_file
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'data/other/forms/raw_data/grierson_naiki_1906'
SOURCE='grierson1906naiki'
def test_actual_canonical_parser_preserves_all_694_rows():
 path=ROOT/'data/other/forms/20260927-grierson-naiki.csv'
 assert path.read_bytes()==(P/'metadata-addendum/whole-review-proposal.csv').read_bytes()
 raw={r[10]:r for r in csv.reader(path.open())};err=io.StringIO()
 parsed,_=make_cldf.parse_file(str(path),err)
 assert len(parsed)==len(raw)==694 and not err.getvalue()
 for r in parsed:
  row=raw[r.entry_key]
  assert r.old_form==row[2] and r.notes==row[6] and r.native==row[4] and r.ipa==row[5]
  assert set(row[14].split())<=set(r.tags.split())
  assert '\ufffd' not in r.form
  assert set(make_refs.source_ids(row[7]))=={SOURCE}
 assert all(not row[11] and not row[12] and not row[13] for row in raw.values())
def test_exact_installed_profile_dialect_reference_and_identity():
 assert (ROOT/'conversion/grierson-naiki-1906.txt').read_bytes()==(P/'proposal-profile.txt').read_bytes()
 dialect=next(csv.DictReader((P/'proposed-dialect.csv').open()))
 actual={r['ID']:r for r in csv.DictReader((ROOT/'cldf/dialects.csv').open())}
 assert actual[dialect['ID']]==dialect
 assert not dialect['Latitude'] and not dialect['Longitude']
 source=parse_file(str(ROOT/'cldf/sources.bib')).entries[SOURCE]
 assert source==parse_file(str(P/'metadata-addendum/proposed-reference.bib')).entries[SOURCE]
 formatted=pybtex.PybtexEngine().format_from_string(source.to_string('bibtex'),'plain',output_backend='markdown')
 assert '1906' in formatted and 'Grierson' in formatted
 assert 'seven complete source blocks' in source.fields['included']
 receipt=json.loads((P/'installation-report-20260927.json').read_text())
 assert hashlib.sha256((ROOT/'data/form-identities.csv').read_bytes()).hexdigest()==receipt['durable_identity_sha256_unchanged']

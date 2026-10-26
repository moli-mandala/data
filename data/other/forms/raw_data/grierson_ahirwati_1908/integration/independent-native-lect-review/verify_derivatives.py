"""Read-only exact acceptance/projection check; writes only this review directory."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent.parent
DATA = PACKAGE.parents[4]
sys.path.insert(0, str(DATA))
from dialects import dialect_tag, normalize_dialect

pins = {}
def pin(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    pins[str(path)] = digest
    return digest
def records(rel):
    p = PACKAGE / rel
    pin(p)
    return [json.loads(s) for s in p.read_text().splitlines()]
def object_(rel):
    p = PACKAGE / rel
    pin(p)
    return json.loads(p.read_text())
def csv_(path, headers=True):
    pin(path)
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream) if headers else csv.reader(stream))

def main():
    approval_rel = 'specimen38/independent-native-review/root-native-provenance-acceptance.json'
    approval = object_(approval_rel)
    assert pins[str(PACKAGE / approval_rel)] == '0ca9166f71a86d5983a32dd5dfedff45118884f85fc47e09000c1674c36ba4d3'
    status = object_('integration/specimen38-native-provenance-r1/status.json')
    for rel, digest in status['inputs'].items():
        assert pin(PACKAGE / rel) == digest, rel
    for rel, digest in status['outputs'].items():
        assert pin(PACKAGE / 'integration/specimen38-native-provenance-r1' / rel) == digest, rel
    original = records('specimen38/native-review/native-atoms-first-reading.jsonl')
    revised = records('integration/specimen38-native-provenance-r1/native-atoms-reviewed.jsonl')
    decisions = records('specimen38/independent-native-review/all-194-keyed-native-decisions.jsonl')
    assert len(original) == len(revised) == len(decisions) == 194
    assert [r['key'] for r in original] == [r['key'] for r in revised]
    corrections = []
    holds = []
    editable = {'raw_native', 'typed_uncertainties', 'stage'}
    for old, new, decision in zip(original, revised, decisions):
        assert new['historical_source_record'] == old
        assert all(new[k] == v for k, v in old.items() if k not in editable), old['key']
        assert new['raw_native'] == decision['proposed_native']
        assert new['typed_uncertainties'] == decision['current_typed_holds']
        if new['raw_native'] != old['raw_native']:
            corrections.append({'key': old['key'], 'old': old['raw_native'], 'new': new['raw_native']})
        if new['typed_uncertainties']:
            holds.append(old['key'])
    assert corrections == [{'key': 'lsi9.2:p237:native-line:4:word:15', 'old': 'گھروالے', 'new': 'گھروالو'}]
    assert set(holds) == {h['source_unit_key'] for h in approval['current_typed_holds']}
    assert len(holds) == 3
    old_spans = records('specimen38/native-review/native-roman-span-alignment-proposal.jsonl')
    spans = records('integration/specimen38-native-provenance-r1/broad-script-spans-reviewed.jsonl')
    roman = records('specimen38/root-reviewed-revision/p238-atomic-reviewed.jsonl')
    nmap = {r['key']: r for r in revised}
    rmap = {r['source_unit_key']: r for r in roman}
    assert len(old_spans) == len(spans) == 8
    refreshed = []
    for old, new in zip(old_spans, spans):
        assert new['historical_source_record'] == old
        allowed = {'raw_native_span', 'raw_roman_span', 'printed_interlinear_analysis', 'status'}
        assert all(new[k] == v for k, v in old.items() if k not in allowed), old['key']
        assert new['raw_native_span'] == ' '.join(nmap[k]['raw_native'] for k in new['native_keys'])
        assert new['raw_roman_span'] == ' '.join(rmap[k]['raw_form'] for k in new['roman_keys'])
        assert new['printed_interlinear_analysis'] == ' '.join(rmap[k]['raw_gloss'] for k in new['roman_keys'])
        assert set(new['current_native_hold_keys']) == set(new['native_keys']) & set(holds)
        if old['raw_roman_span'] != new['raw_roman_span']:
            refreshed.append(new['key'])
    assert len(refreshed) == 4
    assert Counter(k for s in spans for k in s['native_keys']) == Counter(nmap.keys())
    assert Counter(k for s in spans for k in s['roman_keys']) == Counter(rmap.keys())

    assembly = PACKAGE / 'integration/review-assembly-r2'
    assembly_status = object_('integration/review-assembly-r2/status.json')
    for name, digest in assembly_status['outputs'].items():
        assert pin(assembly / name) == digest, name
    rows = csv_(assembly / 'combined-review-candidate.csv', False)
    assert len(rows) == 653 and all(len(r) == 15 for r in rows)
    no38 = [r for r in rows if r[10].startswith('lsi9.2:p238:') or r[10].startswith('lsi9.2:specimen38:')]
    assert len(no38) == 155
    assert all(not r[4] for r in no38), 'Narrow Native promotion'
    dialects = csv_(assembly / 'dialects-proposed.csv')
    languages = csv_(DATA / 'cldf/languages.csv')
    registered = {r['ID'] for r in languages}
    assert len(dialects) == len({r['ID'] for r in dialects}) == 7
    aliases = {r['Source_Language_ID']: r for r in dialects}
    for d in dialects:
        assert d['Language_ID'] in registered
        assert d['ID'] == d['Source_Language_ID']
        assert d['Tag'] == dialect_tag(d['Language_ID'], d['Source_Language_ID'], d['Name'])
        assert all(not d[f] for f in ['Glottocode', 'Latitude', 'Longitude'])
        assert normalize_dialect(d['Source_Language_ID'], '', aliases, registered) == (d['Language_ID'], d['Tag'])
    regional_ids = ['lsi1908_ahirwati_bawal_nabha', 'lsi1908_ahirwati_kot_kasam_jaipur']
    regional_tags = {aliases[k]['Tag'] for k in regional_ids}
    regional = [r for r in rows if 'Bawal-Kot-Kasam-regional-copulas' in r[10]]
    assert len(regional) == 4
    assert [r[2] for r in regional] == ['sō', 'sī', 'sā', 'thō']
    for r in regional:
        assert r[0] == 'mewati_jhambaus'
        assert {t for t in r[14].split() if t.startswith('dialect:')} == regional_tags
        assert not {'m', 'f'} & set(r[14].split()), 'Invented parenthetical gender pairing'
    assert all(not (regional_tags & set(r[14].split())) for r in rows if r not in regional)
    for path, digest in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, ('Input drift', path)
    result = {'status': 'pass', 'scope': 'Exact derivative and lect routing compatibility, not new original glyph acceptance or installation approval',
              'inputs_sha256': pins, 'counts': {'native_atoms': 194, 'broad_spans': 8, 'roman_atoms_conserved': 133,
               'secure_native_corrections': 1, 'retained_native_holds': 3, 'refreshed_roman_spans': 4,
               'No38_rows_with_blank_Native': 155, 'historical_lect_proposals': 7, 'joint_regional_rows': 4},
              'exact_corrections': corrections, 'retained_hold_keys': holds, 'refreshed_span_keys': refreshed,
              'regional_rows': [{'key': r[10], 'form': r[2], 'tags': r[14]} for r in regional],
              'registration_still_pending': [d['ID'] for d in dialects], 'canonical_changes': False}
    destination = HERE / 'derivative-and-lect-review.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'report_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()

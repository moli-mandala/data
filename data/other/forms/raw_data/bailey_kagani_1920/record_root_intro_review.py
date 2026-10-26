"""Save root's exhaustive original-page review of introductory source units."""
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent/'whole-source-recovery'


def main():
    files, findings = {}, []
    corrections = {
        'bailey1920kagani:p89:consonants:cerebral-l': dict(
            type='literal-context-symbol',forms=['l'],
            reason='Original prints plain italic l while describing it as cerebral. Preserve plain l in raw evidence; do not turn the prose classification into an unprinted underdot. This remains excluded phonetic context.'),
        'bailey1920kagani:p90:compound-verbs:leave-1': dict(
            type='comparison-scope',scope='comparison_control',forms=['chăḍḍṇā'],
            reason='The phrase as in Panjabi introduces the comparison chăḍḍṇā; the following chōhṛnā is used clause identifies the target lexical leave, contrasting intensive chŭṛnā. Keep the comparator in the audit rather than asserting another target head. First vowel appears closed a with breve; owner should confirm narrowly before applying.')}
    for page in (89,90):
        path = P/f'intro-p{page}-first-reading-20260927.jsonl'
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            unit = json.loads(line)
            correction = corrections.get(unit['source_unit_key'])
            findings.append(dict(source_unit_key=unit['source_unit_key'],
                original_unit=unit,verdict='correction_required' if correction else 'pass',
                correction=correction,
                evidence='Root compared complete original printed89–90 in6x top/bottom crops; separately inspected13x habit/causative crops.'))
    assert len(findings) == 61
    report = dict(status='complete reading review; two corrections required before final-output audit',
        date='2026-09-27',reviewer='root',input_sha256=files,source_units=61,
        passed_source_units=59,correction_units=2,per_unit=findings,
        confirmed_readings=['kărnā; dĭk̲h̲tā kărnā in habit paragraph',
            'kărna; kărāṇā; cărnā; cărāṇā in causal paragraph',
            'thē̩ṇā on89 has retroflex n and differs from thē̩nā on99',
            'hŭqq is literal on89; preserve distinction from narrative hăqq'],
        scope_limit='Original-vs-first-reading evidence only; final integrated CSV/profile/metadata and independent output audit remain required.')
    (P/'root-intro-independent-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('61intro units reviewed:59pass,2corrections; no canonical installation.')


if __name__ == '__main__':
    main()

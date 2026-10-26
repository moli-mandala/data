"""Record root's completed original-image audit of the exact fourth frozen output."""
import json
from pathlib import Path
from freeze_full_reread_for_audit import sha

P = Path(__file__).resolve().parent
NOTES = {
    100: 'p145 iron lōah matches both columns.',
    226: 'p147 noun well kūo matches both columns; kept separate from preceding adverb.',
    97: 'p145 ignorant bēsi / kănŭṇḍ matches both columns and diacritics.',
    70: 'p145 fourteen tsŏuda matches both columns with numeral scope.',
    253: 'p131 house singular ablative expands the printed stem correctly: gauhrā kā / gauhr ănda.',
    424: 'p137 shared fall infinitive lōṭṇo matches heading, both retroflex dots retained.',
    447: 'p137 Rampuri masculine past singular/plural lōṭau / lōṭē correctly expands printed endings.',
    504: 'p139 Rampuri first present remain rōũ confirmed in10x crop, including nasal mark.',
    399: 'p136 why kīlĕ / kaī matches adverb/interrogative scope.',
    286: 'p132 second-person agent row matches all four number/lect cells and nasal vowels.',
    522: 'p139 shared drink heading pīṇo matches both target exports.',
    337: 'p134 numeral5 pāndz / pŭndz matches the two lect columns.',
    567: 'p141 Baghi sentence9 checked in10x crop: Mōẽ ēhrau tshōṭu bŏri pīṭṭau; complete translation and expression scope retained.',
    552: 'p141 Rampuri sentence2 checked in10x crop: Eu gōhṛi kĕtri ŭmŭr ā; both u breves and question scope retained.',
    591: 'p142 Baghi sentence21 checked in10x crop: Eh tōẽ kŏs ănda mōllĕ gĭnau; complete question and translation retained.',
    627: 'p116 Rampuri feminine inflection -ī belongs only to Rampuri subsection; morphology has blank lexical gloss.',
    617: 'p115 generic Kochi gōhṛo printed with h; source says h is silent with high falling tone. Generic attribution and comparator controls retained, no lect assignment invented.',
    628: 'p116 Rampuri -ondau has plain o in this occurrence, as exported and documented.',
    603: 'p143 Baghi pīero is passive drunk of the thing consumed, explicitly preserved in Notes and tags.',
    597: 'p143 Bēṭhŏndau preserves initial capital and breve; seated stative analysis refers to Rampuri sentence11.',
}


def main():
    freeze = P/'whole-paired-pass4-freeze-20260927.json'
    sample_path = P/'root-frozen-sample-20260927-pass4.json'
    sample = json.loads(sample_path.read_text())
    assert sample['freeze_sha256'] == sha(freeze)
    manifest = json.loads(freeze.read_text())
    for name, files in manifest['packages'].items():
        for filename, digest in files.items():
            assert sha(P.parent/f'bailey_{name}_1920'/filename) == digest
    assert set(NOTES) == {unit['audit_index'] for unit in sample['selected_units']}
    units = [dict(audit_index=unit['audit_index'], printed_page=unit['printed_page'],
                  source_kind=unit['source_kind'], verdict='pass', evidence=NOTES[unit['audit_index']],
                  actual_rows_checked=sum(len(value['actual_csv_rows']) for value in unit['packages'].values()))
             for unit in sample['selected_units']]
    report = dict(status='fresh independent output audit passed; source-stage installation gates remain',
                  reviewer='root', freeze_sha256=sha(freeze), sample_sha256=sha(sample_path),
                  sample_size=20, material_errors=0, passed_units=20,
                  actual_rows_checked=sum(unit['actual_rows_checked'] for unit in units), units=units,
                  method='Root compared actual frozen rows with original full pages at3x, with10x crops for three sampled sentences and remain. Sample excludes all three prior samples.',
                  prior_error_recheck=dict(entry_key='paired:sentence:142:16:baghi:1',
                      form='Ĭs ănda rupaue mŏṅga ōruh', result='corrected final ōruh confirmed in10x original crop',
                      related_occurrences='Both return expressions ōru āṇo rechecked on original p146 and match. No other ōrah or ōruh occurrences in these emitted proposals.'),
                  deferred=['Final metadata reconciliation and guarded source installation', 'Focused installed-source checks',
                            'Full CLDF/database build and graph validation: user has not authorized rebuild',
                            'App/browser QA against rebuilt source: deferred with rebuild'])
    (P/'root-independent-whole-output-audit-20260927-pass4.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(f"20/20 sampled units passed; {report['actual_rows_checked']} exact emitted rows checked")


if __name__ == '__main__':
    main()

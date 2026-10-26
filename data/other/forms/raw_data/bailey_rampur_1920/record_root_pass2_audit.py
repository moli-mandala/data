"""Record root's completed fresh pass2 original-image audit; no installation approval."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
NOTES={
    21:'Both bread forms rōṭṭi match printed144, including doubled underdotted t and plain final i.',
    91:'Rampuri horse incorrectly exports gōhro; narrow8x printed145 rect44,503,133,516 clearly shows gōhṛo. Baghi gōhro has plain r and is correct.',
    4:'Both ass forms gaddha/gaddho match original paired columns.',
    134:'Both now forms ēbhī match printed146.',
    505:'Rampuri rōe matches second present form; ordered-person inference disclosed in Notes.',
    530:'Baghi dĕūlau matches explicit future example; no unprinted person forms added.',
    417:'Rampuri pāḍ / Baghi thŭl match under and paired attribution.',
    329:'Shared -e oblique masculine singular ending explicitly described in original134; appropriate bound morphology, blank lexical gloss.',
    503:'Shared rauhṇo heading governs both explicitly marked R/B paradigms and matches literal.',
    366:'Narrow8x printed135 confirms distinct tsŏāau / tshŭṭau, including Baghi short a and underdot; num ord appropriate.',
    502:'Baghi ḍūe matches second printed past form and retained paradigm scope.',
    303:'Feminine genitive forms match; source stacked i uncertainty explicitly retained in Rampuri row, Baghi īã ro preserved.',
    571:'Complete Baghi sentence11 matches source; sat rather than Rampuri seated is correctly selected in translation.',
    589:'Complete Baghi sentence20 matches source; narrow8x confirms tū̃ stack. Lect-specific question retained.',
    563:'Complete Baghi sentence7 matches; no missing copula invented.',
    623:'Rampuri lĕ dative preposition correctly attributed under Rampuri heading.',
    624:'Rampuri kā ablative preposition correctly attributed under Rampuri heading.',
    618:'Generic Koci bāhī explicitly printed without English gloss; emitted once under kc with attribution uncertainty, not twice under specific lects.',
    600:'Baghi transitive ending -ero has plain e in this occurrence; source passive-participle interpretation supports tags.',
    594:'Baghi Gāhri / gāhrā and upon gloss match original142 explanatory note; postposition/adjective analysis explicitly stated.'
}


def main():
    sample_path=BASE/'root-frozen-sample-20260927-pass2.json'
    sample=json.loads(sample_path.read_text())
    freeze=BASE/'whole-paired-output-freeze-20260927-pass2.json'
    assert hashlib.sha256(freeze.read_bytes()).hexdigest()==sample['freeze_sha256']
    assert set(NOTES)=={r['audit_index'] for r in sample['selected_units']}
    checked=[]
    for unit in sample['selected_units']:
        i=unit['audit_index']
        checked.append(dict(audit_index=i,printed_page=unit['printed_page'],source_kind=unit['source_kind'],
                            actual_rows_reviewed=sum(len(p['actual_csv_rows']) for p in unit['packages'].values()),
                            verdict='material error' if i==91 else 'pass',evidence=NOTES[i]))
    report=dict(status='failed; full inherited-glossary reread required before next freeze',reviewer='root',
                freeze_sha256=sample['freeze_sha256'],sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                units_reviewed=20,material_errors=1,passed_units=19,
                actual_rows_reviewed=sum(r['actual_rows_reviewed'] for r in checked),checked_units=checked,
                required_correction=dict(entry_key='bailey1920rampur:p145:item:33',before='gōhro',after='gōhṛo'),
                required_class_review='Reread every inherited glossary source cell for both lects against originals; second independent sample again found an inherited diacritic omission.',
                installation_approved=False)
    (BASE/'root-independent-whole-output-audit-20260927-pass2.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"20units / {report['actual_rows_reviewed']}rows; 1material error; installation not approved")


if __name__=='__main__':
    main()

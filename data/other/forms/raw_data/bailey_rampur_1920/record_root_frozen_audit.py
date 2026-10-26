"""Record root's completed original-image review of the first frozen output sample."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
NOTES={
    185:'Both sun readings match: Rampuri ū versus Baghi ŭ; both final ă. Checked full p146 plus narrow 8x crop.',
    86:'Rampuri kukhḷi and Baghi kūkhṛi/mŭrgi match. Baghi wild hen does not: exported dhŭhĭ versus printed dhūhĭ, confirmed narrow 8x crop.',
    242:'All six yesterday/day-before/day-before-that readings and gloss alignments match p147.',
    220:'Both walk readings preserve printed underdots and vowel lengths; source columns match.',
    338:'Narrow 8x numeral crop confirms Rampuri tshan (n arch), Baghi tshau (u trough); both mean 6.',
    272:'Four nominative sister rows correctly expand baih- with singular ṇ and plural ṇī; genders/numbers and lect columns match.',
    392:'Both downwards readings and vowel quantities match source, including Baghi underdot ṭ.',
    511:'Baghi remain present răū matches original and appropriate lect; grammatical interpretation disclosed in notes.',
    482:'Rampuri go infinitive năs̲h̲ṇo matches. Adjacent p139 gives separate Baghi go heading, supporting no duplicate attribution.',
    477:'Baghi future row 2 ālau / āle agrees with the printed ordered list; person-order interpretation disclosed.',
    388:'Both hence forms īdrā agree with original and correct paired columns.',
    526:'Baghi pīero matches original stative; passive/transitive tags supported by explicit p143 explanation.',
    590:'Complete Rampuri sentence21 and its separate English question match, including mālle and lēau.',
    561:'Complete Baghi sentence6 checked at 8x: kăkā, baahṇi, săttrau, baih retained; translation and attribution match.',
    550:'Complete Rampuri sentence1 checked narrow 8x: Tēro nāŭ kā; question translation and tags match.',
    626:'Rampuri masculine inflection -ā explicitly printed under Rampuri heading, empty lexical gloss appropriate.',
    636:'Baghi transitive stative -ēro explicitly printed before Surkhuli heading; passive interpretation supported p143.',
    622:'Rampuri genitive suffix -o and adjectival analysis explicitly printed.',
    602:'Baghi khāero eaten matches explicit p143 example and participial attribution.',
    598:'Baghi khāero matches p143 explicit passive-state explanation; full source gloss retained.',
}


def main():
    sample_path=BASE/'root-frozen-sample-20260927.json'
    sample=json.loads(sample_path.read_text())
    freeze=BASE/'whole-paired-output-freeze-20260927.json'
    assert hashlib.sha256(freeze.read_bytes()).hexdigest()==sample['freeze_sha256']
    assert {u['audit_index'] for u in sample['selected_units']}==set(NOTES)
    checked=[]
    for unit in sample['selected_units']:
        index=unit['audit_index']
        checked.append(dict(audit_index=index,printed_page=unit['printed_page'],
                            source_kind=unit['source_kind'],
                            actual_rows_reviewed=sum(len(p['actual_csv_rows']) for p in unit['packages'].values()),
                            verdict='material error' if index==86 else 'pass',evidence=NOTES[index]))
    result=dict(status='failed; correction and new frozen verification required before installation',
                reviewer='root',freeze_sha256=sample['freeze_sha256'],
                sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                units_reviewed=20,material_errors=1,passed_units=19,
                actual_rows_reviewed=sum(u['actual_rows_reviewed'] for u in checked),
                checked_units=checked,
                required_correction={'entry_key':'bailey1920baghi:p145:item:28:answer3',
                                     'before':'dhŭhĭ','after':'dhūhĭ',
                                     'evidence':'Original printed p145 / PDF171, rect23,420,142,443 at8x; agent independently confirmed at10x.'},
                separate_findings=['Inherited grammatical labels in ten glossary glosses require a documented keyed metadata correction, outside this random sample.'],
                installation_approved=False)
    (BASE/'root-independent-whole-output-audit-20260927-pass1.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(f"20 units / {result['actual_rows_reviewed']} rows reviewed; 1 material error; no installation approval")


if __name__=='__main__':
    main()

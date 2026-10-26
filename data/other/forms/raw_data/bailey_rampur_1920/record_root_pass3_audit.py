"""Record root's completed third independent original-vs-output paired audit."""
import hashlib
import json
from pathlib import Path

P=Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sample=json.loads((P/'root-frozen-sample-20260927-pass3.json').read_text())
    freeze=P/'whole-paired-full-reread-freeze-20260927.json'
    assert sha(freeze)==sample['freeze_sha256']
    manifest=json.loads(freeze.read_text())
    for name, files in manifest['packages'].items():
        for filename,digest in files.items():
            assert sha(P.parent/f'bailey_{name}_1920'/filename)==digest
    notes={
        144:'Both plain noun forms sŏllŏ and noun label match p146.',
        135:'Printed -o versus ro distinction and bound suffix retained.',
        120:'p145 milk distinguishes dūdh from dūddh as exported.',
        162:'p146 seven/seventh forms and numeral/ordinal tags match both columns.',
        246:'p131 genitive horse paradigm expanded from printed stem; both number columns and Baghi word space retained.',
        454:'p138 shared be/become heading has both o macrons and n underdot.',
        336:'p134 cardinal4 tsār matches both lect columns.',
        444:'p137 conjunctive participles and Baghi indeclinable annotation retained.',
        267:'p131 girl genitive forms match printed feminine stem and separate singular/plural cells.',
        359:'p134 twenty-and-half expressions preserve ḍ and underlined Baghi sh.',
        481:'p138 Baghi conjunctive āiā and having-come meaning match.',
        465:'p138 four imperative forms preserve respective initial breves, word endings and Baghi underlining.',
        574:'p142 sentence13 Rampuri complete expression and shared literal translation match;8x inspected.',
        581:'Final Baghi hither word is ōruh, not exported ōrah; original8x Rect175,189,209,205. Independent rohru_finish confirmation received.',
        575:'p142 sentence13 Baghi complete expression and shared literal translation match;8x inspected.',
        633:'p116 Baghi ablative ănda and from annotation match explicit heading.',
        635:'p117 first stative ending -ŏndo belongs to intransitive verbs; blank lexical gloss correct.',
        634:'p117 separate future ending -ūlo and bound-morpheme scope match.',
        599:'p143 -ŏndau explicitly the Baghi intransitive stative ending; comparative other-lect endings not substituted.',
        596:'p142 note9 original8x confirms Băṛŏ, bŏri and bŏhri. Different-word claim preserved; only the explicit Baghi alternatives are linked.'}
    units=[]
    for selected in sample['selected_units']:
        index=selected['audit_index']
        units.append(dict(audit_index=index,printed_page=selected['printed_page'],source_kind=selected['source_kind'],
                          actual_rows_checked=sum(len(p['actual_csv_rows']) for p in selected['packages'].values()),
                          verdict='material_error' if index==581 else 'pass',evidence=notes[index]))
    assert len(units)==20 and set(notes)=={u['audit_index'] for u in units}
    report=dict(status='failed fresh independent output audit; no installation approval',reviewer='root',
                freeze_sha256=sha(freeze),sample_sha256=sha(P/'root-frozen-sample-20260927-pass3.json'),
                sample_size=20,material_errors=1,passed_units=19,
                actual_rows_checked=sum(u['actual_rows_checked'] for u in units),units=units,
                corrections=[dict(entry_key='paired:sentence:142:16:baghi:1',
                                  old_form='Ĭs ănda rupaue mŏṅga ōrah',form='Ĭs ănda rupaue mŏṅga ōruh',
                                  reason='Final source vowel is u, independently confirmed in original crop.')],
                method='Root compared all selected actual emitted rows against complete original pages3x; sentence and commentary forms additionally at8x. Both previous samples excluded.',
                next_gate='Correct source sentence and regenerate without overwriting this failed freeze; inspect the same error class and run another fresh output audit.')
    (P/'root-independent-whole-output-audit-20260927-pass3.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"20units / {report['actual_rows_checked']} actual rows:19pass,1material error; no install approval")


if __name__=='__main__':main()

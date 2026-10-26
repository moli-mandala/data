"""Record root's completed original-page comparison of the frozen whole output."""
import json
from pathlib import Path
from freeze_reviewed_for_audit import sha

P = Path(__file__).resolve().parent
NOTES = {
    167:'p187 shepherd băṛā‘lā retains the printed raised mark and vowel quantities.',
    66:'p185 fruit phăl confirmed at10x; a breve retained.',
    27:'p185 budno and explicit contrast not -no confirmed at10x. The printed comparison has plain n; note retained correctly.',
    3:'p185 pīṭṭh back retains both consonant underdots and noun scope.',
    390:'p177 kīcch something agrees with the pronoun section and printed macron.',
    645:'p182 paihlau is explicitly first in ordinal list, preserving num and ord tags.',
    570:'p180 khāṛū preserves source stative label and passive English meaning, not active eat.',
    464:'p178 right first present cell explicitly has ŏsso or ŏ; both exported alternatives, copula category, and unlabeled position note preserved.',
    541:'p180 third right future cell ḍēōlē preserves initial underdot and three printed vowel macrons; no person assigned from unlabeled position.',
    518:'p179 feminine pluperfect example pŏṛe thī retained as complete printed expression; etc. not expanded.',
    557:'p180 bōṭṭhā past sit retains two underdots and both macrons.',
    545:'p180 jā imperative belongs to the go verb explicitly used in composition; note retained.',
    245:'p172 bauihṇ sister verified at10x under Barari attribution, not the following Bishshau comparator.',
    554:'p180 buṭṭho right imperative retains plain u and two consonant underdots.',
    673:'p183 sentence17 verified at10x: corrected rŏs̲h̲īo macron, full imperative expression and literal translation all match.',
    661:'p183 sentence5 verified at10x: Āũ, dūrŏ, hoṇḍĕo marks and complete translation match.',
    670:'p183 sentence14 verified at10x: Mērā bāp tshōṭe gau’ro dī rauo matches, including raised mark and plain final auo.',
    660:'p183 sentence4 verified at10x: Tā’re bappŏ re kētti lŏṛke matches complete question and interrogative scope.',
    685:'p184 note11 bōṭhondā seated has one ṭ and plain o in this occurrence, unlike other occurrences; stative analysis retained in Notes.',
    683:'p184 note9 lāī verified at10x; source says agreement with understood word for blow. Blank lexical gloss and Hindi-only control preserved.',
}


def main():
    sample_path = P/'root-whole-output-sample-20260927-pass1.json'
    sample = json.loads(sample_path.read_text())
    freeze = P/'whole-reviewed-output-freeze-20260927.json'
    assert sample['freeze_sha256'] == sha(freeze)
    manifest = json.loads(freeze.read_text())
    for filename,digest in manifest['files'].items():
        assert sha(P/filename) == digest,filename
    assert set(NOTES) == {unit['audit_index'] for unit in sample['selected_units']}
    units = [dict(audit_index=unit['audit_index'],source_cell_key=unit['audit_unit']['source_cell_key'],
                  actual_rows_checked=len(unit['actual_csv_rows']),verdict='pass',evidence=NOTES[unit['audit_index']])
             for unit in sample['selected_units']]
    report = dict(status='fresh whole-output source comparison passed; installation gates remain',
                  reviewer='root',sample_size=20,material_errors=0,passed_units=20,
                  actual_rows_checked=sum(unit['actual_rows_checked'] for unit in units),units=units,
                  freeze_sha256=sha(freeze),sample_sha256=sha(sample_path),
                  method='Seeded sample of exact frozen CSV rows across glossary, grammar/introduction, sentences and commentary. Original full pages3x, targeted10x crops. Notes, translation, attribution and tags compared as well as literal forms.',
                  complementary_review='Separate agent reviewed all grammar drafts plus all sentences/context and43legacy metadata corrections before this frozen-output comparison.',
                  deferred=['Reproducible guarded whole-source installation and metadata reconciliation',
                            'Installed-source focused checks',
                            'Full database/graph/reference-rendering/build/browser gates deferred by user no-build instruction'])
    (P/'root-whole-output-audit-20260927-pass1.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"20/20 units passed, {report['actual_rows_checked']} exact emitted rows checked")


if __name__ == '__main__':
    main()

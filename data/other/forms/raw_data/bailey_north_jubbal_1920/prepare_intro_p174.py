"""Original-image first reading of the Barari portion of printed174."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / 'whole-source-recovery'


def main():
    units = [
        ('stative:transitive', '-ēṛu', 'stative participle ending', ['participle','tr'],
         'Source says this ending can occur separately before the verb root; retain its printed hyphen here.'),
        ('stative:intransitive', '-ŏndau', 'stative participle ending', ['participle','intr'],
         'Initial o has breve, not tilde; no vowel length supplied elsewhere.'),
        ('go:compound', 'jāṇā', 'go', ['verb'],
         'Source contrasts use in composition with other verbs against the separate action-of-going form.'),
        ('go:action', 'ḍēūṇo', 'go', ['verb'],
         'Both d and n underdots, separate e/u macrons; final o plain.'),
        ('infinitive:purpose', 'tsārne khi', 'for grazing; for the purpose of grazing', ['multiword-expression'],
         'Complete declined verbal-noun phrase retained; no unsupported word-level alignment.'),
        ('ability:ending:1', '-ūo', 'ability participle ending', ['participle'],
         'Author calls these apparently passive; preserve qualified analysis, not an unqualified passive tag.'),
        ('ability:ending:2', '-ūā', 'ability participle ending', ['participle'],
         'Alternative of the preceding ending; source gender/number assignment not given here.'),
        ('ability:ending:3', '-īdā', 'ability participle ending', ['participle'],
         'Author calls these apparently passive. Plain d is printed; do not add underdot.'),
        ('ability:sentence', 'mēre rōṭī na khāīndi', 'I cannot eat bread.', ['sentential','neg'],
         'Source explains genitive subject and participle agreement with logical object. Preserve the complete response; do not infer internal word alignment.'),
    ]
    records = [{'source_unit_key': f'bailey1920northjubbal:p174:{key}',
                'printed_page':174, 'pdf_page':200, 'lect':'Barari',
                'form':form, 'gloss':gloss, 'tags':tags, 'source_analysis':note,
                'status':'owner-original-first-reading; independent review pending'}
               for key,form,gloss,tags,note in units]
    assert len(records) == len({r['source_unit_key'] for r in records}) == 9
    report = {'status':'page174 target first reading; not installed', 'reviewer':'root',
              'evidence':'Original full page at3x and entire Barari portion at8x.',
              'target_units':records,
              'controls':[{'source_unit_key':f'bailey1920northjubbal:p174:control:{section}',
                           'lect':'Bishshau', 'decision':'Section under explicit BISHSHAU heading; not attributed to Barari.'}
                          for section in ['nouns','pronouns','verbs']],
              'continuation':'Final Bishshau ability example continues onto175; retain exclusion until NORTH JUBBAL OR BARARI heading.',
              'pending':'Independent reading, whole-source reconciliation, profile and tests.'}
    (OUT / 'intro-p174-first-reading-20260927.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'{len(records)} target units and3other-lect blocks; no canonical writes')


if __name__ == '__main__':
    main()

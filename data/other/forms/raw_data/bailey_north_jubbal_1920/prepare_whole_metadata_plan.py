"""Reconcile expanded source metadata without changing shared registry files."""
import csv
import hashlib
import json
from pathlib import Path
import yaml
from pybtex.database import parse_file

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[4]
STEM = '20260925-bailey-north-jubbal'
SOURCE = 'bailey1920northjubbal'


def main():
    settings = DATA / f'data/other/forms/{STEM}.yaml'
    meta = yaml.safe_load(settings.read_text())
    assert meta['defaults']['identity']['append_order'] == 53
    assert meta['sources'][SOURCE]['identity']['dedupe_by_entry_key'] is True
    assert meta['sources'][SOURCE]['forms']['split_alternates'] is False
    with (DATA/'cldf/languages.csv').open(newline='') as stream:
        language = next(row for row in csv.DictReader(stream) if row['ID'] == 'Barari')
    bib = parse_file(str(DATA/'cldf/sources.bib'))
    assert SOURCE in bib.entries
    entry = bib.entries[SOURCE]
    assert entry.fields['year'] == '1920' and entry.fields['ocr'] == 'Yes'
    inclusion = ('North Jubbal/Barari introduction, grammar, paradigms and complete translated responses, '
                 'printed pp. 172–184; complete glossary pp. 185–188; explicitly North Jubbal suffix '
                 'comparison on p. 143; applicable vowel-notation guidance on p. 114. '
                 'The currency discussion on p. 184 refers to p. 167, note 13, whose uncertainty is retained. '
                 'Other named lects and comparative controls remain separately attributed and excluded from Barari output.')
    fields = dict(entry.fields)
    fields['included'] = inclusion
    fields['provenance'] = (f'data/other/forms/{STEM}.csv; '
                            'data/other/forms/raw_data/bailey_north_jubbal_1920/full-transcription.tsv; '
                            'data/other/forms/raw_data/bailey_north_jubbal_1920/whole-source-recovery; '
                            'final whole-output audit path to be supplied at approved installation')
    plan = dict(status='metadata installation plan only; review and source acceptance pending',
                source_id=SOURCE, language=language,
                language_decision='Reuse canonical Barari under the existing explicit editorial decision. North Jubbal is the source name for this language; no narrower stable survey site or new dialect is asserted.',
                coordinates_decision='Retain existing quality-C approximate regional point and its caveat; no new coordinates inferred.',
                transcription=dict(profile='bailey-north-jubbal-1920', input='form', preserve_hyphens=True),
                identity=meta['defaults']['identity'], dedupe_by_entry_key=True, split_alternates=False,
                reference_fields_proposal=fields,
                reference_editors=meta['sources'][SOURCE]['reference'],
                settings_before_sha256=hashlib.sha256(settings.read_bytes()).hexdigest(),
                pending=['Independent source reviews and approved whole-output freeze',
                         'Replace glossary-only importer route with guarded whole-source installer',
                         'Use final installed whole-output audit path in bibliography provenance',
                         'Update settings importer note and README counts from final approved manifest',
                         'Regenerated references and compiled graph checks at user-authorized full build'],
                canonical_changes=False)
    (HERE/'whole-metadata-plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print('Canonical Barari, existing quality-C coordinates, source identity and bibliography verified; whole metadata plan saved; no shared edits')


if __name__ == '__main__':
    main()

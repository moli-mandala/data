"""Draft exact source-local YAML/reference changes; never edit shared metadata."""
import csv,hashlib,json
from pathlib import Path
import yaml

P=Path(__file__).resolve().parent;D=P.parents[4]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    language_path=D/'cldf/languages.csv'
    languages={r['ID']:r for r in csv.DictReader(language_path.open())}
    assert {k:languages[k]['Name'] for k in ['ramp','ba','kc']}=={'ramp':'Rampuri','ba':'Baghi','kc':'Kochi'}
    reports={}
    for name,count,legacy,append_order in [('rampur',647,288,73),('baghi',698,296,51)]:
        package=P.parent/f'bailey_{name}_1920';stem=f'20260925-bailey-{name}';source=f'bailey1920{name}'
        path=D/f'data/other/forms/{stem}.yaml'
        before=yaml.safe_load(path.read_text());after=yaml.safe_load(path.read_text())
        after['defaults']['transcription'].update(input='form',preserve_hyphens=True)
        after['defaults']['importer']['note']=(f'Whole paired-source {name.title()} recovery: shared introduction113–115 and lect-specific116–117, grammar131–140,22complete translated responses141–142, commentary142–143 and glossary144–147. {count}rows preserve all{legacy}previously installed rows and stable keys. Printed morphology, variants and typed uncertainties remain explicit; full compiled/browser gates deferred.' + (' Four generic Kochi witnesses occur once under kc; body jĕā́ recovered with typed interpretation uncertainty.' if name=='rampur' else ' Shared generic Kochi witnesses are carried once in the companion Rampuri source, not duplicated.'))
        assert after['defaults']['identity']=={'legacy_ids':'stem','append_order':append_order}
        assert after['sources'][source]['identity']['dedupe_by_entry_key'] is True
        assert after['sources'][source]['forms']['split_alternates'] is False
        assert after['sources'][source]['reference']==before['sources'][source]['reference']
        changed_legacy=8 if name=='rampur' else 3
        intro='116' if name=='rampur' else '116–117'
        after['defaults']['importer']['note']=after['defaults']['importer']['note'].replace('lect-specific116–117',f'lect-specific{intro}').replace(f'preserve all{legacy}previously installed rows and stable keys',f'preserve all{legacy}stable legacy keys, with {changed_legacy}explicit source-reviewed legacy corrections')
        (package/'whole-proposed.yaml').write_text(yaml.safe_dump(after,sort_keys=False,allow_unicode=True))
        included=(f'Complete {"Rampuri" if name=="rampur" else "Baghi"} source scope: shared introduction pp.113–115, lect-specific introduction pp.116–117, grammar pp.131–140,22whole translated responses pp.141–142, commentary pp.142–143, and glossary pp.144–147. {count}rows retain all{legacy}prior glossary rows exactly. Source morphology, full expressions and qualified glyph readings are preserved. Other-lect comparisons, shared context and printed cross-references are explicitly accounted.' + (' Four unglossed generic Koci witnesses occur once under existing Kochi; the former body hold is recovered literally as jĕā́ with interpretation uncertainty.' if name=='rampur' else ' Shared generic Koci witnesses are retained only in companion bailey1920rampur; no unpaired Rampuri glossary answer is attributed to Baghi.'))
        included=included.replace('lect-specific introduction pp.116–117',f'lect-specific introduction p.{intro}' if name=='rampur' else 'lect-specific introduction pp.116–117').replace(f'retain all{legacy}prior glossary rows exactly',f'retain all{legacy}prior glossary keys; {legacy-changed_legacy}legacy rows are unchanged and {changed_legacy}have explicit source-reviewed field corrections')
        reference=dict(citation_id=source,changed_fields={'included':included,'provenance':f'data/other/forms/{stem}.csv; data/other/forms/raw_data/bailey_{name}_1920/audit.jsonl; data/other/forms/raw_data/bailey_rampur_1920/whole-paired-output-freeze-20260927-pass2.json; data/other/forms/raw_data/bailey_rampur_1920/whole-paired-second-reading-20260927.json; data/other/forms/raw_data/bailey_rampur_1920/whole-legacy-corrections-20260927-pass2.json'},preserve_other_fields=True,source_bibliography='cldf/sources.bib',note_policy='Retain original geographic and paired-primary-source provenance, license, URL, editor and OCR fields; do not create a new independent field-event claim.')
        (package/'whole-proposed-reference-fields.json').write_text(json.dumps(reference,ensure_ascii=False,indent=2)+'\n')
        reports[name]=dict(current_yaml_sha256=sha(path),proposed_yaml_sha256=sha(package/'whole-proposed.yaml'),identity_unchanged=after['defaults']['identity'],source_settings_unchanged=after['sources'],changed_yaml_fields=['defaults.transcription.input','defaults.transcription.preserve_hyphens','defaults.importer.note'],proposed_reference_fields=reference['changed_fields'])
    report=dict(status='source-local metadata proposals only; shared metadata unchanged',packages=reports,existing_registry={k:languages[k] for k in ['ramp','ba','kc']},registry_sha256=sha(language_path),registry_changes=[],coordinate_changes=[],dialect_changes=[],source_label_policy='Generic Koci maps once to existing kc with attribution uncertainty; explicit Rampuri/Baghi headings map to existing ramp/ba. No inferred site or population coordinates.',source_stage_deferred=['independent output audit acceptance','guarded installation and real canonical parser checks','fullCLDF/database/build/browser gates'])
    (P/'whole-metadata-reconciliation-proposal-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v['changed_yaml_fields'] for k,v in reports.items()}))


if __name__=='__main__':main()

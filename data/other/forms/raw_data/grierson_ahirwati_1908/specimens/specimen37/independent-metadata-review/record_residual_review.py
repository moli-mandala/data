"""Root-requested intrinsic grammar sweep; isolated proposals, no owner mutation."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPECIMEN = HERE.parent
PACKAGE = SPECIMEN.parents[1]
OUT = HERE / 'residual-revision4-proposal'
OUT.mkdir(exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def literal(s):
    return s.strip(' .;,?!‘’“”"').lower()

original = SPECIMEN/'independent-review/projection-revision3/candidate.csv'
assert sha(original) == 'a8141e9f2d2291d7f50929b3cb9fc3574cbf77669a2a19a92f4b4ee0d670fd99'
base = list(csv.reader(original.open(newline='')))
priorpath = HERE/'protected-field-proposed.csv'
current = list(csv.reader(priorpath.open(newline='')))
assert len(base) == len(current) == 390
grammar_path = PACKAGE/'grammar/child-witnesses-reviewed.jsonl'
grammar = read(grammar_path)
assert sha(grammar_path) == '5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7'
all_decisions, deltas = [], []
person = {'1sg','1pl','2sg','2pl','3sg','3pl','first-person','second-person','third-person'}
for old, row in zip(base,current):
    assert old[10] == row[10]
    protected = [i for i in range(15) if i not in {6,7,14}]
    assert all(old[i] == row[i] for i in protected)
    tags = row[14].split()
    form = literal(row[2])
    reasons, support, removed = [], [], []
    if 'verb' in tags and not row[10].endswith(':expression'):
        allowed = set()
        matched = [g for g in grammar if literal(g['source_form']) == form]
        if form == 'bōlyō':
            allowed |= {'pret','participle'}
            support.append('p237 explicitly calls bōlyō a past participle; person is not specified by that class claim.')
        if form == 'karᵃṇō':
            allowed |= {'fut','pass','participle','inf'}
            support.append('p51 explicitly classifies exact karᵃṇō as future passive participle, identical to infinitive.')
        copula = form.rsplit('-',1)[-1]
        if copula in {'thō','thā','thī'}:
            if form == copula: allowed.add('pret')
            support.append('p51 explicitly classifies '+copula+' as past copula; visible hyphen-compound comparison stays in audit/Notes and does not license a whole-form tag.')
        if copula in {'sai','sā','sū̃','sā̃','sō','saĩ'}:
            if form == copula: allowed.add('pres')
            support.append('p51 printed present-copula paradigm supplies '+copula+'; no inferred stem morphology.')
        if form == 'sū̃':
            allowed |= {'1sg','first-person'}
            support.append('p51 explicitly gives visible sū̃ as first singular present copula.')
        if form == 'sā':
            allowed |= {'2sg','second-person'}
            support.append('p51 explicitly gives sā as second singular present copula; source printed art agrees.')
        if '-' in form and copula in {'thō','thā','thī','sai','sā','sū̃','sā̃','sō','saĩ'}:
            note = 'The printed final component '+copula+' matches a copula in Grierson’s p. 51 paradigm; this comparison does not classify the entire compound.'
            row[6] = (row[6]+' '+note).strip()
            if 'grammatical comparison p. 51' not in row[7]:
                assert row[7].endswith(']')
                row[7] = row[7][:-1]+', grammatical comparison p. 51]'
        if 'impv' in tags:
            allowed.add('impv')
            support.append('Exact quotation is a command addressed to an interlocutor; source functional context supports imperative, without inferred suffix or person.')
        intrinsic = {'pret','pass','fut','pres','inf','participle','conjunctive-participle','relative-participle','modal'} | person
        removed = [t for t in tags if t in intrinsic and t not in allowed]
        if removed:
            tags = [t for t in tags if t not in removed]
            if 'uncertain' not in tags: tags.append('uncertain')
            reasons.append('Withhold English-only intrinsic classifications: '+', '.join(removed)+'. Printed interlinear time, voice and subject wording remains exact; occurrence-specific target morphology is not established.')
            note = 'The printed interlinear English supplies the time, voice or subject interpretation; the corresponding intrinsic form classification is not specified.'
            if note not in row[6]: row[6] = (row[6]+' '+note).strip()
        row[14] = ' '.join(tags)
    changed = [i for i in range(15) if old[i] != row[i]]
    assert set(changed) <= {6,7,14}
    decision = dict(source_unit_key=row[10], old_row=old, proposed_row=row, changed_columns=changed,
                    removed_intrinsic_tags=removed, retained_source_class_evidence=support,
                    typed_analysis_uncertainties=[dict(type='intrinsic-grammatical-class-not-established', scope='whole lexical form only', withheld_tags=removed, printed_english_preserved=row[3])] if removed else [],
                    reasons=reasons, status='root-requested source-local metadata proposal; final exact-delta review pending')
    all_decisions.append(decision)
    if changed: deltas.append(decision)
assert [r[10] for r in current] == [r[10] for r in base]
for name, records in [('all390-keyed-decisions.jsonl',all_decisions),('guarded-deltas.jsonl',deltas)]:
    (OUT/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
with (OUT/'candidate.csv').open('w',newline='') as stream: csv.writer(stream).writerows(current)
with (OUT/'all390-current-tags.tsv').open('w',newline='') as stream:
    writer=csv.writer(stream,delimiter='\t')
    writer.writerow(['Key','Form','Gloss','Current Tags','Proposed Tags','Removed English-only intrinsic tags','Printed class evidence'])
    for d in all_decisions:
        writer.writerow([d['old_row'][10],d['old_row'][2],d['old_row'][3],d['old_row'][14],d['proposed_row'][14],','.join(d['removed_intrinsic_tags']),'; '.join(d['retained_source_class_evidence'])])
report=dict(status='guarded residual metadata proposal; no owner/canonical changes',rows=390,
            combined_changed_rows=len(deltas), residual_intrinsic_rows=sum(bool(d['removed_intrinsic_tags']) for d in all_decisions),
            protected_indices=[i for i in range(15) if i not in {6,7,14}], ordered_keys_forms_glosses_native_unchanged=True,
            inputs={str(p.relative_to(PACKAGE)):sha(p) for p in [original,priorpath,grammar_path,HERE/'independent-full-metadata-review.json',HERE/'precision-source-class-packet-status.json']},
            outputs={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'status.json'},
            limitations='Initial84 and34-question packets unchanged. No smooth translation, literal repair, source morpheme split, new graph or guessed IPA. Typographic holds remain; added uncertain marks a typed intrinsic-analysis hold, not a changed reading.')
(OUT/'status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['rows','combined_changed_rows','residual_intrinsic_rows']}))

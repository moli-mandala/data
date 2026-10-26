"""Guarded residual scope correction. Preserve the complete first review as history."""
from pathlib import Path
import json, hashlib, collections

P = Path(__file__).resolve().parent
OUT = P / 'residual-revision2'
OUT.mkdir(exist_ok=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

old = read(P / 'all-132-keyed-metadata-decisions.jsonl')
assert len(old) == 132
new = []
deltas = []
for original in old:
    r = json.loads(json.dumps(original))
    f = r['exact_form_guard'].strip('‘’.,;?')
    g = r['exact_gloss_guard'].strip('‘’.,;?')
    tags = list(r['proposed_tags'])
    reasons = r['source_scoped_basis']
    holds = r['additional_scope_holds']
    # Connective ki/Ki is NOT the demonstrative meaning assigned to independent
    # English that. Keep the exact target/source-context adjudication decisive.
    if f in {'ki', 'Ki'}:
        tags = [t for t in tags if t not in {'pron', 'demonstrative', 'personal', 'verb'}]
        reasons = [s for s in reasons if not s.startswith('Explicit printed pronoun/deictic meaning')]
        reasons.append('All ki/Ki occurrences introduce quoted content; no demonstrative/pronoun or supplied he-replied verb is projected onto the target connective.')
    if f == 'bōlā':
        tags = [t for t in tags if t not in {'pret', 'participle'}]
        reasons = [s for s in reasons if 'past-participle bōlō/bōlyō class and mixed Bangaru bōlā' not in s]
        reasons.append('p237 identifies bōlā as Bangaru, but the explicit past-participle/y-dropping analysis is scoped to bōlō/bōlyō; only lexical verb POS is secure here.')
        holds.append({'type': 'TAM-analysis-scope', 'detail': 'Printed said supplies lexical past-time interpretation, not an independently established bōlā morphological tense/participle classification.'})
    # Exact chapter form classes license TAM only on thō, sai and bōlō. Other
    # temporal English analyses remain in raw gloss, not intrinsic target Tags.
    removed_tam = [t for t in tags if t in {'pret', 'fut', 'pres'} and f not in {'thō', 'sai', 'bōlō'}]
    if removed_tam:
        tags = [t for t in tags if t not in removed_tam]
        holds.append({'type': 'TAM-analysis-scope', 'detail': f'Printed {g} supplies a lexical temporal interpretation. The exact target form has no independently identified chapter tense-class analysis here; intrinsic {", ".join(removed_tam)} withheld.'})
        reasons = [s for s in reasons if not any(q in s for q in [
            'gives past occurrence meaning', 'explicitly includes was',
            'supplies future meaning', 'Explicit printed it-was-said passive analysis'])]
        reasons.append('Printed temporal English is lexical-time evidence only; exact target morphology is not independently corroborated by the chapter, so no intrinsic tense tag is proposed.')
    if f == 'sai' and g == 'art':
        tags = [t for t in tags if t not in {'2sg', 'second-person'}]
        reasons = [s for s in reasons if not s.startswith('Exact source art')]
        reasons.append('p51 explicitly gives sai in both second- and third-person singular present copula slots; English art here does not identify intrinsic second-person morphology.')
    # Do not transfer English nominal number to a source form where the source
    # chapter does not identify this exact form's number morphology.
    if g in {'years', 'weddings'}:
        tags = [t for t in tags if t != 'pl']
        reasons = [s for s in reasons if not s.startswith('Printed plural nominal sense')]
        reasons.append('Printed plural lexical interpretation years/weddings is retained, but no target plural-morphology claim is independently supplied for baras/biyāh.')
    if 'verb' in tags and f not in {'thō', 'sai', 'bōlō'}:
        removed_person = [t for t in tags if t in {'1sg', '2sg', 'first-person', 'second-person'}]
        if removed_person:
            tags = [t for t in tags if t not in removed_person]
            holds.append({'type': 'verbal-person-scope', 'detail': f'Printed {g} identifies a speech role, retained in the aligned English; no independently established inflectional person category assigned to this target verb form.'})
            reasons.append('Speaker/addressee in aligned English is preserved as source meaning, without imposing target inflectional person absent exact source-class corroboration.')
    if holds and 'uncertain' not in tags:
        tags.append('uncertain')
    r['proposed_tags'] = list(dict.fromkeys(tags))
    r['source_scoped_basis'] = list(dict.fromkeys(reasons))
    r['additional_scope_holds'] = holds
    r['scope_distinction'] = 'Lexical POS from exact source sense/context is separate from intrinsic grammatical morphology. Source-explicit grammar only where independently scoped; temporal English and speech roles stay literal meanings/audit qualifications.'
    assert r['source_unit_key'] == original['source_unit_key']
    assert r['exact_form_guard'] == original['exact_form_guard']
    assert r['exact_gloss_guard'] == original['exact_gloss_guard']
    assert r['old_tags'] == original['old_tags']
    if r['proposed_tags'] != original['proposed_tags']:
        deltas.append({'source_unit_key': r['source_unit_key'], 'exact_form_guard': r['exact_form_guard'],
            'exact_gloss_guard': r['exact_gloss_guard'], 'previous_review_tags': original['proposed_tags'],
            'revised_review_tags': r['proposed_tags'], 'revised_scope': r['source_scoped_basis'],
            'additional_scope_holds': r['additional_scope_holds']})
    new.append(r)
assert {r['source_unit_key'] for r in old} == {r['source_unit_key'] for r in new}
for r in new:
    f = r['exact_form_guard'].strip('‘’.,;?')
    if f in {'ki', 'Ki'}:
        assert not {'pron', 'demonstrative', 'verb'} & set(r['proposed_tags'])
    if f == 'bōlā':
        assert not {'pret', 'participle'} & set(r['proposed_tags'])
    if f == 'sai':
        assert not {'1sg', '2sg', '3sg'} & set(r['proposed_tags'])
    if f not in {'thō', 'sai', 'bōlō'}:
        assert not {'pret', 'fut', 'pres'} & set(r['proposed_tags'])
for name, values in [('all-132-keyed-metadata-decisions.jsonl', new),
    ('keyed-tag-change-proposals.jsonl', [r for r in new if r['old_tags'] != r['proposed_tags']]),
    ('exact-residual-review-deltas.jsonl', deltas)]:
    (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
first = json.loads((P / 'independent-full-metadata-boundary-review.json').read_text())
report = dict(first)
report['status'] = 'Complete independent residual source-scope review; guarded metadata and anchored-key owner derivative proposed, no frozen source fields edited'
report['counts'] = dict(first['counts'])
report['counts']['tag_changes_proposed'] = sum(r['old_tags'] != r['proposed_tags'] for r in new)
report['counts']['review_recommendation_rows_changed_from_r1'] = len(deltas)
report['historical_review_sha256'] = sha(P / 'independent-full-metadata-boundary-review.json')
report['historical_all132_sha256'] = sha(P / 'all-132-keyed-metadata-decisions.jsonl')
report['root_adjudication'] = 'Parent reviewed full132 compact recommendations: ki connective versus demonstrative, bōlā scope, sai shared person surface, English plural and lexical-time versus target morphological TAM. This derivative applies those exact class-wide corrections and additionally withholds uncorroborated verbal person inflection.'
report['outputs'] = {f.name: sha(f) for f in OUT.iterdir() if f.is_file()}
report['boundary_review_unchanged_sha256'] = sha(P / 'all-23-keyed-boundary-decisions.jsonl')
report['review_script_sha256'] = sha(Path(__file__))
report['intrinsic_TAM_proposals_remaining'] = [{'key': r['source_unit_key'], 'form': r['exact_form_guard'],
    'tags': r['proposed_tags']} for r in new if set(r['proposed_tags']) & {'pret', 'pres', 'fut'}]
(OUT / 'independent-full-residual-metadata-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report['counts']))

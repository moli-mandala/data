"""Independent exact-key metadata and original-backed boundary review. No source edits."""
from pathlib import Path
import hashlib, json, collections, sys

HERE = Path(__file__).resolve().parent
P = HERE.parent / 'native-review'
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'data'))
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    return [json.loads(s) for s in p.read_text().splitlines()]

def write(name, rows):
    (HERE / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

pins = {
    'atomic-emissions-proposal.jsonl': '4828536ab952ffea007de5fa3b7d8cedccaa17aab35bd040d3ca4529547a20eb',
    'whole-utterances-proposal.jsonl': 'ff82e810297d785d0e14ec2fa147d9788d656934ca2ac087bf297e01b7c87b6a',
    'emission-proposal-status.json': 'd46575fed7f261ddbcfe5e32c24d6785d5a07334f4cff209fcffebaa6a47213b',
}
for name, expected in pins.items():
    assert sha(P / name) == expected
status = json.loads((P / 'emission-proposal-status.json').read_text())
for name, expected in status['output_hashes'].items():
    assert sha(P / name) == expected
for name, expected in status['input_hashes'].items():
    assert sha(ROOT / name) == expected
atoms = read(P / 'atomic-emissions-proposal.jsonl')
wholes = read(P / 'whole-utterances-proposal.jsonl')
physical = read(HERE.parent / 'root-reviewed-revision/p238-atomic-reviewed.jsonl')
by_physical = {r['source_unit_key']: r for r in physical}
assert len(atoms) == 132 and len(wholes) == 23 and len(physical) == 133
assert [k for a in atoms for k in a['physical_source_keys']] == list(by_physical)

# Exact printed interlinear senses, individually checked in all three original
# Roman crops. These are keyed reviewer decisions, not a reusable prose parser.
nouns = {'Ahīr', 'the-son-in-law', 'day', 'the-Ahīr-to', 'recovery',
         'The-Ahīr', 'brother-to', 'guest', 'husband', 'son-in-law', 'brother',
         'soul-of', 'mouth-from', 'son-in-law-by', 'corn-rake', 'ring-to', 'piece',
         'years', 'uncle', 'hand-of', 'liver-of', 'weddings', 'The-Ahīr-of'}
adjectives = {'sick', 'red-turban-person', 'narrow', 'four-pronged', 'beautiful'}
# red-turban-person is a nominal referent, not an independently glossed adjective.
adjectives.remove('red-turban-person')
nouns.add('red-turban-person')
adverbs = {'gradually', 'to-day', 'how', 'very'}
pronouns = {
    'Him-of': ['pron', 'poss', '3sg', 'third-person'],
    'he': ['pron', 'personal', '3sg', 'third-person'],
    'He': ['pron', 'personal', '3sg', 'third-person'],
    'his-own': ['pron', 'poss', 'refl'],
    'thy': ['pron', 'poss', '2sg', 'second-person'],
    'This': ['pron', 'demonstrative', 'nom'],
    'this': ['pron', 'demonstrative'],
    'That': ['pron', 'demonstrative'],
    'that': ['pron', 'demonstrative'],
    'thou': ['pron', 'personal', '2sg', 'second-person'],
    'Thou': ['pron', 'personal', '2sg', 'second-person'],
    'I': ['pron', 'personal', '1sg', 'first-person'],
    'Me': ['pron', 'personal', '1sg', 'first-person'],
    'my': ['pron', 'poss', '1sg', 'first-person'],
    'My': ['pron', 'poss', '1sg', 'first-person'],
    'you': ['pron', 'personal', 'second-person'],
    'me-to': ['pron', 'personal', '1sg', 'first-person', 'dat'],
    'Thee-to': ['pron', 'personal', '2sg', 'second-person', 'dat'],
    'to-me': ['pron', 'personal', '1sg', 'first-person', 'dat'],
    'who': ['pron', 'interr'],
    'Who': ['pron', 'interr'],
    'which': ['pron', 'relative'],
    'Which-on': ['pron', 'relative'],
    'On-what': ['pron', 'relative'],
    'something': ['pron', 'indef'],
}
past_verbs = {'came', 'become-is', 'became', '(the-word-)issued', 'spoiled-are'}
# Present perfect-like translations do not by themselves establish target past.
past_verbs -= {'become-is', 'spoiled-are'}
kinship = {'the-son-in-law', 'brother-to', 'husband', 'son-in-law', 'brother', 'son-in-law-by', 'uncle'}
genitives = {'Him-of', 'The-Ahīr-of', 'soul-of', 'hand-of', 'liver-of', 'dying-of', 'Hukamlō-of'}

decisions = []
for a in atoms:
    key = a['emission_key']
    g = a['emission_gloss'].strip('‘’.,;?')
    f = a['emission_form'].strip('‘’.,;?')
    new = list(a['tags'])
    reasons = []
    holds = []
    notes = []
    def add(tags, reason):
        new.extend(tags)
        reasons.append(reason)
    if g in nouns:
        add(['noun'], 'Exact printed nominal lexical sense and this narrative referent support noun; no grammatical gender inferred.')
        if g in kinship:
            add(['kinship'], 'Printed kinship relation, including son-in-law, husband, brother or uncle.')
        if g in {'years', 'weddings'}:
            add(['pl'], 'Printed plural nominal sense, not inferred from the numeral alone.')
    if g in adjectives:
        add(['adj'], 'Printed property adjective in this exact source construction.')
    if g in adverbs:
        add(['adv'], 'Printed temporal/manner/degree adverbial function in this exact source context.')
        if g == 'how':
            add(['interr'], 'Source gives how within a printed direct question.')
    if g in pronouns:
        add(pronouns[g], 'Explicit printed pronoun/deictic meaning; person/number only where stated by I/thou/he/thee, never number from you.')
    if g == 'An':
        add(['determiner', 'indef'], 'Source interlinear An is an indefinite determiner here; no cardinal value added merely from Ek resemblance.')
    if f in {'ki', 'Ki'}:
        # The source sometimes expands a bare connective as (He-replied-) that.
        new = [t for t in new if t != 'verb']
        add(['conj', 'complementizer'], 'Printed ki/Ki introduces quoted content. Parenthetic English (He-replied-) is not a printed target verb.')
    if g in {'and', 'then', 'Then', 'if'}:
        add(['conj'], 'Exact printed connective meaning and clause scope; no extra finite morphology.')
    if g == 'if':
        add(['conditional'], 'The printed if-connective introduces a condition; this is not a lexical verb tag.')
    if g == 'not':
        add(['negator'], 'Printed negation; no inferred negative inflection or tense.')
    if g == 'yes-Sir':
        add(['interj'], 'Printed short affirmative speech answer; Sir does not license a sociolinguistic honorific-rank tag.')
    if g == 'with':
        add(['adposition'], 'Printed independent with expression in the source phrase; no forced noun case or positional subtype.')
    if g in {'three', 'twenty-one'}:
        add(['num'], 'Explicit cardinal numeral meanings; repeated three tokens retained independently.')
    if g == 'said':
        add(['verb', 'pret', 'participle'], 'p237 explicitly describes the past-participle bōlō/bōlyō class and mixed Bangaru bōlā; exact occurrence is glossed said.')
        notes.append('Grierson describes past-participle bōlō as a Rohtak form with y dropped from bōlyō; he also identifies bōlā as a Bangaru form in this mixed specimen.' if f == 'bōlō' else 'Grierson identifies bōlā as a Bangaru form occurring alongside Ahirwati forms in this mixed specimen.')
    if g in past_verbs:
        add(['verb', 'pret'], 'Printed came/became/issued gives past occurrence meaning; no person, gender or participial subtype inferred.')
    if g in {'fallen', 'seated', 'dying-of', 'keeping'}:
        add(['verb'], 'Printed verbal/nonfinite meaning; English fallen/seated/dying/keeping alone does not define target participial subtype.')
        holds.append({'type': 'nonfinite-subtype', 'detail': 'Source verbal meaning retained without imposing a target participial subtype from English morphology alone.'})
    if g in {'occurring-was', 'hanging-was'}:
        add(['verb', 'pret'], 'The printed complex verbal expression explicitly includes was; no segmented target component rows or progressive morphology inferred.')
    if g == 'it-was-said':
        add(['verb', 'pret', 'pass'], 'Explicit printed it-was-said passive analysis; no supplied person or independently segmented auxiliary.')
    if g in {'become-is', 'spoiled-are'}:
        add(['verb'], 'Printed complete verbal expression; English is/are does not establish an intrinsic tense for every component.')
    if g in {'will-ask-for', 'wilt-thou-give', 'I-shall-give'}:
        add(['verb', 'fut'], 'Printed will/wilt/shall occurrence analysis explicitly supplies future meaning.')
        if g == 'wilt-thou-give':
            add(['2sg', 'second-person'], 'Thou is explicitly included in this word-aligned English analysis.')
        if g == 'I-shall-give':
            add(['1sg', 'first-person'], 'I is explicitly included in this word-aligned English analysis.')
    if g in {'ask-for', 'give-away'}:
        add(['verb', 'impv'], 'Exact printed direct instruction, preserved in the quoted speech.')
    if g == 'give':
        add(['verb'], 'Conditional if-you-give protasis; not an imperative simply because English give is bare.')
    if g == 'I-give-away':
        add(['verb', '1sg', 'first-person'], 'Printed I-give-away within a question; no future/modal/imperative forced.')
    if g in genitives:
        add(['gen'], 'Printed of analysis and the chapter genitive source class corroborate genitive function; no forced suffix-component segmentation.')
    if g == 'Hukamlō-of':
        add(['noun', 'proper-noun'], 'Named individual in the printed uncle Hukamlō phrase; raw Form/gloss discrepancy retained.')
    if g in {'me-to', 'Thee-to'}:
        reasons.append('p51 explicitly defines nai as dative/agent; this occurrence is printed to, distinguishing it from by.')
    if g == 'to-me':
        notes.append('The chapter describes mērai as a genitive-locative form used for the dative; this specimen prints mērai with to-me.')
        reasons.append('p51 explicit genitive-locative-for-dative claim; add dat function without labelling the word intrinsically genitive or locative.')
    if g in {'enquiry-for', 'the-Ahīr-to', 'brother-to', 'Jai-kaḷi-to', 'ring-to', 'mouth-from', 'Which-on', 'On-what'}:
        holds.append({'type': 'case-scope', 'detail': f'Printed relational meaning {g} retained; mixed-lect suffix analysis is not established by an English relation alone.'})
    if g == 'enquiry-for':
        add(['noun'], 'Printed enquiry nominal meaning; for does not itself establish target case.')
    if g == 'son-in-law-by':
        notes.append('The interlinear source explicitly gives an agent/by interpretation; Grierson describes nai as a dative and agent marker.')
        reasons.append('Agent/by retained as source interpretation; no modern ergative/instrumental split inferred.')
    if g == 'which':
        holds.append({'type': 'case-scope', 'detail': 'Source jaĩh-nai is word-aligned only as which; no dat/agent or hypothetical oblique number imported from the general chapter.'})
    if g == 'his-own':
        reasons.append('p51 explicitly gives apᵃṇ- as own; no third-person possessor/gender copied from English his-own onto reflexive form.')
    if g == 'art':
        add(['2sg', 'second-person'], 'Exact source art (distinct from is), aligned in explicit thou speech; copular surface is not globally person-specific.')
    if g == 'red-turban-person':
        reasons.append('Complete person-denoting referent remains noun; red does not justify making this full form an adjective or splitting unprinted lexical bases.')
    new = list(dict.fromkeys(new))
    if holds and 'uncertain' not in new:
        new.append('uncertain')
    assert set(new) <= GRAMMATICAL_TAGS | GENDER_TAGS, (key, new)
    assert new, (key, g, 'Unclosed lexical scope')
    decisions.append({'source_unit_key': key, 'exact_form_guard': a['emission_form'],
        'exact_gloss_guard': a['emission_gloss'], 'old_tags': a['tags'], 'proposed_tags': new,
        'proposed_source_notes': notes, 'source_scoped_basis': reasons or ['Existing source-explicit metadata accepted without extension.'],
        'additional_scope_holds': holds, 'literal_uncertainties_preserved': a['typed_uncertainties'],
        'decision': 'keyed metadata proposal only; literal/Form/gloss/Native/Phonemic and source keys protected'})
write('all-132-keyed-metadata-decisions.jsonl', decisions)
write('keyed-tag-change-proposals.jsonl', [r for r in decisions if r['old_tags'] != r['proposed_tags']])

positions = {k: n for n, k in enumerate(by_physical)}
expected_join = read(P / 'physical-break-join-proposal.jsonl')[0]
assert expected_join['physical_source_keys'] == [
    'lsi9.2:p238:aligned-line:4:word:10', 'lsi9.2:p238:aligned-line:5:word:1']
assert expected_join['emission_form'] == 'Jai-kaḷi-kai'
boundary_decisions = []
for w in wholes:
    ks = w['physical_source_keys']
    assert ks == list(by_physical)[positions[ks[0]]:positions[ks[-1]] + 1]
    assert w['first_word_key'] == ks[0] and w['last_word_key'] == ks[-1]
    expected_english = ' '.join(by_physical[k]['raw_gloss'] for k in ks)
    assert w['emission_gloss'] == w['printed_interlinear_analysis'] == expected_english
    selected = [a for a in atoms if set(a['physical_source_keys']) & set(ks)]
    assert all(set(a['physical_source_keys']) <= set(ks) for a in selected)
    assert w['emission_form'] == ' '.join(a['emission_form'] for a in selected)
    assert w['emission_native'] == '' and w['phonemic'] == '' and not w['graph_claims']
    boundary_decisions.append({'historical_whole_key': w['key'], 'proposed_stable_key': ks[0] + ':expression',
        'first_key': ks[0], 'last_key': ks[-1], 'physical_source_keys': ks,
        'exact_form_guard': w['emission_form'], 'exact_gloss_guard': w['emission_gloss'],
        'source_scope': w['scope'], 'tags_decision': 'Existing tags accepted; short affirmative spans remain without sentential.',
        'boundary_decision': 'Original-backed sentence/quotation span accepted, with intentional nested overlap; physical rows do not independently define sentences.',
        'identity_correction': 'Replace ordinal whole key by first immutable physical key + :expression; retain historical key and exact contributors in audit.'})
assert len({r['proposed_stable_key'] for r in boundary_decisions}) == 23
assert set(k for w in wholes for k in w['physical_source_keys']) == set(by_physical)
write('all-23-keyed-boundary-decisions.jsonl', boundary_decisions)

# Retain a payload hash for every protected lexical and whole proposal. These
# are independent guards, not a rewritten derivative of the author's inputs.
protected = [{'key': a['emission_key'], 'kind': 'lexical',
              'protected_payload_sha256': hashlib.sha256(json.dumps(a, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
              'form': a['emission_form'], 'gloss': a['emission_gloss'], 'physical_keys': a['physical_source_keys']}
             for a in atoms]
protected += [{'key': w['key'], 'kind': 'whole',
               'protected_payload_sha256': hashlib.sha256(json.dumps(w, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
               'form': w['emission_form'], 'gloss': w['emission_gloss'], 'physical_keys': w['physical_source_keys']}
              for w in wholes]
write('all-155-protected-source-payloads.jsonl', protected)
image_manifest = json.loads((P / 'original-crop-provenance.json').read_text())
roman_images = [r for r in image_manifest if '/original-roman-' in r['path']]
for r in roman_images:
    assert sha(ROOT / r['path']) == r['sha256']
    assert sha(ROOT / r['source']) == r['source_sha256']
grammar = HERE.parents[1] / 'grammar/child-witnesses-reviewed.jsonl'
assert sha(grammar) == '5ac645b13ae1a3132ac094431fbce9376037705f93c0c59d9861e6e7ef999ee7'
commentary = HERE.parent / 'root-documentary-review/p237-linguistic-commentary-native.png'
report = {'status': 'Full independent metadata and whole-boundary review finished; keyed source-local corrections required before final freeze',
    'reviewer': '/root/asur_finish',
    'counts': {'physical_Roman_atoms': 133, 'lexical_emissions_reviewed': 132, 'whole_spans_reviewed': 23,
               'tag_changes_proposed': sum(r['old_tags'] != r['proposed_tags'] for r in decisions),
               'whole_identity_changes_proposed': 23, 'literal_fields_changed': 0,
               'source_glyph_holds_preserved': 10, 'Native_space_groups_NOT_certified': 194},
    'scope': ['Viewed all three retained native-pixel Roman crops for the complete p238 narrative and dialogue.',
              'Viewed p237 original linguistic commentary; source explicitly calls the first sentence good Ahirwati and second good Bangaru, with mixed forms throughout.',
              'Cross-checked exact p51 grammar claims without globally copying hypothetical or regional pronoun morphology.',
              'All132 lexical records and23 source-bound expressions reviewed; no broad ordinary-English grammatical extraction proposed.'],
    'findings': ['Clear lexical POS/person meanings were omitted across the full sparse proposal; exact-key additions are listed, without English object-case or automatic -ing morphology.',
                 'Whole English already correctly preserves exact concatenated interlinear analysis; no smooth translation supplied or fabricated.',
                 'One cross-line Jai-kaḷi- / kai printer continuation is source-supported; every133 physical Roman key occurs once in132 lexical emissions.',
                 'All23 current source-bounded spans accepted; ordinal IDs must be replaced by first physical anchors.',
                 'No intrinsic case forced on kai/sō/from/on relational meanings without source-specific evidence; source nai dative/agent and mērai used-for-dative claims retained precisely.'],
    'Native_limit': 'All lexical and whole Native fields remain blank. Eight broad script provenance spans and194 Persian spacegroups are structural/source evidence only. This report DOES NOT accept or certify194 Persian glyph readings or equate broad Native text to narrower Roman rows.',
    'protected_fields': 'No source proposal, freeze, raw literal/gloss/key/Native/Phonemic/graph field edited. Keyed whole identity changes are proposed separately with full historical-key guards.',
    'input_hashes': {str((P / n).relative_to(ROOT)): h for n, h in pins.items()},
    'additional_evidence': {str(grammar.relative_to(ROOT)): sha(grammar), str(commentary.relative_to(ROOT)): sha(commentary)},
    'Roman_originals': roman_images,
    'outputs': {f.name: sha(f) for f in HERE.glob('*.jsonl')},
    'review_script_sha256': sha(Path(__file__)),
    'deferred': ['Parent scope adjudication and guarded owner metadata/anchored-key derivative.',
                 'Independent Native glyph acceptance where required; final combined output review and source-stage installation.',
                 'DB/CLDF/full tests/browser/publication remain outside this review.']}
(HERE / 'independent-full-metadata-boundary-review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(report['counts'], ensure_ascii=False))

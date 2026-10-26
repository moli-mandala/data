"""Whole-chapter metadata/export proposal, never canonical installation."""
import csv
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = next(p for p in HERE.parents if (p / 'source_meta.py').exists())
sys.path.insert(0, str(DATA))
from tags import GRAMMATICAL_TAGS, GENDER_TAGS

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
FIRST = HERE / 'physical-units-first-reading.jsonl'
assert sha(FIRST) == '7bd8af78668aeff2453ebb99ae9fd0e8ad57212f10af6f3867164ea8e29f51c0'
units = [json.loads(s) for s in FIRST.read_text().splitlines()]
rows, audit, metadata = [], [], []
by_key = {}


def classify(unit, w, index):
    section, form, gloss = unit['section'], w['literal_first_reading'], w['printed_gloss']
    tags, notes, etymology = [], [], ''
    affix_sections = {'masculine-suffixes': ['m'], 'feminine-suffixes': ['f'], 'plural-suffixes': ['pl'],
        'accusative-suffix-cells': ['acc'], 'instrumental-suffix-cells': ['instr'], 'dative-suffix-cells': ['dat'],
        'genitive-suffix-cells': ['gen'], 'locative-suffix-cells': ['loc'], 'ablative-cells': ['abl'],
        'directive-cells': ['dir'], 'present-suffixes': ['verb', 'pres'], 'future-suffixes': ['verb', 'fut'],
        'relative-participle-marker': ['relative', 'participle']}
    case_sections = {'accusative-example': 'acc', 'instrumental-examples': 'instr', 'dative-examples': 'dat',
        'genitive-examples': 'gen', 'locative-examples': 'loc', 'ablative-examples': 'abl', 'directive-examples': 'dir'}
    if section in affix_sections:
        tags = ['affix', 'suffix'] + affix_sections[section]
    elif section in case_sections:
        tags = [case_sections[section]]
        if section == 'accusative-example':
            tags += ['pron', '1sg']; gloss = 'I'
            notes.append('Printed grammatical label: (Acc.).')
        if section == 'genitive-examples' and w['source_scope'] == 'Mullu Kurumba target':
            tags += ['pron', 'poss', '3sg', 'm' if index in (3, 4) else 'f']
        if section == 'dative-examples' and gloss == 'to us': tags += ['pron', '1pl']
        if section == 'locative-examples' and gloss == 'in me': tags += ['pron', '1sg']
        if section == 'locative-examples' and gloss == 'in that': tags += ['pron', 'demonstrative']
        if section == 'ablative-examples' and w['source_scope'] == 'Mullu Kurumba target':
            tags += ['uncertain']
            notes.append("Printed gloss is 'from the trap', despite preceding en- pronoun examples. Preserve the source conflict; no first-person analysis assigned.")
    elif section in ('masculine-examples', 'feminine-examples', 'plural-examples'):
        tags = ['noun', 'm' if section == 'masculine-examples' else 'f' if section == 'feminine-examples' else 'pl']
        if gloss in ('wife', 'daughter'): tags += ['kinship']
    elif section.startswith('pronouns-'):
        tags = ['pron']
        if 'first-' in section: tags += ['first-person', 'sg' if 'singular' in section else 'pl']
        if 'second-' in section: tags += ['second-person', 'sg' if 'singular' in section else 'pl']
        if 'first-plural' in section: tags += ['inclusive' if index == 1 else 'exclusive']
        if 'third-' in section:
            tags += ['third-person', 'sg' if 'singular' in section else 'pl', 'demonstrative']
            tags += ['prox' if index <= (2 if section == 'pronouns-third-plural' else 1) else 'dist']
            if 'masculine' in section: tags += ['m']
            if 'feminine' in section: tags += ['f']
            if 'neuter' in section: tags += ['n']
        if form in ('en-', 'nin-'): tags += ['stem', 'obl']
    elif section in ('interrogative-bases', 'interrogative-pronouns'):
        tags = ['pron', 'interr']
        if section == 'interrogative-bases': tags += ['stem']
        if gloss == 'who-he': tags += ['m']
        if gloss == 'who-she': tags += ['f']
    elif section in ('past-examples', 'present-examples', 'future-examples'):
        tags = ['verb', {'past-examples': 'pret', 'present-examples': 'pres', 'future-examples': 'fut'}[section]]
        if gloss.endswith('-I'): tags += ['1sg']
        if gloss.endswith('-he'): tags += ['3sg', 'm']
        if gloss.endswith('-it'): tags += ['3sg', 'n']
    elif section == 'relative-participle-examples': tags = ['verb', 'relative', 'participle']
    elif section == 'verbal-participle-examples': tags = ['verb', 'participle']
    elif section.startswith('suffix-'):
        tags = ['affix', 'suffix']
        if 'first-person' in section: tags += ['first-person', 'sg' if index == 1 else 'pl']
        if 'second-person' in section: tags += ['second-person', 'sg' if index == 1 else 'pl']
        if 'third-' in section:
            tags += ['third-person', 'pl' if section == 'suffix-third-plural-cell' else 'sg',
                'f' if 'feminine' in section else 'm']
    if section == 'feminine-suffixes':
        etymology = 'Nair describes these as reflexes of Tamil and Malayalam feminine suffixes; no independently printed donor form or Jambu ancestry link is supplied.'
    if section == 'plural-suffixes': notes.append('One shared printed suffix list is explicitly attributed to both Malayalam and Mullu Kurumba; language-scoped witnesses do not imply duplicate physical printing.')
    if section == 'relative-participle-marker': notes.append('The one printed marker is explicitly stated to occur as in Malayalam; the two language-scoped witnesses share this printed attestation.')
    if section in ('genitive-examples', 'future-examples', 'relative-participle-examples', 'verbal-participle-examples'):
        notes.append(unit['source_claim'])
    if section == 'suffix-third-plural-cell': notes.append('Printed in masculine row of the pronominal-suffix table; feminine plural cell is empty. This is distinct from the gender-unspecified pronoun variants on p56.')
    if section == 'pronouns-third-plural': notes.append('Plural proximal/distal variants; no separate masculine/feminine assignment is supplied for these shared pronoun forms.')
    if ' ' in form: tags += ['multiword-expression']
    tags = sorted(set(tags))
    assert set(tags) <= GRAMMATICAL_TAGS | GENDER_TAGS, (w['witness_key'], tags)
    return gloss, tags, ' '.join(n for n in notes if n), etymology


for unit in units:
    emissions, decisions = [], []
    for index, witness in enumerate(unit['source_witnesses'], 1):
        key = witness['witness_key']
        disposition = witness['disposition']
        if disposition not in ('target', 'explicit_comparison'):
            decisions.append(dict(witness_key=key, disposition=disposition, entry_keys=[], reason=unit.get('note') or unit['source_claim']))
            continue
        language = 'Mullu Kurumba' if disposition == 'target' else 'Malayalam'
        # Six explicitly repeated commentary forms reuse the exact earlier table
        # witnesses, with physical occurrences and source claims retained in audit.
        if unit['section'] == 'pronoun-commentary-occurrences':
            match = [r for r in rows if r[0] == language and r[2] == witness['literal_first_reading'] and r[3] == '']
            assert len(match) == 1
            entry_key = match[0][10]
            emissions.append(entry_key)
            decisions.append(dict(witness_key=key, disposition='exact_repeated_attestation_reuse', entry_keys=[entry_key],
                reason='Same explicitly printed pronoun and scope as preceding table; repeated source occurrence retained here.'))
            continue
        gloss, tags, notes, etymology = classify(unit, witness, index)
        row = [language, '', witness['literal_first_reading'], gloss, '', '', notes,
            f"nair2013mullukurumba[p. {unit['printed_page']}]", '', etymology, key, '', '', '', ' '.join(tags)]
        rows.append(row); by_key[key] = row; emissions.append(key)
        decisions.append(dict(witness_key=key, disposition='proposed_lexical_emission', entry_keys=[key],
            language_id=language, raw_printed_gloss=witness['printed_gloss'], grammar_tags=tags))
        metadata.append(dict(witness_key=key, printed_page=unit['printed_page'], source_section=unit['section'],
            source_claim=unit['source_claim'], raw_gloss=witness['printed_gloss'], proposed_gloss=gloss, tags=tags))
    audit.append(dict(source_unit_key=unit['source_unit_key'], printed_pages=[unit['printed_page']],
        entry_keys=emissions, witness_dispositions=decisions, source_record=unit,
        accepted_for_installation=False, review='Complete candidate; independent literal/metadata review and root approval pending'))

edges = []
for child, parent in [("nair2013mullukurumba:p56:pronouns-third-plural:w2", "nair2013mullukurumba:p56:pronouns-third-plural:w1"),
    ("nair2013mullukurumba:p56:pronouns-third-plural:w4", "nair2013mullukurumba:p56:pronouns-third-plural:w3")]:
    by_key[child][11] = parent
    by_key[child][14] = ' '.join(sorted(set(by_key[child][14].split()) | {'alternate'}))
    edges.append(dict(child_key=child, variant_of_key=parent, evidence='p56 prose explicitly says ivicca/avicca are in free variation with ivaru/avaru, preserving proximal/distal pairing.'))
assert len(rows) == len(by_key) == 135 and len(audit) == 65
assert set(k for a in audit for k in a['entry_keys']) == set(by_key)
assert len([d for a in audit for d in a['witness_dispositions']]) == 164
assert all(not r[4] and not r[5] and not r[12] and not r[13] for r in rows)
with (HERE / 'proposed.csv').open('w', newline='') as stream: csv.writer(stream).writerows(rows)
(HERE / 'proposal-audit.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False, sort_keys=True) + '\n' for a in audit))
(HERE / 'metadata-proposal.json').write_text(json.dumps(dict(keyed_records=metadata, accepted=False), ensure_ascii=False, indent=2) + '\n')
(HERE / 'graph-proposal.json').write_text(json.dumps(dict(proposals=edges, accepted=False,
    exclusions='Morphological analysis arrows, source reflex/comparison claims and unlocated ethnonyms do not become automatic ancestry/variant edges.'), indent=2) + '\n')
(HERE / 'candidate-status.json').write_text(json.dumps(dict(status='Whole-source review proposal; not installed',
    rows=len(rows), target_rows=sum(r[0]=='Mullu Kurumba' for r in rows), comparison_rows=sum(r[0]=='Malayalam' for r in rows),
    audit_groups=65, scoped_witness_records=164, exact_commentary_reuses=6, variant_proposals=2,
    inputs={'first_reading': sha(FIRST)}, outputs={n: sha(HERE/n) for n in ('proposed.csv', 'proposal-audit.jsonl', 'metadata-proposal.json', 'graph-proposal.json')},
    canonical_changes=False, pending=['independent full literal/scope review', 'source metadata/profile/parser', 'fresh original output audit', 'root acceptance and guarded installation']), indent=2) + '\n')
print(json.dumps({'rows': len(rows), 'audit_groups': 65, 'canonical_changes': False}))

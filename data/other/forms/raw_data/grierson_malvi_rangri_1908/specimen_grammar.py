"""Conservative grammatical interpretation of explicit interlinear English labels.

Source forms and translations are never changed here. Case readings which English
alone cannot distinguish (notably by = agent/instrument and to = acc/dat/goal)
remain typed audit observations rather than invented specific case tags.
"""
import re

PRONOUNS = {
    'i': ['pron', '1sg'], 'me': ['pron', '1sg'], 'my': ['pron', '1sg', 'poss'],
    'we': ['pron', '1pl'], 'us': ['pron', '1pl'], 'our': ['pron', '1pl', 'poss'],
    'thou': ['pron', '2sg'], 'thee': ['pron', '2sg'], 'thy': ['pron', '2sg', 'poss'],
    'thine': ['pron', '2sg', 'poss'], 'you': ['pron', 'second-person'],
    'your': ['pron', 'second-person', 'poss'], 'he': ['pron', '3sg'],
    'him': ['pron', '3sg'], 'his': ['pron', '3sg', 'poss'],
    'himself': ['pron', '3sg', 'refl'], 'his-own': ['pron', '3sg', 'poss', 'refl'],
    'they': ['pron', '3pl'], 'them': ['pron', '3pl'],
    'your-honour': ['pron', 'honorific', 'second-person'],
    'anyone': ['pron', 'indef'],
}
NUMERALS = {'one', 'two', 'twelve', 'twenty', 'thirty', 'thousands', 'twenty-five', 'two-and-a-half'}
INFINITIVES = {'to-do', 'to-fall', 'to-go', 'to-laugh', 'to-live', 'to-make', 'to-celebrate', 'to-entreat', 'to-be-called-for', 'to-be-shown'}
PAST_VERBS = {'came', 'fell', 'began', 'became', 'became-audible', 'lived', 'remained', 'gushed-out', 'took-place', 'fixed-remained', 'he-sat-down', 'broken-fell-down'}
SIMPLE_VERBS = {'bring', 'do', 'entertain', 'give-out', 'listen-o', 'make', 'makes', 'put-on', 'robs', 'see', 'he-goes'}
PARTICIPLES = {'being', 'coming', 'running', 'standing', 'while-coming-walking'}
TEMPORAL = {'then', 'when', 'now', 'to-day', 'afterwards', 'again', 'always', 'ever', 'at-last', 'this-time'}
SPATIAL = {'there', 'here', 'somewhere', 'here-from', 'there-from', 'that-place-from', 'near', 'in-front', 'in-that-place', 'onwards', 'back', 'aloof'}


def classify(unit):
    gloss = unit.get('emission_gloss', unit['gloss'])
    g = gloss.lower()
    tags = list(unit.get('tags', []))
    observations = []
    # A grouped emission has its own clear composite gloss; unreconciled cells do not.
    if 'uncertain' in tags and 'transposition' in unit.get('note', '').lower() and 'emission_gloss' not in unit:
        return tags, ['Source interlinear transposition retained; no grammatical inference from the mismatched atomic gloss.']
    if g.endswith('-having') or g.startswith(('having-', 'having ')):
        tags += ['verb', 'conjunctive-participle']
    elif g in INFINITIVES:
        tags += ['verb', 'inf'] + (['pass'] if g.startswith('to-be-') else [])
    elif g in {'am', 'is', 'was', 'were', 'are'}:
        tags += ['verb', 'copula', 'pret' if g in {'was', 'were'} else 'pres']
    elif g in {'not-am', 'there-is'}:
        tags += ['verb', 'copula', 'pres'] + (['neg'] if g == 'not-am' else [])
    elif re.match(r'^(?:it-|they-)?(?:was|were)-', g):
        tags += ['verb', 'pret', 'pass']
    elif g.startswith('is-') or g.endswith(('-is', '-was', '-were', '-had', '-am', '-art')):
        tags += ['verb', 'pret' if g.endswith(('-was', '-were', '-had')) else 'pres']
        if g.endswith('-had'):
            tags += ['perfect']
        if g in {'given-is', 'obtained-is', 'is-found', 'is-got', 'is-met', 'killed-was', 'saved-was'}:
            tags += ['pass']
        if any(x in g for x in ['doing-', 'dying-', 'eating-', 'lying-', 'witnessing-']):
            tags += ['progressive']
    elif g == 'has-been-squandered':
        tags += ['verb', 'perfect', 'pass']
    elif '-will-' in g or g.startswith('will-'):
        tags += ['verb', 'fut']
    elif '-may-' in g:
        tags += ['verb', 'subjunctive']
    elif g in {'could-have-lived', 'would-have-lived'}:
        tags += ['verb', 'modal', 'perfect']
    elif g in PAST_VERBS:
        tags += ['verb', 'pret']
    elif g in SIMPLE_VERBS:
        tags += ['verb']
    elif g in PARTICIPLES:
        tags += ['verb', 'participle']
    elif g.startswith('being-'):
        tags += ['verb', 'participle', 'pass']
    if 'verb' in tags:
        if g.startswith('i-') or g.endswith('-am'):
            tags.append('1sg')
        elif g.startswith('we-'):
            tags.append('1pl')
        elif g.startswith('he-'):
            tags.append('3sg')
        elif g.startswith('they-'):
            tags.append('3pl')
        elif g.endswith('-thou-art') or g.endswith('-art'):
            tags.append('2sg')
        elif g.startswith('you-'):
            tags.append('second-person')
        if 'causing-to-' in g:
            tags.append('caus')
    if g in {'and', 'but', 'or', 'if', 'because-that'}:
        tags.append('conj')
    elif g == 'not':
        tags.append('negator')
    elif g in TEMPORAL:
        tags += ['adv', 'temporal']
    elif g in SPATIAL:
        tags += ['adv', 'spatial']
    elif g in {'indeed', 'verily', 'even', 'also', 'too'}:
        tags.append('part')
    if g in PRONOUNS:
        tags += PRONOUNS[g]
    # Only remove explicit relational English markers to recognize an attested
    # pronoun; do not infer a citation form or alter the installed form/gloss.
    base = g
    for prefix in ['by-', 'of-']:
        if base.startswith(prefix):
            base = base[len(prefix):]
    for suffix in ['-from', '-of', '-to', '-in', '-among']:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
    if base in PRONOUNS:
        tags += PRONOUNS[base]
    if g in NUMERALS or g.removesuffix('-of') in NUMERALS:
        tags.append('num')
    if g.endswith('-of') or g.startswith('of-') or g.endswith('’s'):
        tags.append('gen')
    if g.endswith('-from'):
        tags.append('abl')
    if g.endswith('-in'):
        tags.append('loc')
    if g.startswith('by-') or g.endswith('-by'):
        observations.append('English by does not by itself distinguish source agent case from instrumental or another construction; no specific case invented.')
    if g.endswith('-to'):
        observations.append('English to does not by itself distinguish accusative, dative or goal; no specific case invented.')
    return list(dict.fromkeys(tags)), observations

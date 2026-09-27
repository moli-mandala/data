"""Source-scoped grammatical gloss parsing; preserve unrecognized English."""
import re

LABELS = {'IMP': ['impv'], 'ABS': ['abs'], 'HH': ['honorific', 'high-honorific'],
          'PROS': ['prospective'], 'CLF': ['classifier'], 'PST': ['pret']}


def parse(gloss, form):
    exact = {'particle.F': ('', ['part', 'f']), 'particle.M': ('', ['part', 'm']),
             'across/COND.PART': ('across', ['conditional', 'part']),
             'PROS': ('', ['prospective']), '3SG': ('', ['3sg']),
             'past tense marker': ('past tense', ['pret']),
             'non-past tense marker': ('non-past tense', ['non-past']),
             'causative suffix': ('causative', ['caus'])}
    if gloss in exact:
        cleaned, tags = exact[gloss]
        tags = list(tags)
    elif gloss.startswith('ONO, '):
        cleaned, tags = gloss[5:], ['onomatopoeia']
    else:
        match = re.fullmatch(r'(.+?)[.\-]\s*(IMP|ABS|HH|PROS|CLF|PST)', gloss)
        cleaned, tags = (match[1], list(LABELS[match[2]])) if match else (gloss, [])
    if form.startswith('-'):
        tags.append('suffix')
    elif form.endswith('-'):
        tags.append('stem')
    return cleaned, list(dict.fromkeys(tags))

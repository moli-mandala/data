"""Explicit relational analyses from original Naiki specimens and grammar p570.

Only the source's expressly described genitive -nē and locative -lōpul are
structured here. Other English prepositions do not automatically establish a
morphological case. Whole forms and original translations remain in the audit.
"""

GENITIVES = '''
tōlēnē|brother-of|brother
bānē|father-of|father
akāsnē|heaven-of|heaven
bāykōnē|women-of|women
'''

LOCATIVES = '''
dēsām-lōpul|country-in|country
vāvar-lōpul|field-in|field
kāl-lōpul|foot-on|foot
ārasā-lōpul|glass-in|glass
kayi-lōpul|hand-on|hand
chākar-lōpul|servants-in|servants
'''

def decisions():
    result = {}
    for text, case, source in [(GENITIVES, 'gen', 'genitive'),
                               (LOCATIVES, 'loc', 'locative')]:
        for line in text.strip().splitlines():
            form, gloss, lexical = line.split('|')
            pair = form.casefold(), gloss.casefold()
            assert pair not in result
            result[pair] = dict(tags=['noun', case], lexical_gloss=lexical,
                support=f'grierson1906naiki:p570:prose:{source}',
                basis='Original p570 explicitly identifies this case ending; original specimen form and aligned translation identify its scope. No automatic number or gender analysis.',
                scope='Only the listed exact form/gloss pair; preserve whole source form and source translation in audit.')
    return result

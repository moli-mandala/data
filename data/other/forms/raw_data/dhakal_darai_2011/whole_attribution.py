"""Source-explicit Darai attribution, independent of future registry installation."""
import copy
import json
from pathlib import Path

P = Path(__file__).resolve().parent
HISTORICAL_WITNESSES = {
    "Hodgson's (1985) list, as cited by Dhakal",
    'Kotapish and Kotapish (1975), cited by Dhakal; table source note on page172.',
    'Darai (1996:38), cited by Dhakal; explicitly assigned Tanahun in the following paragraph.',
    'Kotapish and Kotapish (1975: 140–141), reproduced by Dhakal (2011: 375)',
}
CURRENT_WITNESSES = {'Word list today'}
OVERRIDES = {
    'dhakal2011:p2:ch12:l29:a1': ('historical cited witness; locality unspecified', 'Hodgson list, as cited by Dhakal'),
    'dhakal2011:p5:ch12:l9:a1': ('historical cited witness; locality unspecified', 'Paudyal (2008), as cited by Dhakal'),
    'dhakal2011:p17:ch12:l5:a1': ('cited cultural term; target-language and dialect attribution uncertain', 'Darai (1996:40), as cited by Dhakal'),
    'dhakal2011:p37:ch12:l8:a1': ('radio programme title; dialect unspecified', 'Vijaya FM programme, reported by Dhakal'),
}
for suffix in ['p3:ch12:l20:a1','p3:ch12:l21:a1','p3:ch12:l21:a2','p3:ch12:l21:a3','p3:ch12:l30:a1']:
    OVERRIDES['dhakal2011:'+suffix] = ('historical cited witness; locality unspecified', 'Kotapish (1973), as cited by Dhakal')


def apply_attribution(rows):
    contexts = json.loads((P / 'appendix5-source-context-20260927.json').read_text())['text_sets']
    by_key = {r['source_unit_key']: r for r in rows}
    resolved = {}

    def resolve(key, trail=()):
        if key in resolved:
            return resolved[key]
        assert key not in trail, ('attribution ancestry cycle', trail, key)
        row = by_key[key]
        claims = row['source_claims']
        parent = row.get('parent_expression_key') or row.get('physical_source_unit_key')
        inherited = resolve(parent, trail+(key,)) if parent in by_key and parent != key else None
        lect = claims.get('lect')
        witness = claims.get('witness', claims.get('attribution'))
        if witness:
            assert witness in HISTORICAL_WITNESSES | CURRENT_WITNESSES, ('Unreviewed witness scope', witness)
        if row['disposition'].startswith('audit-only'):
            attribution = {'scope':'audit-only control','lect':lect}
        elif key in OVERRIDES:
            scope, source = OVERRIDES[key]
            attribution = {'scope':scope,'lect':None,'witness':source}
        elif lect in {'Chitwan','Tanahun','Nawalparasi'}:
            attribution = {'scope':'explicit source dialect','lect':lect}
        elif witness in HISTORICAL_WITNESSES:
            attribution = {'scope':'historical cited witness; locality unspecified','lect':None}
        elif witness in CURRENT_WITNESSES:
            attribution = {'scope':'explicit contemporary witness; author Chitwan default','lect':'Chitwan'}
        elif inherited and inherited['scope'] != 'audit-only control':
            attribution = copy.deepcopy(inherited)
        else:
            attribution = {'scope':'author Chitwan default, printed page 8','lect':'Chitwan'}
        if witness:
            attribution['witness'] = witness
        if claims.get('consultant_provenance'):
            attribution['consultant_provenance'] = claims['consultant_provenance']
        for context in contexts:
            if key.startswith('dhakal2011:appendix5:'+context['prefix']+'.'):
                attribution['corpus_context'] = copy.deepcopy(context)
                attribution['recording_location_is_not_a_dialect_assignment'] = True
                break
        resolved[key] = attribution
        return attribution

    for row in rows:
        row['source_attribution'] = resolve(row['source_unit_key'])
    return rows

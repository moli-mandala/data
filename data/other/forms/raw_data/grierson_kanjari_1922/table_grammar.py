"""Explicit standard-list prompt grammar, shared LSI numbering; no inferred lemma."""
def grammar(item):
    """Only categories explicitly supplied by source prompts or paradigm contrasts."""
    tags=[]
    if item<=13: tags=['num']
    elif item<=31:
        tags=['pron']
        if item in [15,18,21,24,27,30]: tags+=['gen']
        if item in [16,19,22,25,28,31]: tags+=['poss']
        person='first-person' if item<=19 else 'second-person' if item<=25 else 'third-person'
        tags+=[person]
    elif 32<=item<=76: tags=['noun']
    elif 77<=item<=85: tags=['verb']
    elif 86<=item<=91: tags=['adv']
    elif 92<=item<=94: tags=['interr']
    elif 95<=item<=97: tags=['conj']
    elif item==100: tags=['interj']
    elif 101<=item<=118:
        tags=['noun']
        if item in [102,107,111,116]:tags+=['gen']
        elif item in [103,108,112,117]:tags+=['dat']
        elif item in [104,109,113,118]:tags+=['abl']
        if item in [105,106,107,108,109,114,115,116,117,118]:tags+=['pl']
    elif 119<=item<=131:
        tags=['multiword-expression']
        if item in [120,125]:tags+=['gen']
        elif item in [121,126]:tags+=['dat']
        elif item in [122,127]:tags+=['abl']
    elif 132<=item<=137:tags=['adj']
    elif 138<=item<=155:tags=['noun']
    elif 156<=item<=219:
        tags=['verb']
        if item in [169,176]:tags+=['inf']
        if item in [170,177,218]:tags+=['participle']
        if item in [171,178]:tags+=['conjunctive-participle']
        if 185<=item<=190:tags+=['pret']
        if item in [173,195,196,197,198,199,200,204]:tags+=['fut']
        if item in [202,203,204]:tags+=['pass']
    else:tags=['multiword-expression']
    return tags

def explicit_table_tags(item):
    tags = grammar(item)
    if 14 <= item <= 31:
        tags += ['pl' if item in {17,18,19,23,24,25,29,30,31} else 'sg']
    if 138 <= item <= 155:
        if item in {138,142,146,150,153}: tags += ['m','sg']
        if item in {139,143,147,151,154}: tags += ['f','sg']
        if item in {140,141,144,145,148,149,152,155}: tags += ['pl']
        if item in {141,145,149}: tags += ['f']
    for start in (156,162,179,185,195,205,211):
        if start <= item < start+6:
            offset=item-start
            tags += [['1sg','2sg','3sg','1pl','2pl','3pl'][offset], 'sg' if offset<3 else 'pl']
    if item in {172,173,174,191,192,193,194,201,202,203,204}: tags += ['1sg','sg']
    if 162<=item<=167 or 211<=item<=216: tags += ['pret']
    if item in {191,192}: tags += ['progressive']
    if item in {192,193,203}: tags += ['pret']
    if item == 219: tags += ['participle']
    if 101 <= item <= 118:
        tags += ['m' if item <= 109 else 'f']
        if item in {101,102,103,104,110,111,112,113}: tags += ['sg']
    if 119 <= item <= 131:
        tags += ['noun', 'f' if item in {128,130,131} else 'm']
        tags += ['pl' if item in {123,124,125,126,127,130} else 'sg']
    if any(start <= item < start+6 for start in (156,179,205)): tags += ['pres']
    return list(dict.fromkeys(tags))

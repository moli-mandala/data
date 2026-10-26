"""Explicit, owner-read source heading scopes; no prose keyword classification.

Writes a review decision file only, without changing the frozen proposal.
"""
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
SUFFIX_KEYS = {
    'p305:ch11:prose:l6:a1', 'p305:ch11:prose:l7:a1', 'p305:ch11:prose:l7:a2',
    'p310:ch11:prose:l37:a1', 'p314:ch11:prose:l50:a1',
    'p316:ch11:prose:l42:a1', 'p316:ch11:prose:l42:a2',
    'p319:ch11:prose:l60:a1', 'p319:ch11:prose:l60:a2', 'p319:ch11:prose:l60:a3',
    'p321:ch11:prose:l41:a1', 'p326:ch11:prose:l4:a1', 'p342:ch12:prose:l54:a1',
    'p350:ch13:prose:l25:a1', 'p350:ch13:prose:l27:a1',
    'p353:ch13:prose:l22:a1', 'p353:ch13:prose:l26:a1',
    'p354:ch13:prose:l89:a1', 'p358:ch13:prose:l5:a1',
    'p354:ch13:table13.2:Darai:col1', 'p354:ch13:table13.2:Darai:col5',
}


def decide(row):
    page, kind = row['printed_page'], row['kind']
    key = row['source_unit_key']
    match = re.search(r':l(\d+)', key)
    line = int(match[1]) if match else None
    tags, note = [], ''
    if kind == 'listed-head-group':
        if page == 126:
            tags, note = ['adj'], 'Source §5.1.1 lists this as a dimension adjective.'
        elif page == 127:
            category = 'age' if line < 21 else 'value' if line < 43 else 'colour'
            tags, note = ['adj'], f'Source §5.1 lists this as an {category} adjective.'
        elif page == 128:
            category = 'colour' if line < 8 else 'physical-property'
            tags, note = ['adj'], f'Source §5.1 lists this as a {category} adjective.'
        elif page == 129:
            tags, note = ['adj'], 'Source §5.1.6 lists this as a human-propensity adjective.'
        elif page == 130 or page == 131 and line < 25:
            tags, note = ['num'], 'Source §5.1.7 lists this as a cardinal numeral.'
        elif page == 131:
            tags, note = ['adj', 'quantifier'], 'Source §5.1.7 calls these adjectival non-numeral quantifiers, used attributively or predicatively.'
        elif page == 137:
            columns = {61:['interr'],63:['demonstrative','prox'],67:['demonstrative','dist'],69:['relative'],
                       73:['interr'],76:['demonstrative','prox'],79:['demonstrative','dist'],82:['relative'],
                       86:['interr'],88:['demonstrative','prox'],90:['demonstrative','dist'],92:['relative']}
            tags = ['adj'] + columns[line]
            note = 'Source Table 5.2 lists this among pronominal determiners (pronominal adjectives), under its explicit interrogative, demonstrative or relative column.'
        elif page == 138:
            derived = line in {18,31,45}
            tags = ['adj','derived'] if derived else ['noun','proper-noun']
            note = ('Source example 23 explicitly derives this adjective from the neighbouring place name.' if derived
                    else 'Source example 23 labels this as a place name, the base of a derived adjective.')
        elif page in {206,207,208,209,210,211,212,217}:
            tags = ['adv']
            if page == 206:
                tags += ['manner']; note = 'Source §7.1.1 lists this as a manner adverb.'
            elif page in {207,208} or page == 209 and line < 38:
                tags += ['temporal']; note = 'Source §7.1.2 lists this among time, aspectuality or calendrical adverbs.'
            elif page == 209:
                tags += ['temporal']; note = 'Source §7.1.3 lists this as a frequency adverb.'
            elif page == 210:
                tags += ['emph'] if line < 48 else ['degree']
                note = ('Source §7.1.4 lists this as an emphatic adverb; its footnote cautiously suggests Nepali borrowing.' if line < 48
                        else 'Source §7.1.4 lists this as a degree adverb.')
            elif page == 211:
                tags += ['spatial']; note = 'Source §7.1.5 lists this among adverbs of location and direction.'
            elif page == 212:
                tags += ['interr']; note = 'Source §7.1.6 lists this as an interrogative adverb.'
            else:
                tags += ['spatial']
                if line in {36,40,45,50}: tags += ['emph']
                note = 'Source example 30 contrasts spatial adverbs with forms bearing the emphatic marker -n.'
        elif page in {231,232,233}:
            # Already mapped from exact source-category fields. Record coverage
            # explicitly rather than adding an unsupported POS to ideophones.
            tags = row['proposed_grammar']['tags'][:]
            note = 'Existing exact category mapping verified against §§8.3–8.4; non-sound manner words are distinguished from onomatopoeia.'
        elif page in {277,278}:
            outputs = {277:{35,41,45,48,54,59,63},278:{7,11,16,20}}
            tags = ['verb']
            if line in outputs[page]: tags += ['caus','derived']
            if page == 277: tags += ['tr'] if line in outputs[page] else ['intr']
            note = ('Source example 72 explicitly pairs this causative output with its base.' if line in outputs[page]
                    else 'Source example 72 supplies this verb as the base of the paired causative.')
            if page == 277: note += ' The following prose explicitly contrasts intransitive bases with transitive outputs.'
        else:
            raise AssertionError(('Unreviewed listed-head scope', key))
    elif kind == 'paradigm-cell' and page == 173:
        tags = ['verb','intr','non-past']
        if 'non-honorific' in row['source_claims']['source_grammar']: tags += ['non-honorific']
        note = 'Source pp. 172–173, Table 6.2: non-past agreement of intransitive bosike ‘to sit’; person, number and honorific scope follow the printed row.'
    elif kind == 'paradigm-cell' and page == 178:
        tags = ['suffix']
        role = 'nominative or ergative' if 'Set1' in row['source_claims']['source_grammar'] else 'genitive or dative'
        note = f'Source Table 6.3: agreement suffix set selected by {role} argument marking; these are agreement suffixes, not case endings on the verb.'
    elif kind == 'agreement-matrix-cell' and page == 186:
        tags = ['suffix']
        note = 'Source Table 6.4 explicitly lists agreement suffix combinations. Subject and object features remain separately scoped.'
    elif kind == 'pronominal-adverb-table' and page == 215:
        tags = ['adv']; note = 'Source Table 7.1 explicitly lists pronominal adverbs; semantic category and interrogative/demonstrative/relative column are separately retained.'
    elif kind == 'relative-correlative-table' and page == 307:
        tags = ['relative'] if row['source_claims']['source_role'] == 'antecedent' else ['correlative']
        note = 'Source Table 11.2 contrasts relative antecedent forms with their main-proposition correlatives; no single POS is imposed on the mixed pronoun/adjective/adverb heading.'
    elif kind in {'paradigm-root','paradigm-inflection'}:
        assert 367 <= page <= 383
        tags = ['verb']
        note = ('Source Appendix 3 explicitly places this root in a verb paradigm; the root itself does not inherit the inflected columns’ tense/person features.'
                if kind == 'paradigm-root' else
                'Source Appendix 3 explicitly supplies this as an inflected verb; argument features retain their separately recorded roles.')
    elif kind == 'comparative-pronoun-cell' and page == 352:
        tags = ['pron']
        note = 'Source Table 13.1 explicitly identifies this Darai-column form as a personal pronoun.'
    elif kind == 'attributed-comparison' and page == 145:
        tags = ['pron','poss','1pl'] if line in {19,21} else ['adj']
        note = 'Source example 38 compares adjectives and possessive pronouns across the cited historical and contemporary witnesses; the possessive pair is kept distinct from the adjectives.'
    elif key.removeprefix('dhakal2011:') in SUFFIX_KEYS:
        tags = ['suffix']
        note = 'Source explicitly identifies this isolated bound marker; its grammatical function is retained separately.'
    elif key == 'dhakal2011:p222:ch8:prose:l29:a1':
        tags = ['clitic']
        note = 'Source §8.1 explicitly calls -ei an emphatic/focus clitic, attachable to several word classes.'
    elif key == 'dhakal2011:p158:ch6:prose:l37:a1':
        tags = ['multiword-expression']
        note = 'Source Table 6.1 supplies the complete high-grade honorific construction, not an isolated suffix.'
    return tags, note


if __name__ == '__main__':
    decisions = []
    for line in (P/'whole-expanded-metadata-review.jsonl').open():
        row = json.loads(line)
        if row['disposition'].startswith('audit-only'): continue
        tags, note = decide(row)
        remove = []
        if row['kind'] in {'prior-proposal-attestation','aligned-word'}:
            outputs = {'adverbializer':'adv','adjectivizer':'adj','nominalizer':'noun'}
            codes = {'ADVZ':'adverbializer','ADJV':'adjectivizer','ADJZ':'adjectivizer',
                     'NOML':'nominalizer','NOMN':'nominalizer'}
            markers = list(dict.fromkeys(codes[label] for label in row['proposed_grammar'].get('labels', []) if label in codes))
            if markers:
                assert row['source_gloss'], row['source_unit_key']
                tags += [outputs[tag] for tag in markers] + ['derived']
                remove = markers
                note += ' Source interlinear analysis marks derivation within this complete word; the output category is distinguished from the isolated derivational marker.'
        if row['kind'] in {'bound-morpheme','bound-morpheme-repeat','bound-morphology','derivation-affix','comparative-tense-cell'}:
            # A TAM marker is not itself a lexical verb. Its source grammatical
            # labels remain structured, but generic interlinear POS inference
            # must not erase the distinction between a word and its marker.
            remove += ['verb']
            note += ' This source item is an isolated grammatical marker, not a complete lexical verb.'
        if note:
            decisions.append({'source_unit_key':row['source_unit_key'], 'form':row.get('root_boundary_decision',{}).get('before_form',row['form']),
                              'printed_page':row['printed_page'], 'add_tags':tags,
                              'remove_tags':remove,
                              'source_heading_note':note})
    report = {'status':'Owner-read source heading reconciliation draft; not applied to frozen output',
              'policy':'Exact enumerated source sections, table columns and line positions, never generic prose scanning or punctuation-based POS guessing',
              'decisions':decisions, 'records':len(decisions)}
    (P/'whole-heading-scope-review-draft.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(len(decisions))

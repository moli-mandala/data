"""Review-layer punctuation and narrowly source-backed interlinear metadata.

The source's actual pronoun-table forms provide the grammatical evidence.
Neither spelling similarity nor English tense alone licenses tag propagation.
Unmatched tokens remain explicitly pending for further contextual review.
"""
import json
from pathlib import Path
from specimen_context import decisions as context_decisions
from specimen_relations import decisions as relation_decisions
from reconcile_specimen_metadata import decisions as reviewed_decisions

HERE = Path(__file__).resolve().parent

# Exact specimen form and translation -> explicit grammar-page evidence and tags.
# Cases with different spelling or conflicting contextual translations remain
# pending. Person features are not transferred from prose examples: p570 says
# the verb generally does not change for person.
PROSE_MATCHES = [
    ('oko', 'one', 570, 'one', 'num', None),
    ('iroṭēr', 'two', 570, 'two', 'num', None),
    ('yeraṇḍi', 'two', 570, 'two-neuter', 'num n', None),
    ('nāli', 'four', 570, 'four', 'num', None),
    ('pōrākōr', 'children', 570, 'children', 'noun pl', None),
    ('pōrākōr', 'sons', 570, 'children', 'noun pl', None),
    ('turrēl', 'swine', 570, 'swine', 'noun pl', None),
    ('sirlā', 'buffaloes', 570, 'buffaloes', 'noun pl', None),
    ('bānun', 'father-to', 570, 'father', 'noun dat', 'father'),
    ('pōrīn', 'girl-to', 570, 'daughter', 'noun dat', 'girl'),
    ('pōṭāl', 'belly-to', 570, 'belly', 'noun', None),
    ('bānēkil', 'father-to', 570, 'towards-father', 'noun', None),
    ('aun', 'which', 570, 'relative', 'pron relative uncertain', None),
    ('ēn', 'who', 570, 'who', 'pron interr', None),
    ('yēnī', 'anyone', 570, 'anyone', 'pron indef', None),
    ('tānī', 'anything', 570, 'anything', 'pron indef', None),
    ('add', 'they', 570, 'they', 'pron 3pl', None),
    ('anlēn', 'am', 570, 'am-art-is', 'verb', None),
    ('anlēn', 'art', 570, 'am-art-is', 'verb', None),
    ('anlēn', 'is', 570, 'am-art-is', 'verb', None),
    ('paṭṭul', 'fell', 570, 'fell', 'verb pret', None),
    ('paṭṭun', 'fell', 570, 'fell', 'verb pret', None),
    ('siylēn', 'gives', 571, 'gives', 'verb pres', None),
    ('seddēn', 'went', 571, 'went', 'verb pret', None),
    ('yeṇḍēn', 'said', 571, 'said', 'verb pret', None),
    ('antē', 'was', 571, 'was-ante', 'verb pret', None),
    ('sērēn', 'went-not', 571, 'went-not', 'verb pret neg', 'went'),
    ('siyētēn', 'gave-not', 571, 'gave-not', 'verb pret neg', 'gave'),
    ('mōḍāpētān', 'broke-not', 571, 'broke-not', 'verb pret neg', 'broke'),
    ('sī', 'give', 571, 'give', 'verb impv', None),
    ('īḍ', 'keep', 571, 'put', 'verb impv', None),
    ('jhagḍīlnēr', 'quarrel-not', 571, 'quarrel-not', 'verb impv neg', 'quarrel'),
    ('mēpēn', 'to-feed', 571, 'feed', 'verb inf', 'feed'),
    ('eṇḍēn-sāṭī', 'saying-for', 571, 'saying-for', 'verb inf', 'say'),
    ('kakēl', 'to-make', 571, 'do', 'verb inf', 'make'),
    ('kakkēn', 'to-make', 571, 'do', 'verb inf', 'make'),
    ('jamāktun', 'having-gathered', 571, 'collected', 'verb conjunctive-participle', 'gather'),
    ('seddun', 'having-gone', 571, 'gone', 'verb conjunctive-participle', 'go'),
    ('tinnun', 'eating', 571, 'eating-other', 'verb participle', None),
    ('tindunu', 'eating', 571, 'eating-other', 'verb participle', None),
    ('tōnān', 'taking', 571, 'taking', 'verb participle', None),
    ('kaktēn', 'doer', 571, 'agency-noun', 'noun uncertain', None),
    ('iūn', 'said', 571, 'said-questioned', 'verb pret uncertain', None),
    ('aṇḍur', 'was', 571, 'was-andur', 'verb pret uncertain', None),
    ('antēr', 'was', 571, 'was-anter', 'verb pret uncertain', None),
]


def read(name):
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line]


def lexical_text(text):
    # The printed sentence delimiters and speaker dash are not lexical material.
    # Internal hyphens, vowel quantities, nasal marks, and capitalization remain.
    return text.removesuffix('.—').rstrip('.,;?!')


def main():
    pronouns = {r['raw_form'].casefold(): r for r in read('pronoun-table-reviewed.jsonl')}
    grammar = {r['source_unit_key']: r for r in read('pronoun-grammatical-proposal.jsonl')}
    prose = {r['source_unit_key']: r for r in read('prose-reviewed.jsonl')}
    matches = {(form, gloss): (f'grierson1906naiki:p{page}:prose:{name}', tags.split(), lexical)
               for form, gloss, page, name, tags, lexical in PROSE_MATCHES}
    assert len(matches) == len(PROSE_MATCHES)
    assert all(key in prose for key, _, _ in matches.values())
    contextual = context_decisions()
    relations = relation_decisions()
    reviewed = reviewed_decisions()
    output = []
    for source in read('specimens-second-reading.jsonl'):
        form, gloss = lexical_text(source['form']), lexical_text(source['gloss'])
        lexical_gloss = gloss
        key = f'grierson1906naiki:p{source["page"]}:specimen:line{source["line"]}:word{source["word"]}'
        tags, support, note = [], [], ''
        uncertainty = None
        source_caveat = ''
        context = None
        # Exact form plus compatible translation is required. E.g. annē with
        # interlinear "me" is intentionally not assigned the table's genitive.
        pronoun = pronouns.get(form.casefold())
        if pronoun:
            table_gloss = pronoun['raw_gloss'].casefold()
            compatible = gloss.casefold() == table_gloss
            dative = gloss.casefold() == table_gloss + '-to'
            if compatible or dative:
                tags = list(grammar[pronoun['source_unit_key']]['tags'])
                if dative:
                    # p570's syncretism does not make both cases true in every
                    # token; this occurrence's explicit -to selects dative.
                    tags = [tag for tag in tags if tag != 'acc']
                    if 'dat' not in tags:
                        tags.append('dat')
                    lexical_gloss = gloss[:-3]
                elif table_gloss in {'me', 'thee', 'him'}:
                    # English object translation alone does not resolve which
                    # of the source's syncretic cases is used in this occurrence.
                    tags = [tag for tag in tags if tag not in {'acc', 'dat'}]
                support = [pronoun['source_unit_key']]
                note = 'Exact source pronoun-table form and compatible interlinear translation; no propagation to spelling variants.'
        match = matches.get((form.casefold(), gloss.casefold()))
        if match and not support:
            prose_key, tags, lexical = match
            tags = list(tags)
            support = [prose_key]
            source_caveat = prose[prose_key]['source_caveat']
            lexical_gloss = lexical if lexical is not None else gloss
            note = 'Source-explicit analysis on grammar page, applied only to the listed specimen form and translation; person features not propagated.'
            if 'uncertain' in tags:
                uncertainty = dict(type=('source-transcription' if prose_key.endswith(':said-questioned')
                                         else 'source-analysis'), reason=source_caveat)
        if not support:
            relation = relations.get((form.casefold(), gloss.casefold()))
            if relation:
                assert relation['support'] in prose
                tags = list(relation['tags'])
                support = [relation['support']]
                lexical_gloss = relation['lexical_gloss']
                note = relation['basis']
        if not support:
            context = contextual.get((form.casefold(), gloss.casefold()))
            if context:
                tags = list(context['tags'])
                note = context['basis']
                if context.get('uncertainty'):
                    uncertainty = dict(context['uncertainty'])
                    source_caveat = uncertainty['reason']
                    if 'uncertain' not in tags:
                        tags.append('uncertain')
        independent = reviewed.get(key)
        if independent:
            assert not support and not context, key
            assert (form, gloss) == (independent['form'], independent['gloss']), key
            tags = list(independent['proposed_tags'])
            lexical_gloss = independent['proposed_gloss']
            support = list(independent['support'])
            uncertainty = independent['uncertainty']
            source_caveat = uncertainty['reason'] if uncertainty else ''
            note = independent['reason'] + ' ' + independent['owner_decision']
        record = dict(source_unit_key=key, raw_form=source['form'], raw_gloss=source['gloss'],
                      proposed_form=form, proposed_gloss=lexical_gloss, tags=tags,
                      punctuation_cleaned_gloss=gloss,
                      punctuation_decision='Remove terminal sentence delimiters and speaker-label dash only; preserve internal punctuation and spelling.',
                      removed_form_suffix=source['form'][len(form):],
                      removed_gloss_suffix=source['gloss'][len(gloss):],
                      grammatical_support=support, grammatical_reason=note,
                      source_caveat=source_caveat, uncertainty=uncertainty,
                      contextual_decision=context,
                      independent_context_decision=independent,
                      metadata_review_pending=not bool(support or context or independent),
                      independent_review_pending=True)
        assert form and gloss
        assert form + record['removed_form_suffix'] == source['form']
        assert gloss + record['removed_gloss_suffix'] == source['gloss']
        output.append(record)
    assert len(output) == 537
    by_key = {r['source_unit_key']: r for r in output}
    dative = by_key['grierson1906naiki:p572:specimen:line2:word6']
    assert dative['proposed_gloss'] == 'me'
    assert 'dat' in dative['tags'] and 'acc' not in dative['tags']
    conflict = by_key['grierson1906naiki:p573:specimen:line19:word3']
    assert conflict['raw_form'] == 'annē' and conflict['raw_gloss'] == 'me'
    assert not conflict['metadata_review_pending']
    assert conflict['uncertainty']['type'] == 'source-analysis'
    assert not {'acc','dat','gen'} & set(conflict['tags'])
    nasal = by_key['grierson1906naiki:p572:specimen:line14:word8']
    assert nasal['proposed_form'] == 'nāhī̃' and nasal['removed_form_suffix'] == '.'
    compound = by_key['grierson1906naiki:p572:specimen:line1:word6']
    assert compound['proposed_form'] == 'Aunnē-lōpul'
    (HERE / 'specimen-grammatical-proposal.jsonl').write_text(''.join(
        json.dumps(r, ensure_ascii=False) + '\n' for r in output))
    status = dict(aligned_units=len(output),
                  punctuation_changed=sum(r['raw_form'] != r['proposed_form'] for r in output),
                  source_table_supported=sum(any(':pronoun-table:' in key for key in r['grammatical_support']) for r in output),
                  source_prose_supported=sum(any(':prose:' in key for key in r['grammatical_support']) for r in output),
                  contextual_pos_reviewed=sum(bool(r['contextual_decision']) for r in output),
                  independent_context_reviewed=sum(bool(r['independent_context_decision']) for r in output),
                  contextual_metadata_pending=sum(r['metadata_review_pending'] for r in output),
                  independent_review_pending=True, canonical_changed=False)
    (HERE / 'specimen-structure-status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()

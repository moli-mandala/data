"""Reconcile the bounded residual-label census from failed whole audit 205.

Only explicit reviewed interlinear/prose labels are parsed. Sentence-internal
person/register descriptions remain scoped Notes, never whole-sentence tags.
The two genuinely ambiguous interlinear groups retain their evidence untouched.
"""
import gzip
import json
import re
from pathlib import Path
from interlinear_grammar import parse
P = Path(__file__).resolve().parent
MARKERS = {'suffix', 'prefix', 'affix', 'infix', 'clitic'}
NORMALIZED = {
    'remain/COP.PST': 'remain-COP.PST',
    'COP/AUX.NPST': 'COP.AUX.NPST',
    'tap-ACC/DAT': 'tap-ACC.DAT',
    'river-bank-LOC (EMPH)': 'river-bank-LOC.EMPH',
    'you (HH)': 'you-HH',
    'ask ABS': 'ask-ABS',
}
SCOPED_SENTENCES = {
    'dhakal2011:p107:ch4:surface:l38': ('You (PL) cooked snacks.', 'You cooked snacks.', 'Source translation marks the addressee as plural.'),
    'dhakal2011:p281:ch10:surface:l45': ('The act of sitting was done/(I) sat (PASS).', 'The act of sitting was done/(I) sat.', 'Source labels the construction passive.'),
    'dhakal2011:p371:appendix3c1:whole-expression:l24': ('I gave you (SG.H).', 'I gave you.', 'Source translation marks the addressee as singular honorific.'),
    'dhakal2011:p373:appendix3c1:whole-expression:l35': ('He gave you (SG.H).', 'He gave you.', 'Source translation marks the addressee as singular honorific.'),
    'dhakal2011:p378:appendix3e:l18:whole-expression:0': ('You are a (Darai) priest (H.SG).', 'You are a (Darai) priest.', 'Source translation marks the subject as singular honorific.'),
    'dhakal2011:p378:appendix3e:l38:whole-expression:0': ('He is a (Darai) priest (H.SG).', 'He is a (Darai) priest.', 'Source translation marks the subject as singular honorific.'),
    'dhakal2011:p379:appendix3e:l46:whole-expression:0': ('You are at home (SG.H).', 'You are at home.', 'Source translation marks the addressee as singular honorific.'),
    'dhakal2011:p380:appendix3e:l25:whole-expression:0': ('You are hungry(H.SG).', 'You are hungry.', 'Source translation marks the addressee as singular honorific.'),
}


def build():
    # Read the immutable failed proposal, not the repaired derivative, so this
    # census remains reproducible after grammatical labels leave lexical Gloss.
    with gzip.open(P/'whole-freeze-before-interlinear-repair/whole-proposed-audit.jsonl.gz', 'rt') as f:
        rows = [json.loads(line) for line in f]
    decisions = []
    for row in rows:
        if row['disposition'].startswith('audit-only'):
            continue
        gloss = row['proposed_grammar']['gloss']
        if not re.search(r'\b(?:[A-Z]{2,}|[123](?:SG|PL))\b', gloss):
            continue
        key = row['source_unit_key']
        d = dict(source_unit_key=key, form=row['form'], source_gloss=row['source_gloss'],
                 previous_gloss=gloss, action='retain', reason='', note='', add_tags=[])
        if key in SCOPED_SENTENCES:
            before, after, note = SCOPED_SENTENCES[key]
            assert gloss == before
            d.update(action='scoped-translation-label', gloss=after, note=note,
                     reason='Explicit sentence-internal grammatical label; no role propagation to whole construction.')
            if key == 'dhakal2011:p281:ch10:surface:l45': d['add_tags']=['pass','verb']
        elif row['kind']=='whole-expression':
            if re.search(r'\([A-Za-z]+\.[A-Z]+\.\d+\)$', gloss):
                d.update(action='text-provenance', gloss=re.sub(r'\s*\([A-Za-z]+\.[A-Z]+\.\d+\)$', '', gloss),
                    note='Source text identifier: '+re.search(r'\(([^()]*)\)$',gloss)[1]+'.',
                    reason='Corpus text ID is provenance rather than a lexical definition.')
            else:
                assert ' VS.' in gloss, (key,gloss)
                d['reason']='VS is the printed calendar-era designation in a date, not a grammatical label.'
        elif key in {'dhakal2011:p78:ch4:surface:l93:word:l100:w1', 'dhakal2011:p91:ch4:surface:l62:word:l67:w1'}:
            d['reason']='Genuinely ambiguous multiword source alignment remains typed; no individual-word grammar is inferred.'
        elif key == 'dhakal2011:p305:ch11:prose:l9:a1':
            assert gloss == 'crush-PFV fermented rice-COM'
            d.update(action='scoped-phrase-label',gloss='crush fermented rice',
                     note='Source analyses the first component as PFV and fermented rice as COM.',
                     reason='Component grammatical labels remain scoped to their printed components, not to the entire phrase.')
        elif gloss == 'there-ABL onwards':
            d.update(action='explicit-interlinear',grammar=parse('there-ABL'),
                     reason='ABL follows there; onwards remains lexical English.')
            d['grammar']['gloss']='there onwards'
        elif gloss == 'that way (CORR)':
            d.update(action='explicit-interlinear',grammar={'gloss':'that way','tags':['correlative'], 'labels':['CORR']},
                     reason='Explicit parenthetical CORR label scopes to this attested form.')
        elif key == 'dhakal2011:p190:ch6:surface:l28:word:l31:w1':
            assert gloss == '1SG plough'
            d.update(action='scoped-phrase-label',gloss='plough',add_tags=['uncertain'],
                     note='Source-layout uncertainty: the printed label 1SG precedes plough under hor, conflicting with the lexical translation. No first-person analysis is assigned.',
                     reason='Retain the literal anomalous group in audit and its scoped uncertainty in Notes.')
        elif row['proposed_grammar'].get('held_analysis'):
            # Exact original-page review decisions resolve these old geometry
            # warnings in prepare_whole_metadata; do not override them here.
            d.update(action='resolved-alignment',reason='Use exact retained original-page alignment adjudication, not the stale geometry flag.')
        else:
            parsed=parse(NORMALIZED.get(gloss,gloss))
            assert parsed['labels'] and not re.search(r'\b[A-Z]{2,}\b',parsed['gloss']), (key,gloss,parsed)
            if set(row['proposed_grammar']['tags']) & MARKERS:
                parsed['tags']=[t for t in parsed['tags'] if t!='verb']
            d.update(action='explicit-interlinear',grammar=parsed,
                     reason='Explicit source form/gloss pair or quoted linguistic analysis; literal source analysis retained in audit and Notes.')
        decisions.append(d)
    assert len(decisions)==100
    return dict(scope='All 100 residual uppercase/person-number Gloss cases in frozen seed205, individually classified; no arbitrary prose keyword extraction.', decisions=decisions)


if __name__=='__main__':
    report=build()
    (P/'whole-interlinear-scope-decisions-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('Residual cases:', len(report['decisions']))

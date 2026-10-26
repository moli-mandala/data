"""Source-position grammatical decisions for Naiki's two grammar pages.

No inference from spelling or cross-language resemblance; decisions are keyed
to the reviewed source groups. This is a proposal sidecar, not installation.
"""
import ast
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
# Stable source-group suffix -> exact source-scoped tags. Context caveats below
# remain attached even when a broad search tag is useful.
TAGS={
 'noun-plural-rational':'suffix pl','noun-plural-other':'suffix pl',
 'children':'noun pl','swine':'noun pl','buffaloes':'noun pl',
 'me-to-me':'pron 1sg acc dat','case-suffixes':'suffix acc dat',
 'daughter':'noun dat','father':'noun dat','field':'noun dat',
 'servants':'noun pl dat','belly':'noun','towards-father':'noun',
 'genitive':'suffix gen','locative':'suffix loc','heaven':'noun gen',
 'country':'noun loc','one':'num','two':'num','two-neuter':'num n',
 'four':'num','one-proposed':'num uncertain','two-proposed':'num uncertain',
 'she-it':'pron 3sg','she-it-genitive':'pron 3sg gen',
 'they':'pron 3pl','this':'pron demonstrative','relative':'pron relative uncertain',
 'relative-clause':'multiword-expression','who':'pron interr','what':'pron interr',
 'indefinite-suffix':'suffix indef','anyone':'pron indef','anything':'pron indef',
 'am-art-is':'verb','shall-do':'verb fut 1pl','were':'verb pret 3pl',
 'fell':'verb pret 3sg','present-marker':'affix pres uncertain','past-marker':'affix',
 'gives':'verb pres 3sg','gave':'verb pret 3sg','went':'verb pret 3sg',
 'said':'verb pret 3sg','doing':'verb pres 1sg','will-go':'verb fut 1sg',
 'personal-ending':'suffix','plural-personal-ending':'suffix pl','was-ante':'verb pret 3sg',
 'was-andur':'verb pret uncertain','was-anter':'verb pret uncertain',
 'said-questioned':'verb pret 3sg uncertain','negative-marker':'affix neg',
 'went-not':'verb pret 3sg neg','gave-not':'verb pret 3sg neg',
 'broke-not':'verb pret 1sg neg','gavest-not':'verb pret 2sg neg',
 'give':'verb impv','put':'verb impv','let-see':'verb 1pl',
 'put-not':'verb impv neg','quarrel-not':'verb impv neg',
 'infinitive-suffixes':'suffix inf','feed':'verb inf','saying-for':'verb inf',
 'do':'verb inf','conjunctive-suffixes':'suffix conjunctive-participle',
 'collected':'verb conjunctive-participle','gone':'verb conjunctive-participle',
 'eating-conjunctive':'verb conjunctive-participle','called':'verb participle',
 'eating-other':'verb participle','eating-tinchin':'verb participle',
 'playing':'verb participle','taking':'verb participle',
 'agency-noun':'noun uncertain','done-phrase':'multiword-expression',
 'sin-phrase':'multiword-expression',
}
GRAMMAR_ONLY={'noun-plural-rational','noun-plural-other','case-suffixes',
 'genitive','locative','relative','indefinite-suffix','present-marker',
 'past-marker','personal-ending','plural-personal-ending','negative-marker','infinitive-suffixes',
 'conjunctive-suffixes','agency-noun'}
CAUTIONS={
 'noun-plural-rational':('source-analysis','Rational restriction is qualified as seems by source.'),
 'past-marker':('source-analysis','Source also says t occurs in present/future; do not label every t-bearing form past.'),
 'was-andur':('source-analysis','Translation is singular he/she was; source calls r-ending forms properly plural. Do not collapse that conflict.'),
 'was-anter':('source-analysis','Translation is singular it was; source calls r-ending forms properly plural.'),
 'one-proposed':('source-transcription','Author conjectures vowel quantity; not a second independently observed token.'),
 'two-proposed':('source-transcription','Author conjectures vowel quantity; not a second independently observed token.'),
 'said-questioned':('source-transcription','Author explicitly calls the printed reading probably wrong.'),
 'agency-noun':('source-analysis','Apparently a noun of agency; do not gloss the isolated form as the entire perfect construction.'),
 'relative':('source-analysis','Author qualifies relative-pronoun function as seems.'),
 'present-marker':('source-analysis','Author qualifies present marker as seems.'),
}

def main():
    registry=ast.parse((HERE.parents[4]/'tags.py').read_text())
    allowed=set()
    for statement in registry.body:
        if isinstance(statement,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id in {'GRAMMATICAL_TAGS','GENDER_TAGS'}
            for t in statement.targets):
            allowed.update(ast.literal_eval(statement.value))
    assert allowed
    output=[]
    for line in (HERE/'prose-reviewed.jsonl').read_text().splitlines():
        row=json.loads(line)
        name=row['source_unit_key'].rsplit(':',1)[-1]
        control=row['disposition'].startswith('control-')
        if control:
            tags=[]
            action='exclude-control'
        else:
            assert name in TAGS, name
            tags=TAGS[name].split()
            action='propose-with-source-caveats'
        caution=CAUTIONS.get(name)
        if caution and 'uncertain' not in tags:
            tags.append('uncertain')
        assert set(tags)<=allowed, (name,set(tags)-allowed)
        output.append(dict(source_unit_key=row['source_unit_key'],
            tags=tags,disposition=action,
            proposed_gloss='' if name in GRAMMAR_ONLY else row['raw_gloss_or_analysis'],
            source_analysis=row['raw_gloss_or_analysis'] if name in GRAMMAR_ONLY else '',
            source_caveat=row['source_caveat'],
            uncertainty=dict(type=caution[0],reason=caution[1]) if caution else None,
            independent_review_pending=False,final_output_review_pending=True))
    assert len(output)==84
    assert sum(r['disposition']=='exclude-control' for r in output)==5
    (HERE/'prose-grammatical-proposal.jsonl').write_text(''.join(
        json.dumps(r,ensure_ascii=False)+'\n' for r in output))
    table=[]
    for line in (HERE/'pronoun-table-reviewed.jsonl').read_text().splitlines():
        row=json.loads(line)
        col,position=row['column'],row['row']
        tags=['pron','personal',f'{col}'+('pl' if position==4 else 'sg')]
        if col==3: tags.append('m')
        if position==3: tags.extend(['gen','poss'])
        elif position==2: tags.extend(['acc','dat'])
        assert set(tags)<=allowed
        table.append(dict(source_unit_key=row['source_unit_key'],tags=tags,
            proposed_gloss=row['raw_gloss'],source_analysis=row['analysis'],
            reasoning='Source personal-pronoun table columns and translations; source tentatively says accusative/dative do not appear to be distinguished.',
            independent_review_pending=False,final_output_review_pending=True))
    assert len(table)==11
    (HERE/'pronoun-grammatical-proposal.jsonl').write_text(''.join(
        json.dumps(r,ensure_ascii=False)+'\n' for r in table))
    print('84 prose groups and 11 pronoun cells structured; 5 controls retained; all tags registered; no canonical installation')

if __name__=='__main__': main()

from pathlib import Path
import csv,json,hashlib
P=Path(__file__).resolve().parent;F=P/'whole-proposal-r1';OUT=P/'revision2-preparation';OUT.mkdir(exist_ok=True)
rows={r[10]:r for r in csv.reader((F/'proposal.csv').open())};audit=[json.loads(s) for s in(F/'proposal-audit.jsonl').read_text().splitlines()]
checks=[];changes=[]
pron={
 'i':['pron','personal','first-person','sg'],'me':['pron','personal','first-person','sg'],
 'my':['pron','personal','first-person','sg','gen'],'mine-is':['pron','personal','first-person','sg','gen'],
 'thou':['pron','personal','second-person','sg'],'thy':['pron','personal','second-person','sg','gen'],
 'thine-is':['pron','personal','second-person','sg','gen'],
 'he':['pron','personal','third-person','sg'],'him':['pron','personal','third-person','sg'],
 'it':['pron','third-person','sg'],'them':['pron','personal','third-person','pl'],
 'this':['pron','demonstrative','prox','sg'],'that':['pron','demonstrative','dist','sg'],
}
for unit in audit:
    inp=unit.get('input',{});section=inp.get('section','')
    if section not in {'parable-aligned','parable-parenthetical','story-aligned'}:continue
    raw=inp.get('raw_alignment',inp.get('gloss',''));low=raw.lower()
    for key in unit.get('entry_keys',[]):
        row=rows[key];old=row[14].split();new=list(old);gloss=row[3];basis=[]
        if 'ing' in low:
            before=set(new)
            if 'participle' in new:
                if row[2]=='tŭrdā' and section=='story-aligned':
                    basis.append('Source105 walking modifies traveller; literal -dā matches explicit90 present-participle ending after a sonant, here r. Retain this source-supported participle analysis.')
                else:
                    new=[t for t in new if t not in {'participle','pres'}]
                    basis.append('English -ing alone does not identify a Kagani present participle. Remove ungrounded subtype; retain literal interlinear analysis.')
                    if row[2] in {'gēṇā','băjēṇā'}:
                        new+=['inf'];basis.append('Original103 nominal singing/playing uses -ṇā;90 explicitly identifies -ṇā as infinitive. Whole printed form retained.')
                    if row[2]=='kărā':
                        if 'caus' not in new:new+=['caus']
                        basis.append('Original103 getting-made uses kărā;90 explicitly supplies kărāṇā causative do. Do not infer passive from English gloss.')
            if row[2]=='k̲h̲ărc' or (row[2]=='k̲h̲ărc'.replace('\u0332','')):
                new=[t for t in new if t!='verb'];gloss=raw
                basis.append('Original101 kharc followed by kar forms compound spending/making; the first member is not independently established as a verb. Preserve raw gloss rather than infer a verbal lemma.')
            checks.append(dict(entry_key=key,form=row[2],raw_analysis=raw,classification='reviewed English ing-containing analysis',decision=basis or ['No generic present-participle tag assigned; finite/conjunctive analysis remains separately supported.']))
        if low in pron:
            new=list(dict.fromkeys(new+pron[low]));basis.append('Explicit interlinear pronoun meaning supplies person/number; source93–94 pronoun paradigms corroborate classes. No case inferred for unmarked personal pronouns; source spelling retained.')
        if low in {'you','you-of'}:
            if row[2] in {'tŭsī̃','tŭsdā','tŭs dā'}:
                new=list(dict.fromkeys(new+['pron','personal','second-person','pl']+(['gen'] if low=='you-of' else [])))
                basis.append('Exact tŭsī̃ and tŭs dā under source93 second-person plural headings; fused tŭsdā keeps printed boundary. English you alone is not evidence.')
            else:basis.append('English you remains number-ambiguous without exact source-paradigm support.')
        if low in pron or low in {'you','you-of'}:
            checks.append(dict(entry_key=key,form=row[2],raw_analysis=raw,classification='pronoun metadata',decision=basis))
        if new!=old or gloss!=row[3]:changes.append(dict(entry_key=key,old_form=row[2],old_gloss=row[3],new_gloss=gloss,old_tags=row[14],new_tags=' '.join(dict.fromkeys(new)),basis=basis))
report={'status':'r2 review proposal only; r1 unchanged','input_sha256':{n:hashlib.sha256((F/n).read_bytes()).hexdigest() for n in ['review-manifest.json','proposal.csv','proposal-audit.jsonl']},'method':'Review all narrative/parenthetical ing-containing English analyses and all simple personal/demonstrative pronoun glosses in frozen actual output; source103 original3x rechecked and source90/93–94 prior full original readings used. No inferred token segmentation.','checks':checks,'changes':changes,'pending':'Root r1 failed audit completion, then apply outside immutable freeze and validate/fresh audit.'}
(OUT/'interlinear-metadata-class-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('checks',len(checks),'changes',len(changes))

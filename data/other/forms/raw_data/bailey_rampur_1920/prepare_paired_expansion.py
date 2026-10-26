"""Explicit source-supported paired paradigm expansion; no canonical writes.

Raw abbreviated cells remain in every audit unit. Rules below are restricted to
the printed stem/ending and ditto paradigms, never general morphological inference.
"""
import hashlib
import json
import re
from pathlib import Path

P = Path(__file__).resolve().parent
FIELDS = ('rampur_sg','rampur_pl','baghi_sg','baghi_pl')
NOUNS = {'horse':'m','house':'m','house-heading':'m','father':'m','elephant':'m','girl':'f','girl-heading':'f','sister':'f'}
PRONOUNS = {'first-person','second-person','third-person-masculine','third-person-feminine','this-masculine','this-feminine','who-interrogative','who-relative','other-pronouns'}
CONTEXT = {'adjective-class-context','fall-future-reuse','fall-present-conditional-reuse','tense-function-commentary'}
PERSON = {1:'first-person',2:'second-person',3:'third-person'}


def load(name):
    return [json.loads(x) for x in (P/name).read_text().splitlines()]


def clean_note(text):
    """Remove procedural draft instructions, keeping source facts in the audit.

    Only complete review-process sentences are omitted; unresolved glyphs are
    represented separately, not hidden by this presentation cleanup.
    """
    kept=[]
    for sentence in re.split(r'(?<=[.!?])\s+',text):
        if re.search(r'pending|requires? (?:independent|second)|provisional|verify independently|do not invent|no unseen|never emit|no analogical|preserv(?:e|ed|ing).*?(?:literal|printed|source|occurrence)|raw .*?retained|generate no|expanded only|not harmonized|not silently|no additional unseen|account by reuse|no missing cells',sentence,re.I):
            continue
        if sentence.startswith(('Paired numeral columns:','Fine breve','Printed source quantities','Whole translated response;','Separate headings;','Two explicit forms then etc.;')):
            continue
        kept.append(sentence)
    return ' '.join(kept).strip()


def base_metadata(row,field,form):
    s=row['section']; label=row['source_label']; item=int(row['item']); lect='Rampuri' if field.startswith('rampur') else 'Baghi'
    tags=[];notes=[];gloss=row['gloss'];unc=[]
    if s in NOUNS:
        tags += ['noun',NOUNS[s]]
    elif s in PRONOUNS:
        tags+=['pron']
        if s.startswith(('first-person','second-person')):tags += [s,'personal']
        if s.startswith('third-person'):tags+=['third-person','personal']
        if s.startswith('this-'):tags+=['demonstrative','prox']
        if s.endswith('masculine'):tags+=['m']
        if s.endswith('feminine'):tags+=['f']
    elif s in {'genitive-morphology','adjective-morphology','ordinal-suffix','vowel-root-ending','other-root-ending','fall-future-feminine-ending'}:
        tags+=['suffix' if s!='genitive-morphology' or lect=='Rampuri' else 'postp']
        if s=='genitive-morphology':
            tags+=['gen','adj', 'f' if item==2 else 'm','pl' if field.endswith('_pl') else 'sg']
            if item==3:tags+=['obl']
            notes.append('The genitive agrees with its noun in gender and number; masculine singular changes in oblique agreement, as described by Bailey.')
        if s=='adjective-morphology':
            tags+=['adj','f' if item==2 else 'm']
            tags+=['sg','obl'] if item==3 else ['pl'] if item==1 else ['sg','pl']
            notes.append('Source explicitly identifies adjective agreement endings; other adjectives do not inflect unless used substantivally.')
        if s=='ordinal-suffix':tags+=['num','ord']
        if s in {'vowel-root-ending','other-root-ending'}:
            tags+=['verb','pres','ind']
            notes.append('Bailey contrasts vowel-root -ũ with other-root -ū; the printed marks are retained literally.')
        if s=='fall-future-feminine-ending':
            tags+=['verb','f','fut']
            notes.append('Bailey says the feminine future ending e is almost i; the printed e is not normalized.')
    elif s in {'cardinal','fractional','fractional-morphology','ordinal'}:
        tags+=['num']
        if s=='ordinal':tags+=['ord']
    elif s=='pronominal-adjectives':tags+=['pron','adj']
    elif s.endswith('adverbs'):
        tags+=['adv']
        if s=='time-adverbs':tags+=['temporal']
        if s=='place-adverbs':tags+=['spatial']
    elif s=='prepositions':
        tags+=['postp']
        if item==10:tags+=['instr']
        if item==1 and lect=='Rampuri':tags+=['suffix','gen']
    elif s in {'house-in','comparison'}:
        tags+=['adj'] if s=='comparison' else ['multiword-expression']
        if s=='comparison':notes.append('Bailey reports comparison using an ablative expression with the positive adjective, without a distinct comparative or superlative inflection.')
    elif s=='become-aspiration-comment':
        tags+=['prefix']
        notes.append('Bailey says a slight h is frequently prefixed throughout the Baghi verb; no unattested prefixed paradigm is generated.')
    else:
        tags+=['noun'] if s.endswith('-agent') else ['verb']
        if 'heading' in s:tags+=['inf']
        if 'substantive' in s:tags+=['copula']
        if 'present' in s or s=='give-vowel-root-example':tags+=['pres','ind']
        if 'future' in s:tags+=['fut']
        if 'imperfect' in s:tags+=['ipfv']
        elif 'past' in s:tags+=['pret']
        if 'conditional' in s:tags+=['conditional']
        if 'imperative' in s:tags+=['impv','second-person']
        if 'negative' in s:tags+=['neg']
        if 'stative' in s:
            tags+=['participle'];notes.append('Source labels this a stative participle.')
        if 'past-participle' in s:tags+=['participle']
        if 'conjunctive' in s:tags+=['conjunctive-participle']
        if s=='fall-conjunctive' and lect=='Baghi':tags+=['indecl']
        if s=='fall-future' and lect=='Baghi':
            tags+=['m']
            notes.append('These are the printed masculine future cells; Bailey specifies feminine e throughout separately.')
        if lect=='Baghi' and s in {'eat-stative','drink-stative','take-stative','bring-stative','take-away-stative'}:
            tags+=['pass','tr']
            notes.append('Bailey’s p143 commentary identifies the stative participle of transitive verbs as passive, contrasting it with the intransitive ending.')
        if 'source phonetic commentary' in label:
            notes.append('Bailey explicitly describes the initial ā as very long; this note does not add an unprinted length mark or assign a new tense.')
        if 'habitual' in s:notes.append('Source labels the construction habitual.')
        if 'masculine' in label:tags+=['m']
        if 'feminine' in label:tags+=['f']
        if re.search(r'person row [123]|first-person|second-person|third-person',label):
            n=int(re.search(r'person row ([123])',label)[1]) if 'person row' in label else (1 if 'first-person' in label else 2 if 'second-person' in label else 3)
            tags += [PERSON[n], 'pl' if field.endswith('_pl') else 'sg']
            if 'person row' in label:notes.append('Person follows the printed three-row paradigm order; the source does not print a separate person number on each row.')
        if any(x in s for x in ['imperative','-masculine','-feminine']):
            tags+=['pl' if field.endswith('_pl') else 'sg']
        if s in {'come-habitual','give-vowel-root-example'}:tags+=['sentential','first-person','sg']
        if s in {'eat-present','drink-present','give-present','give-future','bring-future','take-away-future'}:
            notes.append('The source gives this example followed by an abbreviated or regular paradigm statement; no additional unprinted person forms are supplied.')
        if 'present' in s and 'substantive' not in s:
            notes.append('Bailey states that present indicative and present conditional coincide in both lects; Rampuri also uses this form for the future. These are alternative source functions, not a claim of a composite tense.')
        if s=='beat-heading':notes.append('Bailey says this verb is regular in both lects and agrees with its object in past tenses.')
    if s in set(NOUNS)|PRONOUNS:
        for marker,tag in [('Nom.','nom'),('Gen.','gen'),('Dat.','dat'),('Acc.','acc'),('Abl.','abl'),('Agent','erg'),('Voc.','voc')]:
            if marker in label:tags.append(tag)
        if s in {'horse','girl','sister'} or s in PRONOUNS-{'other-pronouns','who-interrogative','who-relative'}:
            tags+=['pl' if field.endswith('_pl') else 'sg']
        elif s in {'house','father','elephant'} and ('Nom.' in label or s!='house-heading'):
            tags+=['sg']
        if s=='girl' and item==3:notes.append('The source appears to print Act. in this paradigm, beside Dat.; Acc. is interpreted from the parallel preceding horse paradigm.')
    for marker,tag in [('interrogative','interr'),('relative','relative'),('correlative','correlative'),('indefinite','indef'),('indeclinable','indecl')]:
        if re.search(r'(?<![A-Za-z])'+marker+r'(?![A-Za-z])',label):tags.append(tag)
    gloss=gloss.rstrip('?')
    if ' ' in form:tags+=['multiword-expression']
    if 'ī̃' in form or 'ē̃' in form or 'tāī̃' in form:
        unc.append('Source vowel has layered upper ink marks; the literal length-plus-nasal encoding is provisional, not a phonological assertion.')
    if s=='cardinal' and item==11 and lect=='Baghi':
        unc.append('Source gaira has a normal i dot plus a faint isolated upper-left rectangular mark; retain gaira, without asserting whether the extra ink is a diacritic or a scan blemish.')
    if s=='substantive-negative' and lect=='Baghi':
        unc.append('The final vowel in nĕhĩ is clearly i; its upper mark is read provisionally as a tilde rather than breve. Root independently prefers the tilde, but exact mark shape remains uncertain.')
    if s=='cardinal' and item==2 and lect=='Rampuri':notes.append('Source footnote says ū is very long; no extra unprinted length mark is added.')
    if s=='cardinal' and item==16 and lect=='Baghi':notes.append('Bailey explicitly specifies plain l, not ḷ, for this occurrence.')
    if s=='other-adverbs' and item in {2,3} and lect=='Baghi':notes.append('Answer to a question.' if item==2 else 'Answer to a call.')
    note=clean_note(row['note'])
    if note and note not in notes:notes.append(note)
    return dict(gloss=gloss,tags=list(dict.fromkeys(tags)),notes=notes,typed_uncertainty=unc)


def expand_grammar(rows):
    units=[]
    # Explicit stems are only used in the printed abbreviation classes below.
    stems={'horse':'gōh','house':'gauh','girl':'tshōṭ','sister':'baih'}
    previous={}
    for row in rows:
        s=row['section'];key=row['source_unit_key'];item=int(row['item'])
        unit=dict(source_unit_key=key,printed_page=int(row['page']),source_kind='grammar',source_review=row,candidates=[],disposition='emitted')
        if s in CONTEXT:
            unit['disposition']='source grammar/context preserved, no distinct literal response'
            unit['context_only_reason']='Class-defining vowels or explicit reuse/function statement; supported function is recorded on applicable emitted paradigms.'
            units.append(unit);continue
        cells=[(f,row[f]) for f in FIELDS if row[f]]
        if row['shared_form']:
            # The Baghi go heading is explicitly replaced by ḍĕūṇo on p139.
            lects=['rampur'] if s=='go-heading' else ['rampur','baghi']
            cells += [(l+'_shared',row['shared_form']) for l in lects]
        for field,raw in cells:
            parts=raw.split(';')
            ambiguous = s=='become-future' and item==3 and field=='rampur_pl'
            if ambiguous:parts=['ōā','auī']
            for index,part in enumerate(parts,1):
                form=part;expansion=[]
                if s in stems:
                    stem=stems[s]
                    if form.startswith('-'):form=stem+form[1:];expansion.append('Source supplies '+stem+'- as the shared printed stem.')
                    elif form.startswith(stem+'-'):form=stem+form[len(stem)+1:];expansion.append('Join the printed stem and ending across the source morphological hyphen.')
                elif s.startswith('fall-imperfect-') and s!='fall-imperfect-component':
                    if form=='〃':
                        form=previous[(s,field)]
                        expansion.append('Explicit ditto repeats the earlier cell in this source column and gender row.')
                    if s=='fall-imperfect-feminine':
                        masculine=previous[('fall-imperfect-masculine',field)]
                        component=masculine.rsplit(' ',1)[0]
                        if ' ' not in form:
                            form=component+' '+form
                            expansion.append('Source prints the feminine auxiliary in place of the immediately preceding masculine auxiliary; retain its full printed lexical component.')
                        elif item>1:
                            form=component+' '+form.rsplit(' ',1)[1]
                            expansion.append('Explicit feminine ditto repeats the auxiliary; lexical component follows this person row, as printed.')
                    previous[(s,field)]=form
                elif s.startswith('fall-') and any(x in s for x in ['conditional-masculine','conditional-feminine','stative-','past-masculine','past-feminine','past-participle-']):
                    stem='lōṭṭ' if 'past-participle' in s else 'lōṭ'
                    if form.startswith(stem+'-'):form=stem+form[len(stem)+1:]
                    else:form=stem+form.lstrip('-')
                    expansion.append('Expand the explicitly printed '+stem+'- stem and gender/number ending, retaining the raw abbreviated cell.')
                elif s=='go-present' and field.startswith('rampur') and form.startswith('-'):
                    form='năs̲h̲'+form[1:]
                    expansion.append('Source gives năs̲h̲ū and a six-ending sequence; substitute only the printed ending.')
                assert form and '〃' not in form,(key,field,form)
                meta=base_metadata(row,field,form)
                if ambiguous:
                    meta['typed_uncertainty'].append('Printed future third-plural cell is unpunctuated ōā auī. Two alternatives are an editorial segmentation supported by the adjacent comma-separated parallel cells; no confident variant edge is asserted.')
                candidate=dict(candidate_key=key+':'+field+':'+str(index),form=form,lang='ramp' if field.startswith('rampur') else 'ba',source_field=field,raw_cell=raw,printed_alternative=index,expansion=expansion,**meta)
                if len(parts)>1 and not ambiguous:
                    candidate['variant_of_key']=key+':'+field+':1' if index>1 else ''
                unit['candidates'].append(candidate)
        assert unit['candidates'],key
        units.append(unit)
    keyed={u['source_unit_key']:u for u in units}
    for unit in units:
        row=unit['source_review'];s=row['section']
        if s in {'fall-future-reuse','fall-present-conditional-reuse'}:
            lects=['ramp'] if s=='fall-future-reuse' else ['ramp','ba']
            unit['stated_reuse']=[dict(candidate_key=c['candidate_key'],source_function='future' if s=='fall-future-reuse' else 'present conditional') for u in units if u['source_review']['section']=='fall-present' for c in u['candidates'] if c['lang'] in lects]
            assert len(unit['stated_reuse'])==(6 if s=='fall-future-reuse' else 12)
        if s in {'third-person-feminine','this-feminine'}:
            counterpart=s.replace('feminine','masculine')
            target=[u for u in units if u['source_review']['section']==counterpart and u['source_review']['item']==row['item']]
            assert len(target)==1
            unit['stated_reuse']=[dict(candidate_key=c['candidate_key'],source_function='feminine plural explicitly same as masculine plural') for c in target[0]['candidates'] if c['source_field'].endswith('_pl')]
        if s=='sister' and int(row['item'])>1:
            unit['stated_reuse']=[dict(candidate_key=c['candidate_key'],source_function='plural etc., as singular, per printed paradigm statement; no separate physical plural cell') for c in unit['candidates']]
        if s=='go-heading':unit['scope_note']='The following explicit Baghi heading is ḍĕūṇo on p139; năs̲h̲ṇo is kept in the preceding Rampuri paradigm, not duplicated as an unattested Baghi head.'
    return units


def expand_sentences(rows):
    units=[]
    for row in rows:
        key=row['source_unit_key'];item=int(row['item']);lect=row['lect'];form=row['form'];gloss=row['literal_translation'];notes=[];unc=[]
        if item==12:gloss='His brother (own) sister from (than) big, i.e. is bigger.'
        if item==8 and lect=='Rampuri':gloss='His back upon saddle tighten.'
        if item==10:gloss='Hill upon cows goats is grazing.' if lect=='Rampuri' else '(He) jungle in cows goats is grazing.'
        if item==11:gloss='That tree under horse upon seated.' if lect=='Rampuri' else 'That tree under horse upon sat.'
        if item==16 and lect=='Rampuri':gloss='Him from rupees ask.'
        if item in {3,6,8,9,10,11,12,16,17,20,21}:notes.append('Bailey’s complete translated response is retained; inline English qualifications are interpreted for the explicitly marked lect, without inferring unprinted words.')
        if item==19 and lect=='Baghi':unc.append('Initial Mū̃ has a compact upper stack. A separate macron below the tilde is plausible but not completely secure; retain this provisional literal encoding, distinct from an assertion about pronunciation.')
        forms=[form]
        if item==18 and lect=='Baghi':
            assert '(or pāṇi)' in form
            forms=[form.replace('cīs̲h̲ (or pāṇi)','cīs̲h̲'),form.replace('cīs̲h̲ (or pāṇi)','pāṇi')]
            notes.append('Source explicitly permits pāṇi in place of cīs̲h̲ in this response; the unchanged surrounding words are retained in both full variants.')
        tags=['sentential','multiword-expression']+(['interr'] if gloss.endswith('?') else [])
        unit=dict(source_unit_key=key,printed_page=int(row['page']),source_kind='sentence',source_review=row,candidates=[],disposition='emitted')
        for i,value in enumerate(forms,1):
            unit['candidates'].append(dict(candidate_key=key+':'+str(i),lang='ramp' if lect=='Rampuri' else 'ba',form=value,gloss=gloss,tags=tags,notes=notes,typed_uncertainty=unc,raw_cell=form,expansion=['Explicit alternate noun in complete response.'] if len(forms)>1 else [],variant_of_key=key+':1' if i>1 else ''))
        units.append(unit)
    return units


def expand_prose(rows,kind):
    units=[]
    for row in rows:
        key=row['source_unit_key'];scope=row['scope'];lect=row['source_lect']
        unit=dict(source_unit_key=key,printed_page=row['printed_page'],source_kind=kind,source_review=row,candidates=[],disposition='source context/control preserved')
        target=scope.startswith('target') or scope=='shared lexical witness'
        if not target or not row['forms']:
            unit['context_only_reason']='Explicitly other-lect comparison, general source commentary, or class-defining orthographic context; no unprinted target lexical form inferred.'
            units.append(unit);continue
        tagged=[];notes=[row['source_claim']];gloss=row['gloss']
        if 'morphology' in scope:
            tagged+=['suffix'] if row['forms'][0].startswith('-') else ['postp']
        if 'stative' in key or key.endswith(('intransitive-ending','transitive-ending','ending-comparison-head')):
            tagged+=['participle']
            if 'transitive' in key and 'intransitive' not in key:tagged+=['tr','pass']
            if 'intransitive' in key:tagged+=['intr']
        if 'future' in key:tagged+=['verb','fut']
        if 'genitive' in key:
            tagged+=['gen']
            if lect=='Rampuri':tagged+=['adj']
        if 'dative' in key:tagged+=['dat']
        if 'ablative' in key:tagged+=['abl']
        if 'inflection' in key:tagged+=['f' if '-fem-' in key else 'm']
        if key.endswith('8-gahri'):tagged+=['postp','adj']
        if key.endswith('9-distinct'):tagged+=['adv','degree']
        if kind=='commentary' and key.endswith(('11-seated','passive-example','fallen','eaten','drunk')):
            tagged+=['verb','participle']
            tagged+=['pass','tr'] if key.endswith(('passive-example','eaten','drunk')) else ['intr']
        if kind=='introduction' and scope=='target':tagged+=['postp']
        for i,form in enumerate(row['forms'],1):
            lang='kc' if scope=='shared lexical witness' else 'ramp' if lect=='Rampuri' else 'ba'
            if key.endswith('9-distinct'):lang='ramp' if i==1 else 'ba'
            tags=tagged+(['multiword-expression'] if ' ' in form else [])
            uncertainty=['Source gives this as generic Koci without specifying which constituent lect; registered Kochi kc is used once, not duplicated as a specific Rampuri or Baghi witness.'] if lang=='kc' else []
            candidate=dict(candidate_key=key+':'+str(i),lang=lang,form=form,gloss=gloss,tags=list(dict.fromkeys(tags)),notes=notes,typed_uncertainty=uncertainty,raw_cell=form,expansion=[])
            if key.endswith('9-distinct') and i==3:candidate['variant_of_key']=key+':2'
            unit['candidates'].append(candidate)
        unit['disposition']='emitted';units.append(unit)
    return units


def main():
    paths=['whole-paired-grammar-reviewed-20260927.jsonl','whole-paired-sentences-reviewed-20260927.jsonl','whole-paired-commentary-reviewed-20260927.jsonl','whole-paired-introduction-reviewed-20260927.jsonl']
    grammar,sentences,commentary,introduction=map(load,paths)
    units=expand_grammar(grammar)+expand_sentences(sentences)+expand_prose(commentary,'commentary')+expand_prose(introduction,'introduction')
    candidates=[c for u in units for c in u['candidates']]
    keys={c['candidate_key'] for c in candidates}
    assert len(keys)==len(candidates)
    assert all(not c.get('variant_of_key') or c['variant_of_key'] in keys for c in candidates)
    (P/'whole-paired-expansion-draft-20260927.jsonl').write_text(''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units))
    counts={lang:sum(c['lang']==lang for c in candidates) for lang in ['ramp','ba','kc']}
    report=dict(status='source-local expansion and metadata draft; final self-review and independent output audit pending',source_groups=len(units),candidate_rows=len(candidates),per_language=counts,context_groups=sum(not u['candidates'] for u in units),variants=sum(bool(c.get('variant_of_key')) for c in candidates),uncertain_rows=sum(bool(c['typed_uncertainty']) for c in candidates),input_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in paths})
    (P/'whole-paired-expansion-summary-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()

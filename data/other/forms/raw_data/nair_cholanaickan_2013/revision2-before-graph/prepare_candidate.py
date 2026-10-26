"""Reproducible 15-column source-stage proposal; source review remains pending."""
import csv,hashlib,io,json,unicodedata
from pathlib import Path
HERE=Path(__file__).resolve().parent
units=[json.loads(s) for s in (HERE/'physical-units-first-reading.jsonl').read_text().splitlines()]
COLUMNS=['Language_ID','Parameter_ID','Form','Gloss','Native','Phonemic','Notes','Source','Cognateset','Etymology','Entry_Key','Variant_Of_Key','Borrowed_From_Key','Derivation_Parent_Keys','Tags']
tags={
'name-marker':'suffix','plural-example':'multiword-expression num','accusative-markers':'suffix acc','accusative-example':'noun acc','instrumental-markers':'suffix instr','instrumental-example':'noun instr','ablative-markers':'suffix abl','ablative-examples':'abl','genitive-markers':'suffix gen','genitive-examples':'pron gen poss','locative-markers':'suffix loc','locative-examples':'loc','purposive-markers':'suffix purposive','purposive-examples':'noun purposive','vocative-markers':'suffix voc','personal-pronouns':'pron','interrogatives':'interr','masculine-markers':'suffix m','feminine-markers':'suffix f','gender-prefixes':'prefix','gender-examples':'noun','future-person-markers':'suffix','negative-marker':'auxiliary neg','negative-examples':'verb neg','causative-permissive':'suffix','causative-permissive-examples':'verb','hortative-marker':'suffix hortative','hortative-examples':'verb hortative','lexical-list':''}
# These are visible documentary or zero-exponent mentions, not emitted sound forms.
audit_only={('instrumental-markers',2):'identical second -indu is an unresolved typographic variant claim',('genitive-markers',1):'zero exponent; no phonetic form',('vocative-markers',5):'stem ending condition -annu, not a vocative marker',('feminine-markers',7):'zero exponent; no phonetic form',('feminine-markers',8):'South Dravidian comparative reconstruction, not separately attributed target form',('feminine-markers',9):'Tamil/Kodagu/Kannada retention comparison, not separately attributed target form',('future-person-markers',4):'zero exponent; no phonetic form'}
# Exact table-cell contexts; do not infer values from generic English prose.
pron_features=['1sg','1pl','1pl','1sg gen poss','1pl gen poss','2sg','2pl','2pl','refl','refl','refl pl','refl pl','3sg m dist','3pl dist','3sg m prox','3pl prox','3sg f dist','3sg f prox','3sg n dist','3sg n prox']
interr_features=['interr','interr quantifier','interr m','interr f','interr n','interr','interr spatial','interr manner','interr spatial','interr temporal','interr temporal','interr manner']
lex_pos=['noun','noun','noun','verb','noun','noun','noun','noun','noun','noun','noun','noun','noun','noun','noun','noun','noun']
assert len(pron_features)==20 and len(interr_features)==12 and len(lex_pos)==17
rows=[];audit=[]
for u in units:
    short=u['unit_key'].split(':')[-1]
    for i,form in enumerate(u['source_forms'],1):
        key=f"{u['unit_key']}:form{i}"
        reason=audit_only.get((short,i))
        if u['scope']!='target':reason=u['scope']
        row=None
        if not reason:
            tag=tags[short]
            if short=='personal-pronouns':tag+=' '+pron_features[i-1]
            if short=='interrogatives':tag=('pron ' if i in (1,3,4,5,6) else '')+tag+(' '+interr_features[i-1].replace('interr','').strip() if interr_features[i-1].replace('interr','').strip() else '')
            if short=='lexical-list':tag=lex_pos[i-1]
            if short=='accusative-markers' and i==3:tag='affix' # source calls it an inflectional increment; exact placement uncertain
            if short=='vocative-markers' and i==4:tag='suffix voc' # source prose -ē contrasts preceding -e
            if short=='future-person-markers':tag+=' '+('fut' if i in (1,2) else 'pret pres')
            if short=='causative-permissive':tag+=' '+('caus' if i==1 else 'possibility')
            if short=='causative-permissive-examples':tag+=' '+('caus' if i==1 else 'possibility')
            if short=='gender-examples' and ' ' in form:tag+=' multiword-expression'
            if short=='ablative-examples':tag=('adv ' if i==1 else 'noun ')+tag
            if short=='locative-examples':tag=('noun ' if i==1 else 'pron ')+tag
            if short=='negative-examples' and "won’t" in u['source_glosses'][i-1]:tag+=' fut'
            if short=='gender-prefixes':tag+=' '+('m' if i==1 else 'f')
            if short=='gender-examples':tag+=' '+('f' if i==1 else 'm')
            note=''
            if short=='instrumental-markers':note='Source prints an identical alleged instrumental variant; distinction unresolved.'
            if short=='vocative-markers' and i==4:note='The source first lists -e then describes -ē after -u or -e stems; vowel quantity uncertain in this claim.'
            if short=='hortative-examples' and i==2:note='Source gloss reads “left (someone) open” in a hortative section; source inconsistency retained.'
            if short=='interrogatives' and i in (10,11):note='Source gives the combined gloss “when /now”; alternate-to-gloss alignment unspecified.'
            if short=='accusative-example':note='Printed gloss is “child” within accusative examples; case function from heading.'
            if short=='future-person-markers' and i==3:note='Source reports -adu as common past and present personal termination.'
            if short=='feminine-markers' and i in (8,9):note='Repeated marker in comparative commentary; retained as a separately cited source mention.'
            if short in ('locative-markers','genitive-markers') and i==2:note='Source describes a case-marker variant; host restrictions retained in source audit.'
            if short=='negative-marker':note='Source calls -illa a negative existential auxiliary, attached to an infinitive or verbal noun.'
            if short=='name-marker':note='Source says community members affix this to their names; target-language attribution remains provisional.'
            if short=='plural-example':note='Source explicitly says target speech has no plural marker in these examples.'
            if short=='gender-examples':note='Source says gender prefix is used; segmentation not asserted by this row.'
            if short=='lexical-list' and i==8:note='Source spelling/gloss hiccough retained.'
            if short=='masculine-markers':note='Source compares this marker to Proto-Dravidian suffixes; no resolvable graph parent.'
            if short=='feminine-markers' and i<=6:note='Source gives cross-Dravidian comparison claims; no source-local donor edge.'
            if short=='future-person-markers' and i==1:note='Source describes -mu as an independent innovation.'
            row=['Cholanaickan','',form,u['source_glosses'][i-1] if u['source_glosses'] else '', '', '',note,
                 f"nair2013cholanaickan[p. {u['printed_page']}, {short.replace('-', ' ')}]",'', '',key,'','','',(' '.join([tag.strip(),'uncertain']) if (short=='instrumental-markers' or (short=='vocative-markers' and i==4) or (short=='interrogatives' and i in (10,11)) or (short=='name-marker')) else tag.strip())]
            assert len(row)==15 and unicodedata.is_normalized('NFC',form)
            rows.append(row)
        audit.append({'unit_key':u['unit_key'],'witness_key':key,'printed_page':u['printed_page'],'pdf_page':u['pdf_page'],
                      'raw_form_first_reading':form,'raw_gloss_first_reading':u['source_glosses'][i-1] if u['source_glosses'] else '',
                      'scope':u['scope'],'disposition':'proposal-unreviewed' if row else 'audit-only',
                      'reason':reason or 'independent literal/metadata review pending','proposed_entry_key':key if row else None,
                      'source_claim':u['source_claim'],'source_note':u['note']})
    if not u['source_forms']:
        audit.append({'unit_key':u['unit_key'],'witness_key':None,'printed_page':u['printed_page'],'pdf_page':u['pdf_page'],
                      'scope':u['scope'],'disposition':'documentary','reason':'No target form printed','source_claim':u['source_claim']})
assert len(audit)==117
out=io.StringIO(newline='');csv.writer(out).writerows(rows)
(HERE/'proposed.csv').write_text(out.getvalue())
(HERE/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False)+'\n' for a in audit))
status={'status':'unreviewed-whole-chapter-proposal','physical_blocks':len(units),'physical_form_mentions':116,
        'audit_records':len(audit),'proposed_rows':len(rows),'audit_only_form_mentions':116-len(rows),
        'csv_sha256':hashlib.sha256((HERE/'proposed.csv').read_bytes()).hexdigest(),
        'audit_sha256':hashlib.sha256((HERE/'proposal-audit.jsonl').read_bytes()).hexdigest(),
        'canonical_changes':False,'independent_audit':'pending'}
(HERE/'proposal-status.json').write_text(json.dumps(status,indent=2)+'\n');print(status)

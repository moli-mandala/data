"""Stage complete Holiya proposal and audit; never writes canonical installation."""
from pathlib import Path
import csv
import hashlib
import json
import unicodedata
from specimen_grammar import classify

P=Path(__file__).resolve().parent
SOURCE='grierson1906holiya'
CAVEAT=('Source transcription warning (p.386): the editor supplied e/o quantity from Standard Kanarese and admits possible errors; printed vowel lengths, aspiration and consonant doubling are inconsistent. Literal source notation is preserved.')

def read(name):return [json.loads(x) for x in (P/name).read_text().splitlines() if x]
def nfc(s):return unicodedata.normalize('NFC',s)

def build():
    prose=read('prose-reviewed-staged.jsonl');spec=read('specimens-reviewed-staged.jsonl')
    assert len(prose)==98 and len(spec)==1184
    rows=[];audit=[]
    by_key={u['source_unit_key']:u for u in spec}
    def emit(unit,forms,gloss,tags,notes,locator):
        keys=[]
        site=unit['site']
        if site:tags=tags+[f'dialect:Holiya:lsi1906-holiya:{site}']
        for i,form in enumerate(forms,1):
            key=unit['source_unit_key']+(f':alternate:{i}' if i>1 else '')
            actual=list(tags)
            if ' ' in form:actual+=['multiword-expression']
            if any(c in unicodedata.normalize('NFD',form.lower()) for c in 'eo'):actual+=['uncertain']
            rows.append(['Holiya','',nfc(form),nfc(gloss),'','',nfc(notes),f'{SOURCE}[{locator}]','','',key,'','','',' '.join(dict.fromkeys(actual))])
            if i>1:
                rows[-1][11]=unit['source_unit_key']
                rows[-1][14]+=' alternate'
            keys.append(key)
        return keys
    for u in prose:
        a={**u,'entry_keys':[], 'source_transcription_caveat': {'kind':'source-editor-quantity','printed_page':386,'claim':CAVEAT,'row_flag_policy':'Forms containing e/o carry uncertain because their quantities were supplied by the editor; this is distinct from literal glyph uncertainty.'}}
        if u['scope']=='holiya':
            notes=' '.join(x for x in [u['note'],u['source_claim'],CAVEAT] if x)
            a['entry_keys']=emit(u,u['forms'],u['gloss'],u['tags'],notes,f"p. {u['printed_page']}, grammatical discussion, {u['source_unit_key'].rsplit(':',1)[1]}")
            a['disposition']='emitted-target'
        else:a['disposition']='audit-only-'+u['scope']
        audit.append(a)
    for u in spec:
        a={**u,'entry_keys':[], 'source_transcription_caveat': {'kind':'source-editor-quantity','printed_page':386,'claim':CAVEAT,'row_flag_policy':'Forms containing e/o carry uncertain because their quantities were supplied by the editor; this is distinct from literal glyph uncertainty.'}}
        forms=list(u['forms']);gloss=u['gloss'];notes=[CAVEAT,'Aligned specimen expression; translation is local to this occurrence.']
        if u.get('source_group_keys'):
            group=u['source_group_keys']
            if u['source_unit_key']!=group[0]:
                a['entry_keys']=[group[0]];a['disposition']='physical-continuation-reused';audit.append(a);continue
            assert [by_key[k]['forms'][0] for k in group]==['Khōlī-','dā']
            forms=['Khōlī-dā'];gloss='room-in'
            notes.append('Source splits this word across two interlinear lines; both physical atoms are retained in the audit.')
            a['emission_group_keys']=group;a['emission_forms']=forms;a['emission_gloss']=gloss
        if any('(sic.)' in f for f in forms):
            forms=[f.replace('(sic.)','') for f in forms]
            notes.append('Source explicitly marks this printed form sic; retained without correction.')
        typed=[]
        for uncertainty in u.get('typed_uncertainties', []):
            typed.append('uncertain')
            notes.append('Source-typography uncertainty: '+uncertainty['observation'])
        if '(?)' in gloss:
            gloss=gloss.replace('(?)','');typed.append('uncertain')
            notes.append('The source explicitly questions the printed English gloss.')
        tags,observations=classify({**u,'emission_gloss':gloss})
        a['grammatical_observations']=observations
        a['emission_tags']=tags+typed
        loc=f"p. {u['printed_page']}, specimen {u['section']}, line {u['line']}, word {u['word']}"
        if a.get('emission_group_keys'):loc+=' and line 9, word 1'
        a['entry_keys']=emit(u,forms,gloss,tags+typed,' '.join(notes),loc)
        a['disposition']='emitted-target'
        audit.append(a)
    # Equality includes complete analysis, site and all scholarly notes. Physical
    # citations and keys are the only differing columns allowed in exact reuse.
    representatives={};aliases={};out=[]
    for row in rows:
        fingerprint=tuple(v for i,v in enumerate(row) if i not in {7,10})
        if fingerprint not in representatives:
            representatives[fingerprint]=row;out.append(row)
        rep=representatives[fingerprint];aliases[row[10]]=rep[10]
        if rep is not row:rep[7]+='; '+row[7]
    for row in out:
        if row[11]:row[11]=aliases[row[11]]
    for a in audit:
        a['pre_reuse_entry_keys']=a['entry_keys']
        a['entry_keys']=[aliases.get(k,k) for k in a['entry_keys']]
        if a['entry_keys']!=a['pre_reuse_entry_keys']:a['exact_attestation_reuse']=True
    with (P/'proposal.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerows(out)
    clusters=set()
    for row in out:
        current=''
        for char in row[2]:
            if unicodedata.combining(char) and current:current+=char
            else:
                if current:clusters.add(current)
                current=char
        if current:clusters.add(current)
    (P/'proposal-profile.txt').write_text('Grapheme\tIPA\n'+''.join(c+'\t'+('#' if c==' ' else c.lower().replace('ṅ','ŋ').replace('w','v'))+'\n' for c in sorted(clusters)))
    (P/'proposal-audit.jsonl').write_text(''.join(json.dumps(a,ensure_ascii=False,sort_keys=True)+'\n' for a in audit))
    report={'status':'complete scope and full independent rereading reconciled; final output audit pending','physical_units':len(audit),'candidate_rows_before_exact_reuse':len(rows),'proposal_rows':len(out),'reused_rows':len(rows)-len(out),'audit_only_controls_or_bound_morphology':sum(not a['entry_keys'] for a in audit),'native':'not printed by original source; inapplicable','reviewed_proposal':False,'hashes':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['proposal.csv','proposal-audit.jsonl','proposal-profile.txt']}}
    (P/'proposal-summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':build()

"""Full specimen37 source-only physical ledger and conservative alignment scaffold."""
import csv
import hashlib
import json
import unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[8]
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
    if (HERE/'first-reading-freeze.json').exists():
        assert p.exists() and p.read_text()==value, f'Frozen first-reading output differs: {p.name}; prepare a separate reviewed derivative'
        return
    p.write_text(value)
def dump(p,rows):
    write(p,''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))

# These are physical continuations, not two independently printed lexical heads.
JOINS=[((234,6,9),(234,7,1),'kar-chukyō,','was-made-completely,'),
       ((235,2,11),(235,3,1),'dēkhᵃtā̃-hī','on-seeing-even'),
       ((235,3,10),(235,4,1),'lagā-liyō,','he-was-attached,')]
UNCERTAINTIES={
    (234,2,8):'Initial i quantity requires narrow corroboration; this specimen prints a potentially distinct witness from grammar p51 diṇū̃.',
    (234,5,2):'Raised vowel/retroflex lateral glyph requires second reading; raw source notation retained.',
    (234,6,4):'Quantity of vowel following raised-a fragment requires second reading.',
    (234,13,3):'Raised-a and nasal attachment in labourers form require second reading.',
    (234,15,8):'Printed English below khōṭ is unresolved evil-aped versus evil-acted: joined/blurred middle consonants prevent certification; current evil-aped remains provisional.',
    (235,2,3):'Near-form first vowel is disputed ā/ī: owner narrow view suggests pālᵃnē, root narrow view provisionally pīlᵃnē but acknowledges ambiguous lower vowel body. Retain pīlᵃnē provisionally; native पाहने does not authorize changing the Roman witness.',
    (235,7,2):'Initial consonant and raised-a/retroflex-r cluster require narrow source glyph check.',
    (235,20,4):'Native/Roman initial consonant of companions differs or is typographically unclear; no harmonization.',
    (235,6,10):'Line ends achchhā- and next begins taĩ-achchhā. Native prints two words अच्छातैं अच्छा; keep full two-token scope pending emission boundary review.',
}
NATIVE_UNCERTAINTIES={
    'सगको':'Middle consonant क/ळ shape needs independent font-control reading. Roman sagᵃḷō does not authorize replacing the printed native witness.',
    'तकी-नैं':'Middle native consonants require independent reading; the paired Roman tamhī-nē̃ differs and is retained separately.',
    'क्यूंके':'Nasal sign shape (anusvara versus candrabindu) remains a native glyph hold; retain current encoded reading provisionally.',
    'जिमीदारा-के':'Native ending appears to differ from Roman jimīdār-kē; source script witnesses retained separately without harmonization.',
    'खूयो':'Native and Roman lost-form spelling differs or is unclear; retain both source-script witnesses pending independent reading.',
}
UTTERANCES=[
 ((234,1,1),(234,1,5),'narrative sentence'),
 ((234,1,6),(234,3,2),'narrative and embedded speech'),
 ((234,2,3),(234,3,2),'embedded direct speech; overlapping physical span'),
 ((234,3,3),(234,4,4),'narrative sentence'),
 ((234,4,5),(234,6,5),'narrative sentence'),
 ((234,6,6),(234,9,1),'narrative sentence'),
 ((234,9,2),(234,9,8),'narrative sentence'),
 ((234,10,1),(234,12,2),'narrative with cause clause'),
 ((234,12,3),(235,1,7),'narrative and quoted multi-sentence thought'),
 ((234,12,8),(235,1,7),'quoted multi-sentence thought; overlapping physical span'),
 ((235,1,8),(235,2,4),'narrative sentence'),
 ((235,2,5),(235,4,5),'narrative sentence'),
 ((235,4,6),(235,6,3),'narrative and embedded direct speech'),
 ((235,4,9),(235,6,3),'embedded direct speech; overlapping physical span'),
 ((235,6,4),(235,9,10),'narrative and quoted commands/explanation'),
 ((235,6,10),(235,9,10),'quoted commands/explanation; overlapping physical span'),
 ((235,10,1),(235,10,5),'narrative sentence'),
 ((235,11,1),(235,11,5),'narrative sentence'),
 ((235,11,6),(235,15,2),'narrative question and reply'),
 ((235,13,1),(235,13,3),'quoted question; overlapping physical span'),
 ((235,13,7),(235,15,2),'quoted reply; overlapping physical span'),
 ((235,15,3),(235,16,2),'narrative sentence'),
 ((235,16,3),(235,16,8),'narrative sentence'),
 ((235,17,1),(236,1,6),'narrative and multi-sentence direct speech'),
 ((235,17,6),(236,1,6),'multi-sentence direct speech; overlapping physical span'),
 ((236,1,7),(236,5,2),'narrative and multi-sentence reply'),
 ((236,2,3),(236,5,2),'multi-sentence reply; overlapping physical span'),
]

def main():
    manifest=json.loads((PACKAGE/'scope/native-specimen-image-manifest.json').read_text())
    images={x['printed_page_expected']:x for x in manifest['pages']}
    for pg in [233,234,235,236]: assert sha(ROOT/images[pg]['lossless_png_path'])==images[pg]['lossless_png_sha256']
    rows=list(csv.DictReader((HERE/'roman-first-reading.tsv').open(),delimiter='\t'))
    assert [sum(r['page']==str(pg) for r in rows) for pg in [234,235,236]]==[16,21,5]
    lines=[];atoms=[];index={}
    for row in rows:
        page,line=int(row['page']),int(row['line']); forms=row['source_forms_pipe_separated'].split('|'); gloss=row['printed_glosses_pipe_separated'].split('|')
        assert len(forms)==len(gloss),(page,line)
        key=f'lsi9.2:p{page}:aligned-line:{line}'
        wordkeys=[]
        for rank,(form,g) in enumerate(zip(forms,gloss),1):
            keyw=key+f':word:{rank}'; wordkeys.append(keyw)
            atom={'key':keyw,'parent_key':key,'printed_page':page,'physical_line':line,'word_rank':rank,
                  'raw_form':form,'raw_gloss':g,'source_script':'Roman','specimen':37,
                  'source_language_label':'Ahīrwāṭī','source_district':'Gurgaon',
                  'source_image':images[page]['lossless_png_path'],'source_image_sha256':images[page]['lossless_png_sha256'],
                  'typed_uncertainties':[UNCERTAINTIES[(page,line,rank)]] if (page,line,rank) in UNCERTAINTIES else [],
                  'stage':'owner first reading; independent original review pending'}
            atoms.append(atom);index[(page,line,rank)]=atom
        lines.append({'key':key,'printed_page':page,'physical_line':line,'raw_source_form':' '.join(forms),
                      'printed_interlinear_analysis':' '.join(gloss),'word_keys':wordkeys,'source_free_translation':None,
                      'source_image':images[page]['lossless_png_path'],'source_image_sha256':images[page]['lossless_png_sha256']})
    assert len(atoms)==367 and len({x['key'] for x in atoms})==367
    joined=[];tails=set();heads={}
    for first,last,form,g in JOINS:
        a,b=index[first],index[last]; assert a['raw_form'].endswith('-'); assert a['raw_form']+b['raw_form']==form
        joined.append({'emission_key':a['key'],'physical_keys':[a['key'],b['key']],'source_form':form,
                       'source_analysis':g,'scope':'physical line-break continuation; raw two alignment gloss fragments preserved',
                       'stage':'proposed join, independent original review pending'})
        heads[a['key']]=joined[-1]; tails.add(b['key'])
    roman_groups=[]
    for a in atoms:
        if a['key'] in tails:continue
        roman_groups.append({'roman_keys':heads[a['key']]['physical_keys'] if a['key'] in heads else [a['key']],
                             'roman_form':heads[a['key']]['source_form'] if a['key'] in heads else a['raw_form']})
    native=list(csv.DictReader((HERE/'native-first-reading.tsv').open(),delimiter='\t'))
    assert len(native)==27
    native_lines=[];native_atoms=[]
    for row in native:
        page,line=int(row['page']),int(row['line']); key=f'lsi9.2:p{page}:native-line:{line}'; keys=[]
        for rank,form in enumerate(row['source_text'].split(),1):
            kw=key+f':word:{rank}';keys.append(kw)
            native_atoms.append({'key':kw,'parent_key':key,'raw_native':form,'printed_page':page,'physical_line':line,
                                 'word_rank':rank,'source_script':'Devanagari','source_image':images[page]['lossless_png_path'],
                                 'source_image_sha256':images[page]['lossless_png_sha256'],
                                 'typed_uncertainties':[NATIVE_UNCERTAINTIES[form]] if form in NATIVE_UNCERTAINTIES else [],
                                 'stage':'owner full physical reading; independent original review pending'})
        native_lines.append({'key':key,'raw_source_native':row['source_text'],'native_word_keys':keys,
                             'source_image':images[page]['lossless_png_path'],'source_image_sha256':images[page]['lossless_png_sha256']})
    assert len(native_atoms)==365
    native_groups=[];skip=False
    for n,a in enumerate(native_atoms):
        if skip:skip=False;continue
        if a['raw_native']=='संभार-':
            b=native_atoms[n+1]; assert b['raw_native']=='के'; skip=True
            native_groups.append({'native_keys':[a['key'],b['key']],'native_form':'संभार-के'})
        else:native_groups.append({'native_keys':[a['key']],'native_form':a['raw_native']})
    assert len(native_groups)==len(roman_groups)==364
    # Printed segmentation differs locally. Do not propagate a transient one-word
    # offset through the narrative merely because counts later cancel.
    def combine(groups, field, keysfield, first, following, joined_form, reason):
        at=[n for n,r in enumerate(groups) if r[field]==first]
        assert len(at)==1,(field,first,at)
        n=at[0]; assert groups[n+1][field]==following,(first,groups[n+1])
        groups[n:n+2]=[{field:joined_form,keysfield:groups[n][keysfield]+groups[n+1][keysfield],
                         'segmentation_note':reason}]
    combine(roman_groups,'roman_form','roman_keys','bā̃ṭ','diyō.','bā̃ṭ diyō.',
            'Two Roman alignment words correspond to one hyphenated native word; keep both physical keys and English glosses.')
    combine(native_groups,'native_form','native_keys','जमा','कर-के','जमा कर-के',
            'Two native words correspond to Roman jamā-kar-kē; spacing retained, no inferred morpheme pairing.')
    combine(roman_groups,'roman_form','roman_keys','‘achchhā-','taĩ-achchhā','‘achchhā-taĩ-achchhā',
            'Roman source compound breaks across lines; full native two-word comparative span kept jointly.')
    combine(native_groups,'native_form','native_keys','अच्छातैं','अच्छा','अच्छातैं अच्छा',
            'Native comparative expression is two words; no arbitrary per-word matching to Roman physical fragments.')
    assert len(native_groups)==len(roman_groups)==362
    assert [k for r in roman_groups for k in r['roman_keys']]==[a['key'] for a in atoms]
    assert [k for r in native_groups for k in r['native_keys']]==[a['key'] for a in native_atoms]
    alignment=[{'key':f'lsi9.2:specimen37:script-alignment:{rank}',**r,**n,
                'method':'ordered source-span candidate after explicit physical-break and source-segmentation accounting',
                'status':'owner reviewed full order/anchors; native/Roman fine glyph differences remain for independent review'} for rank,(r,n) in enumerate(zip(roman_groups,native_groups),1)]
    native_by_key={a['key']:a for a in native_atoms}
    roman_by_key={a['key']:a for a in atoms}
    for span in alignment:
        span['typed_uncertainties']=[u for k in span['roman_keys'] for u in roman_by_key[k]['typed_uncertainties']]+[u for k in span['native_keys'] for u in native_by_key[k]['typed_uncertainties']]
    order={a['key']:n for n,a in enumerate(atoms)}
    expressions=[]
    for rank,(start,end,scope) in enumerate(UTTERANCES,1):
        first,last=index[start]['key'],index[end]['key']; lo,hi=order[first],order[last]; assert lo<=hi
        selected=atoms[lo:hi+1]; keys={a['key'] for a in selected}
        relevant=[r for r in alignment if set(r['roman_keys']) & keys]
        assert all(set(r['roman_keys']).issubset(keys) for r in relevant),'Expression boundary bisects source span'
        parts=[]
        for a in selected:
            if a['key'] in tails:continue
            parts.append(heads[a['key']]['source_form'] if a['key'] in heads else a['raw_form'])
        expressions.append({'key':f'lsi9.2:specimen37:whole-utterance:{rank}',
             'first_word_key':first,'last_word_key':last,'roman_word_keys':[a['key'] for a in selected],
             'native_word_keys':[k for r in relevant for k in r['native_keys']],
             'raw_roman_source_parts':[{'key':a['key'],'form':a['raw_form'],'gloss':a['raw_gloss']} for a in selected],
             'proposed_whole_form':' '.join(parts),'printed_interlinear_analysis':' '.join(a['raw_gloss'] for a in selected),
             'source_free_translation':None,'lexical_free_gloss':None,
             'translation_type':'concatenated printed interlinear analysis, not author-supplied free translation',
             'scope':scope,'provenance':'Owner boundaries from printed sentence and speech punctuation; physical lines not treated as sentences.',
             'typed_uncertainties':list(dict.fromkeys(u for r in relevant for u in r['typed_uncertainties'])),
             'stage':'whole source quotation proposal; independent source-boundary review pending'})
    assert set(k for e in expressions for k in e['roman_word_keys'])==set(order), 'Whole/speech spans must account every Roman physical atom'
    outputs={'roman-physical-lines-first-reading.jsonl':lines,'roman-atoms-first-reading.jsonl':atoms,
             'native-physical-lines-first-reading.jsonl':native_lines,'native-atoms-first-reading.jsonl':native_atoms,
             'physical-break-joins-proposal.jsonl':joined,'script-alignment-proposal.jsonl':alignment,
             'whole-utterances-first-reading.jsonl':expressions}
    for name,data in outputs.items():dump(HERE/name,data)
    for atom in atoms:
        assert not any(unicodedata.category(c)=='Co' or c=='\ufffd' for c in atom['raw_form'])
    report={'status':'complete physical owner first-reading draft; cross-script/whole-utterance review not yet closed',
            'counts':{'roman_lines':42,'roman_physical_atoms':367,'native_lines':27,'native_physical_atoms':365,
                      'roman_join_groups':3,'script_alignment_candidates':362,'whole_utterances':len(expressions)},
            'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'roman-first-reading.tsv',HERE/'native-first-reading.tsv',PACKAGE/'scope/native-specimen-image-manifest.json']},
            'output_hashes':{name:sha(HERE/name) for name in outputs},'script_sha256':sha(Path(__file__)),
            'planning_count_reconciliation':{'initial_roman_space_atom_count':366,'fine_roman_atom_count':367,
              'reason':'p234 fine aligned first reading accounts 141 atoms rather than initial planning space count 140; planning census is preserved.'},
            'known_limits':['Ordinal equal counts do not establish lexical alignment; all362 source-span anchors owner-read, independent review still required.',
                            'Native source and Roman source differences must remain separate evidence; no silent transliteration repair.',
                            'Printed Devanagari quotation punctuation differs from Roman quotation; no inferred speech labels.',
                            'Printed interlinear analysis is not a supplied free English translation.',
                            'Native font confusions including ळ/क, nasal signs, and final joined letters remain owner fine-review tasks.',
                            'Whole utterance boundaries and proposed atomic joins require independent review before proposal freeze.']}
    write(HERE/'first-reading-draft-status.json',json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    write(HERE/'script-alignment-review.tsv','rank\tRoman\tNative\n'+''.join(f'{n}\t{r["roman_form"]}\t{r["native_form"]}\n' for n,r in enumerate(alignment,1)))
    print(json.dumps(report['counts']))
if __name__=='__main__':main()

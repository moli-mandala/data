"""Outside-list first reading with complete scope accounting; no installation mode."""
from pathlib import Path
import collections, csv, hashlib, json, re
import fitz

H=Path(__file__).resolve().parent
P=H.parent.parent
W=next(p for p in H.parents if (p/'data/tags.py').exists())
PDF=W/'tmp/pdfs/korwa-kodaku-2022/source.pdf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PDF)=='a8efbe88405e27024a7a6ec786cd6fde3e382f0eaf0d0081197d3880ed97eb0c'
scope=[json.loads(l) for l in (H.parent/'scope/outside-lexical-scope-inventory.jsonl').read_text().splitlines()]
assert len(scope)==26
doc=fitz.open(PDF)
def exact_rectangles(page, form):
    """Case-sensitive whole-token/phrase matches, not search_for substrings."""
    chars=[]; coordinates=[]; line_ids=[]; line_id=0
    for b in page.get_text('rawdict')['blocks']:
        for line in b.get('lines',[]):
            line_id+=1
            for span in line['spans']:
                for c in span['chars']:
                    chars.append(c['c']); coordinates.append(c['bbox']); line_ids.append(line_id)
            chars.append('\n'); coordinates.append(None); line_ids.append(line_id)
    pattern=r'(?<!\w)'+r'\s+'.join(re.escape(s) for s in form.split())+r'(?!\w)'
    occurrences=[]
    for m in re.finditer(pattern,''.join(chars)):
        by_line={}
        for k in range(m.start(),m.end()):
            if coordinates[k] is not None:
                box=fitz.Rect(coordinates[k]); lid=line_ids[k]
                by_line[lid]=box if lid not in by_line else by_line[lid]|box
        occurrences.append([list(box) for box in by_line.values()])
    return occurrences
# Every entry is (literal form, exact gloss if printed, gloss basis). Empty gloss
# deliberately separates contextual descriptions from lexical translations.
groups={
1:[('Kora','womb','explicit parenthetical')],
2:[('kor','man','explicit quoted meaning')],
3:[('dodawas','earth diggers','explicit meaning')],
4:[('Kodwa','digger of the soil','explicit meaning')],
5:[('Zamindars','landlords','parenthetical'),('Rajas','kings','parenthetical')],
6:[('hadia','','context describes indigenous alcohol')],
7:[('Pahadi/Pahariya Korwa','hill dwellers','joint quoted gloss'),('Dihari/Dihariya Korwa','plains dwellers','joint quoted gloss')],
8:[('Gotra','','context: clans which they call Gotra')],
9:[('Hazeda','bamboo','quoted')],10:[('Edigwar-Kachi','a creeper','quoted')],
11:[('Samat','a tree','quoted')],12:[('Madhiya-Dohs','head','quoted')],
13:[('Ginu','ant hill soil','quoted')],
14:[('mangni','','context: agreeing to a marriage proposal')],
15:[('Hadia','','feast name'),('hadia','','beverage mentioned in the same sentence')],
16:[('Panchbandhi','','feast name')],17:[('puja path','ritual','quoted')],
18:[('Pitra Puja','','name of ritual')],19:[('Chuman','','programme following wedding')],
20:[('mahato','','village headman'),('mahato','','hereditary post; second occurrence')],
21:[(x,'','Kodaku clan name') for x in ['Kumharia','Kumbaria','Palhe','Charghat','Haddae','Dumaria','Sommat','Sammat']],
22:[('chatti','','sixth-day naming ceremony')],
23:[(x,'','deity name in community account') for x in ['Khuria Rani','Dulha Deo','Thakur Deo','Gohel Dharti Mahadevi','Devi Bansapti','Chalta']],
24:[(x,'','festival name in community account') for x in ['Karma','Holi','Diwali','Dussehra','Ram Navami']],
25:[(x,'','report-specific notation statement') for x in ['ə','ɐ','ʌ']],
26:[('ku','','explicit plural marker; comparison grouping procedure')]
}
anchors={
1:'According to mythology, the Korwa tribe originated from the womb',
2:'According to another legend, the Korwa were created when god created man',
3:'Other sources state that the name Korwa is',4:'Similarly, the name Kodaku is derived',
5:'They were believed to be Zamindars',6:'Drinking an indigenous alcohol',
7:'The ones who still live in the hills',8:'Socially, the Pahadi Korwa people',
9:'Hazeda',10:'Edigwar-Kachi',11:'Samat',12:'Madhiya-Dohs',13:'Ginu',
14:'Agreeing to a proposal is called mangni',15:'there is a feast known as Hadia',
16:'called Panchbandhi',17:'puja path',18:'Pitra Puja',19:'programme called Chuman',
20:'a mahato as the village headman',21:'Kumharia or Kumbaria',22:'naming ceremony called chatti',
23:'The Korwa people worship various tribal gods',24:'The Karma festival',
25:'The wordlists were transcribed using different phonetic systems',26:'Plural marker [ku]'}
authorities={1:'Powercity Korba n.d.c',2:'Singh 1994:1826',3:'Singh 1994:613',4:'Singh 1994:546',
5:'Powercity Korba n.d.b',6:'No separate term-specific authority; paragraph ends Singh 1994:547',
7:'Narayan 1990:18 cited at paragraph end',8:'CTDP 2002:n.p.',9:'CTDP 2002:n.p.',10:'CTDP 2002:n.p.',
11:'CTDP 2002:n.p.',12:'CTDP 2002:n.p.',13:'CTDP 2002:n.p.',
14:'J.P., p.c. 2004',15:'J.P., p.c. 2004',16:'J.P., p.c. 2004',17:'J.P., p.c. 2004',18:'J.P., p.c. 2004',19:'J.P., p.c. 2004',
20:'No term-specific citation in paragraph',21:'No term-specific citation in paragraph',22:'J.P., p.c. 2004',
23:'Singh 1994:614 for Korwa; 1994:546 for Kodaku',24:'No term-specific citation',25:'Behera et al. report note8',26:'Behera et al. grouping procedure'}
claims={
1:'The report attributes Kora meaning womb to the Korwa language within an origin myth, and connects the baby/name Korwa to that myth. This is attributed folklore, not an established etymology.',
2:'The report calls a relation between kor and the name Korwa plausible. Preserve the qualification; no unqualified derivation edge.',
3:'The report attributes derivation of the name Korwa from dodawas to Singh1994:613, and says the mother tongue is called by the same name. It does not identify the language of the etymological term.',
4:'The report attributes derivation of Kodaku from Kodwa to Singh1994:546. Language of the etymological term is not stated.',
25:'Source note8 says ə, ɐ and ʌ are interchangeable for the report and its appendices and that no phonemic distinction is assumed. Preserve original spellings; this does not itself authorize literal replacement.',
26:'The grouping procedure says plural marker [ku] is not considered in comparisons97 and98. The statement does not specify which of the25 lists supply it; no assigned language or inflectional reconstruction.'}
records=[]; units=[]
for i,s in enumerate(scope,1):
    p=s['pdf_page']; page=doc[p-1]; text=page.get_text()
    blocks=page.get_text('blocks')
    matching=[dict(bbox=list(b[:4]),raw_text=b[4]) for b in blocks if anchors[i] in ' '.join(b[4].split())]
    # Each exact literal occurrence also retains original PDF coordinates. A line
    # wrap may split a phrase; search_for returns all corresponding rectangles.
    witnesses=[]
    for n,(form,gloss,basis) in enumerate(groups[i],1):
        eligible=i in (1,2)
        documentary=i in (5,7,21,23,24,25)
        disposition='eligible-source-attributed-lexical' if eligible else ('documentary' if documentary else 'held-language-attribution')
        reason=('Korwa-language attribution in explicit language wording and scoped legend; supported by independent identity reviewer.' if eligible else
                'Notation symbols are documentary, not lexical witnesses.' if i==25 else
                'Names/English historical titles do not establish the language of these forms.' if documentary else
                'Community context or a source etymology does not uniquely establish lexical language; preserve the attestation without guessing.')
        occurrences=exact_rectangles(page,form)
        if i==20 and occurrences: occurrences=[occurrences[n-1]]
        positions=[box for occurrence in occurrences for box in occurrence]
        key=s['source_unit_key']+f':w{n}'
        tags=['noun'] if eligible else (['affix','pl'] if i==26 else [])
        etymology={
            1:'Behera (2022:1), citing Powercity Korba (n.d.c), reports an origin myth relating the name Korwa to Kora ‘womb’.',
            2:'Behera (2022:1), citing Singh (1994:1826), describes a relation between kor ‘man’ and the name Korwa as plausible.',
            3:'Behera (2022:1), citing Singh (1994:613), reports that the name Korwa derives from dodawas ‘earth diggers’ and that the mother tongue is called by the same name.',
            4:'Behera (2022:1), citing Singh (1994:546), reports that the name Kodaku derives from Kodwa ‘digger of the soil’.'
        }.get(i,'')
        notes=((claims.get(i,'') if not etymology else '')+' '+('Source community context: '+('Korwa' if i in range(7,20) else 'Kodaku' if i in (20,21,22) else 'Korwa and Kodaku')+'. ' if not eligible and i!=25 and i!=26 else '')).strip()
        witnesses.append(key)
        records.append(dict(entry_key=key,source_unit_key=s['source_unit_key'],witness_number=n,
            form=form,original=form,gloss=gloss,gloss_basis=basis,language='kw' if eligible else None,
            dialect=None,tags=tags,notes=notes,etymology=etymology,disposition=disposition,disposition_reason=reason,
            source_claim=claims.get(i),cited_authority=authorities[i],
            pdf_page=p,printed_page=p-10,original_literal_rectangles=positions,
            original_literal_occurrences=occurrences,rectangle_method='case-sensitive whole-token/phrase native-character coordinates; repeated exact occurrences retained separately',
            locator=f'printed{p-10}; outside-group{i}; witness{n}',
            source=f'behera2022korwakodaku[p. {p-10}]',
            native='',phonemic='',graph_edges=[],
            uncertainty=[] if eligible or documentary else [{'type':'language-attribution','reason':reason}],
            accepted_for_installation=False,review_status='owner original reading complete; independent acceptance pending'))
    units.append(dict(source_unit_key=s['source_unit_key'],scope_input=s,pdf_page=p,printed_page=p-10,
        witnesses=witnesses,source_context_blocks=matching,anchor=anchors[i],
        authority=authorities[i],source_claim=claims.get(i),
        original_image=f'original-pdf{p:03d}-150dpi.png',original_image_sha256=sha(H/f'original-pdf{p:03d}-150dpi.png'),
        status='all source witnesses retained; no installation accepted'))
assert len(units)==26 and len({r['entry_key'] for r in records})==len(records)
assert {u['source_unit_key'] for u in units}=={s['source_unit_key'] for s in scope}
assert all(u['source_context_blocks'] for u in units)
assert all(r['original_literal_rectangles'] for r in records)
assert sorted(k for u in units for k in u['witnesses'])==sorted(r['entry_key'] for r in records)
assert all(r['language'] is None for r in records if r['disposition']!='eligible-source-attributed-lexical')
for name,rows in [('outside-candidate.jsonl',records),('outside-audit.jsonl',units)]:
    (H/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
baseline=W/'data/data/other/forms/20260828-sil-korwa-kodaku.csv'
with baseline.open() as f:
    reader=csv.reader(f); base=list(reader)
# Only literal raw-cell candidates are reported, never automatic cross-lect reuse.
matches=[]
for r in records:
    if r['language']:
        for row in base:
            if r['form'] in row:
                matches.append(dict(candidate_key=r['entry_key'],raw_baseline_row=row,disposition='not automatically reused; exact scope/lect/gloss check required'))
(H/'legacy-exact-string-candidates.json').write_text(json.dumps(matches,ensure_ascii=False,indent=2)+'\n')
pins={str(p.relative_to(W)):sha(p) for p in [PDF,H.parent/'scope/outside-lexical-scope-inventory.jsonl',H.parent/'identity-review/all-source-routing-proposal.json',baseline,Path(__file__)]}
status=dict(status='complete26-group outside first-reading proposal; independent review pending',
    inputs=pins,scope_groups=26,witnesses=len(records),dispositions=dict(collections.Counter(r['disposition'] for r in records)),
    candidate_outputs={n:sha(H/n) for n in ['outside-candidate.jsonl','outside-audit.jsonl','legacy-exact-string-candidates.json']},
    source_claims_preserved=True,canonical_changes=False,
    independent_identity_consultation='Asur agreed Kora/kor kw and reported no stronger language evidence for remaining ritual/clan/headman/ku attestations.',
    graph_policy='All graph claims remain attributed evidence; no inferred source-language targets, ancestry or borrowing edges.',
    transcription_policy='Printed49 explicitly states that all wordlists and transcribed texts are field transcriptions without phonological and grammatical analysis. Preserve source notation and avoid treating generic English translations as analyzed morphology. Historical prose spellings have blank Native/Phonemic.',
    validation={'exact_26_scope_key_coverage':True,'all_witness_keys_closed':True,
                'all_original_contexts_and_literal_rectangles_present':True,
                'no_language_inferred_for_held_or_documentary_witness':True,
                'existing_exact_literal_string_matches':len(matches)},
    residuals=['All held lexical witnesses remain recoverable with exact source contexts, not dropped.',
               'No phonetic value invented for historical prose romanizations; Native/Phonemic blank.',
               'No canonical installation, parser/full build or browser acceptance claimed.'])
(H/'status.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:status[k] for k in ['scope_groups','witnesses','dispositions']},ensure_ascii=False))

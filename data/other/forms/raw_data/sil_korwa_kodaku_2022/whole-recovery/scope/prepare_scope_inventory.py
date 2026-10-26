"""Reproducible scope inventory only; native text is not accepted transcription."""
from pathlib import Path
import collections, hashlib, json, re

H = Path(__file__).resolve().parent
def dump(name, obj):
    (H/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')
def jsonl(name, rows):
    (H/name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
pages = {p: (H/f'page-{p:03d}-native.txt').read_text() for p in range(1,116)}
texts=[]; starts=[]; total=0
for p in range(93,109):
    starts.append((total,p)); texts.append(pages[p]); total+=len(pages[p])
whole=''.join(texts)
def page_at(offset):
    return max((p for start,p in starts if start<=offset), default=93)
stories=[('C.2','Job story (Korwa story)','Pahadi Korwa','Rakkaya'),
         ('C.3','Marriage story (Kodaku)','Kodaku','Kanda'),
         ('C.4','Life story (Local Sadri)','Local Sadri','Dumortoli'),
         ('C.5','Brave Hunter story','Standard Sadri','Kesramal')]
boundaries=[re.search(re.escape(s[0])+r'\s+'+re.escape(s[1]),whole).end() for s in stories]+[len(whole)]
records=[]; summaries=[]
for i,(section,title,language,site) in enumerate(stories):
    base=boundaries[i]; end=boundaries[i+1]
    if i<3: end=whole.rfind(stories[i+1][0],base,end)
    raw=whole[base:end]
    events=list(re.finditer(r'(?m)^\s*(?:\\([npgɡfqa])\s*|QUESTION\s+(\d+)\s*)',raw))
    number=None; question=None; counts=collections.Counter(); numbered=[]
    for j,e in enumerate(events):
        stop=events[j+1].start() if j+1<len(events) else len(raw)
        value=raw[e.end():stop].strip()
        raw_marker=e.group(1) or 'question-label'
        kind='g' if raw_marker=='ɡ' else raw_marker
        if kind=='n':
            m=re.match(r'\d+(?:\.\d+)*',value); assert m,(section,value)
            number=m.group(); numbered.append(number); question=None
        if kind=='question-label': question=e.group(2)
        counts[kind]+=1
        key=f'behera2022korwakodaku:{section}:native-block:{j+1}'
        records.append(dict(source_unit_key=key,section=section,marker=kind,raw_marker=raw_marker,
            number_label=number,question_label=question,
            pdf_pages=list(range(page_at(base+e.start()),page_at(base+stop-1)+1)),
            source_language=language,source_site=site,raw_native_text=value,
            disposition=('target-question-transcription-candidate' if kind=='p' and question else
                         'target-story-transcription-candidate' if kind=='p' else
                         'answer-scope-review-including-delimited-target-forms' if kind=='a' else
                         'alignment-or-documentary'),
            accepted_transcription=False,requires_original_alignment=True))
    summaries.append(dict(section=section,title=title,source_language=language,source_site=site,
        pdf_pages=list(range(page_at(base),page_at(end-1)+1)),number_labels=numbered,
        marker_counts=dict(counts),attribution_evidence='PDF27 printed17 Table7; PDF28 printed18 for C.5 prior source'))
jsonl('interlinear-native-block-inventory.jsonl',records)
dump('four-text-scope.json',dict(status='scope only; exact word and expression alignment pending',stories=summaries,
    notes=['Do not equate physical p blocks with complete sentences.',
           'C.5 contains target-language questions and target answers inside slash delimiters; recover these too.',
           'English q/a in C.2–C.4 are controls, not invented target translations.',
           'School story was tested then discontinued; no transcript is printed in this report.',
           'Native extraction can include page folios and line wraps; retain raw evidence and verify originals.']))

ranges=[(1,5,'frontmatter','bibliography/licence/abstract; no accepted lexical claim'),
 (6,6,'contents','navigation'),(7,10,'maps','site provenance'),
 (11,14,'introductory lexical and ethnographic material','quoted forms require attribution review'),
 (15,30,'survey analysis','language identity/site methodology; English questions and statistics'),
 (31,33,'A.1 questionnaire','English controls'),(34,34,'A.2 abbreviations','survey coding'),
 (35,40,'A.2.2 biodata','respondent provenance'),(41,55,'A.2 responses','English/coded survey responses'),
 (56,57,'B.1 IPA key','source notation, not lexical entries'),
 (58,59,'B.2/B.3 procedures','comparison rules and explicit ku plural marker; no inferred cognacy'),
 (60,65,'B.4 wordlist biodata','all 25 list labels and sites'),
 (66,90,'B.5 wordlists','25 lists x 210 prompts, including blanks and alternatives'),
 (91,92,'C.1 method','test methodology'),(93,108,'C.2–C.5 four interlinear texts','all p/g/f/q/a blocks'),
 (109,112,'C.6 biodata','test-subject provenance, not narrative language'),
 (113,113,'C.7/C.8 questions','English controls'),(114,115,'references','auxiliary source claims')]
dispositions=[]
for lo,hi,section,disposition in ranges:
    for p in range(lo,hi+1):
        dispositions.append(dict(pdf_page=p,printed_page=p-10 if p>=11 else None,section=section,
            scope_disposition=disposition,native_text_sha256=hashlib.sha256(pages[p].encode()).hexdigest(),
            extraction_accepted=False))
assert [r['pdf_page'] for r in dispositions]==list(range(1,116))
jsonl('whole-page-dispositions.jsonl',dispositions)

outside=[
 (11,'Kora','womb','explicit their-language claim; historical quotation'),
 (11,'kor','man','qualified source name-origin hypothesis'),
 (11,'dodawas','earth diggers','source name-origin claim; preserve cited authority'),
 (11,'Kodwa','digger of the soil','Kodaku name-origin claim'),
 (12,'Zamindars / Rajas','landlords / kings','historical comparison terms; language unassigned'),
 (12,'hadia','indigenous alcohol','shared ethnographic context; language attribution pending'),
 (13,'Pahadi / Pahariya; Dihari / Dihariya','hill dwellers; plains dwellers','names; not automatically endonyms'),
 (13,'Gotra','clans','source calls-them claim; attribution review'),
 (13,'Hazeda','bamboo','glossed clan name'),(13,'Edigwar-Kachi','a creeper','glossed clan name'),
 (13,'Samat','a tree','glossed clan name'),(13,'Madhiya-Dohs','head','glossed clan name'),
 (13,'Ginu','ant hill soil','glossed clan name'),
 (13,'mangni','', 'marriage proposal ceremony context'),(13,'Hadia','', 'marriage feast/drink context'),
 (13,'Panchbandhi','', 'feast context'),(13,'puja path','ritual','explicit quoted gloss'),
 (13,'Pitra Puja','', 'name for ritual'),(13,'Chuman','', 'post-wedding gift programme'),
 (13,'mahato','headman','two occurrences; preserve source locators'),
 (13,'Kumharia or Kumbaria; Palhe; Charghat; Haddae; Dumaria; Sommat or Sammat','', 'Kodaku clan-name list; alternatives explicit'),
 (13,'chatti','', 'sixth-day naming ceremony context'),
 (14,'Khuria Rani; Dulha Deo; Thakur Deo; Gohel Dharti Mahadevi; Devi Bansapti; Chalta','', 'deity names; documentary unless linguistic attribution established'),
 (14,'Karma; Holi; Diwali; Dussehra; Ram Navami','', 'festival names in English ethnographic prose; documentary'),
 (57,'ə / ɐ / ʌ','', 'source says interchangeable for this report; no phonemic distinction assumed'),
 (59,'ku','plural marker','explicit morphology, language/list attribution unresolved; crossref prompts97/98')]
out=[]
for i,(p,forms,gloss,note) in enumerate(outside,1):
    out.append(dict(source_unit_key=f'behera2022korwakodaku:scope:outside:{i}',pdf_page=p,
        printed_page=p-10,scope_form_lead= forms,gloss_or_context=gloss,scope_note=note,
        accepted_transcription=False,language_assignment=None,required_action='original reading and attribution disposition'))
jsonl('outside-lexical-scope-inventory.jsonl',out)
dump('scope-status.json',dict(status='whole report scope inventory; extraction and final audit pending',
    pages_accounted=115,wordlist_cells=5250,wordlist_lists=25,comparison_lists_previously_excluded=7,
    physical_interlinear_blocks=len(records),stories=summaries,outside_scope_groups=len(out),
    canonical_rows_preserved=4458,legacy_reconciliation='../legacy-review/reconciliation.json',
    exact_reuse_policy='Only exact form, sense, lect and source provenance agreement may reuse an existing key; citations must include the new occurrence. No automatic form-only merging.',
    pending=['Original visual verification of scope leads and section boundaries',
      'All seven excluded lists and four full texts require complete literal recovery',
      'Source language identities/routes pending independent identity-review',
      'Existing 18 lists need source-supported grammatical metadata review',
      'Full word/expression alignment, source-claim graph review, profile/metadata, independent audit, focused parser/tests and installation gates not performed']))
print(json.dumps(dict(pages=115,native_blocks=len(records),stories=summaries,outside_groups=len(out)),ensure_ascii=False))

"""Record completed original-image structure review; no lexical output or installation."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
def evidence(relative):
    path = ROOT / relative
    return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
def save(name, data):
    (HERE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

blocks = []
ranges = [(1,25),(26,52),(53,79),(80,106),(107,133),(134,160),(161,187),(188,214),(215,241)]
for physical, (first,last) in zip(range(668,701,4), ranges):
    target = 'tmp/kui-648.png' if physical == 668 else f'tmp/kui-physical-{physical}.png'
    english = 'tmp/kui-649.png' if physical == 668 else f'tmp/kui-physical-{physical+1}.png'
    blocks.append(dict(target_physical=physical,target_printed=physical-20,
                       english_physical=physical+1,english_printed=physical-19,
                       first_prompt=first,last_prompt=last,
                       header='Kui (Khondmals)',header_verified=True,
                       target_image=evidence(target),english_image=evidence(english)))
save('independent-table-structure-review-20260927.json', {
    'reviewer':'norton_finish','date':'2026-09-27',
    'status':'complete independent original-image structure review; not literal transcription approval',
    'source':evidence('tmp/pdfs/LSI-V4.djvu'),'source_physical_pages':701,
    'coverage':{'paired_blocks':9,'prompts':241,'direct_literal_prompt_cells':233,
                'cross_referenced_future_prompts':6,'printed_no_response_cells':2},
    'blocks':blocks,
    'controls':[
        {'prompts':[133,136],'finding':'Better and Higher respectively have printed no-response leaders, clearly a series of dots in the sharper alternate scan (earlier low-resolution census described dashes). Retain physical records; no invented comparative forms.'},
        {'prompts':list(range(195,201)),'finding':'One physical bracket says As the present tense. Explicitly covers all six future prompts.',
         'approved_structural_policy':'Keep one physical bracket witness; resolve each future attestation to the corresponding accepted present form, preserve distinct future source key and both citations. These are source-explicit uses, not six printed tokens.',
         'reference_map':{str(n):n-16 for n in range(195,201)},'literal_reference_acceptance':'pending owner transcription and literal audit'},
        {'prompts':[49,50],'finding':'Older/younger parenthetical kinship senses distinguish witnesses; do not flatten into generic alternatives.'},
        {'prompts':[97],'finding':'If cell includes source suffix qualification; retain bound grammatical analysis. Exact lexical reading remains owner task.'},
        {'prompts':list(range(185,191)),'finding':'Past Tense paradigm spans physical692 and696; heading governs both parts.'},
        {'prompts':list(range(220,242)),'finding':'Exactly22 complete sentence/expression prompts. Items215–219 finish preceding verbal section.'},
        {'prompts':[207,211],'finding':'Paired English697 prints He goes and item211 correctly. Do not transfer another table-page typo or addendum correction here.'},
        {'prompts':[1,2,3,4,5,6,7,26,29,32,46,47,66,81,93,132,209,233],
         'finding':'Visible comma/parenthesis/semicolon/or/qualification structures require literal scope retention. Alternatives not independently counted or transcribed by this structural review.'}
    ],
    'boundary':'Last table leaf physical701 ends241 and printer imprint. This table census does not substitute for chapter457–471 and source-wide mention recovery.',
    'attribution':'Target middle column only, between Malto and Gondi(Mandla); English from next leaf. All nine headers explicitly Khondmals.',
    'canonical_changes':False
})

save('independent-bibliography-addenda-review-20260927.json', {
    'reviewer':'norton_finish','date':'2026-09-27',
    'status':'original authorities pages458–459 and later bound-in pages17–18 visually reviewed; auxiliary works not independently consulted',
    'original_authorities_images':[evidence('tmp/kui-458.png'),evidence('tmp/kui-459.png')],
    'existing_primary_key':'grierson1906lsi4',
    'addendum':{
        'existing_key':'grierson_addenda_minora_iv_boundin',
        'separate_scan_identifier':'in.ernet.dli.2015.42222',
        'not_part_of':'701-page tmp/pdfs/LSI-V4.djvu',
        'witnesses':[
            dict(evidence('tmp/holiya-scope-review/addenda-leaf706.jpg'),printed_page=17,archive_leaf=706,url='https://archive.org/download/in.ernet.dli.2015.42222/page/n706.jpg'),
            dict(evidence('tmp/korvi-full-scope/alternate-leaf708.jpg'),printed_page=18,archive_leaf=708,url='https://archive.org/download/in.ernet.dli.2015.42222/page/n708.jpg')],
        'date_policy':'Undated later bound-in witness. Do not call this1906 or assert unverified1927 SupplementII equivalence.',
        'p459_addition':[
            {'author':'Friend-Pereira, J. E.','title':'A Grammar of the Kūi Language','address':'Calcutta','year':'1909'},
            {'author_as_printed':'Schultze, F. V. P.','title':'A Grammar of the Kuvi Language, with Copious Examples','address':'Madras','year':'1911'},
            {'author_as_printed':'Schultze, F. V. P.','title':'Vocabulary of the Kuvi-Kond Language, with short Sentences on General Subjects for Conversational Purposes','address':'Madras','year':'1913','existing_key':'S-Kuwi','existing_author_spelling':'Schulze','policy':'Reuse existing work identity; preserve printed author spelling in secondary citation audit, no duplicate work.'}],
        'historical_context':'Calls Kuvi a dialect of Kui, spoken by some three hundred thousand people in Vizagapatam Agency and District. Preserve as attributed historical classification; do not merge modern language identities.',
        'lexical_examples':'None in this Kui/Kuvi addendum paragraph; titles and language names only.',
        'lexical_corrections':'No Kui-column or chapter457–471 lexical correction on the two inspected sheets. p18 corrections are explicitly other columns/languages or shared English prompts on other printed leaves.'},
    'priority_auxiliary_identifications':[
        {'author':'Lingum Letchmajee','title':'An Introduction to the Grammar of the Kui or Kandh Language','first_publication':'Calcutta Christian Observer, May and June1853','edition':'Second edition, revised and corrected by Rev.J.M.Descombes and Dr.G.A.Grierson, Calcutta1902','role':'Explicit grammar and quantity-analysis authority; qualify source assertions and differing witnesses.'},
        {'author_as_printed':'Smith, Major J. McD.','title':'A Practical Handbook of the Khond Language','address':'Cuttack','year':'1876','role':'Explicit source of p464 tale, plus orthographic comparison; Grierson supplies interlinear analysis. Do not claim original Handbook independently consulted.'},
        {'author':'Frye, John Percival','title':'A Grammar and Progressive Reading Lessons in the Kondh Language with an English Translation','address':'Cuttack','year':'1851','role':'Listed authority on p458.'},
        {'author':'Friend-Pereira, J. E.','title':'Marriage Customs of the Khonds','journal':'Journal of the Asiatic Society of Bengal','volume':'71','part':'iii','year':'1902','pages':'18ff','role':'p459 says contains a Kui invocation with translation; not a generic vocabulary claim.'}
    ],
    'other_listed_authorities':'Hodgson1857/reprint1880 (H.Newill vocabulary); Lyall1868; Carmichael1869; Dalton1872(V.Ball vocabulary); Campbell1874; Manual of Administration of Madras PresidencyII1885. These contextual authority listings do not establish direct consultation or new attestations.',
    'registration_recommendation':'Register minimal identified auxiliary works only where imported analysis/secondary quotations depend on them, with unconsulted-secondary status. Reuse S-Kuwi and existing primary/addendum identities. Do not infer publishers or precise page endpoints. Dedicated Kui source metadata should explain Grierson editorial interlinear and witness distinctions.',
    'canonical_changes':False
})
print('Wrote independent table structure and bibliography/addenda reviews.')

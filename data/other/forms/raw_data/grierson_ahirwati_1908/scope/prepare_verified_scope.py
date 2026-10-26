"""Record original-reviewed scope only. No lexical rows or canonical mutations."""
from pathlib import Path
import csv, hashlib, json, re, unicodedata

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[6]
TMP=ROOT/'tmp/jambu-next-source/ahirwati-scope-20260927'
DISC=ROOT/'tmp/jambu-next-source/ahirwati-discovery-20260927'
TEXT=ROOT/'tmp/pdfs/lsi-v9-2/LSI-V9-2-alternate-djvu.txt'

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def rel(p): return str(p.relative_to(ROOT))
def save(name,value,jsonl=False):
    data=(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in value) if jsonl
          else json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode()
    p=BASE/name
    if p.exists(): assert p.read_bytes()==data, f'Immutable census changed: {name}'
    else: p.write_bytes(data)
def evidence(p): return {'path':rel(p),'sha256':sha(p)}
def fold(s): return ''.join(c for c in unicodedata.normalize('NFKD',s).lower() if not unicodedata.combining(c))

# The text's original line coordinates are retained; no automatic printed-page offset.
lines=TEXT.read_text().splitlines()
pattern=re.compile(r'ahir|hirwat|hirwal|irwati|irwali|aliir|ahii|abirw')
def disposition(n):
    if 8480<=n<=8640: return 'core_description_and_grammar', [49,50,51]
    if 25840<=n<=25900: return 'core_gurgaon_specimen_and_script', [233]
    if 26180<=n<=26320: return 'core_mixed_rohtak_specimen_and_translation', [237,238,239]
    if 17380<=n<=17460: return 'outside_documentary_geography', [147,148]
    if 17530<=n<=17560: return 'outside_bagri_comparison_scope', [149]
    if 17780<=n<=17860: return 'outside_bagri_copula_comparison_scope', [151]
    if n==1496 or n==1551: return ('documentary_classification' if n==1496 else 'false_positive_OCR_inhabitant'), [3]
    if 7740<=n<=7920: return 'documentary_mewati_distribution_no_target_quotation', [43,44,45]
    if 8860<=n<=8890: return 'documentary_distinct_malvi_Ahiri_name', [53,54]
    if 20740<=n<=20910: return 'social_Ahir_character_in_Bagri_story_not_language_label', [176,177]
    if 25250<=n<=25830: return 'social_Ahir_character_in_Mewati_story_not_language_label', [226,228,230,232]
    if n==48648: return 'distinct_Cutch_Ayari_Ahiri_explicitly_Bhil_languages', [424]
    return 'false_positive_OCR_or_non_Ahirwati_lexical_string', []
leads=[]
for i,l in enumerate(lines):
    if pattern.search(fold(l)):
        d,p=disposition(i+1)
        leads.append({'key':f'ia-djvutxt:line:{i+1}','raw_line_number':i+1,
          'raw_line':l,'raw_context':'\n'.join(lines[max(0,i-2):i+3]),
          'disposition':d,'scope_printed_pages':p,
          'literal_transcription_accepted':False,
          'page_scope_caveat':'Context association, not a mechanical OCR-line to printed-page map. Originals govern lexical readings.'})
assert len(leads)==78
save('volume-wide-text-lead-census.jsonl',leads,True)

tables=[]
start=[1,26,53,80,107,134,161,188,215]
end=[25,52,79,106,133,160,187,214,241]
for p in range(304,322):
    cols=(['English','Marwari','Marwari (Thali of Jaisalmer)','Jaipuri','Mewati'] if p%2==0
          else ['Malvi (Rangri)','Malvi (when different from Rangri)','Nimari (Nimar)','English'])
    j=(p-304)//2
    tables.append({'key':f'lsi9.2:p{p}:standard-table-header','printed_page':p,
      'djvu_physical_one_based':p+15,'first_prompt':start[j],'last_prompt':end[j],
      'columns':cols,'ahirwati_column_present':False,
      'header_original':evidence(TMP/f'table-header-footer-printed-{p}-physical-{p+15}.png'),
      'footer_original':evidence(TMP/f'table-footer-printed-{p}-physical-{p+15}.png'),
      'review':'Every header and footer viewed; paired pages duplicate English prompt scope, not an Ahirwati wordlist.'})
save('standard-table-census.jsonl',tables,True)

# Structural slots precede any accepted fine transcription. Paragraph/cell counts
# are distinct from quoted form counts; blank/output/reuse decisions are later gates.
units=[]
def unit(p,label,kind,scope,**extra):
    units.append({'key':f'lsi9.2:p{p}:{label}','printed_page':p,'kind':kind,
      'scope':scope,'stage':'physical census; no accepted lexical transcription',**extra})
for p,labels in {
49:['distribution-1','mewati-boundary-2','nabha-narnaul-mixture-3','classification-4','Abhira-history-5','Ahir-history-6','footnote-name-hyphens','footnote-Dajana-unreturned','footnote-Narnaul-list-classification'],
50:['history-continuation-and-Khandeshi-comparison','population-table-introduction','literature-and-authorities','script-provenance-unprinted-Gurmukhi','grammar-standard-and-copula-comparison','nominal-ending-comparison-continuation','population-footnote'],
51:['inherited-conjugation-closing'],
233:['two-specimen-script-and-lect-introduction'],
237:['folktale-provenance','mixed-Ahirwati-Bangaru-quoted-comparisons']}.items():
    for label in labels: unit(p,label,'paragraph_or_footnote','Whole documentary and quoted-form scope; quotations receive children during fine reading.')
for i,label in enumerate(['Gurgaon','Pataudi','Delhi','Rohtak-Jhajjar','South-Nabha','Narnaul-Patiala'],1):
    unit(50,f'population-cell:{i}','documentary_table_cell',label)
grammar=[
('locative-genitive-whole-example','whole example and contrasted possessive marker'),
('locative-noun-alternatives','horse alternatives'),('consonant-noun-locative','house example'),
('dative-agent-suffix-alternatives','two same-cell markers with dative and agent functions'),
('genitive-suffix','marker compared with Mewati'),('agent-passive-whole-example','complete future passive example'),
('genitive-locative-dative-use','to-me example'),('occasional-neuter-example','qualified that-to-be-given example'),
('personal-pronoun-me-alternatives','two printed alternatives'),('possessive-pronoun-your','quoted possessive'),
('own-pronoun-alternatives','two printed alternatives'),
('demonstrative-this-nominative','two masculine alternatives plus feminine'),
('demonstrative-this-oblique','two singular alternatives and plural'),
('demonstrative-that-nominative','two masculine alternatives plus feminine'),
('demonstrative-that-oblique','printed singular alternatives and plural'),
('relative-when-example','whole then/when statement'),
('qualified-oblique-relative-interrogative','explicit conjecture; author has not encountered examples'),
('past-copula-three-gender-number-forms','three printed past forms'),
('Bawal-Kot-Kasam-regional-copulas','three parenthetical alternatives plus repeated past form'),
('Bagri-second-plural-borrowing-claim','source claim; no inferred donor edge'),
('Panjabi-comparison-whole-copulas','quoted Ahirwati and Panjabi expressions')]
for label,scope in grammar: unit(51,label,'grammar_quoted_group',scope)
for person in range(1,4):
    for number in ['sg','pl']: unit(51,f'present-copula:p{person}:{number}','paradigm_cell',f'Printed person {person}, {number}; alternatives preserved as one physical cell.')
for label,scope in [
 ('quoted-first-Ahirwati-sentence','Printed good-Ahirwati quotation; reconcile exact matching narrative span, no duplicate lexical head merely because quoted twice.'),
 ('quoted-second-Bangaru-sentence','Printed equally-good-Bangaru quotation; route explicit comparator to kaithal, reconcile narrative span without changing whole mixed specimen identity.'),
 ('local-past-participle-spelling-comparisons','Quoted bolo/bolyo versus Bangaru bola, with literal spelling distinctions and source locality qualification.'),
 ('quoted-oblique-pronoun','Printed alternative to nominative this, separately source-attested.')]:
    unit(237,label,'quoted_comparison_group',scope)

# Explicitly counted physical aligned rows, not sentence boundaries. Space-separated
# source atom estimates are recorded separately for rechecking by the fine reader.
atom_counts={234:[8,9,10,8,7,8,10,8,8,9,8,8,9,9,9,12],
235:[9,11,10,9,12,10,9,9,10,5,9,7,9,8,6,8,9,10,10,8,10],
236:[7,9,10,10,2],238:[9,9,10,10,10,8,9,10,8,10,9,11,9,9,2]}
for p,counts in atom_counts.items():
    for i,n in enumerate(counts,1):
        unit(p,f'aligned-line:{i}','physical_interlinear_line',
          'Entire target transliteration and corresponding printed English analysis; preserve cross-line compounds.',
          source_atom_count_initial=n,atom_count_status='Owner visual space-group planning count; must be reconciled with fine token ledger before claiming extraction completeness.',
          specimen_number=37 if p<237 else 38,
          image=evidence(DISC/f'printed-{p}.png'))
for p,n,script in [(233,27,'Devanagari'),(237,11,'Persian')]:
    for i in range(1,n+1): unit(p,f'native-line:{i}','physical_native_script_line',script,
      image=evidence(DISC/f'printed-{p}.png'),alignment='Same whole specimen; no invented token pairing before source alignment.')
unit(238,'free-translation-opening','printed_free_translation','First two physical English lines; continues on239.')
unit(239,'free-translation-completion','printed_free_translation','Whole continuation through final question; not a new target-language text.')
for p,kind,note in [
 (3,'documentary','Historical classification, no Ahirwati lexical quotation'),
 (43,'documentary','Distribution/headquarters/population'),(44,'documentary','Mewati boundary and mixture; Bighota label belongs Mewati'),
 (45,'documentary','Questioned Delhi language returns'),(53,'documentary','Distinct Malvi Ahiri alias'),
 (54,'documentary','Name commentary referring back49; does not make Malvi examples Ahirwati'),
 (147,'documentary','Bagri boundary'),(148,'documentary','Narnaul classification; no lexical example assigned to Ahirwati'),
 (149,'comparative_paragraph','Bagri b versus w/v in explicitly Ahirwati-related influence paragraph'),
 (151,'comparative_paragraph','Sirsa present copula six cells and Hissar past three forms; source Bagri attribution, not Ahirwati')]:
    unit(p,'outside-lead-scope',kind,note,image=evidence(TMP/f'outside-printed-{p}-physical-{p+15}.png'))
for person in range(1,4):
    for number in ['sg','pl']: unit(151,f'outside-Bagri-present:p{person}:{number}','comparison_paradigm_cell',
      'Printed Sirsa Bagri copula affected by Bangaru or Ahirwati; do not infer target origin.')
unit(151,'outside-Bagri-past-group','comparison_quoted_group','Three Hissar Bagri past forms; source influence claim retained.')
unit(424,'Cutch-Ayari-Ahiri','documentary_control','Explicitly referred to Bhil languages IX.3, not Ahirwati.',
     image=evidence(TMP/'boundary-physical-440.png'))
unit(50,'Gurmukhi-not-printed','documentary_unavailable_witness','Author explicitly declines to print received Gurmukhi specimens; not an available missing extraction.')
save('physical-source-unit-census.jsonl',units,True)

sources=[ROOT/'tmp/pdfs/lsi-v9-2/LSI-V9-2.djvu',ROOT/'tmp/pdfs/lsi-v9-2/LSI-V9-2-alternate.pdf',
         TEXT,ROOT/'tmp/pdfs/lsi-v9-2/alternate-metadata.json',ROOT/'tmp/pdfs/lsi-sprakbanken.xml.bz2']
supp=[ROOT/f'tmp/jambu-next-source/ahirwati-sup2-{p:03}.jpg' for p in [2,3,4,35,39,41,43,44,45,46,47]]
save('source-and-errata-review.json',{
 'stage':'Scope/edition review, no accepted lexical output',
 'main':{'author':'George Abraham Grierson','year':1908,'volume':'IX','part':'II',
   'printed_title':'Linguistic Survey of India. Indo-Aryan Family, Central Group. Specimens of the Rajasthani and Gujarati',
   'later_title_correction':'1927 Addenda p41 says omit THE before Rajasthani and Gujarati; original title evidence retained.',
   'publisher':'Superintendent Government Printing, India','address':'Calcutta',
   'title_image':evidence(TMP/'boundary-physical-4.png'),
   'license':'Public-domain original1908; Commons source record declares PDM1.0. IA metadata separately says Copyright Permitted.',
   'primary_urls':['https://commons.wikimedia.org/wiki/File:Linguistic_Survey_of_India_Vol_9_Part_2.djvu','https://archive.org/details/in.ernet.dli.2015.34951','https://dsal.uchicago.edu/books/lsi/']},
 'text_derivative':{'url':'https://archive.org/download/in.ernet.dli.2015.34951/2015.34951.Indo-aryan-Family-Central-Group-Part-Ii-Specimens-Of-The-Rajasthani-And-Gujrati_djvu.txt',
   'acquired':'2026-09-27','bytes':TEXT.stat().st_size,'sha256':sha(TEXT),'sha1':hashlib.sha1(TEXT.read_bytes()).hexdigest(),
   'policy':'Locator aid only. No form-feed page boundaries. Never accepted as transcription without original review.'},
 'later_errata':{'work':'LSI Volume I Supplement II: Addenda et Corrigenda Minora','year':1927,
   'publisher':'Government of India Central Publication Branch','address':'Calcutta',
   'title_original':evidence(supp[0]),'contents_original':evidence(supp[2]),
   'IX2_printed_pages':[41,42],'next_section_page':43,
   'all_IX2_item_pages':[5,19,26,53,60,123,307,308,311,313,314,315,316,317,318,319,320,321,329,337,344,346,361,427,468,470,473,475],
   'core_or_outside_comparator_literal_corrections':[],
   'relevant_documentary_changes':['Title omit THE','p53 Sondwari name diacritic correction; distinct Malvi population table'],
   'evidence':[evidence(p) for p in supp],
   'boundary':'Original p43 starts IX.3, closing complete IX.2 minor list.',
   'other_supplement_gate':'The contents says important addenda separately SupplementI; applicability must be checked before final whole-source acceptance, not silently equated to the minor list.'},
 'original_volume_end':{'physical493':'printed477 terminal Gujarati wordlist','physical494':'colophon only; no bound-in errata',
   'evidence':[evidence(TMP/f'boundary-physical-{p}.png') for p in [493,494]]},
 'mapping_cautions':['DjVu printed+15 individually verified49–51, outside3/43–54/147–151 and304–321; not global.',
   'DjVu physical440=printed424,493=477 (offset16). Historical filename outside-printed477-physical492 actually shows476.',
   'AlternatePDF printed+17 verified49–52/233–240/304–305; exploratory purported321 is319. No global offset.',
   'Sparse retainedDjVu text has138 emitted chunks for494physical leaves, so initial empty searches cannot close volume scope. Historical receipts preserved.'],
 'inputs':[evidence(p) for p in sources],
 'summary':{'text_leads':len(leads),'standard_header_pages':18,'standard_prompt_pairs':9,'Ahirwati_standard_cells':0,
   'core_pages':10,'physical_census_records':len(units),'interlinear_lines':sum(map(len,atom_counts.values())),
   'initial_physical_Roman_atom_count':sum(map(sum,atom_counts.values())),
   'native_lines':38,'exact_child_token_count':'Planning count awaiting independent fine-ledger reconciliation; not accepted extraction count.'}})

registry={r['ID']:r for r in csv.DictReader((ROOT/'data/cldf/languages.csv').open())}
save('historical-identity-proposal.json',{'stage':'Proposed, root source decision incorporated; no registration',
 'base':registry['mewati_jhambaus'],'Bangaru_comparator_base':registry['kaithal'],
 'new_lects':[{'ID':'lsi1908_ahirwati_gurgaon','Language_ID':'mewati_jhambaus','Name':'Ahirwati (Gurgaon, LSI1908)',
   'Location':'District Gurgaon, historical survey','Latitude':'','Longitude':'','Glottocode':'',
   'source_scope':'Specimen37; grammar follows explicit Gurgaon-standard rule on50, unless individually differently attributed.'},
 {'ID':'lsi1908_ahirwati_rohtak_jhajjar','Language_ID':'mewati_jhambaus','Name':'Ahirwati (Jhajjar/Rohtak, mixed LSI1908)',
   'Location':'Jhajjar Tahsil, district Rohtak, historical survey','Latitude':'','Longitude':'','Glottocode':'',
   'source_scope':'Specimen38 historically mixed with Bangaru; preserve qualification, do not guess per-token genealogy.'}],
 'policy':['No Jhambaus coordinates inherited','No modern lect Glottocode inferred','Explicitly quoted Bangaru comparator to existing kaithal/Haryanvi',
   'Bagri outside quotes retain Bagri identity','Whole mixed Rohtak specimen retains qualified historical Ahirwadi routing',
   'No donor/borrowing edges from mixture or historical influence alone'],
 'evidence':[evidence(BASE/'ROOT_IDENTITY_DECISION.md')],
 'existing_raw_coverage':{'20230521-rajasthani.csv':220,'20260913-zoller-linguistic-data.csv':1,
   'generic_LSI_Ahirwadi_rows':0,'count_type':'Canonical raw rows in current discovery, not compiled unique forms.'}})

outputs=['volume-wide-text-lead-census.jsonl','standard-table-census.jsonl','physical-source-unit-census.jsonl','source-and-errata-review.json','historical-identity-proposal.json']
save('verified-scope-manifest.json',{'status':'reviewable scope planning; extraction not started; SupplementI applicability gate open',
 'files':{name:sha(BASE/name) for name in outputs},'script_sha256':sha(Path(__file__))})
print(json.dumps({'records':len(units),'leads':len(leads),'aligned_lines':57,'initial_atoms':sum(map(sum,atom_counts.values()))}))

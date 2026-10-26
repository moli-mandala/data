"""Keyed original-context first reading of every known outside-chapter Kui span."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[5]
rows=[]
def add(page,slug,form,gloss='',analysis=None,note='',uncertainty=None):
 rows.append({'source_unit_key':f'grierson1906kui:p{page}:outside:{slug}',
  'printed_page':page,'djvu_physical_page':229 if page==210 else page+20,
  'language_label':'Kui','source_lect':'unlocalized comparative witness; no automatic Khondmals/Gumsur assignment',
  'form':form,'gloss':gloss,'explicit_analysis':analysis or [],'note':note,
  'uncertainty':uncertainty,'review_status':'original-image first reading; independent acceptance pending'})
add(210,'suffix','añju',analysis=['suffix'],note='Explicitly called Kui suffix; compared with Juang ñje. Possible first-person origin discussed for Juang, not established for this Kui suffix.')
add(210,'father','abhañju','father',note='Immediately following Compare after the explicit Kui suffix discussion; no repeated language label.',uncertainty={'type':'attribution','reason':'Probably exemplifies immediately preceding Kui añju; language is not repeated. Retain contextual uncertainty, do not silently assign to Juang or Kui as certain.'})
add(290,'woman','kuāli','a Kui woman',analysis=['noun'],note='Bishop Caldwell compares with a feminine ending; retain attributed comparison, not an ancestry edge.')
for n,f in enumerate(['gā','ṅgā','skā'],1):add(291,f'plural-{n}',f,analysis=['suffix','plural'],note='Explicit plural suffix comparison. Source says probably originally neuter, now also used for rational nouns; do not turn conjecture into categorical gender.')
add(291,'young-man','lāvēñju','a young man',analysis=['noun'],uncertainty={'type':'transcription','alternatives':['lāvēñju','lāveñju'],'reason':'Narrow original detail appears to include e macron; Nordic locator omits it. Root adjudication requested.'})
add(291,'young-man-dative','lāvēni-ki','a young man',analysis=['noun','dative'],note='Same lexical referent explicitly introduced as dative; retain hyphen and source declension analysis.')
add(293,'you','īru','you',analysis=['pronoun','second-person','plural'],note='Explicit second-person plural paragraph; no singular inference.')
add(293,'we-exclusive','āmu','we',analysis=['pronoun','first-person','plural','exclusive'],note='Actual comparative table under Kui, exclusive row; absent from sentence-only locator extraction.')
add(293,'we-inclusive','āju','we',analysis=['pronoun','first-person','plural','inclusive'],note='Actual comparative table under Kui, inclusive row; no long final u visible.')
add(295,'he-struck','ēañju pāgiteñju','he struck',analysis=['whole-expression','third-person','singular','past'],note='Whole source-aligned comparison clause. No invented glossing of individual components.')
add(296,'thou-doest','īnu gi-d-ī','thou doest',analysis=['whole-expression','second-person','singular','present'],note='Explicit indefinite present discussion, said generally usable as present and often future. Do not assign a distinct future attestation without scoped evidence.')
add(296,'past-conjunctive','t',analysis=['suffix','past','conjunctive-participle'],note='Printed isolated t in comparison of conjunctive-participle suffixes used to form past. No added hyphen in literal form; not lexical verb.')
add(413,'ablative','ṭai',analysis=['suffix','ablative'],note='Kui comparison to Kurukh ablative tī; connection described as apparent, no derivation link inferred.')
for n,f in enumerate(['gā','skā'],1):add(479,f'plural-{n}',f,analysis=['suffix','plural'],note='Explicit comparative plural suffixes; preserve new occurrence separately from p291.')
add(481,'five','siṅgi','five',analysis=['numeral'],note='Compared to Gondi numeral beginning with s.')
add(481,'six','sajgi','six',analysis=['numeral'],note='Compared to Gondi numeral beginning with s.')
add(481,'thou','īmu','thou',analysis=['pronoun','second-person','singular'],note='Original visibly prints m. Preserve source variant/possible typo; do not harmonize to other-page īnu.')
add(481,'he','ēañju','he',analysis=['pronoun','third-person','singular'],note='Explicitly labelled Kui, compared with Telugu vāḍu.')
add(484,'honorific','mu',analysis=['suffix','honorific'],note='Explicit honorific suffix in Telugu and Kui, compared with Tamil um. No lexical verb/pronoun POS inferred.')
add(566,'he','eañju','he',analysis=['pronoun','third-person','singular'],note='Original initial e appears unmarked; preserve difference from p481. Gloss supplied by immediate he-pronoun context, not direct word-by-word translation.',uncertainty={'type':'gloss-scope','reason':'Context is comparison of he-pronouns; immediate clause does not repeat English gloss after Kui form.'})
for slug,f,g,a in [('i','ānu','I',['first-person','singular']),('we','āmu','we',['first-person','plural']),('thou','īnu','thou',['second-person','singular'])]:
 add(570,slug,f,g,analysis=['pronoun']+a,note='All three explicitly assigned to Kui; following same forms usual in Kolami is a separate source comparison, not geographic Kui attribution.')
add(623,'from','ṭai','from',analysis=['suffix','ablative'],note='Source compares several forms for from; instrumental origin/connection remains qualified by perhaps.')
add(623,'goat','oḍā','a goat',analysis=['noun'],note='Source comparison with Tamil, Gondi and Brahui; no cognacy edge inferred.')
for n,f in enumerate(['i','ni'],1):add(623,f'genitive-{n}',f,analysis=['suffix','genitive'],note='Both literal forms explicitly assigned to Kui in genitive comparison; source etc. retained in context, no additional invented forms.')
add(623,'locative','ta',analysis=['suffix','locative'],note='Kui has ta in attributed Dr.Trumpp locative comparison; no inferred length or underdot.')
add(624,'that-base','ē','that',analysis=['demonstrative-base'],note='Explicitly called base ē in Kui; no automatic affix POS or full pronoun expansion.')
for slug,f,g in [('beat','pāgiĩ','I beat'),('do','giĩ','I do')]:
 add(624,slug,f,g,analysis=['verb','first-person','singular','present'],note='Two additional source forms at very bottom of page, omitted from Nordic mention sentences. Clear alternate IA leaf645 confirms letters and final curved upper mark.',uncertainty={'type':'source-mark-encoding','reason':'Final curved upper i-mark encoded provisionally as tilde consistently with table. No extra length or phonemic nasal inference.'})
assert len(rows)==34
controls=[
 {'page':210,'context':'Juang ñje and examples konā-ñje, kanu-niñje, būi-ñje precede comparison to Kui añju. Possible first-person suffix interpretation remains attributed and tentative.','attribution_policy':'Juang controls; ambiguous abhañju separately retained with uncertainty.'},
 {'page':290,'context':'Caldwell feminine comparison: Telugu kōḍalu daughter-in-law and āḍu female; Kurukh āli woman; feminine suffix aḷ.','attribution_policy':'Do not assign these to Kui or convert scholarly comparison into derivation.'},
 {'page':291,'context':'Plural suffix list assigns Tamil gaḷ with colloquial variants; Kanarese gaḷu and colloquial variants; Gondi k, ṅg, sk; Telugu lu; Brahui āk,k. Oblique comparison includes Telugu tammuḍu, hypothetical *tammunu, tammuni.','attribution_policy':'Source-specific control-language labels and hypothetical asterisk retained; not target Kui rows.'},
 {'page':293,'context':'Inline Tamil nīr/nīṅgaḷ, Tulu īr, Telugu mīru/oldīru. Two-row table also has Tamil, Malayalam, Kurukh and Telugu columns (eight control cells), all separate from Kui.','attribution_policy':'Control table cells accounted as comparison evidence; do not infer Kui forms from their placement or similarity. Original physical313 table is authority for fine control typography.'},
 {'page':295,'context':'Tamil avaṉ aḍeigiṟāṉ he gets; Kanarese avanu māḍidanu he did; Gondi ōr kītōr he did; Telugu vāḍu koṭṭināḍu he struck.','attribution_policy':'Whole comparison clauses, not Kui variants or donor claims; fine non-target typography remains documentary.'},
 {'page':296,'context':'Kanarese māḍ-uttā-ne, Kurukh es-d-an, Gondi nannā kīā-t-ōn-ā, Telugu nēnu koṭṭu-t-unnānu / nēnu koṭṭu-d-unu; comparative past/conjunctive suffixes in Tamil, Kanarese, Kurukh, Gondi, Telugu and Brahui.','attribution_policy':'Controls remain explicitly attributed. Hypothetical common ku-suffix origin is author conjecture, not confirmed etymology.'},
 {'page':413,'context':'Kurukh ablative tī compared with Tamil iṉṟu/iṇḍe and Kanarese inda; Kui ṭai comparison labelled apparently connected.','attribution_policy':'No inferred graph edge or transfer of Kurukh examples into Kui.'},
 {'page':479,'context':'Gondi plural k/ṅg with examples kāl-k feet, maṭṭā-ṅg mountains; Korava ṅga.','attribution_policy':'Historical source calls Korava a Tamil dialect; retain attribution without overriding modern registry.'},
 {'page':481,'context':'Gondi saiyūṅg five/sārūṅg six, im(m)-pronouns; Telugu vāḍu; additional numeral cognate comparisons in Kanarese, Telugu and Tamil.','attribution_policy':'Do not conflate Kōi/Gondi with Kui.'},
 {'page':484,'context':'Tamil honorific um, Telugu mu, nearby Gondi imperative forms.','attribution_policy':'Kui has only explicit comparative suffix mu in this span.'},
 {'page':566,'context':'Bhili of Basim avan/avand he and oblique forms; Kolami and derived from avnd; Telugu vāḍu.','attribution_policy':'Retain historical Bhili/Kolami context and hypothetical analysis separately; target Kui comparison is eañju only.'},
 {'page':570,'context':'Naiki pronoun table and following statement that same forms are usual in Kolami.','attribution_policy':'Kui three quoted forms are labelled explicitly; do not copy the Naiki table or create unprinted new Kolami alternatives.'},
 {'page':623,'context':'From-series: Tamil iṉṟu, Korava iṇḍe/uṇḍ, Kurukh tī, Kōi aggaḍa. Goat-series: Tamil āḍ, Gondi yēṭī, Brahui hēṭ. Genitive-series: Burgandi e/ne, Naiki/Kolami nē, Gondi ā. Locative comparison: Brahui ṭī and Tamil iḍei place.','attribution_policy':'All controls keep printed labels; Dr.Trumpp attribution and perhaps/apparently qualifications remain documentary.'},
 {'page':624,'context':'Brahui ō/ē demonstratives and Tamil avaṉ; later Brahui khaniv/khanev and Malto bandin/daryin are compared with the two final Kui verbs.','attribution_policy':'No transfer of Brahui conjunctive/subjunctive meaning onto Kui merely from comparison; retain printed Kui English I beat/I do.'}
]
report={'status':'all14 candidate original pages reviewed;34 target/contextual units proposed, including four target units absent from sentence-only snapshot; independent lexical acceptance pending',
 'reviewer':'norton_finish','source':{'path':'tmp/pdfs/LSI-V4.djvu','sha256':hashlib.sha256((ROOT/'tmp/pdfs/LSI-V4.djvu').read_bytes()).hexdigest()},
 'page_coverage':[{'printed':n,'djvu_physical':229 if n==210 else n+20,'original_full_page_read':True} for n in [210,290,291,293,295,296,413,479,481,484,566,570,623,624]],
 'literal_review_limits':'Small compressed marks specifically typed, not normalized from main chapter. Comparative controls preserve identity/scope but are not newly installed lexical inventories.',
 'units':rows,'comparative_controls':controls,
 'additional_discoveries':['p293 two Kui table cells āmu/āju beyond inlineīru','p624 final pāgiĩ/giĩ beyond demonstrativeē'],
 'bibliography':'p296 explicitly says Caldwell comparative grammar was consulted throughout inquiry; record this secondary analysis provenance. Dr.Trumpp comparisons on623 require source attribution; no claim of independent consultation.',
 'canonical_changes':False}
(HERE/'outsidechapter-original-reviewed.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
(HERE/'outsidechapter-original-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(len(rows),'target/contextual units;',len(controls),'comparison-control groups')

"""Account for every original page, keeping raw evidence and scope distinct."""
import json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
def loadjl(name):return [json.loads(s) for s in (HERE/name).read_text().splitlines()]
def decision(p):
 if p<=0:return 'front-matter','Title, publication/permission statement, abstract, contents and lists; no target lexical example. Bibliographic evidence retained.'
 if p<=3:return 'background','Survey introduction, population and geography; conventional nomenclature documented, no additional phonetic lexical list.'
 if p in {4,5}:return 'ethnographic-prose','Explicit headman/cultural-name groups inventoried separately; generic English clothing prose and Hindu festival controls accounted without target-language inference.'
 if p<=8:return 'language-background','Classification, vitality, orthography and literature survey; script name recorded as documentary evidence. Bibliographic titles are not new lexical attestations.'
 if p<=14:return 'methods','Survey questions, sample selection and fieldwork methodology; English descriptions are not reconstructed Kurux responses.'
 if p in {15,16}:return 'nomenclature','All explicit phonetic/native autonyms and conventional/outsider spellings inventoried separately; no unprinted pairing among alternative lists.'
 if p==17:return 'nomenclature-context','Repeated conventional names and translated opinions on naming. Source-language quotations are absent; recorded names are covered in pp15–16 inventory.'
 if p<=34:return 'sociolinguistic-results','Attitudes, intelligibility, orthography, vitality, conclusions and translated respondent opinions. English quotations have no printed Kurux originals and are not back-translated.'
 if p<=36:return 'nepali-control','Nepali translation of report conclusions, explicitly credited to Man Jang Magarati; not a Kurux text.'
 if p==37:return 'abbreviation-key','Administrative and language-name abbreviation table; Kux/Urw are codes, not target phonetic headwords.'
 if p==38:return 'ipa-chart','General IPA chart, credited to International Phonetic Association2018; metalinguistic inventory, not Kurux lexical examples.'
 if p in {39,40}:return 'comparison-method','Similarity thresholds and illustrative phonetic segments inventoried as documentary notation. No genuine cognate analysis is claimed; no graph edges inferred.'
 if p in {41,42}:return 'morphology','All66 physical morphology/inline/qualification units retained, including seven zero witnesses. Transcription and consultant qualifications recorded.'
 if p==43:return 'consent-control','Wordlist informed-consent script explicitly supplied in English and Nepali, not Kurux. No target translation is inferred.'
 if p<=56:return 'wordlist','Every physical cell in all six Kurux columns reviewed; English/Nepali prompts are controls. Blanks remain audit records.'
 if p<=59:return 'rtt-procedure-control','Recorded-text-test instructions and trial material in English/Nepali; no Kurux transcript to reconstruct.'
 if p==60:return 'narrative-provenance','Two recorded-story provenance descriptions; Jharkhand transcript follows. West Bengal transcript is not printed, explicitly accounted as unavailable in this publication.'
 if p==61:return 'narrative-preface','Hometown/recorded-test labels and abbreviation key: Ku, Eng, Free, EMPH, PREP, POSTP, POSS, PT, Q, A. Lexical aligned text begins62; grammar-label expansions follow this key.'
 if p<=67:return 'native-narrative','All35 complete segments and377 aligned physical groups reviewed; explicit font mapping and source inconsistencies retained.'
 if p<=73:return 'rtt-results','Comprehension scoring and questionnaire responses; no additional target transcript. English translations cannot supply missing Kurux originals.'
 if p==74:return 'orthography-procedure','Orthography test protocol; passage and translation are on75.'
 if p==75:return 'native-quotation','All11 native image lines and full English translation retained as one quoted paragraph; no word-level alignment invented.'
 if p<=77:return 'orthography-results','Reading-test results and assessment; no additional target passage.'
 if p<=80:return 'questionnaire-control','Bilingual English/Nepali biodata/interview questionnaire and administrative definitions; not target lexical responses.'
 if p<=83:return 'biodata-nomenclature','Respondent demographic, caste, mother-tongue and religion fields; exact quoted occupation/religion labels retained in documentary inventory. Site alone does not establish lexical language.'
 if p<=105:return 'interview-results','Respondent tables on language use, proficiency, family and attitudes. Nomenclature codes are documentary labels; no target sentence originals supplied.'
 if p<=108:return 'insider-questionnaire-control','English/Nepali insider questionnaire; no translation into Kurux inferred.'
 if p<=112:return 'insider-results','English report of insider interview answers, names and institutions. No additional aligned/translated Kurux lexical examples.'
 if p<=122:return 'mapping-material','Dialect-mapping protocol, maps and geographic labels. Place labels are documentary geographic evidence, not inferred Kurux vocabulary.'
 if p<=125:return 'vitality-framework','Sustainable Use Model/EGIDS framework and analysis categories; not target language data.'
 if p==126:return 'geographic-map','Kurukh/Uranw-speaking-area map; documentary geography only.'
 return 'bibliography','Auxiliary identification and source roles retained; cited titles are not independent lexical attestations.'
def main():
 census=json.loads((HERE/'native-page-census.json').read_text())
 units=[]
 for name in ['wordlist-reviewed.jsonl','morphology-reviewed.jsonl','narrative-first-reading.jsonl','prose-reviewed.jsonl']:units+=loadjl(name)
 units.append(json.loads((HERE/'orthography-p75-reviewed.json').read_text()))
 pages=[]
 for raw in census['page_inventory']:
  p=raw['pdf_page']-8;category,note=decision(p)
  keys=[r.get('source_unit_key',r.get('key')) for r in units if r.get('printed_page')==p]
  pages.append(dict(pdf_page=raw['pdf_page'],printed_page=p if p>0 else None,category=category,decision=note,evidence_unit_keys=keys,native_text_sha256=hashlib.sha256(raw['text'].encode()).hexdigest(),source_image_count=raw['image_count']))
 assert len(pages)==137 and {r['pdf_page'] for r in pages}==set(range(1,138))
 report=dict(status='whole-document scope decisions; final prose review/output audit still pending',source_sha256=census['source_sha256'],method='Original lexical pages and raw native-text census jointly inspected. Scope decisions preserve every page and explicitly distinguish non-target questionnaires, translated-only opinions, chart notation, biodata labels and lexical-bearing material. This is not a claim that every administrative cell received independent visual transcription.',pages=pages)
 (HERE/'whole-page-decisions.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(len(pages),'pages accounted')
if __name__=='__main__':main()

"""Original-reviewed page/paragraph/table-cell/line census, before form parsing.

One census unit is a physical paragraph, table prompt cell, or interlinear line.
Inline forms and alternatives become child witnesses of these units; these unit
counts must never be described as final lexical/token counts.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'data/source_meta.py').exists())


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    rows=[]
    def unit(page,section,index,kind,scope,note=''):
        directory='kolami-discovery-originals' if page in (561,564,568,569) else 'basim-identity-originals'
        path=ROOT/f'tmp/jambu-next-source/{directory}/printed-{page}.png'
        rows.append(dict(source_unit_key=f'grierson1906kolamibasim:p{page}:{section}:{index}',
            printed_page=page,physical_page_onebased=page+20,section=section,physical_index=index,
            unit_type=kind,source_scope=scope,source_image=str(path.relative_to(ROOT)),source_image_sha256=sha(path),
            inventory_note=note,accepted_transcription=False,disposition='original first-reading required'))
    for section in ['shared-introduction','kolami-distribution','pusad-population','community-classification','linguistic-classification']:
        unit(561,section,1,'documentary-paragraph','shared' if section=='shared-introduction' else 'Kolami')
    for index in range(1,5):unit(561,'population-table',index,'documentary-table-cell','Kolami','Three district estimates plus total; no lexical head.')
    for index in range(1,3):unit(562,'authorities',index,'bibliography-record','reported auxiliary authority')
    for section,kind,note in [
        ('source-basis','documentary-paragraph','Grierson supplies printed interlinear English; two other versions and standard list are mentioned, not printed here.'),
        ('principal-features','documentary-paragraph','Introductory statement.'),
        ('noun-plural','lexical-prose-paragraph','Every printed suffix and target/comparison example must be a child witness; tentative ur/r analysis retained.'),
        ('gender','documentary-paragraph','No feminine noun examples; Naiki comparison and probable Kolami gender claim, not invented forms.'),
        ('accusative-dative','lexical-prose-paragraph','Two suffixes plus three examples; shared case claim retained.'),
        ('genitive-locative','lexical-prose-paragraph','All four quoted endings, whole examples and explicitly printed field base retained separately.'),
        ('numerals','lexical-prose-paragraph','All target numeral alternatives plus five comparison witnesses; no reconstructed numbers.'),
        ('pronoun-number-caveat','documentary-paragraph','Plural apparently seldom used; singular forms substituted.'),
        ('other-pronouns','lexical-prose-paragraph','All demonstrative/genitive/interrogative forms with printed scope.'),
        ('verb-number-caveat','documentary-paragraph','Plural used singular and vice versa; source makes complete mapping difficult.'),
        ('present-copula','lexical-prose-paragraph','Four printed functional groups including optional notation; no unprinted plural reconstruction.')]:
        unit(562,section,1,kind,'Kolami',note)
    for index in range(1,13):unit(562,'personal-pronouns',index,'paradigm-cell','Kolami','Three columns, four/three/five gloss groups; alternatives remain separate child witnesses.')
    for index in range(1,7):unit(562,'past-copula',index,'paradigm-cell','Kolami','Two number columns, three explicitly numbered persons each.')
    for section,note in [
        ('finite-verbs','Past d/t; future d tentative; finite examples and neuter comparison whole clause retained.'),
        ('imperative','Printed plural suffix r and all imperative examples.'),
        ('negative','Printed tōten/e morphology and every negative example; no invented bases.'),
        ('verbal-nouns','Whole printed forms and quoted ād ending; tentative analyses preserved.'),
        ('conjunctive-participle','Printed nā and all three examples.'),
        ('further-details','Documentary reference to following specimen.')]:
        unit(563,section,1,'documentary-paragraph' if section=='further-details' else 'lexical-prose-paragraph','Kolami',note)
    for page,count,scope,specimen in [(564,17,'Kolami / District Wun',75),(565,18,'Kolami / District Wun',75),(568,17,'Basim / so-called Bhili',76),(569,12,'Basim / so-called Bhili',76)]:
        if page in (564,568):unit(page,'specimen-heading',specimen,'documentary-heading',scope,'Family, source label, specimen number and district are provenance.')
        for index in range(1,count+1):unit(page,f'specimen{specimen}-line',index,'physical-interlinear-line',scope,
            'Preserve every target atom and aligned English, punctuation/quotes; source-bounded expression spans may cross physical lines/pages. No separate free translation printed.')
    unit(565,'basim-introduction',1,'documentary-paragraph','Basim','Begins below specimen75 and continues onto566; do not assign preceding specimen to Basim.')
    for section,kind,note in [
        ('basim-introduction-continuation','documentary-paragraph','Continuation from565; Bhil counts include speech not necessarily this lect.'),
        ('pronunciation-a-u','lexical-prose-paragraph','Source letter contrast plus two examples; no global conversion inferred.'),
        ('pronunciation-l-n','lexical-prose-paragraph','Quoted letter substitution, target/comparison forms, whole said-to-father phrase.'),
        ('pronunciation-n-nd-d','lexical-prose-paragraph','All variants, tentative Kolami amd/avnd derivation and explicit Kui/Telugu comparisons.'),
        ('nouns','lexical-prose-paragraph','Two plural and two dative suffixes, four examples; source omission caveat retained.'),
        ('numerals','lexical-prose-paragraph','Two numbers plus printed two-daughters expression and gender hypothesis.'),
        ('possessive-agreement','lexical-prose-paragraph','Quoted possessives, demonstrative and whole thy-son example; source analysis scoped.'),
        ('other-pronouns','lexical-prose-paragraph','Five explicitly glossed forms.'),
        ('verbs-introduction','documentary-paragraph','Conjugation introduction.'),
        ('present-copula-comparison','lexical-prose-paragraph','Printed additional neuter form; do not synthesize other equivalent Kolami cells.'),
        ('past-copula','lexical-prose-paragraph','Two explicitly glossed forms and tentative was/dead/agency analyses all retained.')]:
        unit(566,section,1,kind,'Basim',note)
    for index in range(1,16):unit(566,'personal-pronouns',index,'paradigm-cell','Basim','Three five-cell columns; optional (d)/(n) literal notation retained, no unprinted expansions.')
    for index in range(1,8):unit(566,'present-copula',index,'paradigm-cell','Basim','Four singular/three plural cells. Third singular printed label apparently 2.m.; retain typed source-label conflict pending sharp review.')
    for section,kind,note in [
        ('finite-verbs','lexical-prose-paragraph','Explicit s/t endings and beat verb base printed before table.'),
        ('present-suffix-comparison','lexical-prose-paragraph','Source s ending and two Telugu suffix comparands.'),
        ('negative','lexical-prose-paragraph','Printed tōdī and all negative examples with infinitive analysis.'),
        ('conjunctive-participle','lexical-prose-paragraph','Quoted ūn/an endings, all three examples.'),
        ('summary','documentary-paragraph','Refers to following parable; no lexical form invented.')]:unit(567,section,1,kind,'Basim',note)
    for index in range(1,19):unit(567,'beat-paradigm',index,'paradigm-cell','Basim','Three TAM columns × two number columns × three numbered persons; literal labels retained.')
    rows.sort(key=lambda r:(r['printed_page'],r['section'],r['physical_index']))
    assert len(rows)==len({r['source_unit_key'] for r in rows})==169
    (HERE/'physical-unit-census.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(status='Complete nine-page physical structural census; no accepted lexical transcription',
        pages=list(range(561,570)),structural_units=len(rows),counts_by_page=dict(sorted(Counter(r['printed_page'] for r in rows).items())),
        counts_by_type=dict(Counter(r['unit_type'] for r in rows)),interlinear_lines=64,
        specimen75_lines=35,specimen76_lines=29,source_original='tmp/pdfs/LSI-V4.djvu',
        source_original_sha256=sha(ROOT/'tmp/pdfs/LSI-V4.djvu'),canonical_changes=False,
        count_definition='Paragraphs, table cells, and aligned physical lines; not final source-form/token/emission counts. Every printed inline morphology and alternative must become an explicit child witness during complete first reading.',
        root_independent_line_counts='Root independently confirmed all64 physical lines.',
        pending=['Complete literal child-witness census/first reading and atomic counts','Original-controlled punctuation and expression spans','Independent glyph/metadata/graph/output review','Asur outside-chapter closure','Sound/reference/lect metadata and actual-parser validation','Guarded approved source-stage installation; full build/browser gates deferred'])
    (HERE/'physical-census-status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()

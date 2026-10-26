"""Original-page adjudication of all 25 geometric warnings (20 expressions).

Preserve printed segmented strings, including typographic spaces. These records
are evidence, not automatically lemmatized or installed lexical output. The
remaining candidates still require review; a clean geometry score is not approval.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Each group identifies zero-based candidate positions within the full expression.
# None leaves its literal candidate form/gloss unchanged. Merges preserve source
# spaces; missing ordinary-font items are explicitly added with native line IDs.
REPAIRS = {
 (79,33): [([0,1], 'dzeṭ ʰi', 'eldest')],
 (86,45): [([0,1], 'beṭ isəb', 'daughter-PL')],
 (89,31): [([1], None, 'run away-SEQ'), ([2], None, 'go-PST')],
 (90,74): [([0,1], 'ram- kə', 'Ram-GEN'), ([2,3], 'bʰai- k', 'brother-3SG.POSS')],
 (91,4): [([0,1,2], 'ram - kə', 'Ram-GEN')],
 (91,28): [([1,2], 'cʰaho- m', 'son-1SG.POSS')],
 (91,44): [([1,2], 'cʰahi- m', 'daughter-1SG.POSS')],
 (91,62): [([2,3,4], 'kʰəna tə u-rə', 'PART he-[OBL]GEN')],
 (93,16): [([1,2], 'cʰə̃uḍ i', 'girl')],
 (93,42): [([1], None, 'nectar'), ([2], None, 'fall-PST')],
 (95,49): [([0], None, 'then'), ([4], None, 'COP.NPST-3SG')],
 (100,27): [([2], None, 'he-[OBL]DAT'), ([3], None, 'kill-PST')],
 (102,59): [([1], 'kʰəna-jə̃', None), ([3], 'rati-jə̃', None),
             ([4], None, 'come out-PST'), ([5], None, 'tiger')],
 (104,19): [([3], None, 'he-[OBL]DAT'), ([4], None, 'rice')],
 (105,5): [([2], None, 'some'), ([6], None, 'COP.NPST-3PL')],
 (105,46): [([7], None, 'add-INF'), ([8], None, 'become-3SG-NPST')],
 (108,28): [([2,3], 'bʰeṭ -larə', 'meet- PROS'), ([4], 'ṭʰaũ-jə̃', None)],
 (110,46): [([2], None, 'one'), ([3], None, 'give-NPST-1PL')],
 (115,69): [], # Printed knowledge conflicts with the free translation's wood name; literal retained.
 (116,78): [([0], 'dzono-ĩ', None), ([3,4], 'u- ĩ', 'he-ERG'), ([5], None, 'money')],
}
ADDITIONS = {(105,5): [(20,1,'kami-se','ironsmith-ABL')],
             (110,46): [(50,1,'mana','a unit'), (50,2,'caur','rice')]}


def build():
    source = HERE / 'chapter4-interlinear-alignment-candidates.jsonl'
    candidates = [json.loads(x) for x in source.read_text().splitlines()]
    expressions = {}
    for candidate in candidates:
        expressions.setdefault(candidate['parent_expression_key'], []).append(candidate)
    result = []
    review = []
    for parent, rows in expressions.items():
        page = rows[0]['printed_page']; line = int(parent.rsplit(':l',1)[1]); pair = (page,line)
        if pair not in REPAIRS:
            for row in rows:
                result.append(dict(row, review_status='Unreviewed; geometric candidate only'))
            continue
        changes = {}
        skipped = set()
        for positions, form, gloss in REPAIRS[pair]:
            assert not skipped.intersection(positions)
            chosen = [rows[i] for i in positions]
            record = dict(chosen[0])
            record['source_candidate_keys'] = [x['source_unit_key'] for x in chosen]
            record['source_candidate_evidence'] = chosen
            record['form'] = form if form is not None else record['form']
            record['candidate_gloss'] = gloss if gloss is not None else record['candidate_gloss']
            record['adjudication'] = 'Original PDF printed expression and interlinear lines compared; literal segmentation/gloss retained.'
            changes[positions[0]] = record; skipped.update(positions[1:])
        emitted = []
        for i,row in enumerate(rows):
            if i in skipped:continue
            record = changes.get(i, dict(row))
            record['review_status'] = 'Owner original-image alignment review; independent review pending'
            record['needs_manual_alignment'] = False
            record['source_candidate_keys'] = record.get('source_candidate_keys',[row['source_unit_key']])
            if pair == (91,62) and i == 2:
                record['uncertainty'] = 'Printed PART and he-[OBL]GEN labels do not separately align all three words. Preserve the complete printed group; do not invent a separate gloss for tə.'
                record['needs_manual_alignment'] = True
            if pair == (115,69) and i == 2:
                record['source_semantic_discrepancy'] = 'Interlinear gloss is knowledge; free translation identifies Mantinne as the sought wood. Retain both source claims without choosing a correction.'
            emitted.append(record)
        for native_line,word,form,gloss in ADDITIONS.get(pair,[]):
            page_record=json.loads((HERE/'chapter4-native'/f'p{page:03}.json').read_text())
            emitted.append({'source_unit_key':parent+f':word:l{native_line}:w{word}',
                'parent_expression_key':parent,'printed_page':page,'pdf_page':page+22,
                'native_line':native_line,'form':form,'candidate_gloss':gloss,
                'raw_evidence':page_record['lines'][native_line-1],
                'source_candidate_keys':[], 'review_status':'Owner original-image alignment review; independent review pending',
                'adjudication':'Target interlinear item printed in ordinary font; font-only candidate eligibility omitted it.',
                'needs_manual_alignment':False})
        result.extend(emitted)
        review.append({'parent_expression_key':parent,'printed_page':page,
                       'old_candidates':len(rows),'reviewed_groups':len(emitted),
                       'status':'Original-page owner alignment review complete; source ambiguity preserved where present'})
    assert len(review)==20
    assert len({r['source_unit_key'] for r in result}) == len(result)
    return result, {'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'reviewed_expressions':review,'remaining_unreviewed_groups':sum(r['review_status'].startswith('Unreviewed') for r in result),
                    'canonical_modified':False,'policy':'No automatic grammatical/lexical emission. Preserve source literal analysis, whole clauses, and typed alignment ambiguity.'}

if __name__ == '__main__':
    rows,report=build()
    output=HERE/'chapter4-interlinear-adjudicated-candidates.jsonl'
    output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report['output_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    (HERE/'chapter4-interlinear-warning-review-20260927.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(len(rows),'groups;',report['remaining_unreviewed_groups'],'remain unreviewed')

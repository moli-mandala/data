"""Record review-ready paired proposals and scoped policy evidence."""
import csv,hashlib,json,sys,unicodedata
from pathlib import Path

P=Path(__file__).resolve().parent;D=P.parents[4]
sys.path.insert(0,str(D))
from profile_policy import house_output


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    packages={}
    for name in ['rampur','baghi']:
        p=P.parent/f'bailey_{name}_1920'
        progress=json.loads((p/'whole-proposal-progress-20260927.json').read_text())
        validation=json.loads((p/'whole-parser-validation-20260927.json').read_text())
        assert validation['proposal_sha256']==sha(p/'whole-proposed.csv')
        assert validation['profile_sha256']==sha(p/'whole-proposed-profile.txt')
        rules=dict(csv.reader((p/'whole-proposed-profile.txt').open(),delimiter='\t'));rules.pop('Grapheme')
        violations=[]
        for g,out in rules.items():
            want=house_output(g,out,rules,f'bailey-{name}-1920')
            if unicodedata.normalize('NFC',want)!=unicodedata.normalize('NFC',out):violations.append([g,out,want])
        assert not violations,violations
        profile_report=dict(status='passed',profile_sha256=sha(p/'whole-proposed-profile.txt'),rule_count=len(rules),house_policy_violations=violations,method='Check every proposed mapping with current profile_policy.house_output; actual parser separately verifies NFC/NFD equality for every row. Preserve old mappings; lowercase, w→v and ṅ→ŋ are shared conventions. Breves, underdots, underlines, nasal/length stacks and acute marks remain literal, without invented IPA or phonological interpretation.',canonical_profile_sha256=sha(D/f'conversion/bailey-{name}-1920.txt'))
        (p/'whole-profile-validation-20260927.json').write_text(json.dumps(profile_report,ensure_ascii=False,indent=2)+'\n')
        packages[name]={k:progress[k] for k in ['rows','legacy_rows_preserved','added_rows','recovered_glossary_holds','generic_kochi_rows','audit_units','uncertain_rows','variant_rows']}
        packages[name]['files']={n:sha(p/n) for n in ['whole-proposed.csv','whole-proposed-audit.jsonl','whole-proposed-profile.txt','whole-parser-validation-20260927.json','whole-profile-validation-20260927.json']}
    inputs=['whole-paired-grammar-first-reading-20260927.tsv','whole-paired-sentences-first-reading-20260927.tsv','whole-paired-commentary-first-reading-20260927.jsonl','whole-paired-introduction-first-reading-20260927.jsonl','whole-paired-second-reading-20260927.json','whole-paired-grammar-reviewed-20260927.jsonl','whole-paired-sentences-reviewed-20260927.jsonl','whole-paired-commentary-reviewed-20260927.jsonl','whole-paired-introduction-reviewed-20260927.jsonl','whole-paired-expansion-draft-20260927.jsonl','whole-glossary-hold-recovery-20260927.json','prepare_paired_reviewed.py','prepare_paired_expansion.py','prepare_paired_proposals.py']
    report=dict(status='frozen for independent actual-output source audit; not installed',source_pdf='tmp/pdfs/bailey1920/bailey1920.pdf',source_pdf_sha256='7a1577a2d2a24eca8b95e9b1ed700ef01f716d1069e4468f36a673611d093b39',page_mapping='Printed n = one-based PDF n+26 = zero-based PDF n+25',source_scope='Shared113–115; heading-scoped116–117; paired grammar131–140;44complete responses141–142; commentary142–143; complete glossary144–147; boundary controls115–118,130,148',source_groups=dict(frozen_supplementary_groups=393,additional_lexical_headings=2,reviewed_supplementary_groups=395,grammar=305,sentences=44,commentary=19,introduction=27,context_or_control_groups=23),packages=packages,input_sha256={n:sha(P/n) for n in inputs},source_decisions=['GenericKochi4examples emitted once under kc in Rampuri carrier, with explicit attribution scope and blank glosses.','Explicit suffixes and ditto replacements expanded only from printed stems/context; raw physical cells retained.','Same-as paradigm statements have exact candidate-key reuse accounting; no arbitrary unprinted paradigms generated.','Baghi water-word alternate emits2complete responses; all44physical responses retained.','Future3pl unpunctuated ōā auī split into2tentative alternatives with root-supported segmentation uncertainty and no confident variant edge.','Formerly held glossary body jĕā́ recovered after root original-image confirmation; historical hold retained and interpretation uncertainty explicit.'],validation=dict(focused_test_file='tests/test_bailey_rampur_baghi_whole_recovery.py',focused_tests='6 passed',parser_rows=1345,normalizations=['NFC','NFD'],profile_policy='zero violations',legacy_rows_exact=584),remaining=['independent actual-output source audit and adjudication','guarded canonical importer/metadata/reference/checklist installation and canonical parser checks','fullCLDF/database/build/browser gates explicitly deferred; no remote work authorized'])
    delta_name='whole-legacy-corrections-20260927-pass2.json'
    report['input_sha256'][delta_name]=sha(P/delta_name)
    report['version']='pass2 after independently identified legacy mark and metadata corrections'
    report['previous_freeze_sha256']=json.loads((P/delta_name).read_text())['previous_freeze_sha256']
    report['validation'].update(focused_tests='7 passed',legacy_rows_exact=573,legacy_keys_preserved=584,explicit_legacy_corrections=11)
    report['source_decisions'].append('Preserve all584 legacy keys; apply only11 exact independently documented legacy deltas: one literal quantity and ten grammatical gloss/tag repairs.')
    for name,info in report['packages'].items():
        info['legacy_keys_preserved']=info.pop('legacy_rows_preserved')
        info['legacy_rows_exact']=280 if name=='rampur' else 293
    encoded=json.dumps(report,ensure_ascii=False,indent=2)+'\n'
    (P/'whole-paired-output-freeze-20260927-pass2.json').write_text(encoded)
    (P/'whole-paired-output-freeze-20260927.json').write_text(encoded)
    print(json.dumps(packages,ensure_ascii=False))


if __name__=='__main__':main()

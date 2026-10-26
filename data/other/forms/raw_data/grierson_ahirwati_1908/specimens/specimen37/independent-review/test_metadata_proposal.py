import unittest,json,csv,collections,unicodedata
from pathlib import Path
P=Path(__file__).resolve().parent;S=P.parent;O=P/'metadata-proposal'
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
class CompleteProposal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (O/'candidate.csv').open() as f:cls.csv=list(csv.reader(f))
        cls.by={r[10]:r for r in cls.csv}
        cls.audit=rows(O/'candidate-audit.jsonl');cls.aliases=rows(O/'physical-key-aliases.jsonl')
    def test_complete_unique_keys_and_all_physical_witnesses(self):
        self.assertEqual((len(self.csv),len(self.by)),(389,389))
        expected=rows(S/'roman-atoms-first-reading.jsonl')+rows(S/'native-atoms-first-reading.jsonl')
        self.assertEqual(collections.Counter(a['physical_key'] for a in self.aliases),collections.Counter(r['key'] for r in expected))
        self.assertTrue(all(a['emission_key'] in self.by for a in self.aliases))
    def test_only_explicit_printed_object_label(self):
        for r in self.csv:
            if 'obj' in r[14].split():self.assertIn('(obj.)',r[3])
        r=self.by['lsi9.2:p235:aligned-line:12:word:1']
        self.assertIn('obj',r[14].split())
    def test_source_infinitive_same_form_does_not_erase_passive_function(self):
        r=self.by['lsi9.2:p236:aligned-line:3:word:9']
        self.assertTrue({'verb','participle','pass','fut'}<=set(r[14].split()))
        self.assertIn('identical in form to the infinitive',r[6])
    def test_english_ing_does_not_force_participle(self):
        r=self.by['lsi9.2:p234:aligned-line:13:word:9']
        self.assertEqual(r[3],'dying')
        self.assertNotIn('participle',r[14].split())
    def test_native_joint_spans_conserve_spacing_and_continuations(self):
        a=self.by['lsi9.2:p234:aligned-line:5:word:4']
        self.assertEqual((a[1],a[2]),('जमा कर-के','jamā-kar-kē'))
        a=self.by['lsi9.2:p234:aligned-line:12:word:5']
        self.assertEqual(a[1],'संभार-के')
        a=self.by['lsi9.2:p235:aligned-line:6:word:10']
        self.assertEqual((a[1],a[2]),('अच्छातैं अच्छा','‘achchhā-taĩ-achchhā'))
    def test_whole_keys_are_first_physical_anchor_and_analysis_not_paraphrase(self):
        wholes=[a for a in self.audit if a['kind']=='whole-source-expression']
        self.assertEqual(len(wholes),27)
        old={r['key']:r for r in rows(S/'whole-utterances-first-reading.jsonl')}
        for a in wholes:
            self.assertEqual(a['source_unit_key'],a['roman_physical_keys'][0]+':expression')
            self.assertEqual(a['row'][3],old[a['old_candidate_key']]['printed_interlinear_analysis'])
    def test_literal_form_original_and_blank_phonemic_no_graph_invention(self):
        for r in self.csv:
            self.assertEqual(len(r),15);self.assertEqual(r[2],r[4]);self.assertEqual(r[5],'')
            self.assertEqual(r[11:14],['','',''])
            self.assertEqual(unicodedata.normalize('NFC',unicodedata.normalize('NFD',r[2])),unicodedata.normalize('NFC',r[2]))
if __name__=='__main__':unittest.main()

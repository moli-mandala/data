import copy
import unittest

from atomic_conservation_verifier import validate_atomic_accounting
from revised_boundary_verifier import JOIN_FIRST, JOIN_TAIL


class AtomicGuards(unittest.TestCase):
    def setUp(self):
        self.raw = [
            {'source_unit_key': 'before', 'raw_source_form': 'x', 'raw_interlinear_gloss': 'same'},
            {'source_unit_key': JOIN_FIRST, 'raw_source_form': 'Bāp-', 'raw_interlinear_gloss': 'Father-'},
            {'source_unit_key': JOIN_TAIL, 'raw_source_form': 'phōkān', 'raw_interlinear_gloss': 'near'},
            {'source_unit_key': 'after', 'raw_source_form': 'x', 'raw_interlinear_gloss': 'same'},
        ]
        self.reviewed = copy.deepcopy(self.raw)
        self.output = [
            {**self.raw[0], 'physical_source_keys': ['before'], 'raw_source_record': self.raw[0]},
            {'source_unit_key': JOIN_FIRST, 'physical_source_keys': [JOIN_FIRST, JOIN_TAIL], 'raw_source_form': 'Bāp-phōkān', 'raw_interlinear_gloss': 'Father-near', 'raw_source_record': self.raw[1], 'source_parts': self.reviewed[1:3]},
            {**self.raw[3], 'physical_source_keys': ['after'], 'raw_source_record': self.raw[3]},
        ]
        self.audit = [{'source_unit_key': a['source_unit_key'], 'raw_source_record': a, 'proposed_entry_keys': [JOIN_FIRST if a['source_unit_key'] == JOIN_TAIL else a['source_unit_key']]} for a in self.raw]

    def check(self):
        return validate_atomic_accounting(self.raw, self.reviewed, self.output, self.audit)

    def test_printed_join_and_repeated_occurrences(self):
        self.assertEqual(self.check()['repeated_literal_groups_retained_as_occurrences'], 1)
        self.assertEqual(self.check()['physical_atoms'], 4)

    def test_repeated_token_collapse_rejected(self):
        self.output.pop()
        with self.assertRaises(AssertionError): self.check()

    def test_literal_normalization_rejected(self):
        self.output[0]['raw_source_form'] = 'normalized'
        with self.assertRaises(AssertionError): self.check()

    def test_wrong_alias_target_rejected(self):
        self.audit[2]['proposed_entry_keys'] = ['before']
        with self.assertRaises(AssertionError): self.check()

    def test_historical_raw_overwrite_rejected(self):
        self.audit = copy.deepcopy(self.audit)
        self.audit[0]['raw_source_record']['raw_interlinear_gloss'] = 'emended'
        with self.assertRaises(AssertionError): self.check()


if __name__ == '__main__':
    unittest.main()

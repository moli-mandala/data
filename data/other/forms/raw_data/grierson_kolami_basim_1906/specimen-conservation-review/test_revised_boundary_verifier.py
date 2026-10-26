"""Tiny fixtures for contributor loss, fabricated English and unstable identities."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from revised_boundary_verifier import apply_pinned_approved_corrections, validate_partition


class BoundaryGuardTests(unittest.TestCase):
    def setUp(self):
        self.atoms = [{'source_unit_key': f'grierson1906kolamibasim:p568:specimen76-line:1:word:{i}', 'raw_source_form': form, 'raw_interlinear_gloss': f'gloss{i}', 'printed_page': 568} for i, form in enumerate(['a', 'b.', 'c.', 'd', 'e', 'f.'], 1)]
        self.old = [{'key': self.atoms[start]['source_unit_key'] + ':expression', 'physical_keys': [a['source_unit_key'] for a in self.atoms[start:end]]} for start, end in [(0, 1), (1, 4), (4, 6)]]
        self.expressions = []
        for seq, (start, end) in enumerate([(0, 2), (2, 3), (3, 6)], 1):
            source = self.atoms[start:end]
            keys = [a['source_unit_key'] for a in source]
            self.expressions.append({'source_unit_key': keys[0] + ':expression', 'specimen': 76, 'display_sequence': seq, 'physical_source_keys': keys, 'first_source_key': keys[0], 'last_source_key': keys[-1], 'printed_pages': [568], 'raw_physical_atoms': source, 'raw_source_form': ' '.join(a['raw_source_form'] for a in source), 'raw_aligned_interlinear_gloss': ' '.join(a['raw_interlinear_gloss'] for a in source)})

    def test_merge_and_split_histories(self):
        r = validate_partition(self.atoms, self.expressions, self.old)
        self.assertEqual(r['counts']['current_expressions'], 3)
        self.assertTrue(any(x['relation'] == 'merge' for x in r['current_to_historical']))
        self.assertTrue(any(x['relation'] == 'split' for x in r['historical_to_current']))

    def test_ordinal_identity_rejected(self):
        self.expressions[0]['source_unit_key'] = 'expression:1'
        with self.assertRaises(AssertionError): validate_partition(self.atoms, self.expressions, self.old)

    def test_optional_display_sequence_is_not_identity(self):
        for expression in self.expressions:
            expression.pop('display_sequence')
        self.assertEqual(validate_partition(self.atoms, self.expressions, self.old)['counts']['current_expressions'], 3)

    def test_missing_contributor_rejected(self):
        self.expressions[0]['physical_source_keys'].pop()
        with self.assertRaises(AssertionError): validate_partition(self.atoms, self.expressions, self.old)

    def test_invented_english_rejected(self):
        self.expressions[0]['raw_aligned_interlinear_gloss'] = 'A fluent paraphrase.'
        with self.assertRaises(AssertionError): validate_partition(self.atoms, self.expressions, self.old)

    def test_raw_atom_mutation_rejected(self):
        self.expressions = copy.deepcopy(self.expressions)
        self.expressions[0]['raw_physical_atoms'][0]['raw_source_form'] = 'normalized'
        with self.assertRaises(AssertionError): validate_partition(self.atoms, self.expressions, self.old)

    def test_boundary_mismatch_rejected(self):
        self.expressions[0]['last_source_key'] = self.atoms[0]['source_unit_key']
        with self.assertRaises(AssertionError): validate_partition(self.atoms, self.expressions, self.old)

    def correction_fixture(self, approved, bad_hash=False):
        correction = {'source_unit_key': self.atoms[0]['source_unit_key'], 'field': 'raw_source_form', 'old': 'a', 'new': 'a,'}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            f = Path(directory) / 'fixture-ledger.json'
            f.write_text(json.dumps({'corrections': [correction]}))
            pin = '0' * 64 if bad_hash else hashlib.sha256(f.read_bytes()).hexdigest()
            return apply_pinned_approved_corrections(self.atoms, f, pin, [correction] if approved else [])

    def test_exact_independent_correction(self):
        result = self.correction_fixture(True)
        self.assertEqual(result[0]['raw_source_form'], 'a,')
        self.assertEqual(self.atoms[0]['raw_source_form'], 'a')

    def test_unapproved_correction_rejected(self):
        with self.assertRaises(AssertionError): self.correction_fixture(False)

    def test_changed_correction_ledger_rejected(self):
        with self.assertRaises(AssertionError): self.correction_fixture(True, bad_hash=True)

    def test_list_ledger_retains_approved_change_history(self):
        correction = {'source_unit_key': self.atoms[0]['source_unit_key'], 'field': 'raw_source_form', 'old': 'a', 'new': 'a,'}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            f = Path(directory) / 'fixture-ledger.json'
            f.write_text(json.dumps([correction]))
            pin = hashlib.sha256(f.read_bytes()).hexdigest()
            result = apply_pinned_approved_corrections(self.atoms, f, pin, [correction], metadata_field='native_review_changes')
            self.assertEqual(result[0]['native_review_changes'], [correction])
            self.assertEqual(result[0]['raw_source_form'], 'a,')


if __name__ == '__main__': unittest.main()

"""Protect sampling weights and graph boundaries in lexical contrast mining."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_concept_isoglosses import contrast, distribution, resolve, supported_score, entry_family_map


class IsoglossTests(unittest.TestCase):
    def test_entry_grouping_requires_an_explicit_internal_edge(self):
        forms = {k: {} for k in ['10', '10-2', '10-3', '11']}
        edge = lambda c, p, kind: dict(Child_ID=c, Parent_ID=p, Kind=kind, Rank='1')
        groups = entry_family_map(forms, [edge('10-2','10','derived'), edge('11','10','derived')])
        self.assertEqual(groups['10-2'], '10')
        self.assertEqual(groups['10-3'], '10-3')
        self.assertEqual(groups['11'], '11')
        # Multiple headwords in the same family must not multiply a language's vote.
        distribution_after, n = distribution({'a': {groups[x] for x in ['10','10-2']}, 'b': {'11'}})
        self.assertEqual(distribution_after, {'10': .5, '11': .5})

    def test_unknowns_do_not_reduce_score_but_known_competitors_do(self):
        original = {str(i): {'a'} for i in range(5)}
        with_unknowns = {**original, **{f'u{i}': set() for i in range(100)}}
        p = {'b': 1}
        def score(units):
            d, n = distribution(units)
            separation, a, b = contrast(d, p)
            return supported_score(separation, d[a] * n, p[b] * 5)[0]
        self.assertEqual(score(original), 1)
        self.assertEqual(score(with_unknowns), 1)
        self.assertLess(score({**original, 'crossover': {'b'}}), 1)
        self.assertLess(score({**original, 'different': {'c'}}), 1)
        self.assertEqual(supported_score(1, 1, 5)[0], .2)

    def test_synonyms_split_one_language_vote(self):
        freq, count = distribution({'a': {'x', 'y'}, 'b': {'x'}, 'c': set()})
        self.assertEqual(count, 2)
        self.assertEqual(freq, {'x': .75, 'y': .25})

    def test_shared_distribution_has_no_contrast(self):
        self.assertEqual(contrast({'a': .5, 'b': .5}, {'a': .5, 'b': .5})[0], 0)
        self.assertEqual(contrast({'a': 1}, {'b': 1}), (1, 'a', 'b'))

    def test_ancestry_stops_at_oia_and_filters_borrowing(self):
        forms = {k: {'Language_ID': lang} for k, lang in
                 [('f', 'Sh'), ('v', 'Sh'), ('oia', 'Indo-Aryan'), ('deep', 'Indo-ir')]}
        parents = {'f': ('v', 'variant'), 'v': ('oia', 'reflex'), 'oia': ('deep', 'reflex')}
        self.assertEqual(resolve('f', forms, parents), ('oia', 'linked'))
        parents['v'] = ('oia', 'borrowed')
        self.assertEqual(resolve('f', forms, parents), ('', 'borrowed'))
        self.assertEqual(resolve('f', forms, parents, True), ('oia', 'linked'))

    def test_invalid_graph_and_redirects(self):
        forms = {'f': {'Language_ID': 'Sh'}, 'g': {'Language_ID': 'Sh'},
                 'old': {'Language_ID': 'Indo-Aryan', 'Redirect': 'new'},
                 'new': {'Language_ID': 'Indo-Aryan'}}
        self.assertEqual(resolve('f', forms, {'f': ('g', 'variant'), 'g': ('f', 'variant')}), ('', 'cycle'))
        self.assertEqual(resolve('f', forms, {'f': ('absent', 'reflex')}), ('', 'missing_node'))
        self.assertEqual(resolve('f', forms, {}), ('', 'unlinked'))
        self.assertEqual(resolve('f', forms, {'f': ('old', 'reflex')}), ('new', 'linked'))


if __name__ == '__main__':
    unittest.main()

# Mining Dardic / Plains lexical contrasts

Run from the workspace root (Python standard library only):

```sh
python3 data/analyze_concept_isoglosses.py --family-level entry
python3 data/analyze_concept_isoglosses.py --family-level head --out data/stats/dardic-plains-head
python3 data/analyze_concept_isoglosses.py --min-languages 10 --top 30 --out /tmp/isoglosses
python3 -m unittest discover -s data/tests -p test_concept_isoglosses.py
```

`report.md` shows the highest-ranked candidates plus LONG, with attested forms,
glosses, source references and persistent form IDs. `ranking.csv` contains every
positive contrast (filter `eligible` for the sample threshold); `results.json`
also contains complete family distributions, settings, selected languages and
SHA-256 hashes of the five input tables. `evidence.csv` includes all selected
form–concept records, including unlinked forms and excluded borrowings.

## Definitions

- Dardic: Jambu's Chitrali, Kashmiric, Kohistani, Kunar, Pashai and Shinaic clades.
- Plains: Bhil, Bihari, E. Hindi, Eastern, Gujaratic, Halbic, Lahndic,
  Marathi-Konkani, Punjabic, Rajasthanic, Sindhic and W. Hindi. This is an
  operational comparison set, not a claim that every included language is
  spoken in plains. Pahari, insular, migratory, Nuristani and historical groups
  are outside the default comparison. Domaaki is separately excluded despite
  Jambu's Punjabic classification. Names beginning Old, Middle or Proto- are
  excluded within selected clades. Override with `--dardic-clades`,
  `--plains-clades`, and `--exclude-languages` (comma-separated exact values).
- The sampling unit is a Glottocode, falling back to language ID. Multiple
  survey parents, dialects, spellings and sources do not multiply its vote.
- Concepts come only from the attested form's existing `form_concepts.csv`
  links. An ancestor's meaning is never projected onto its descendants.
- Families follow rank-1 reflex/variant ancestry and redirects, stopping at
  the first Old Indo-Aryan head (following variant links there as well).
  Paths without an OIA head use their terminal accepted ancestor. Derived and
  component edges and alternate hypotheses are not followed. Borrowing on a
  traversed path is excluded unless `--include-borrowed` is passed. This option
  counts a loan under its donor family, not as evidence of inheritance.

## Ranking

For each concept, each language with at least one resolved family contributes
one vote, divided equally among its distinct families. Unlinked synonyms do
not receive a family vote. For opposing families A and B, choose the pair
maximizing `(D(A) - P(A) + P(B) - D(B)) / 2`. This penalizes crossover as well
as competing families; zero means no directional separation. Ties are broken
deterministically by family IDs.

The ranking score is that separation multiplied by
`min(1, D(A)*D_linked/5, P(B)*P_linked/5)`. The last two quantities measure
positive fractional language votes for the focal family on its associated side.
Five such votes on each side give full support; `--support-target` changes this
explicit heuristic threshold. The pair is selected by resolved separation before
applying support. Unknown etymologies do not reduce this score: adding any number
of unresolved-only languages leaves it unchanged. Known competing families and
crossover still reduce separation. Coverage is reported alongside the score,
not multiplied into it. `legacy_coverage_score` and `legacy_rank` preserve the
previous coverage-penalized metric for comparison.
The default eligibility threshold is five linked language units per group.
Own shares are D(A), P(B); cross shares are D(B), P(A). Coverage means at
least one resolved family, not complete analysis of every synonym.

## Interpretation and limits

These are discovery candidates, not proven categorical isoglosses or
statistical significance tests. The data combine dictionaries and wordlists;
attestation does not establish the normal everyday word. Concept mapping can
be overbroad or wrong: inspect the printed glosses in the evidence. Languages
and clades are unevenly represented, and equal language weights do not make
them genealogically independent. Missing links are unknown, not absences.
RAW and UNRIPE, for example, can reuse the same records and are not independent
discoveries. Default entry-family counting groups numbered CDIAL subheads with
their parent article only when an accepted explicit derived/reflex/variant edge
links them. This includes some derivatives, not just strict variants. Use
`--family-level head` to keep separate heads; explicit variant edges are followed
in both modes. Neither mode merges on spelling or a guessed common root.
Choose one stable `--cldf` snapshot for comparisons across modes.

The script reads the compiled snapshot only; pending raw etymology overlays
are not applied, and a stale concept table may omit newer forms. It records
dangling form–concept links but cannot detect all semantic staleness. No source
ingestion, lexical edits, database rebuild or browser refresh is performed.

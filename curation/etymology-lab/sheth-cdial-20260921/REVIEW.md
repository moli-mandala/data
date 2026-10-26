# Sheth 1923 Sanskrit equivalents → CDIAL — research pass, 21 September 2026

Sheth's *Pāia-sadda-mahaṇṇavo* prints a bracketed Sanskrit equivalent for most Prakrit senses.
The 14 September ingestion installed those as 26,191 standalone `Sk` rows plus source
comparisons, with no ancestry. This pass links them, and the Prakrit senses they belong to, to
CDIAL entries. It also aligned the `sheth-ddsa` sound profile with the CDIAL house transcription
(`Ch` → `Cʰ`, vocalic `ṛ` → `r̩`, `ṅ` → `ŋ`, `ḥ` → `ḣ`, candrabindu → nasalised vowel), so the
rebuilt Sheth forms print like Turner's; the pass was generated from that rebuild.

## Rules

1. **Sanskrit rows.** The accent-stripped Sanskrit form (CDIAL `Original`, Sheth `Form`, both
   folded to one alphabet) is looked up among CDIAL entry headwords (`Status = entry`, redirects
   excluded; `*` and trailing `-` ignored; a final visarga may be dropped). Exactly one headword
   → `reflex` edge, rank 1, to that entry. Several headwords → hand review. No headword → left
   unlinked (20,251 rows; mostly classical/Jain vocabulary CDIAL does not carry).
2. **Hand review** (`review-groups.tsv`, `review-decisions.tsv`; 496 groups, 515 Sanskrit rows):
   the homographs, the bound heads Sheth writes `°…`, one- and two-letter Sanskrit forms, and
   forms coinciding with a Sheth work abbreviation. Decisions were made from Sheth's Hindi
   definition against Turner's glosses; a sense that fits neither head, a proper name, or an
   unglossed section head is held. Two `[हे]` brackets were the Hemacandra abbreviation, not
   Sanskrit *hē*; `ka` is Sheth's suffix *-ka*, not the pronoun; *vr̥ṣa* 'bull' is held because
   CDIAL 12083 is only the plant name.
3. **Prakrit / Apabhraṁśa rows.** Every Prakrit row of a linked sense gets the same `reflex`
   edge, except alternate heads that already carry a `variant` edge to their primary head (242;
   they reach the etymon transitively). Where CDIAL does not itself cite that Prakrit form under
   the entry, a consonant-skeleton check (`skeleton.py`, threshold 0.5, calibrated so that 1.2% of
   the CDIAL-confirmed forms fall below it) holds rows whose Sanskrit equivalent is not shaped
   like their etymon, because Sheth sometimes prints a semantic equivalent (bhukta for samāṇia,
   idānīm for ēvvahi). Held Prakrit rows keep their Sanskrit row linked. The check is weak for one-consonant
   skeletons (e.g. aicchiya 'gone' under gata 4008, which Sheth prints but which is not its
   etymon); those few remain as Sheth's printed claim.
4. Reconstructed CDIAL heads (`*…`, 119 Sanskrit rows) are linked with a note that Sheth's
   Sanskrit is a Sanskritisation.

## Counts (`summary.json`)

| | rows |
|---|---|
| Sanskrit rows linked by unique headword match | 5,306 (+119 to reconstructed heads) |
| Sanskrit rows linked by hand review | 359 |
| Sanskrit rows held by hand review | 156 |
| Prakrit rows linked, CDIAL cites the same Prakrit form | 3,137 |
| Prakrit rows linked on Sheth's equivalent alone | 2,472 |
| Prakrit rows held by the skeleton check | 175 |
| Prakrit alternate heads left on their variant edge | 242 |
| **Accepted assignments** | **11,393** to 2,737 CDIAL entries |

Cross-check: 3,137 of the linked Prakrit forms are the very forms Turner cites under the matched
entry, and no Sheth article's senses resolve to different CDIAL entries.

Sampling: 20 random Sanskrit links and 30 random unconfirmed Prakrit links were read before the
skeleton filter was added; the Sanskrit sample had no errors, the Prakrit sample surfaced the
semantic-equivalent class that rule 3 now holds.

## Reproduction

```sh
uv run python curation/etymology-lab/sheth-cdial-20260921/build_decisions.py   # candidates.csv, decisions.json, summary.json
make check-pass DECISIONS=curation/etymology-lab/sheth-cdial-20260921/decisions.json PASS=sheth-cdial
```

Rows were saved with `make save-pass` into `data/other/forms/etymologies/20260914-sheth-sanskrit.csv`
and `data/other/forms/etymologies/20260914-sheth.csv`; the ledgers are beside this file.

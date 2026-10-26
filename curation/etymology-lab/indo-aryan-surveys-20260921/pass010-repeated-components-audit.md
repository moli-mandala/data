# Repeated components

The de-de giving construction repeats the same lexical family in two positions. It is not collapsed into a single inheritance link.

Read-only inspection confirms that `assign_form_ids.apply_assignments` matches component edges on child, parent, kind, rank **and Pos**. The browser database writer (`jambu-static/scripts/build_static_db.py`, `load_edges`) deduplicates on the same five values, and the compact database retains position. Thus two ordered occurrences of one parent survive the implementation, despite the abbreviated two-field key description in AGENTS.md.

The additional scratch-graph test `test_repeated_lexical_member_keeps_both_positions` confirms two edges, contiguous positions, zero changes on repeat application, and preservation of both members in the review manifest. All 17 focused helper tests pass. No database was rebuilt to perform this check.

Construction evidence was read in the labelled Platts dictionary reproduction at [denā](https://www.rekhta.org/urdudictionary?keyword=denaa), which explicitly lists de-denā and discusses the intensive auxiliary use of denā. Full historical stem evidence is cached in `primary-018.json`.

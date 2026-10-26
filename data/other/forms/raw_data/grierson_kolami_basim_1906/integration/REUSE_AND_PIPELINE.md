# Same-occurrence reuse and pipeline limits

The whole-chapter candidate exports 827 new rows. Three existing canonical rows cover two additional physical printings and remain under their existing keys:

| Chapter audit scope | Existing key | Canonical owner |
| --- | --- | --- |
| p566 `amd` | `grierson1906kui:p566:outside-comparand:06` | Kui package, Kolami comparison |
| p623 `nē`, Kolami attribution | `grierson1906kui:p623:outside-comparand:15` | Kui package |
| p623 `nē`, Naiki attribution | `grierson1906naiki:p623:genitive-comparison:naiki` | Naiki package |

`combined-review-audit.jsonl` records the new chapter scope keys, physical source context, and existing `entry_keys`. These are **audit-only scope mappings**, not new canonical source keys, persistent-ID aliases, or additional compiled citations. The p566 reuse preserves the approved old contextual gloss `he`; the chapter's prose comparison supplies no separate gloss and remains blank in its scope evidence. Both p623 language attributions arise from the same printed token, already represented by the two existing source-specific witnesses.

The current compiler does not read this chapter audit as an alias or citation input. `assign_form_ids.py` reads canonical emitted source keys and invokes `source_key_aliases.apply_source_key_aliases`; that helper reads `data/other/form_aliases/*.csv` and redirects **previously registered retired source IDs** to active target keys. The new chapter scope keys have never been emitted or registered, so adding them to that retired-ID mechanism would not create source-key aliases or citations. No such sidecar is proposed.

The installation will preserve these three rows and their current keys/citations. Their original source-owned importers remain responsible for emitting them. This chapter's own importer must revalidate the canonical reuse rows (including the accepted `amd` correction) and maintain the audit mappings without duplicate lexical exports. Current source-local checks prove canonical byte/hash continuity and exact mapping closure; future full-build checks must verify the old keys retain their persistent IDs and the combined graph. No new scope-key persistent-ID or citation behavior is claimed.

# KeraɁ Mundari Living Dictionary: complete raw snapshot

Status: **479-row source stage installed after passing independent review**. See `INSTALLED_SOURCE_CHECKLIST.md` and `installation-report-20260927.json` for authoritative current scope, validation and hashes. The acquisition-stage sections below are historical; the complete importer and metadata installation supersede their pending interpretation list. Full database and browser gates remain deferred, so this is not a claim of completed end-to-end ingestion.

The [public dictionary](https://livingdictionaries.app/kera-mundari) was selected on 2026-09-27 as a new named-variety source: 446 entries, explicitly KeraɁ Mundari, ISO `unr`, Glottocode `kera1257`. Searches of current bibliography, dialect registry and source package names found no Kera Living Dictionary installation. Existing Kharia and Birhor Living Dictionary work was excluded. The current catalogue also offers Alu Kurumba, Gutob, Gta, Ho, Angika and other eligible leads; Kera was selected for distinct variety coverage with a bounded complete snapshot, not as a slice of a larger dictionary.

The [source description](https://livingdictionaries.app/kera-mundari/about) treats Kera as a Mundari variety. Proposed routing is therefore existing `mu` plus a separately reviewed Kera dialect, not a new base and not Kurux. No registry edit has been made. Its public map point must not be presented as a verified elicitation site.

## Reproduction and evidence

`snapshot_source.py` retrieves the complete public snapshot at `https://snapshots.livingdictionaries.app/dictionaries/kera-mundari.db.gz`, preserving exact response bodies, documentary pages, hashes and acquisition dates. The endpoint returned SQLite bytes despite its `.gz` filename; the script checks the actual encoding and retains a hardlinked SQLite reader path without duplicating disk usage. An initial gzip assumption failed safely before any export; no data was discarded. The first response headers were not retained because of that failure; the receipt explicitly marks cached-body responses and records file acquisition times rather than inventing headers.

- Snapshot: 1,572,864 bytes, SHA256 `ea5178108325f961ae5e34484ea6cafef267862e142377e16b49c553e7b95b00`.
- Upstream metadata: dictionary `kera-mundari`; schema `20260723_ignored_forms.sql`; last modified `2026-07-23T06:50:32.652Z`; synced sequence `891`.
- `export_snapshot.py` emits **all 446 entries and 446 senses**, preserving upstream IDs, exact strings, timestamps, POS, gloss languages, domains and annotation associations. Export SHA256 `8999ed134b13a54bf7780098635ec1c6e07a6f142736ee372278cf793c387764`.
- The saved live dictionary HTML advertises 446 entries. Every snapshot table is counted in `snapshot-completeness.json`; unique IDs, entry–sense coverage and annotation foreign keys all pass.
- No pagination is involved: this is the complete public SQLite representation. This proves completeness of that downloaded published snapshot, not access to unpublished content or atomic equivalence to later live edits.
- There are zero sentence, text, grammar-section, source-reference and relationship records. All these empty tables are explicitly accounted for.

The public original includes media metadata and administrative fields; it is ignored by Git and retained as evidence. No media binaries were fetched. The reviewable lexical export omits account identifiers and speaker personal metadata. Media tables are counted (445 audio records and associations), not mistaken for additional lexical entries.

## Rights and scope

Reuse is covered by the user's explicit confirmation that Living Dictionaries is allowed per Sarva, reiterated on 2026-09-27. The source's terms, dated 2026-06-12, are preserved in the snapshot. No Creative Commons or other open licence is invented. The dictionary's copyright field is null in the saved public metadata; that is not a public-domain assertion.

The ingestion checklist is active with dictionary/glossary and website/API addenda. Original source text, native spelling, phonetic interpretation, inflection, borrowing and all source annotations still require complete review. All 446 phonetic fields are blank; no IPA may be inferred merely from Roman spelling. There are 440 nonempty entry Notes and 443 POS fields. The three upstream “dialect” labels are **English Loan**, **Hindi Loan**, and **Possible Hindi Loan**, with 28 assignments. They are source borrowing annotations, not geographical dialects; certainty must survive interpretation.

Remaining gates: complete field interpretation/importer, notation/profile review, auxiliary-source attribution, registered dialect metadata, deterministic staged output, full independent audit, focused source tests and explicit installation approval from the parent reviewer. Full data build and suite remain deferred under the current instruction; browser database construction is not authorized by this source task. No commits, publication, remote execution or canonical changes occurred.

# Root preliminary profile review

Root inspected the source-local tests, proposed YAML/BibTeX, and actual-tokenizer
and actual-parser review reports, then reran the three preliminary profile tests
through the existing local monitored tmux session.

Result: **two conversion tests passed; one pinned-input test failed** in 0.33s.
The failure is expected review-input drift, not a conversion assertion: the
outside census changed `scope-decisions.json` from SHA256
`709ecd3dba7b0fea695e030c1ca9bf61a1a42bd08d6b5a3c68c26bf847455aec`
to `d53cb8d3e2152d3d18bb9a08e207c03c3f13372754ba6b89558fdd6bc03ed291`.
The failure log is retained at `tmp/kolami-root-preliminary-profile-tests.log`.
The old receipt is stale; it must not be presented as verification of current
outside decisions or the final corrected candidate.

The conversion tests compare all805 current fixture cases under NFC and NFD
using the actual segments tokenizer. Difficult controls explicitly distinguish
ṅ from ñ, preserve A/ū quantity distinction, preserve optional parenthetical
notation, meaningful hyphens, source boundaries and punctuation. The inventory
and coverage projection verify exact immutable input witnesses, subject to the
pin failure above. The fixture must be regenerated and rerun once the accepted
full candidate and reviewed corrections are frozen.

The separate actual-parser report covers563 input rows under four route/Unicode
combinations. It reports all15 source fields and Original retained, with121
boundary punctuation display deltas for the profile route. That report is
preliminary source-local evidence, not a full CLDF pipeline gate. Root's display
decision is recorded in `berar-and-display-decisions.md`.

No canonical profile, YAML, bibliography or database was changed. Required full
build and compiled checks remain deferred under the user's instructions.

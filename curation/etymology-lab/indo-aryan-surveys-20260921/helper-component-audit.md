# Shared helper: ordered compound support

The existing `etymology_lab.py` accepted one parent per record and ignored historical `components` lists. This run extends it to expand an ordered list into `component` rows with contiguous `Pos`, validate **every** parent, and retain every component ID and form in per-language manifests. The historical `parent` field is permitted only when it agrees with the first component.

The change preserves the one-analysis-per-target guard and uses the same current-sidecar conflict check, exact-node checks, temporary graph application, idempotence check, unrelated-edge comparison, source resolver and before/after hashes. It does not regenerate shared data.

Validation: `.venv/bin/python -m pytest tests/test_etymology_lab.py -q` — **16 passed**. Tests cover legacy single-parent output, modern and historical compound decisions, malformed component lists and conflicting fields, absent/unlinked/redirected second parents, read-only dry runs, idempotence and complete ordered manifests. All fixtures use tiny temporary graphs. Pass006 also exercises the shared helper on real compiled inputs in a temporary graph and saves its 48 component rows to the existing source sidecars.

No full suite or rebuild was run, as requested. The two pre-existing misplaced sidecar rows remain separately reported by `check-sources`.

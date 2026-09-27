.PHONY: all forms parsers inferred-extensions cdial dedr dedr_params parser-diff ingest sources check-pass save-pass check-etymologies check-sources manual-survey-etymology-check punjabi burushaski-cognates wiktionary-piir wiktionary-piir-refresh

# One interpreter for every stage and importer, and one heavy job at a time (8 GB laptop).
PY := uv run python
export OMP_NUM_THREADS ?= 1
export OPENBLAS_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1

# Regenerate one source CSV from its importer. The commands live in the source's YAML
# (`defaults.importer.commands` in data/other/forms/<stem>.yaml); `make sources` lists them.
#   make ingest SOURCE=20260828-sil-jaunsari
ingest:
	@test -n "$(SOURCE)" || (echo "usage: make ingest SOURCE=<stem>   (see: make sources)"; exit 2)
	$(PY) source_meta.py ingest $(SOURCE)

sources:
	@$(PY) source_meta.py importers

# Validate the per-source YAML settings and the etymology sidecars (cheap; run before a build).
check-sources:
	$(PY) source_meta.py
	$(PY) etymology_assignments.py check

# Etymology-lab research passes. DECISIONS is the pass's decisions JSON; PASS names the ledger
# files written beside it. check-pass validates against the compiled graph without writing.
#   make check-pass DECISIONS=curation/etymology-lab/<dir>/decisions.json PASS=<name>
#   make save-pass  DECISIONS=… PASS=<name> NOTE="Joint review 2026-09-16." [AUTH="…"]
check-pass:
	@test -n "$(DECISIONS)" -a -n "$(PASS)" || (echo "usage: make check-pass DECISIONS=<json> PASS=<name>"; exit 2)
	$(PY) etymology_lab.py save "$(DECISIONS)" --pass "$(PASS)" --note "$(NOTE)" --dry-run

save-pass:
	@test -n "$(DECISIONS)" -a -n "$(PASS)" || (echo "usage: make save-pass DECISIONS=<json> PASS=<name> NOTE=\"…\" [AUTH=\"…\"]"; exit 2)
	$(PY) etymology_lab.py save "$(DECISIONS)" --pass "$(PASS)" --note "$(NOTE)" --authorization "$(AUTH)"

# Legacy shell recipe predating the importer convention.
punjabi:
	cd data/other/forms/raw_data && $(PY) old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

# The two dictionary parsers are real make targets: their CSVs are rebuilt only when the parser
# code or the cached page snapshot changed, and `make all` never builds from a stale parse.
# Each parser writes to a temp file and renames on success, so a crash leaves the old CSV intact
# and make sees it as still out of date.
CDIAL_CSV := data/cdial/cdial.csv
DEDR_CSV := data/dedr/dedr_new.csv

$(CDIAL_CSV): data/cdial/parse.py data/cdial/abbrevs.py data/cdial/references.py data/dedr/parser_utils.py data/cdial/cdial.pickle
	cd data/cdial && $(PY) parse.py

$(DEDR_CSV): data/dedr/parse.py data/dedr/parser_utils.py data/dedr/cleanup.py data/dedr/abbrevs.py data/dedr/dedr.pickle
	cd data/dedr && $(PY) parse.py

# Shape-inferred extended reflexes (-kk-, -ḍ-, -l-, -r-) that Turner left on the bare etymon;
# unify_cldf.py re-homes them. Rebuilt whenever the CDIAL parse or the detector changes.
#   make inferred-extensions            (or: uv run python infer_extensions.py --metrics)
INFERRED_EXT := data/cdial/inferred-extensions.csv

$(INFERRED_EXT): $(CDIAL_CSV) infer_extensions.py
	$(PY) infer_extensions.py

inferred-extensions: $(INFERRED_EXT)

parsers: $(CDIAL_CSV) $(DEDR_CSV) $(INFERRED_EXT)
cdial: $(CDIAL_CSV)
dedr: $(DEDR_CSV)
	cd data/dedr && $(PY) get_params.py

# What did a parser change do? Diffs the parser CSV against HEAD in seconds; no build needed.
#   make parser-diff P=cdial   (or P=dedr; add ARGS="--samples 40")
parser-diff:
	@test -n "$(P)" || (echo "usage: make parser-diff P=cdial|dedr [ARGS=...]"; exit 2)
	$(PY) parser_diff.py $(P) $(ARGS)

# Content-only build: everything that determines forms/glosses/edges, without the slow
# alignment and reference stages. Use it to check a parser or importer change end to end.
forms: parsers
	$(PY) make_cldf.py
	$(PY) link_refs.py
	$(PY) unify_cldf.py
	$(PY) assign_form_ids.py

all: parsers
	$(PY) make_cldf.py
	$(PY) link_refs.py
	$(PY) unify_cldf.py
	$(PY) assign_form_ids.py
	$(PY) concepts.py
	$(PY) align.py
	$(PY) make_refs.py
	$(MAKE) manual-survey-etymology-check

manual-survey-etymology-check:
	$(PY) -m pytest -q tests/test_manual_survey_etymologies.py

# The Proto-Indo-Iranian etymon layer resolves its links against the *built*
# graph, so it needs a complete build to read and a second one to compile what it
# writes. `make all` in between is not optional: the importer refuses to run
# against a half-built cldf/.
wiktionary-piir:
	uv run --with segments python -c "import sys; sys.path.insert(0,'data/other/params/raw_data'); import wiktionary_piir as W; W.write_register('data/other/params/raw_data/20260827-indo-iranian-source-register.csv'); print(W.install()[0].most_common())"
	$(MAKE) all

# Re-snapshot the source from the MediaWiki API before rebuilding it.
wiktionary-piir-refresh:
	$(PY) data/other/params/raw_data/wiktionary_piir.py fetch
	$(MAKE) wiktionary-piir

burushaski-cognates:
	$(PY) burushaski_cognates.py

dedr_params:
	cd data/dedr && $(PY) get_params.py && cd ../..

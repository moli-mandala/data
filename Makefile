punjabi:
	cd data/other/forms/raw_data && python old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

all:
	@if [ -f tmp/pdfs/JLSR2025-005.pdf ]; then \
		echo "regenerating Markodi forms from markodi_etyma.csv"; \
		cd data/other/forms/raw_data && UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/uv-cache} uv run --with pypdf python markodi.py; \
	else echo "skipping Markodi regen (working PDF tmp/pdfs/JLSR2025-005.pdf absent)"; fi
	UV_CACHE_DIR=$${UV_CACHE_DIR:-/tmp/uv-cache} uv run --with segments --with unidecode --with tqdm python make_cldf.py
	uv run python link_refs.py
	uv run python unify_cldf.py
	uv run python align.py

dedr:
	cd data/dedr && uv run --with beautifulsoup4 --with html5lib --with tqdm python parse.py && uv run python get_params.py && cd ../..

dedr_params:
	cd data/dedr && python get_params.py && cd ../..

punjabi:
	cd data/other/forms/raw_data && python old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

ssnp:
	python3 data/other/forms/raw_data/ssnp.py

berger:
	uv run python data/other/forms/raw_data/berger.py --install

sigiri:
	uv run python data/other/forms/raw_data/sigiri.py --install

kullui:
	uv run python data/other/forms/raw_data/kullui_org.py

all:
	@if [ -f tmp/pdfs/JLSR2025-005.pdf ]; then \
		echo "regenerating Markodi forms from markodi_etyma.csv"; \
		cd data/other/forms/raw_data && uv run python markodi.py; \
	else echo "skipping Markodi regen (working PDF tmp/pdfs/JLSR2025-005.pdf absent)"; fi
	uv run python make_cldf.py
	uv run python link_refs.py
	uv run python unify_cldf.py
	uv run python assign_form_ids.py
	uv run python concepts.py
	uv run python align.py
	uv run python make_refs.py

burushaski-cognates:
	python3 burushaski_cognates.py

dedr:
	cd data/dedr && uv run python parse.py && uv run python get_params.py && cd ../..

dedr_params:
	cd data/dedr && python get_params.py && cd ../..

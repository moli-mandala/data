.PHONY: all manual-survey-etymology-check ssnp ssnp04 sil-irula sil-gadaba sil-jaunsari sil-bishnupriya sil-meitei sil-war-jaintia sil-kuki-chin-bangladesh sil-lahul sil-pahari-pothwari

punjabi:
	cd data/other/forms/raw_data && python old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

ssnp:
	python3 data/other/forms/raw_data/ssnp.py

ssnp04:
	python3 data/other/forms/raw_data/ssnp04_1992/extract_ssnp04.py
	python3 data/other/forms/raw_data/ssnp04_1992/import_ssnp04.py

berger:
	uv run python data/other/forms/raw_data/berger.py --install

sigiri:
	uv run python data/other/forms/raw_data/sigiri.py --install

kullui:
	uv run python data/other/forms/raw_data/kullui_org.py

# Needs the SIL ESR 2017-005 PDF at tmp/pdfs/beine-bhatri/source.pdf; the importer checks its hash.
beine-bhatri:
	uv run python data/other/forms/raw_data/beine_bhatri.py --install

# Rebuilds from the pinned page cache. Add --refresh (with curl_cffi) to re-crawl Webonary.
halbi:
	uv run python data/other/forms/raw_data/woods_halbi.py --offline --install

# Rebuilds the checked Irula transcription and complete audit from the pinned OCR scaffold.
sil-irula:
	uv run python data/other/forms/raw_data/sil_irula_2018/import_irula.py --install

# Rebuilds the checked Mudhili Gadaba transcription and complete audit.
sil-gadaba:
	python3 data/other/forms/raw_data/sil_gadaba_2019/import_gadaba.py

# Rebuilds the official SAG-IPA-decoded Jaunsari wordlists and complete audit.
sil-jaunsari:
	python3 data/other/forms/raw_data/sil_jaunsari_2008/import_jaunsari.py

# Rebuilds the image-verified legacy-IPA Bishnupriya wordlists and complete audit.
sil-bishnupriya:
	python3 data/other/forms/raw_data/sil_bishnupriya_2008/import_bishnupriya.py

# Rebuilds the official SAG-IPA-decoded Meitei wordlists and complete audit.
sil-meitei:
	python3 data/other/forms/raw_data/sil_meitei_2008/import_meitei.py

# Rebuilds the preserved official SAG-IPA-decoded War-Jaintia wordlists and audit.
sil-war-jaintia:
	python3 data/other/forms/raw_data/sil_war_jaintia_2007/import_war_jaintia.py

# Rebuilds the public Appendix A SAG-IPA-decoded Bangladesh Kuki-Chin wordlists and audit.
sil-kuki-chin-bangladesh:
	python3 data/other/forms/raw_data/sil_kuki_chin_bangladesh_2011/import_kuki_chin.py

# Rebuilds the Unicode text-layer Lahul Valley wordlists and complete audit.
sil-lahul:
	python3 data/other/forms/raw_data/sil_lahul_2019/import_lahul.py

# Rebuilds the positioned Doulos SIL Pahari/Pothwari wordlists and complete audit.
sil-pahari-pothwari:
	python3 data/other/forms/raw_data/sil_pahari_pothwari_2010/extract_pahari_pothwari.py
	python3 data/other/forms/raw_data/sil_pahari_pothwari_2010/import_pahari_pothwari.py

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
	$(MAKE) manual-survey-etymology-check

manual-survey-etymology-check:
	uv run pytest -q tests/test_manual_survey_etymologies.py

# The Proto-Indo-Iranian etymon layer resolves its links against the *built*
# graph, so it needs a complete build to read and a second one to compile what it
# writes. `make all` in between is not optional: the importer refuses to run
# against a half-built cldf/.
wiktionary-piir:
	uv run --with segments python -c "import sys; sys.path.insert(0,'data/other/params/raw_data'); import wiktionary_piir as W; W.write_register('data/other/params/raw_data/20260827-indo-iranian-source-register.csv'); print(W.install()[0].most_common())"
	$(MAKE) all

# Re-snapshot the source from the MediaWiki API before rebuilding it.
wiktionary-piir-refresh:
	uv run python data/other/params/raw_data/wiktionary_piir.py fetch
	$(MAKE) wiktionary-piir

burushaski-cognates:
	python3 burushaski_cognates.py

dedr:
	cd data/dedr && uv run python parse.py && uv run python get_params.py && cd ../..

dedr_params:
	cd data/dedr && python get_params.py && cd ../..

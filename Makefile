punjabi:
	cd data/other/forms/raw_data && python old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

all:
	python make_cldf.py

dedr:
	cd data/dedr && python parse.py && python get_params.py && cd ../..

dedr_params:
	cd data/dedr && python get_params.py && cd ../..
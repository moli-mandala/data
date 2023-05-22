parse_punjabi:
	cd data/other/forms/raw_data && python old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

make_cldf:
	python make_cldf.py

parse: parse_punjabi make_cldf

all: parse

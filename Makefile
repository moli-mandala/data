punjabi:
	cd data/other/forms/raw_data && python old_punjabi.py && mv old_punjabi.csv ../20230521-old_punjabi.csv && cd ../../../..

cldf:
	python make_cldf.py

ia:
	cd data/cdial && python parse.py && cd ../..

parse: cldf ia

all: parse

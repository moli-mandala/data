basic-transformer:
	python train.py -a 1 -lr 1 -la 1 -he 1 -emb 32 -hid 64 -bs 1024 -e 50 -f all.pickle

basic-gru:
	python train.py -e 50 -f all.pickle

best-transformer:
	python train.py -a 1 -lr 1 -la 3 -he 4 -emb 64 -hid 128 -bs 1024 -e 50 -f all.pickle

best-gru:
	python train.py -a 0 -lr 0.002 -la 4 -emb 64 -bs 1024 -e 50 -f all.pickle
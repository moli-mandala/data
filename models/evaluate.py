import nltk

def eval_model(filename):
    batch_size = 1024
    data, eval_data, to_char = get_data(batch_size=batch_size)
    criterion = nn.NLLLoss(reduction="sum", ignore_index=0)
    model = make_model(num_words, num_words, emb_size=args.emb, hidden_size=args.emb, bidirectional=True if args.bidirectional else False)
    model.load_state_dict(torch.load(filename))
    model.eval()

    fout = open('result.txt', 'w')
    with torch.no_grad(): 
        perplexity = run_epoch(eval_data, model,
                                SimpleLossCompute(model.generator, criterion, None))
        print("Evaluation perplexity: %f" % perplexity)
        print_examples(eval_data, model, fout, n=50, max_len=20, mapping=to_char, plot=2)
    fout.close()
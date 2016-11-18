def eval_score(test_words, rank_table):
    
    test_w = test_words.ix[:,'blank_':]
    test_w.index = test_words['dat2$track_id'].tolist()
    
    results = []
    for i in range(len(test_w.index)):
        ls = test_w.ix[i].nonzero()[0].tolist()
        words_rank = rank_table.ix[i].iloc[ls].tolist()
        rank_bar = sum(words_rank)/(len(ls) * sum(rank_table.ix[i])/5000)
        results.append(rank_bar)
    return(results)
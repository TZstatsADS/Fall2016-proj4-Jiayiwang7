def word_cluster_table(model, df_train):
    
    # return word frequency for each cluster

    pred = model.predict(df_train)
    df_cluster = pd.DataFrame({'file_1': df_train.index.tolist(), 'cluster':pred, 'count': 1})
    df_word = pd.read_csv('lyr.csv')
    
    for i in range(len(pd.Series.unique(df_cluster['cluster']))):
        if i == 0:
            tmp_ini = df_cluster.loc[df_cluster['cluster'] == i]
            result_ini = pd.concat([tmp_ini, df_word], axis=1, join='inner')
            df_s_ini = result_ini.drop('file_1', 1)
            #cnt_ini = len(df_s_ini)
            sum_c_ini = df_s_ini.sum(axis=0)
            ls_wd = sum_c_ini.to_frame()
            #ls_word['cnt'] = cnt_ini
        else:
            tmp = df_cluster.loc[df_cluster['cluster'] == i]
            result = pd.concat([tmp, df_word], axis=1, join='inner')
            df_s = result.drop('file_1', 1)
            cnt = len(df_s)
            sum_c = df_s.sum(axis=0)
            ls_sum = list(sum_c)
            #ls_sum.append
            ls_wd[i] = ls_sum
        ls_word = ls_wd.transpose()
        ls_word = ls_word.drop('dat2$track_id', 1).drop('cluster', 1)
        
    return(ls_word)


    def test_model(cluster_table, model, df_test):
    
    # return predicted word rank
    # pred_test = model.predict(df_test)
    cluster_table['cluster'] = range(len(cluster_table))
    pred_test = model.predict(df_test)
    pred_cluster = pd.DataFrame({'file_1': df_test.index.tolist(), 'cluster':pred_test})

    for i in range(len(pred_cluster)):
        if i == 0:
            pre_c_word = cluster_table.loc[cluster_table['cluster'] == pred_cluster.ix[i][0]]
            pre_c_word['track_id'] = pred_cluster.ix[i][1]
        else:
            tmp = cluster_table.loc[cluster_table['cluster'] == pred_cluster.ix[i][0]]
            tmp['track_id'] = pred_cluster.ix[i][1]
            pre_c_word = pd.concat([pre_c_word, tmp])
    index = pre_c_word['track_id'].tolist()
    word_test = pre_c_word.ix[:,'blank_': 'zwei']
    word_test.index = index
    
    
    for i in range(len(word_test)):
        ls_freq = word_test.iloc[[i]].values.tolist()[0]
        ls_ne = [-x for x in ls_freq]
        rank = rankdata(ls_ne).tolist()
        #seq = sorted(ls_freq, reverse = True)
        #rank = [seq.index(v)+1 for v in ls_freq]
        word_test.iloc[i] = rank
    
    return(word_test)


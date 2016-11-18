import pandas as pd
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()

def topic_model(df_w, n_topic):
    track_id = df_w['dat2$track_id'].tolist()
    df_w.index = track_id
    df_topic = df_w.drop('dat2$track_id', 1)
    word_dic = df_topic.columns.values
    ly_text = []
    en_stop = ['i', 'you', 'the']
    for i in range(len(df_topic)):
        cnt = list(df_topic.ix[i])
        tmp_ls = []
        for j in range(len(cnt)):
            tmp = cnt[j] * [word_dic[j]]
            tmp_ls += tmp
        tmp_ls = [i for i in tmp_ls if not i in en_stop]
        ly_text.append(tmp_ls)
    dictionary = corpora.Dictionary(ly_text)
    corpus = [dictionary.doc2bow(text) for text in ly_text]
    ldamodel = models.LdaModel(corpus, num_topics=18, id2word = dictionary, passes=20)
    return [ldamodel, corpus, dictionary]

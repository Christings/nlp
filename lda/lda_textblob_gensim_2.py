#!/usr/bin/env Python
# coding=utf-8

from textblob import TextBlob
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models, similarities
import pyLDAvis.gensim
import gensim

with open("../corpus/4095_01.txt", "r", encoding="utf-8") as f:
    processed_doc = []
    for doc in f:
        # print("doc:",doc)
        #     doc = doc.replace('\"', '.')
        doc = TextBlob(doc)
        processed_words = []
        for words, tag in doc.lower().tags:
            # print("word:",words,tag)
            if words not in STOPWORDS and len(words) > 3:
                if tag != 'IN' or tag != 'CC' or tag != 'DT' or tag != 'TO':
                    words = words.singularize()
                    processed_words.append(words)
        # print("processed_words:", processed_words)
        processed_doc.append(processed_words)
        # print("processed_doc:", len(processed_doc))

dictionary = corpora.Dictionary(processed_doc)
print("dictionary:", dictionary)

corpus = [dictionary.doc2bow(text) for text in processed_doc]

tfidf = models.TfidfModel(corpus)

corpus_tfidf = tfidf[corpus]

num_topics = 100
lda = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=60)

vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.show(vis_data)
#!/usr/bin/env Python
# coding=utf-8
import pandas as pd
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from pprint import pprint
import re, nltk

# 1.读取文件
f = open('../corpus/4095_01.csv', encoding='gbk')
stop_list = list(STOPWORDS)


# 2.对文档进行分词，并去掉停用词
def tokenize(text):
    # text=text.lower()

    text = re.sub("[^a-zA-Z]", " ", text)  # Removing numbers and punctuation
    text = re.sub(" +", " ", text)  # Removing extra white space
    # text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b", " ", text)  # Removing very long words above 10 characters
    text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b", " ", text)  # Removing single characters (e.g k, K)
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text)
    tokens = nltk.word_tokenize(text.strip())
    tokens = nltk.pos_tag(tokens)
    # Uncomment next line to use stemmer
    # tokens = stem_tokens(tokens, stemmer)

    # words = [w for w in text if w not in stop_list]
    return tokens

texts = []
for doc in f:
    temp_doc = tokenize(doc.strip())
    current_doc = []
    for word in range(len(temp_doc)):
        if temp_doc[word][0] not in stop_list:
            current_doc.append(temp_doc[word][0])

    texts.append(current_doc)

# texts = [tokenize(doc) for doc in f]
# texts = [[word for word in tokenize(line.strip().lower().split()) if word not in stop_list] for line in f]
print("Texts=")
print(texts)

# 3.构建字典
dictionary = corpora.Dictionary(texts)
print(dictionary)

v = len(dictionary)  # 字典长度

# 4.计算每个文档中的TF-IDF的值
# 根据字典，将每行文档都转换为索引的形式
corpus = [dictionary.doc2bow(text) for text in texts]
for line in corpus:
    print(line)

# 计算每篇文档没个词的tf-idf值
corpus_tfidf = models.TfidfModel(corpus)[corpus]
print("TF_IDF:")
for c in corpus_tfidf:
    print(c)

# 5.应用LDA模型
print('\nLDA Model:')
num_topics = 2  # 主题数目
lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, alpha='auto', eta='auto',
                      minimum_probability=0.001)

# 打印每篇文档被分在各个主题的概率
doc_topic = [a for a in lda[corpus_tfidf]]
print('Document-Topic:\n')
print(doc_topic)

# 打印每个主题中，每个词出现的概率
for topic_id in range(num_topics):
    print('Topic', topic_id)
    print(lda.show_topic(topic_id))

# 计算文档与文档之间的相似性
similarity = similarities.MatrixSimilarity(lda[corpus_tfidf])
print('Similarity:')
print(list(similarity))

import pyLDAvis.gensim

vis = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.show(vis)

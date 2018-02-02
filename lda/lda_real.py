#!/usr/bin/env Python
# coding=utf-8

import re
import pandas as pd
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import STOPWORDS
import gensim

## 1.读取数据
df = pd.read_excel("../corpus/4095_01.xlsx", sheet_name='500')
print(df['AB'].head())

## 2.查看数据格式
# print(df.shape)  (4095,66)

## 3.加载停止词
# stopset=stopwords.words('english')
# print(stopset)
# print(len(stopset)) 179个停止词

stopset = STOPWORDS


# print(len(stop)) 337个停止词

## 4.文本处理(大写变小写，去掉无关符号，初步获得词向量，去掉停止词)
# def tokenize(text):
#     text = re.sub("[^a-zA-Z]", " ", text)  # Removing numbers and punctuation
#     text = re.sub(" +", " ", text)  # Removing extra white space
#     # text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b", " ", text)  # Removing very long words above 10 characters
#     text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b", " ", text)  # Removing single characters (e.g k, K)
#     text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text)
#     tokens = nltk.word_tokenize(text.strip())
#     tokens = nltk.pos_tag(tokens)
#     # Uncomment next line to use stemmer
#     # tokens = stem_tokens(tokens, stemmer)
#     return tokens

def tokenize(text):
    # text = text.lower()
    # words = re.sub("\w", " ", text).split()
    words = [w for w in text if w not in stopset]
    return words


processed_docs = [tokenize(doc) for doc in df]
word_count_dict = gensim.corpora.Dictionary(processed_docs)
# word_count_dict.filter_extremes(no_below=20,no_above=0.1)
# word must appear >10 times, and no more than 20% documents


## 5.将文档(词表)转换为词袋(BOW)格式(token_id,token_count)的二元组
bag_of_words_corpus = [word_count_dict.doc2bow(pdoc) for pdoc in processed_docs]

## 6.LDA分析
lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=5, id2word=word_count_dict, passes=5)
# gensim.models.ldamulticore.LdaMulticore(corpus=None, num_topics=100, id2word=None, workers=None, chunksize=2000,
#                                         passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0,
#                                         eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None)
# 利用并行LDA加快处理速度

#输出主题
print(lda_model.print_topic(4))
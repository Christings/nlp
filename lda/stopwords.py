import re, nltk
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
import gensim
import pyLDAvis.gensim

# 1.打开文件
# f = open('../corpus/4095_01.csv', encoding='gbk')
f=["imaging","images"]

# 2.tokenize()函数
def tokenize(text):
    text = re.sub("[^a-zA-Z-]", " ", text)  # Removing numbers and punctuation
    text = re.sub(" +", " ", text)  # Removing extra white space
    # # text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b", " ", text)  # Removing very long words above 10 characters
    # text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b", " ", text)  # Removing single characters (e.g k, K)
    # text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text)
    tokens = nltk.word_tokenize(text.strip())  # 分词
    tokens = nltk.pos_tag(tokens)  # 词性标注
    # # Uncomment next line to use stemmer
    # tokens = stem_tokens(tokens, stemmer)
    return tokens
    # return text

for doc in f:
    x=tokenize(doc)
    print(x)


# # 3.停用词
# stopwords = STOPWORDS
# stopwords=list(stopwords)
# print(type(stopwords))
#
# # freq_words = ['http', 'https', 'amp', 'com', 'co', 'th', 'ji', 'all', 'sampla', 'pQ', 'lot', 'sir', 'pxr', 'ncbn', 'plz', 'qnI', 'way', 'sEkq', 'iAvvaPhp', 'zYGLz', 'tHMJdha']
# # for i in freq_words:
# #     stopwords.append(i)
#
# # 4.LDA分析
# def analyze(fileObj, outputName):
#     text_corpus = []
#     for doc in fileObj:
#         temp_doc = tokenize(doc.strip())
#         # print(temp_doc)
#         # print(len(temp_doc))
#         current_doc = []
#         for word in range(len(temp_doc)):
#             if temp_doc[word][0] not in stopwords and temp_doc[word][1] == 'NN':
#                 current_doc.append(temp_doc[word][0])
#         text_corpus.append(current_doc)
#
#     dictionary = corpora.Dictionary(text_corpus)
#     corpus = [dictionary.doc2bow(text) for text in text_corpus]
#     ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=8, id2word=dictionary, passes=20)
#     print("Topics for ", outputName, "\n")
#     for topics in ldamodel.print_topics(num_topics=8, num_words=5):
#         print(topics, "\n")
#
#     vis_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
#     pyLDAvis.show(vis_data)
#
#
# xx = open("../corpus/4095_01.txt")
# analyze(xx, "xx")

import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from gensim import corpora, models
import gensim
import pyLDAvis.gensim
import pyLDAvis
from gensim.parsing.preprocessing import STOPWORDS
from pprint import pprint

stemmer = SnowballStemmer("english")
# stemmer = PorterStemmer()


# 1.tokenize()函数--文本处理
def tokenize(text):
    # text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)  # Removing numbers and punctuation
    text = re.sub(" +", " ", text)  # Removing extra white space
    # text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b", " ", text)  # Removing very long words above 10 characters
    # text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b", " ", text)  # Removing single characters (e.g k, K)
    # text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text)
    tokens = nltk.word_tokenize(text.strip())  # 分词
    tokens = nltk.pos_tag(tokens)  # 词性标注
    print("tokens:", tokens)
    print(type(tokens))
    # Uncomment next line to use stemmer
    tokens = stem_tokens(tokens, stemmer)
    # print(type(tokens))
    return tokens


# 词干化处理和
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return list(stemmed)


# 2.停用词--nltk+gensim--去重
def get_stop_word_set():
    stopwordset = stopwords.words('english')
    # freq_words = ['http', 'https', 'amp', 'com', 'co', 'th', 'ji', 'all', 'sampla', 'pQ', 'lot', 'sir', 'pxr', 'ncbn', 'plz', 'qnI', 'way', 'sEkq', 'iAvvaPhp', 'zYGLz', 'tHMJdha']
    for i in STOPWORDS:
        stopwordset.append(i)
    stopwordset = list(set(stopwordset))
    print("stopword:", stopwordset)
    return stopwordset


# 3.LDA分析
def analyze(fileObj, outputName):
    stopwordset = get_stop_word_set()

    text_corpus = []
    for doc in fileObj:
        temp_doc = tokenize(doc.strip())
        current_doc = []
        for word in range(len(temp_doc)):
            if temp_doc[word][0] not in stopwordset:
                if temp_doc[word][1] != 'IN' or temp_doc[word][1] != 'CC' or temp_doc[word][1] != 'DT' or \
                        temp_doc[word][1] != 'CD':
                    current_doc.append(temp_doc[word][0])

        text_corpus.append(current_doc)
    print("Text=")
    pprint(text_corpus)

    # 构建生成文档的词典，每个词与一个整型索引值对应
    dictionary = corpora.Dictionary(text_corpus)

    # word must appear >10 times, and no more than 20% documents
    # dictionary.filter_extremes(no_below=20, no_above=0.1)

    # 词频统计，转换成空间向量格式/每个词对应的稀疏向量
    # 将文档（词表）转为词袋（BOW）格式(token_id, token_count)的二元组
    corpus = [dictionary.doc2bow(text) for text in text_corpus]

    # 统计tfidf
    tfidf = models.TfidfModel(corpus)

    print("LDA:")

    # 得到每个文本的tfidf向量，稀疏矩阵
    corpus_tfidf = tfidf[corpus]
    lda = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=15, id2word=dictionary, passes=60)

    # 使用并行LDA加快处理速度
    # lda=gensim.models.ldamulticore.LdaMulticore(corpus=None, num_topics=100, id2word=None, workers=None, chunksize=2000,
    #                                         passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0,
    #                                         eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None)

    # 每个文本对应的LDA向量，稀疏矩阵，元素值是隶属于对应叙述类的权重
    corpus_lda = lda[corpus_tfidf]
    print('Topics for ', outputName, '\n')
    for topics in lda.print_topics(num_topics=15, num_words=7):
        print(topics, "\n")

    vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # vis_data = pyLDAvis.gensim.prepare(corpus_lda, corpus_tfidf, dictionary)
    pyLDAvis.show(vis_data)
    # pyLDAvis.display(vis_data)
    # pyLDAvis.save_html(vis_data, outputName+'.html')


testDocument = open("../corpus/4095_01.txt")
analyze(testDocument, "testDocument")

# manojsinha = open('manojsinha.txt')
# analyze(manojsinha, 'manojsinha')
#
# # others = open('others.txt')
# # analyze(others, 'others')

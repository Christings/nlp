import re, nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import gensim
import pyLDAvis.gensim
import pyLDAvis
from gensim.parsing.preprocessing import STOPWORDS
from pprint import pprint
import time

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


# stemmer = PorterStemmer()


# 一.tokenize()函数--文本处理
def tokenize(text):
    text = text.lower()
    text = re.sub("[^a-zA-Z]", " ", text)  # Removing numbers and punctuation
    text = re.sub(" +", " ", text)  # Removing extra white space
    # text = re.sub("\\b[a-zA-Z0-9]{10,100}\\b", " ", text)  # Removing very long words above 10 characters
    text = re.sub("\\b[a-zA-Z0-9]{0,2}\\b", " ", text)  # Removing single characters (e.g k, K)
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text)
    tokens = nltk.word_tokenize(text.strip())  # 分词

    tokens = stem_tokens(tokens, stemmer)  # 词干化

    tokens = nltk.pos_tag(tokens)  # 词性标注
    # print("tokens:", tokens)
    # Uncomment next line to use stemmer

    # print(type(tokens))
    return tokens


# 二.词干化处理和词干还原
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        temp = stemmer.stem(item)
        temp = lemmatizer.lemmatize(temp)
        # stemmed.append(stemmer.stem(item))
        stemmed.append(temp)
    return stemmed


# 三.停用词--nltk+gensim--去重
def get_stop_word_set():
    stopwordset = stopwords.words('english')
    # freq_words = ['http', 'https', 'amp', 'com', 'co', 'th', 'ji', 'all', 'sampla', 'pQ', 'lot', 'sir', 'pxr', 'ncbn', 'plz', 'qnI', 'way', 'sEkq', 'iAvvaPhp', 'zYGLz', 'tHMJdha']
    for i in STOPWORDS:
        stopwordset.append(i)
    stopwordset = list(set(stopwordset))
    print("stopword:", stopwordset)
    return stopwordset


# 四.LDA分析
def analyze(fileObj, outputName):
    t_start = time.time()

    print("2.初始化停用词表 -------- ")
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
    print("读入语料数据并处理完成，用时%.3f秒" % (time.time() - t_start))
    length = len(text_corpus)
    print("文本树木：%d个" % length)
    print("------Text=------")
    pprint(text_corpus)

    # 构建生成文档的词典，每个词与一个整型索引值对应
    print("3.正在建立词典 -------- ")
    dictionary = corpora.Dictionary(text_corpus)
    print("dictionary:", dictionary)

    # word must appear >10 times, and no more than 20% documents
    # dictionary.filter_extremes(no_below=20, no_above=0.1)

    # 词频统计，转换成空间向量格式/每个词对应的稀疏向量
    # 将文档（词表）转为词袋（BOW）格式(token_id, token_count)的二元组
    print("4.正在计算文本向量 -------- ")
    corpus = [dictionary.doc2bow(text) for text in text_corpus]
    for line in corpus:
        print("line:", line)

    # 统计tfidf
    print("5.正在计算文档TF-IDF -------- ")
    t_start=time.time()
    tfidf = models.TfidfModel(corpus)

    # 得到每个文本的tfidf向量，稀疏矩阵
    corpus_tfidf = tfidf[corpus]
    print("建立文档TF-IDF完成，用时%.3f秒"%(time.time()-t_start))
    print("TF-IDF:")
    for temp in corpus_tfidf:
        print(temp)

    print("6.LDA模型拟合推断 -------- ")
    print("\nLDA Model:")
    num_topics = 15
    lda = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=60)
    print("LDA模型训练完成，训练时间为\t%.3f秒"%(time.time()-t_start))

    # 使用并行LDA加快处理速度
    # lda=gensim.models.ldamulticore.LdaMulticore(corpus=None, num_topics=100, id2word=None, workers=None, chunksize=2000,
    #                                         passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0,
    #                                         eval_every=10, iterations=50, gamma_threshold=0.001, random_state=None)

    # 每个文本对应的LDA向量，稀疏矩阵，元素值是隶属于对应叙述类的权重
    corpus_lda = lda[corpus_tfidf]

    print('Topics for ', outputName, '\n')
    for topics in lda.print_topics(num_topics=num_topics, num_words=7):
        print(topics, "\n")

    print("7.结果--文档的主题分布： -------- ")
    doc_topic = [temp for temp in corpus_lda]
    print("Document-Topic:\n")
    pprint(doc_topic)

    print("8.结果--每个主题词分布： -------- ")
    for topic_id in range(num_topics):
        print("Topic:", topic_id)
        pprint(lda.show_topic(topic_id))

    # 计算文档之间的相似性(通过tf-idf来进行计算)
    similarity = similarities.MatrixSimilarity(corpus_lda)
    print("Simmilarity:")
    pprint(list(similarity))

    vis_data = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
    # vis_data = pyLDAvis.gensim.prepare(corpus_lda, corpus_tfidf, dictionary)
    pyLDAvis.show(vis_data)
    # pyLDAvis.display(vis_data)
    # pyLDAvis.save_html(vis_data, outputName+'.html')


if __name__ == '__main__':
    # 开始的时间
    print("1.开始读入语料数据 -------- ")
    testDocument = open("../corpus/4095_01.txt")
    analyze(testDocument, "testDocument")

# manojsinha = open('manojsinha.txt')
# analyze(manojsinha, 'manojsinha')

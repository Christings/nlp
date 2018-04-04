# import spacy
from textblob import TextBlob
from gensim.parsing.preprocessing import STOPWORDS
# from nltk.stem import WordNetLemmatizer
from gensim import corpora,models,similarities
import pyLDAvis.gensim
import gensim

with open("../corpus/4095_01.txt", "r", encoding="utf-8") as f:
    processed_doc = []
    for doc in f:
        # print("doc:",doc)
        #     doc = doc.replace('\"', '.')
        #
        doc = TextBlob(doc)
        processed_words = []
        for words, tag in doc.lower().tags:
            # print("word:",words,tag)
            if words not in STOPWORDS and len(words) > 3:
                if tag != 'IN' or tag != 'CC' or tag != 'DT' or tag != 'TO':
                    words=words.singularize()
                    processed_words.append(words)
        # print("processed_words:", processed_words)
        processed_doc.append(processed_words)
    # print("processed_doc:", len(processed_doc))


dictionary=corpora.Dictionary(processed_doc)
print("dictionary:",dictionary)

corpus=[dictionary.doc2bow(text) for text in processed_doc]


tfidf=models.TfidfModel(corpus)

corpus_tfidf=tfidf[corpus]

num_topics=100
lda=gensim.models.ldamodel.LdaModel(corpus_tfidf,num_topics=num_topics,id2word=dictionary,passes=60)

vis_data=pyLDAvis.gensim.prepare(lda,corpus,dictionary)
pyLDAvis.show(vis_data)






# with open("../corpus/4095_01.txt", "r", encoding="utf-8") as f:
#     doc = f.read()
#     text = TextBlob(doc)
#
# processed_text = []
# for sentence in text.sentences:
#     print(sentence)
#     # print("11", sentence.noun_phrases)
#     processed_words = []
#     for num, word in enumerate(sentence.lower().words):
#         # print(num,word)
#         if word not in STOPWORDS and len(word) > 3:
#             tags = sentence.tags
#             if tags[num][1] != 'IN' or tags[num][1] != 'CC' or tags[num][1] != 'DT' or tags[num][1] != 'TO':
#                 processed_words.append(word)
#                 print("processed_words:", processed_words)
#     processed_text.append(processed_words)
# print("processed_text:", processed_text)

# nlp = spacy.load('en')
#
# with open("../corpus/4095_01.txt", "r", encoding="utf-8") as f:
#     # docs = f.read()
#     for doc in f:
#         doc = nlp(doc)
#         for num,sentence in enumerate(doc.sents):
#             print("Sentences {}:".format(num+1))
#             print(sentence)
#             print(" ")
#             for word in sentence:
#                 print("token:",word)
#
#         # for num,entity in enumerate(doc.ents):
#         #     print("Entity {}:".format(num+1),entity,"-",entity.label_)
#         #     print(" ")

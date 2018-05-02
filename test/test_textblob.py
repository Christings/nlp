from textblob import TextBlob
from gensim.parsing.preprocessing import STOPWORDS

with open("../corpus/4095_01.txt", "r", encoding="utf-8") as f:
    doc = f.read()
    text = TextBlob(doc)

processed_text = []
for sentence in text.sentences:
    print("sentence",sentence)
    print("noun_phrases", sentence.noun_phrases)
    processed_words = []
    for num, word in enumerate(sentence.lower().words):
        print(num,word)
        if word not in STOPWORDS and len(word) > 3:
            tags = sentence.tags
            if tags[num][1] != 'IN' or tags[num][1] != 'CC' or tags[num][1] != 'DT' or tags[num][1] != 'TO':
                processed_words.append(word)
                print("processed_words:", processed_words)
    processed_text.append(processed_words)
print("processed_text:", processed_text)


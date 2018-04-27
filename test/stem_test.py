from nltk import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import wordnet

# 熟悉nltk中一些常用词干化及还原方法

# 反义词
antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms)
# 同义词集
synonyms = []
for syn in wordnet.synsets("computer"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)

# 词条定义及举例
syn = wordnet.synsets("pain")
print(syn[0].definition())
print(syn[0].examples())

# 词干还原
t = WordNetLemmatizer()
print(t.lemmatize("imaging",pos="v"))

# 词干化
stemmer = PorterStemmer()
print(stemmer.stem("imaging"))


x = wordnet.morphy('needs')
print(x)

# 词干化
y = SnowballStemmer("english")
z = y.stem("imaging")
print(z)

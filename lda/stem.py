from nltk.corpus import wordnet as wn
from nltk import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.stanford_segmenter import StanfordSegmenter
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import wordnet

from nltk.corpus import wordnet

antonyms = []
for syn in wordnet.synsets("small"):
    for l in syn.lemmas():
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(antonyms)

synonyms = []
for syn in wordnet.synsets("computer"):
    for lemma in syn.lemmas():
        synonyms.append(lemma.name())
print(synonyms)

syn = wordnet.synsets("pain")
print(syn[0].definition())
print(syn[0].examples())

t = WordNetLemmatizer()
print(t.lemmatize("imaging",pos="v"))

stemmer = PorterStemmer()
print(stemmer.stem("imaging"))

x = wn.morphy('needs')
print(x)

y = SnowballStemmer("english")
z = y.stem("imaging")
print(z)

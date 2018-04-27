import spacy

nlp = spacy.load('en')

with open("../corpus/4095_01.txt", "r", encoding="utf-8") as f:
    for doc in f:
        doc = nlp(doc)
        for num, sentence in enumerate(doc.sents):
            print("Sentences {}:".format(num + 1))
            print(sentence)
            print(" ")
            for word in sentence:
                print("token:", word)

        for num, entity in enumerate(doc.ents):
            print("Entity {}:".format(num + 1), entity, "-", entity.label_)
            print(" ")

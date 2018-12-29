import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity

text1 = "AI is our friend and it has been friendly"
text2 = "AI and humans have always been friendly"

def extract_words(sentence):
    ignore_words = ['a']
    words = re.sub("[^\w]", " ",sentence ).split()
    words_cleaned = [w.lower() for w in words if w not in ignore_words]
    return words_cleaned

def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = extract_words(sentence)
        words.extend(w)

    words = sorted(list(set(words)))
    return words

def bag_of_words(sentence, words):
    sentence_words = extract_words(sentence)
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1
    return np.array(bag)

sentences = [text1, text2]
vocabulary = tokenize_sentences(sentences)


# print(bag_of_words(sentences[0], vocabulary))
vec = []
vec.append(bag_of_words(sentences[0], vocabulary))
vec.append(bag_of_words(sentences[1], vocabulary))

print(vocabulary)
print(extract_words(sentences[0]))
print(vec[0])
print(extract_words(sentences[1]))
print(vec[1])

vec[0] = vec[0].reshape(1, -1)
vec[1] = vec[1].reshape(1, -1)
sim = cosine_similarity(vec[0], vec[1])

print(sim)

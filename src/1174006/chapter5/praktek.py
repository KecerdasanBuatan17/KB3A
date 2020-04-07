import gensim, logging
gmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
gmodel.similarity('wash', 'motor')

from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
# unsupervised training data
import re
import os
unsup_sentences = []

def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) # strip html tags
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent) # remove apostrophes
    sent = re.sub(r'\W', ' ', sent) # remove punctuation
    sent = re.sub(r'\s+', ' ', sent) # remove repeated spaces
    sent = sent.strip()
    return sent.split()

# source: http://www.cs.cornell.edu/people/pabo/movie-review-data/
for dirname in ["txt_sentoken/pos", "txt_sentoken/neg"]:
    for fname in sorted(os.listdir(dirname)):
        if fname[-4:] == '.txt':
            with open(dirname + "/" + fname, encoding='UTF-8') as f:
                for i, sent in enumerate(f):
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(words, ["%s/%s-%d" % (dirname, fname, i)]))
                    
import random
class PermuteSentences(object):
    def __init__(self, sents):
        self.sents = sents
        
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent
            
permuter = PermuteSentences(unsup_sentences)
model = Doc2Vec(permuter, dm=0, hs=1, size=50)

# done with training, free up some memory
model.delete_temporary_training_data(keep_inference=True)

model.save('reviews.d2v')

model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
    [model.infer_vector(extract_words("Highly recommended."))],
    [model.infer_vector(extract_words("Service sucks."))])

sentences = []
sentvecs = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]: 
    with open("sentiment labelled sentences/%s_labelled.txt" % fname, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split('\t')
            sentences.append(line_split[0])
            words = extract_words(line_split[0])
            sentvecs.append(model.infer_vector(words, steps=10)) # create a vector for this document
            sentiments.append(int(line_split[1]))
# shuffle sentences, sentvecs, sentiments together
combined = list(zip(sentences, sentvecs, sentiments))
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier()

scores = cross_val_score(clf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)

scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)
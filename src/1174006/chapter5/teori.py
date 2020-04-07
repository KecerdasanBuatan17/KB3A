import gensim, logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec

with open("sentiment labelled sentences/yelp_labelled.txt") as f:
    for item_no, line in enumerate(f):
        print(item_no, line)
        
sentences = []
sentiments = []
with open("sentiment labelled sentences/yelp_labelled.txt") as f:
    for item_no, line in enumerate(f):
        line_split = line.strip().split('\t')
        sentences.append((line_split[0], "yelp_%d" % item_no))
        sentiments.append(int(line_split[1]))
        
len(sentences), sentences

import re
sentences = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]:
    with open("sentiment labelled sentences/%s_labelled.txt" % fname) as f:
        for item_no, line in enumerate(f):
            line_split = line.strip().split('\t')
            sent = line_split[0].lower()
            sent = re.sub(r'\'', '', sent)
            sent = re.sub(r'\W', ' ', sent)
            sent = re.sub(r'\s+', ' ', sent).strip()
            #sentences.append(LabeledSentence(sent.split(), ["%s_%d" % (fname, item_no)]))
            sentences.append(TaggedDocument(sent.split(), ["%s_%d" % (fname, item_no)]))
            sentiments.append(int(line_split[1]))
            
sentences

import random
class PermuteSentences(object):
    def __iter__(self):
        shuffled = list(sentences)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent
permuter = PermuteSentences()
        
model = Doc2Vec(permuter, min_count=1)

#model.most_similar('tasty')
model.wv.most_similar('tasty')
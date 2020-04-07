# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:01:08 2020

@author: DZ
"""
#%% Soal no 1

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dz_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

#%%
dz_model['love']
#%%
dz_model['faith']
#%%
dz_model['fall']
#%%
dz_model['sick']
#%%
dz_model['clear']
#%%
dz_model['shine']
#%%
dz_model['bag']
#%%
dz_model['car']
#%%
dz_model['wash']
#%%
dz_model['motor']
#%%
dz_model['cycle']
#%%
dz_model.similarity('wash', 'clear')
#%%
dz_model.similarity('bag', 'love')
#%%
dz_model.similarity('motor', 'car')
#%%
dz_model.similarity('sick', 'faith')
#%%
dz_model.similarity('cycle', 'shine')

#%% Soal no 2

import re 
    
test_string = "A,B"
print ("A atau B : " +  test_string) 
res = re.findall(r'\w+', test_string) 

print ("The list of words is : " +  str(res)) 

#%%
    
import random

sent_matrix = [ ['1', '2'],
                ['3', '4'],
                ['5', '6'],
                ['7', '8']
              ]

result = ""
for elem in sent_matrix:
    result += random.choice(elem) + " "

print (result)

#%% Soal no 3

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

## Exapmple document (list of sentences)
doc = ["1","2","3","4","5","6"]

tokenized_doc = ['numb']
tokenized_doc

print(doc)

#%%
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
tagged_data
## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
# Save trained doc2vec model
model.save("cek.model")
## Load saved doc2vec model
model= Doc2Vec.load("cek.model")
## Print model vocabulary
model.wv.vocab


#%% Soal no 4
import re
import os
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("acldb/" + dirname)):
        if fname[-4:] == '.txt':
            with open("aclImdb/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = (sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))
                

#%% soal no 5
                
mute = (unsup_sentences)

model.delete_temporary_training_data(keep_inference=True)
                
                
#%% soal no 6                
                
model.save('dz.d2v')

model.delete_temporary_training_data(keep_inference=True)             

#%% soal no 7infer vectors with different 'steps' parameters
    infervec1 = model.infer_vector(doc, alpha=0.025, min_alpha=0.01, steps=1)
    infervec2 = model.infer_vector(doc, alpha=0.025, min_alpha=0.01, steps=10)
    infervec3 = model.infer_vector(doc, alpha=0.025, min_alpha=0.01, steps=100)
    infervec4 = model.infer_vector(doc, alpha=0.025, min_alpha=0.01, steps=1000)

#%% soal no 8

import re
import math
from collections import Counter


def get_cosine(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    return Counter(words)


def get_result(content_a, content_b):
    text1 = content_a
    text2 = content_b

    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)

    cosine_result = get_cosine(vector1, vector2)
    return cosine_result


print (get_result)

#%% soal no 9

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))



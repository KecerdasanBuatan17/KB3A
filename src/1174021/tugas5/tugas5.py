# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:01:08 2020

@author: FAHMI-PC
"""
#%% Soal no 1

import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
fahmi_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=500000)

#%%
fahmi_model['love']
#%%
fahmi_model['faith']
#%%
fahmi_model['fall']
#%%
fahmi_model['sick']
#%%
fahmi_model['clear']
#%%
fahmi_model['shine']
#%%
fahmi_model['bag']
#%%
fahmi_model['car']
#%%
fahmi_model['wash']
#%%
fahmi_model['motor']
#%%
fahmi_model['cycle']
#%%
fahmi_model.similarity('wash', 'clear')
#%%
fahmi_model.similarity('bag', 'love')
#%%
fahmi_model.similarity('motor', 'car')
#%%
fahmi_model.similarity('sick', 'faith')
#%%
fahmi_model.similarity('cycle', 'shine')

#%% Soal no 2

import re 
    
test_string = "Muhammad Fahmi,    Anak yang ganteng!!!"
print ("Fakta Membuktikan : " +  test_string) 
res = re.findall(r'\w+', test_string) 

print ("The list of words is : " +  str(res)) 

#%%
    
import random

sent_matrix = [ ['Fahmi', 'Pamungkas'],
                ['Adalah', 'Anak'],
                ['Yang', 'Baik'],
                ['Dan', 'Ganteng']
              ]

result = ""
for elem in sent_matrix:
    result += random.choice(elem) + " "

print (result)

#%% Soal no 3

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

## Exapmple document (list of sentences)
doc = ["I love data science",
        "I love coding in python",
        "I love Poltekpos",
        "This is a good phone",
        "This is a good TV",
        "This is a good laptop"]

tokenized_doc = ['love']
tokenized_doc

print(doc)

#%%
tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(tokenized_doc)]
tagged_data
## Train doc2vec model
model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, workers=4, epochs = 100)
# Save trained doc2vec model
model.save("test_doc2vec.model")
## Load saved doc2vec model
model= Doc2Vec.load("test_doc2vec.model")
## Print model vocabulary
model.wv.vocab


#%% Soal no 4
import re
import os
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir("aclImdb/" + dirname)):
        if fname[-4:] == '.txt':
            with open("aclImdb/" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = (sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))
                

#%% soal no 5
                
#Pengacakan data
mute = (unsup_sentences)

#Pembersihan data
model.delete_temporary_training_data(keep_inference=True)
                
                
#%% soal no 6                
                
#Save data
model.save('fahmi.d2v')

#Delete temporary data
model.delete_temporary_training_data(keep_inference=True)             

#%% soal no 7

model.infer_vector(extract_words("Hayoo mau copas yaa"))

#%% soal no 8

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
        [model.infer_vector(extract_words("Yahhh mau copas lagi ya"))],
        [model.infer_vector(extract_words("Tobat guys tobat"))])

#%% soal no 9

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
print(cross_val_score(lasso, X, y, cv=3))



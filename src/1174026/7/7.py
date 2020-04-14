# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:47:27 2020

@author: User
"""

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoder.fit(integer_encoded)
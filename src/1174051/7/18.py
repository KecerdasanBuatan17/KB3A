# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:53:35 2020

@author: User
"""

import keras.models
model2 = keras.models.load_model("mathsymbols.model")
print(model2.summary())
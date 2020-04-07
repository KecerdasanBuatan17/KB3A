# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:33:05 2020

@author: Sujadi
"""

model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
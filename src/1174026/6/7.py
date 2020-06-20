# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:27:00 2020

@author: User
"""

model = Sequential([
    Dense(100, input_dim=np.shape(train_input)[1]),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])
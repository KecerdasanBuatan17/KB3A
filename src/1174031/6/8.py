# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 23:27:20 2020

@author: User
"""

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
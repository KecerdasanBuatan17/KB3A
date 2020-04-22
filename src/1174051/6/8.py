# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:33:18 2020

@author: Sujadi
"""

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:50:12 2020

@author: User
"""

model.fit(np.concatenate((train_input, test_input)),
          np.concatenate((train_output, test_output)),
          batch_size=32, epochs=10, verbose=2)
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 23:47:27 2020

@author: User
"""

import numpy as np

train_input = np.asarray(list(map(lambda row: row[2], train)))
test_input = np.asarray(list(map(lambda row: row[2], test)))

train_output = np.asarray(list(map(lambda row: row[1], train)))
test_output = np.asarray(list(map(lambda row: row[1], test)))
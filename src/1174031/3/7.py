# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:17:15 2020

@author: ACER
"""

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
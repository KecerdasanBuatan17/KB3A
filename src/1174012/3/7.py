# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:32:46 2020

@author: Damara
"""

from sklearn.metrics import confusion_matrix
y_true = [2, 0, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]
confusion_matrix(y_true, y_pred)
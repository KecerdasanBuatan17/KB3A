# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 13:33:59 2020

@author: Damara
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lontong, pecel_att, pecel_pass, cv=5)
# show average score and +/- two standard deviations away
#(covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))